"""Modelagem estatística de complexidade empírica (Etapa 2 – Dev B).

Ajusta curvas candidatas via ``scipy.optimize.curve_fit`` para tempo e pico de
memória, tanto por algoritmo (``runs.jsonl``) quanto por subrotina
(``profile.jsonl``):

  * ``power``       : a · n^k        (chute inicial por regressão log-log)
  * ``quadratic``   : a · n² + b
  * ``cubic``       : a · n³ + b
  * ``exponential`` : a · c^n        (chute inicial por regressão log-linear)

Para cada modelo reporta R², RMSE e AIC (calculado na escala original de y para
manter comparabilidade direta entre candidatos) e seleciona o melhor por menor AIC.

Uso:
    python -m src.analysis.complexity \
        --runs results/runs.jsonl --profile results/profile.jsonl --out results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.tsp.common import STATUS_OK

from ._io import ensure_output_dirs, load_table


@dataclass(frozen=True)
class FitResult:
    model: str
    params: dict[str, float]
    r2: float
    rmse: float
    aic: float
    n_points: int
    formula: str


def _power_fn(n: np.ndarray, a: float, k: float) -> np.ndarray:
    return a * np.power(n, k)


def _quad_fn(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * (n ** 2) + b


def _cubic_fn(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * (n ** 3) + b


def _exp_rate_fn(n: np.ndarray, a: float, r: float) -> np.ndarray:
    return a * np.exp(r * n)


def _eval_model(model: str, n: np.ndarray, params: dict[str, float]) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    if model == "power":
        return _power_fn(n, params["a"], params["k"])
    if model == "quadratic":
        return _quad_fn(n, params["a"], params["b"])
    if model == "cubic":
        return _cubic_fn(n, params["a"], params["b"])
    if model == "exponential":
        return params["a"] * np.power(params["c"], n)
    raise ValueError(f"Modelo desconhecido: {model}")


def _metrics(y: np.ndarray, y_hat: np.ndarray, n_params: int) -> tuple[float, float, float]:
    """Calcula (R², RMSE, AIC) na escala original de ``y``.

    AIC gaussiano: m · ln(RSS / m) + 2 · (n_params + 1), onde o +1 conta σ².
    """
    m = len(y)
    resid = y - y_hat
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    if tss > 0.0:
        r2 = 1.0 - rss / tss
    else:
        r2 = 1.0 if rss == 0.0 else 0.0
    rmse = float(np.sqrt(rss / m))
    mse = max(rss / m, 1e-300)
    aic = float(m * np.log(mse) + 2 * (n_params + 1))
    return r2, rmse, aic


def _fit_single(
    model: str,
    n: np.ndarray,
    y: np.ndarray,
) -> FitResult | None:
    """Ajusta um único modelo candidato a ``(n, y)`` usando ``curve_fit``."""
    n = np.asarray(n, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(n) < 3 or np.unique(n).size < 3:
        return None

    pos = (n > 0) & (y > 0)
    try:
        if model == "power":
            if np.count_nonzero(pos) >= 2:
                slope, intercept = np.polyfit(np.log(n[pos]), np.log(y[pos]), 1)
                a0 = float(np.clip(np.exp(intercept), 1e-15, 1e15))
                k0 = float(np.clip(slope, -4.0, 15.0))
            else:
                a0, k0 = max(float(np.mean(y)), 1e-9), 1.0
            popt, _ = curve_fit(
                _power_fn,
                n,
                y,
                p0=[a0, k0],
                bounds=([0.0, -5.0], [np.inf, 20.0]),
                maxfev=10_000,
            )
            a, k = float(popt[0]), float(popt[1])
            y_hat = _power_fn(n, a, k)
            params = {"a": a, "k": k}
            formula = f"{a:.2e} * n^{k:.2f}"

        elif model == "quadratic":
            slope, intercept = np.polyfit(n ** 2, y, 1)
            a0 = max(float(slope), 1e-15)
            b0 = float(intercept)
            popt, _ = curve_fit(
                _quad_fn,
                n,
                y,
                p0=[a0, b0],
                bounds=([0.0, -np.inf], [np.inf, np.inf]),
                maxfev=10_000,
            )
            a, b = float(popt[0]), float(popt[1])
            y_hat = _quad_fn(n, a, b)
            params = {"a": a, "b": b}
            formula = f"{a:.2e} * n^2 + {b:.2e}"

        elif model == "cubic":
            slope, intercept = np.polyfit(n ** 3, y, 1)
            a0 = max(float(slope), 1e-15)
            b0 = float(intercept)
            popt, _ = curve_fit(
                _cubic_fn,
                n,
                y,
                p0=[a0, b0],
                bounds=([0.0, -np.inf], [np.inf, np.inf]),
                maxfev=10_000,
            )
            a, b = float(popt[0]), float(popt[1])
            y_hat = _cubic_fn(n, a, b)
            params = {"a": a, "b": b}
            formula = f"{a:.2e} * n^3 + {b:.2e}"

        elif model == "exponential":
            max_n = float(np.max(n))
            r_max = min(5.0, 500.0 / max(max_n, 1.0))
            if np.count_nonzero(pos) >= 2:
                r0, intercept = np.polyfit(n[pos], np.log(y[pos]), 1)
                r0 = float(np.clip(r0, 1e-6, r_max * 0.9))
                a0 = float(np.clip(np.exp(intercept), 1e-15, 1e15))
            else:
                a0, r0 = max(float(np.min(y)), 1e-12), min(0.05, r_max * 0.5)
            popt, _ = curve_fit(
                _exp_rate_fn,
                n,
                y,
                p0=[a0, r0],
                bounds=([0.0, 0.0], [np.inf, r_max]),
                maxfev=10_000,
            )
            a, r = float(popt[0]), float(popt[1])
            c = float(np.exp(r))
            y_hat = _exp_rate_fn(n, a, r)
            params = {"a": a, "c": c}
            formula = f"{a:.2e} * {c:.4f}^n"

        else:
            raise ValueError(f"Modelo desconhecido: {model}")

    except Exception:
        return None

    if not np.all(np.isfinite(y_hat)):
        return None
    r2, rmse, aic = _metrics(y, y_hat, n_params=2)
    return FitResult(
        model=model,
        params=params,
        r2=r2,
        rmse=rmse,
        aic=aic,
        n_points=len(n),
        formula=formula,
    )


CANDIDATE_MODELS: tuple[str, ...] = ("power", "quadratic", "cubic", "exponential")


def fit_candidates(n: np.ndarray, y: np.ndarray) -> tuple[list[FitResult], FitResult | None]:
    """Ajusta todos os modelos candidatos e escolhe o melhor por menor AIC."""
    results: list[FitResult] = []
    for name in CANDIDATE_MODELS:
        res = _fit_single(name, n, y)
        if res is not None:
            results.append(res)
    if not results:
        return [], None
    best = min(results, key=lambda r: r.aic)
    return results, best


def _aggregate_series(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """Agrega por mediana em cada ``(grupo..., n)``, descartando valores ausentes."""
    sub = df.dropna(subset=["n", value_col]).copy()
    if sub.empty:
        return pd.DataFrame(columns=[*group_cols, "n", value_col])
    sub["n"] = pd.to_numeric(sub["n"], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=["n", value_col])
    return sub.groupby([*group_cols, "n"], as_index=False)[value_col].median()


def _fit_grouped(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str,
    metric_label: str,
) -> tuple[pd.DataFrame, dict[tuple, tuple[np.ndarray, np.ndarray, list[FitResult], FitResult]]]:
    agg = _aggregate_series(df, group_cols, metric_col)
    rows: list[dict] = []
    fitted_map: dict[tuple, tuple[np.ndarray, np.ndarray, list[FitResult], FitResult]] = {}

    if agg.empty:
        return pd.DataFrame(), fitted_map

    for key, grp in agg.groupby(group_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        grp = grp.sort_values("n")
        n_vals = grp["n"].to_numpy(dtype=np.float64)
        y_vals = grp[metric_col].to_numpy(dtype=np.float64)
        results, best = fit_candidates(n_vals, y_vals)
        if best is None:
            continue
        fitted_map[key_tuple] = (n_vals, y_vals, results, best)
        base = dict(zip(group_cols, key_tuple))
        for res in results:
            rows.append(
                {
                    **base,
                    "metric": metric_label,
                    "model": res.model,
                    "is_best": res.model == best.model,
                    "r2": res.r2,
                    "rmse": res.rmse,
                    "aic": res.aic,
                    "n_points": res.n_points,
                    "formula": res.formula,
                    "params_json": json.dumps(res.params),
                }
            )
    return pd.DataFrame(rows), fitted_map


def _plot_algorithm_fits(
    fitted_map: dict[tuple, tuple[np.ndarray, np.ndarray, list[FitResult], FitResult]],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    if not fitted_map:
        return
    keys = sorted(fitted_map)
    ncols = min(3, len(keys))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    palette = {"power": "#1f77b4", "quadratic": "#2ca02c", "cubic": "#ff7f0e", "exponential": "#d62728"}

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        n_vals, y_vals, results, best = fitted_map[key]
        ax.scatter(n_vals, y_vals, color="black", s=26, zorder=5, label="mediana empírica")
        n_grid = np.linspace(float(n_vals.min()), float(n_vals.max()), 200)
        for res in results:
            y_grid = _eval_model(res.model, n_grid, res.params)
            is_best = res.model == best.model
            lw = 2.2 if is_best else 1.1
            ls = "-" if is_best else "--"
            alpha = 0.95 if is_best else 0.55
            tag = "★ " if is_best else ""
            ax.plot(
                n_grid,
                y_grid,
                color=palette.get(res.model, "gray"),
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
                label=f"{tag}{res.model} (R²={res.r2:.3f}, AIC={res.aic:.1f})",
            )
        label = " / ".join(str(k) for k in key)
        ax.set_title(f"{label}\nmelhor: {best.formula}", fontsize=9.5)
        ax.set_xlabel("N (cidades)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")

    for idx in range(len(keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_subroutine_breakdown(
    profile_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    agg = _aggregate_series(profile_df, ["algorithm", "section"], metric_col)
    if agg.empty:
        return
    algs = sorted(agg["algorithm"].unique())
    ncols = min(2, len(algs))
    nrows = int(np.ceil(len(algs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.0 * nrows), squeeze=False)

    for idx, alg in enumerate(algs):
        ax = axes[idx // ncols][idx % ncols]
        sub = agg[agg["algorithm"] == alg]
        for sec, grp in sub.groupby("section"):
            grp = grp.sort_values("n")
            ax.plot(grp["n"], grp[metric_col], marker="o", linewidth=1.8, markersize=4, label=str(sec))
        ax.set_title(f"Subrotinas – {alg}", fontsize=10.5)
        ax.set_xlabel("N (cidades)")
        ax.set_ylabel(ylabel)
        if (sub[metric_col] > 0).all():
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8.5, loc="best")

    for idx in range(len(algs), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_complexity(
    runs_df: pd.DataFrame | None = None,
    profile_df: pd.DataFrame | None = None,
    out_dir: str | Path = "results",
) -> dict[str, pd.DataFrame]:
    """Executa o ajuste de complexidade empírica e grava tabelas e figuras."""
    fig_dir, tab_dir = ensure_output_dirs(out_dir)
    outputs: dict[str, pd.DataFrame] = {}

    ok_profile_keys: set[tuple] | None = None
    if runs_df is not None and not runs_df.empty:
        ok_df = runs_df.copy()
        if "status" in ok_df.columns:
            ok_df = ok_df[ok_df["status"] == STATUS_OK]

        if "warm_start" in ok_df.columns:
            ws = ok_df["warm_start"].fillna("").astype(str).str.strip()
            ok_df["algorithm"] = np.where(
                ws != "",
                ok_df["algorithm"].astype(str) + ":" + ws,
                ok_df["algorithm"].astype(str),
            )

        if {"instance", "algorithm"} <= set(ok_df.columns):
            ok_profile_keys = set(zip(ok_df["instance"].astype(str), ok_df["algorithm"].astype(str)))

        alg_tables: list[pd.DataFrame] = []
        if "time_total_s" in ok_df.columns:
            t_df, t_map = _fit_grouped(ok_df, ["algorithm"], "time_total_s", "time_total_s")
            if not t_df.empty:
                alg_tables.append(t_df)
                _plot_algorithm_fits(
                    t_map,
                    ylabel="Tempo total (s)",
                    title="Modelagem de Complexidade Temporal por Algoritmo",
                    out_path=fig_dir / "complexity_algorithms_time.png",
                )

        if "peak_mem_kb" in ok_df.columns:
            m_df, m_map = _fit_grouped(ok_df, ["algorithm"], "peak_mem_kb", "peak_mem_kb")
            if not m_df.empty:
                alg_tables.append(m_df)
                _plot_algorithm_fits(
                    m_map,
                    ylabel="Pico de memória (KB)",
                    title="Modelagem de Complexidade Espacial por Algoritmo",
                    out_path=fig_dir / "complexity_algorithms_memory.png",
                )

        alg_all = pd.concat(alg_tables, ignore_index=True) if alg_tables else pd.DataFrame()
        alg_all.to_csv(tab_dir / "complexity_algorithms.csv", index=False)
        outputs["algorithms"] = alg_all

    if profile_df is not None and not profile_df.empty:
        p_df = profile_df.copy()
        if ok_profile_keys is not None and {"instance", "algorithm"} <= set(p_df.columns):
            # Mantém apenas subrotinas de execuções OK (evita tempos truncados de TIMEOUT)
            known_algs = {a for _, a in ok_profile_keys}
            mask_ok = [
                (a not in known_algs) or ((i, a) in ok_profile_keys)
                for i, a in zip(p_df["instance"].astype(str), p_df["algorithm"].astype(str))
            ]
            p_df = p_df[mask_ok]

        sub_tables: list[pd.DataFrame] = []
        if "time_s" in p_df.columns:
            st_df, st_map = _fit_grouped(p_df, ["algorithm", "section"], "time_s", "time_s")
            if not st_df.empty:
                sub_tables.append(st_df)
                _plot_algorithm_fits(
                    st_map,
                    ylabel="Tempo da subrotina (s)",
                    title="Complexidade Temporal por Subrotina",
                    out_path=fig_dir / "complexity_subroutines_time_fits.png",
                )
            _plot_subroutine_breakdown(
                p_df,
                metric_col="time_s",
                ylabel="Tempo da subrotina (s)",
                title="Decomposição de Tempo por Subrotina (escala log)",
                out_path=fig_dir / "complexity_subroutines_time.png",
            )

        if "peak_kb" in p_df.columns:
            sm_df, sm_map = _fit_grouped(p_df, ["algorithm", "section"], "peak_kb", "peak_kb")
            if not sm_df.empty:
                sub_tables.append(sm_df)
                _plot_algorithm_fits(
                    sm_map,
                    ylabel="Pico de memória da subrotina (KB)",
                    title="Complexidade Espacial por Subrotina",
                    out_path=fig_dir / "complexity_subroutines_memory_fits.png",
                )
            _plot_subroutine_breakdown(
                p_df,
                metric_col="peak_kb",
                ylabel="Pico de memória (KB)",
                title="Decomposição de Memória por Subrotina (escala log)",
                out_path=fig_dir / "complexity_subroutines_memory.png",
            )

        sub_all = pd.concat(sub_tables, ignore_index=True) if sub_tables else pd.DataFrame()
        sub_all.to_csv(tab_dir / "complexity_subroutines.csv", index=False)
        outputs["subroutines"] = sub_all

    return outputs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ajuste de curvas de complexidade empírica (tempo e memória)")
    ap.add_argument("--runs", type=Path, default=Path("results/runs.jsonl"))
    ap.add_argument("--profile", type=Path, default=Path("results/profile.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    runs_df = load_table(args.runs) if args.runs.exists() else None
    profile_df = load_table(args.profile) if args.profile.exists() else None
    if runs_df is None and profile_df is None:
        raise SystemExit("Informe ao menos --runs ou --profile existente.")

    outputs = analyze_complexity(runs_df, profile_df, args.out)
    for name, df in outputs.items():
        n_best = int(df["is_best"].sum()) if "is_best" in df.columns else 0
        print(f"[complexity] {name}: {len(df)} ajustes ({n_best} melhores por AIC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
