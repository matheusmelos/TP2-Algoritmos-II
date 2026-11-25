"""Análise do impacto do warm-start no Branch-and-Bound (Etapa 2 – Dev B).

Gera tabelas e figuras a partir de ``runs.jsonl`` (ou ``.csv``):
  1. Nós expandidos e podados por estratégia de warm-start em função de N (log).
  2. Trade-off entre ``time_heuristic_s`` e tempo de busca do B&B, indicando se
     a heurística inicial se paga frente ao cold start (``trivial``).
  3. Ponto de colapso: menor N em que começam os ``TIMEOUT`` por estratégia.

Uso:
    python -m src.analysis.warmstart --runs results/runs.jsonl --out results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tsp.common import STATUS_OK, STATUS_TIMEOUT

from ._io import ensure_output_dirs, load_table

WARM_ORDER = ("trivial", "nn", "hull", "tatt", "christofides")
WARM_COLORS = {
    "trivial": "#7f7f7f",
    "nn": "#1f77b4",
    "hull": "#2ca02c",
    "tatt": "#ff7f0e",
    "christofides": "#d62728",
}


def _ordered_warm_starts(values: pd.Series) -> list[str]:
    present = list(dict.fromkeys(values.dropna().astype(str)))
    ordered = [w for w in WARM_ORDER if w in present]
    ordered.extend(sorted(w for w in present if w not in ordered))
    return ordered


def _select_bnb_runs(runs_df: pd.DataFrame, algorithm: str | None = None) -> pd.DataFrame:
    df = runs_df.copy()
    if algorithm is not None:
        df = df[df["algorithm"] == algorithm]
    else:
        has_bnb = df["algorithm"].astype(str).str.startswith("bnb")
        has_ws = df["warm_start"].notna() & (df["warm_start"].astype(str) != "")
        df = df[has_bnb | has_ws]

    if df.empty:
        return df

    for col in ("n", "time_total_s", "time_heuristic_s", "expanded", "pruned", "generated", "max_depth", "max_frontier"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time_heuristic_s"] = df["time_heuristic_s"].fillna(0.0)
    df["time_bnb_s"] = np.maximum(0.0, df["time_total_s"] - df["time_heuristic_s"])
    df["prune_ratio"] = np.where(df["generated"] > 0, df["pruned"] / df["generated"], np.nan)
    return df


def _build_summary_and_tradeoff(bnb_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega por (algorithm, instance, n, warm_start) e calcula ganho vs trivial."""
    if bnb_df.empty:
        return pd.DataFrame()

    # Se houver mais de uma repetição, agrega por mediana nas colunas numéricas e pior status
    def _agg_status(s: pd.Series) -> str:
        vals = set(s.dropna())
        if STATUS_TIMEOUT in vals:
            return STATUS_TIMEOUT
        if STATUS_OK in vals:
            return STATUS_OK
        return str(next(iter(vals))) if vals else "UNKNOWN"

    agg = (
        bnb_df.groupby(["algorithm", "instance", "n", "warm_start"], as_index=False)
        .agg(
            status=("status", _agg_status),
            time_total_s=("time_total_s", "median"),
            time_heuristic_s=("time_heuristic_s", "median"),
            time_bnb_s=("time_bnb_s", "median"),
            expanded=("expanded", "median"),
            pruned=("pruned", "median"),
            generated=("generated", "median"),
            prune_ratio=("prune_ratio", "median"),
            max_depth=("max_depth", "max"),
            max_frontier=("max_frontier", "max"),
        )
        .sort_values(["algorithm", "n", "instance", "warm_start"])
    )

    baseline = agg[agg["warm_start"] == "trivial"][
        ["algorithm", "instance", "time_total_s", "time_bnb_s", "expanded"]
    ].rename(
        columns={
            "time_total_s": "trivial_time_total_s",
            "time_bnb_s": "trivial_time_bnb_s",
            "expanded": "trivial_expanded",
        }
    )

    merged = agg.merge(baseline, on=["algorithm", "instance"], how="left")
    merged["bnb_time_saved_s"] = merged["trivial_time_bnb_s"] - merged["time_bnb_s"]
    merged["net_time_saved_s"] = merged["trivial_time_total_s"] - merged["time_total_s"]
    merged["speedup_vs_trivial"] = np.where(
        merged["time_total_s"] > 0,
        merged["trivial_time_total_s"] / merged["time_total_s"],
        np.nan,
    )
    merged["expanded_reduction_pct"] = np.where(
        merged["trivial_expanded"] > 0,
        100.0 * (1.0 - merged["expanded"] / merged["trivial_expanded"]),
        np.nan,
    )
    merged["pays_off"] = np.where(
        merged["net_time_saved_s"].notna(),
        merged["net_time_saved_s"] > 0.0,
        np.nan,
    )
    return merged


def _build_collapse_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Identifica o ponto de colapso (menor N com TIMEOUT) por estratégia."""
    if summary_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (alg, ws), grp in summary_df.groupby(["algorithm", "warm_start"]):
        ok = grp[grp["status"] == STATUS_OK]
        to = grp[grp["status"] == STATUS_TIMEOUT]
        rows.append(
            {
                "algorithm": alg,
                "warm_start": ws,
                "total_instances": int(len(grp)),
                "ok_instances": int(len(ok)),
                "timeout_instances": int(len(to)),
                "max_n_ok": int(ok["n"].max()) if not ok.empty else None,
                "first_timeout_n": int(to["n"].min()) if not to.empty else None,
                "timeout_names": ",".join(to.sort_values("n")["instance"].astype(str)),
            }
        )
    out = pd.DataFrame(rows)
    order_map = {w: i for i, w in enumerate(WARM_ORDER)}
    out["_ord"] = out["warm_start"].map(lambda w: order_map.get(w, 99))
    return out.sort_values(["algorithm", "_ord", "warm_start"]).drop(columns=["_ord"])


def _plot_nodes(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    warm_starts = _ordered_warm_starts(summary_df["warm_start"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for ax, col, ylabel in (
        (axes[0], "expanded", "Nós expandidos (escala log)"),
        (axes[1], "pruned", "Nós podados (escala log)"),
    ):
        for ws in warm_starts:
            sub = summary_df[summary_df["warm_start"] == ws].sort_values("n")
            y = np.maximum(sub[col].to_numpy(dtype=np.float64), 1.0)
            n = sub["n"].to_numpy(dtype=np.float64)
            color = WARM_COLORS.get(ws, None)
            ax.plot(n, y, marker="o", markersize=4.5, linewidth=1.8, color=color, label=ws)

            to_mask = (sub["status"] == STATUS_TIMEOUT).to_numpy()
            if np.any(to_mask):
                ax.scatter(
                    n[to_mask],
                    y[to_mask],
                    marker="x",
                    s=65,
                    linewidths=2.0,
                    color=color,
                    zorder=6,
                )
        ax.set_yscale("log")
        ax.set_xlabel("N (cidades)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="Warm-start", fontsize=8.5)

    axes[0].set_title("Nós Expandidos por Estratégia (× = TIMEOUT)")
    axes[1].set_title("Nós Podados por Estratégia (× = TIMEOUT)")
    fig.suptitle("Impacto do Warm-Start na Árvore do Branch-and-Bound", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_tradeoff(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    warm_starts = _ordered_warm_starts(summary_df["warm_start"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # Painel esquerdo: tempo da heurística vs tempo total do B&B
    ax = axes[0]
    for ws in warm_starts:
        sub = summary_df[summary_df["warm_start"] == ws].sort_values("n")
        color = WARM_COLORS.get(ws, None)
        n = sub["n"].to_numpy(dtype=np.float64)
        t_tot = np.maximum(sub["time_total_s"].to_numpy(dtype=np.float64), 1e-6)
        t_heur = np.maximum(sub["time_heuristic_s"].to_numpy(dtype=np.float64), 1e-6)
        ax.plot(n, t_tot, marker="o", linewidth=1.8, markersize=4, color=color, label=f"{ws} (total)")
        if ws != "trivial":
            ax.plot(n, t_heur, linestyle=":", linewidth=1.2, alpha=0.75, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("N (cidades)")
    ax.set_ylabel("Tempo (s) – sólido: total, pontilhado: heurística")
    ax.set_title("Tempo da Heurística vs Tempo Total")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Painel direito: speedup líquido frente ao trivial (se a heurística se paga)
    ax2 = axes[1]
    non_triv = [w for w in warm_starts if w != "trivial"]
    has_speedup = False
    for ws in non_triv:
        sub = summary_df[(summary_df["warm_start"] == ws) & summary_df["speedup_vs_trivial"].notna()].sort_values("n")
        if sub.empty:
            continue
        has_speedup = True
        ax2.plot(
            sub["n"],
            sub["speedup_vs_trivial"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=WARM_COLORS.get(ws, None),
            label=ws,
        )
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="empate (1.0×)")
    ax2.set_xlabel("N (cidades)")
    ax2.set_ylabel("Speedup = T_total(trivial) / T_total(ws)")
    ax2.set_title("Trade-off: a Heurística se Paga? (> 1 = ganho líquido)")
    if has_speedup:
        ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.suptitle("Trade-off entre Custo da Heurística Inicial e Tempo de B&B", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_collapse(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    warm_starts = _ordered_warm_starts(summary_df["warm_start"])
    inst_order = (
        summary_df[["instance", "n"]]
        .drop_duplicates()
        .sort_values(["n", "instance"])
    )
    instances = inst_order["instance"].tolist()
    ns = inst_order["n"].tolist()

    mat = np.full((len(instances), len(warm_starts)), np.nan)
    labels = [["" for _ in warm_starts] for _ in instances]
    lookup = summary_df.set_index(["instance", "warm_start"])

    for i, inst in enumerate(instances):
        for j, ws in enumerate(warm_starts):
            if (inst, ws) not in lookup.index:
                continue
            row = lookup.loc[(inst, ws)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if row["status"] == STATUS_TIMEOUT:
                mat[i, j] = np.nan
                labels[i][j] = "TIMEOUT"
            else:
                t = float(row["time_total_s"])
                mat[i, j] = np.log10(max(t, 1e-4))
                labels[i][j] = f"{t:.2f}s"

    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * len(warm_starts)), max(4.5, 0.42 * len(instances))))
    cmap = plt.get_cmap("YlOrRd").with_extremes(bad="#2b2b2b")
    im = ax.imshow(mat, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(warm_starts)))
    ax.set_xticklabels(warm_starts)
    ax.set_yticks(range(len(instances)))
    ax.set_yticklabels([f"{inst} (N={int(n)})" for inst, n in zip(instances, ns)], fontsize=8.5)

    for i in range(len(instances)):
        for j in range(len(warm_starts)):
            txt = labels[i][j]
            if not txt:
                continue
            color = "white" if txt == "TIMEOUT" else "black"
            weight = "bold" if txt == "TIMEOUT" else "normal"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7.8, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(tempo total em segundos)")
    ax.set_title("Ponto de Colapso do Branch-and-Bound (OK vs TIMEOUT)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_warmstart(
    runs_df: pd.DataFrame,
    out_dir: str | Path = "results",
    algorithm: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Executa a análise de warm-start e gera tabelas e gráficos."""
    fig_dir, tab_dir = ensure_output_dirs(out_dir)
    bnb_df = _select_bnb_runs(runs_df, algorithm=algorithm)
    summary_df = _build_summary_and_tradeoff(bnb_df)
    collapse_df = _build_collapse_table(summary_df)

    summary_df.to_csv(tab_dir / "warmstart_summary.csv", index=False)
    collapse_df.to_csv(tab_dir / "warmstart_collapse.csv", index=False)

    if not summary_df.empty:
        algs = sorted(summary_df["algorithm"].unique())
        if len(algs) == 1:
            _plot_nodes(summary_df, fig_dir / "warmstart_nodes.png")
            _plot_tradeoff(summary_df, fig_dir / "warmstart_tradeoff.png")
            _plot_collapse(summary_df, fig_dir / "warmstart_collapse.png")
        else:
            for alg in algs:
                sub = summary_df[summary_df["algorithm"] == alg]
                _plot_nodes(sub, fig_dir / f"warmstart_nodes_{alg}.png")
                _plot_tradeoff(sub, fig_dir / f"warmstart_tradeoff_{alg}.png")
                _plot_collapse(sub, fig_dir / f"warmstart_collapse_{alg}.png")

    return {"summary": summary_df, "collapse": collapse_df}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Análise de poda, trade-off e colapso por warm-start")
    ap.add_argument("--runs", type=Path, default=Path("results/runs.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--algorithm", type=str, default=None, help="filtra um algoritmo B&B específico")
    args = ap.parse_args(argv)

    runs_df = load_table(args.runs)
    outputs = analyze_warmstart(runs_df, args.out, algorithm=args.algorithm)
    collapse = outputs["collapse"]
    if not collapse.empty:
        print(collapse.to_string(index=False))
    else:
        print("[warmstart] Nenhuma execução B&B encontrada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
