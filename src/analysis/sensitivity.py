"""Análise de sensibilidade do gap às features das instâncias (Etapa 2 – Dev B).

Avalia como as características geométricas/topológicas extraídas por
``src/tsp/features.py`` influenciam o gap relativo de aproximação para
TATT, Christofides, Vizinho Mais Próximo (NN) e Inserção por Fecho Convexo
(hull) – além de ablações como ``christofides_greedy`` quando presentes.

Métodos:
  1. Correlação de Spearman (ρ e p-valor) por (algoritmo, feature).
  2. Regressão linear múltipla padronizada (coeficientes β).
  3. ``DecisionTreeRegressor(max_depth=3)``, com importâncias e visualização
     da árvore.
  4. Validação por Leave-One-Out (LOOCV), registrando explicitamente a
     limitação de amostra pequena (~16 instâncias).

Uso:
    python -m src.analysis.sensitivity \
        --runs results/runs.jsonl --features results/features.csv --out results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

from src.tsp.common import STATUS_OK
from src.tsp.features import FEATURE_NAMES

from ._io import ensure_output_dirs, load_table

DEFAULT_ALGORITHMS: tuple[str, ...] = ("tatt", "christofides", "nn", "hull")
MODEL_FEATURES: tuple[str, ...] = (
    "aspect_ratio",
    "hull_fraction",
    "spatial_entropy",
    "clark_evans",
    "mst_max_degree",
    "mst_mean_degree",
    "mst_edge_cv",
    "nn_dist_cv",
)
SMALL_SAMPLE_NOTE = (
    "Amostra pequena (tipicamente <= 16 instâncias): resultados de LOOCV e "
    "modelos multivariados têm alta variância e caráter exploratório."
)


def _prepare_gap_table(
    runs_df: pd.DataFrame,
    features_df: pd.DataFrame,
    algorithms: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = runs_df.copy()
    if "status" in df.columns:
        df = df[df["status"] == STATUS_OK]
    df["gap"] = pd.to_numeric(df["gap"], errors="coerce")
    df = df.dropna(subset=["gap", "instance", "algorithm"])

    if algorithms is None:
        present = list(dict.fromkeys(df["algorithm"].astype(str)))
        selected = [a for a in DEFAULT_ALGORITHMS if a in present]
        if "christofides_greedy" in present and "christofides_greedy" not in selected:
            selected.append("christofides_greedy")
    else:
        selected = [a for a in algorithms if a in set(df["algorithm"].astype(str))]

    df = df[df["algorithm"].isin(selected)]
    agg = df.groupby(["instance", "algorithm"], as_index=False)["gap"].median()

    feat_cols = [c for c in FEATURE_NAMES if c in features_df.columns]
    merged = agg.merge(features_df[["instance", *feat_cols]], on="instance", how="inner")
    return merged, selected, feat_cols


def _spearman_analysis(merged: pd.DataFrame, algorithms: list[str], feat_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for alg in algorithms:
        sub = merged[merged["algorithm"] == alg]
        y = sub["gap"].to_numpy(dtype=np.float64)
        for feat in feat_cols:
            x = sub[feat].to_numpy(dtype=np.float64)
            if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
                rho, pval = np.nan, np.nan
            else:
                res = spearmanr(x, y)
                rho, pval = float(res.statistic), float(res.pvalue)
            rows.append(
                {
                    "algorithm": alg,
                    "feature": feat,
                    "spearman_rho": rho,
                    "p_value": pval,
                    "n_instances": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def _loocv_predictions(X: np.ndarray, y: np.ndarray, model_kind: str) -> np.ndarray:
    """Predições out-of-sample por Leave-One-Out sem vazamento de escala."""
    loo = LeaveOneOut()
    preds = np.empty(len(y), dtype=np.float64)
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        if model_kind == "linear":
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            est = LinearRegression()
            est.fit(X_tr_s, y_tr)
            preds[test_idx] = est.predict(X_te_s)
        elif model_kind == "tree":
            est = DecisionTreeRegressor(max_depth=3, random_state=42)
            est.fit(X_tr, y_tr)
            preds[test_idx] = est.predict(X_te)
        else:
            raise ValueError(model_kind)
    return preds


def _plot_spearman_heatmap(spearman_df: pd.DataFrame, out_path: Path) -> None:
    if spearman_df.empty:
        return
    piv_rho = spearman_df.pivot(index="feature", columns="algorithm", values="spearman_rho")
    piv_p = spearman_df.pivot(index="feature", columns="algorithm", values="p_value")
    feats = [f for f in FEATURE_NAMES if f in piv_rho.index]
    piv_rho = piv_rho.loc[feats]
    piv_p = piv_p.loc[feats]

    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(piv_rho.columns) + 1.5), max(4.5, 0.55 * len(feats))))
    im = ax.imshow(piv_rho.to_numpy(dtype=np.float64), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(piv_rho.columns)))
    ax.set_xticklabels(piv_rho.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)

    for i, f in enumerate(feats):
        for j, a in enumerate(piv_rho.columns):
            rho = piv_rho.loc[f, a]
            pval = piv_p.loc[f, a]
            if pd.isna(rho):
                txt = "NA"
            else:
                stars = "**" if (not pd.isna(pval) and pval < 0.01) else ("*" if (not pd.isna(pval) and pval < 0.05) else "")
                txt = f"{rho:+.2f}{stars}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlação de Spearman (ρ)")
    ax.set_title("Correlação de Spearman: Features vs Gap (* p<0.05, ** p<0.01)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_linear_coefs(reg_df: pd.DataFrame, out_path: Path) -> None:
    if reg_df.empty:
        return
    piv = reg_df.pivot(index="feature", columns="algorithm", values="std_coef").fillna(0.0)
    feats = [f for f in MODEL_FEATURES if f in piv.index]
    piv = piv.loc[feats]
    algs = list(piv.columns)

    y_pos = np.arange(len(feats))
    width = 0.8 / max(len(algs), 1)
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.55 * len(feats))))
    for idx, alg in enumerate(algs):
        offset = (idx - (len(algs) - 1) / 2.0) * width
        ax.barh(y_pos + offset, piv[alg].to_numpy(), height=width, label=alg)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Coeficiente padronizado (β)")
    ax.set_title("Regressão Linear Múltipla Padronizada (impacto no gap)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_tree_and_importance(
    tree: DecisionTreeRegressor,
    active_feats: list[str],
    alg: str,
    loocv_r2: float,
    loocv_rmse: float,
    n_inst: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [2.2, 1.0]})
    plot_tree(
        tree,
        feature_names=active_feats,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=axes[0],
    )
    axes[0].set_title(f"Árvore de Decisão (max_depth=3) – {alg}")

    imp = tree.feature_importances_
    order = np.argsort(imp)
    axes[1].barh([active_feats[i] for i in order], imp[order], color="#2ca02c")
    axes[1].set_xlabel("Importância (redução de impureza)")
    axes[1].set_title(f"Importância das Features\nLOOCV R²={loocv_r2:.3f}, RMSE={loocv_rmse:.4f} (n={n_inst})")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.text(0.5, 0.01, f"Nota: {SMALL_SAMPLE_NOTE}", ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_sensitivity(
    runs_df: pd.DataFrame,
    features_df: pd.DataFrame,
    out_dir: str | Path = "results",
    algorithms: tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    """Executa análise de sensibilidade completa e grava tabelas e figuras."""
    fig_dir, tab_dir = ensure_output_dirs(out_dir)
    merged, selected, feat_cols = _prepare_gap_table(runs_df, features_df, algorithms=algorithms)

    spearman_df = _spearman_analysis(merged, selected, feat_cols)
    reg_rows: list[dict] = []
    tree_rows: list[dict] = []
    loocv_rows: list[dict] = []

    model_feats = [f for f in MODEL_FEATURES if f in merged.columns]

    for alg in selected:
        sub = merged[merged["algorithm"] == alg].dropna(subset=[*model_feats, "gap"])
        n_inst = len(sub)
        if n_inst < 4:
            continue

        # Remove features constantes na amostra para evitar divisão por zero no StandardScaler
        active_feats = [f for f in model_feats if float(sub[f].std()) > 0.0]
        if not active_feats:
            continue

        X = sub[active_feats].to_numpy(dtype=np.float64)
        y = sub["gap"].to_numpy(dtype=np.float64)

        # 1. Regressão linear múltipla padronizada + LOOCV
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        lin = LinearRegression()
        lin.fit(X_s, y)
        y_hat_lin = lin.predict(X_s)
        loo_lin = _loocv_predictions(X, y, "linear")

        lin_train_r2 = float(r2_score(y, y_hat_lin)) if np.std(y) > 0 else 1.0
        lin_loo_r2 = float(r2_score(y, loo_lin)) if np.std(y) > 0 else 0.0
        lin_loo_rmse = float(np.sqrt(np.mean((y - loo_lin) ** 2)))

        for feat, coef in zip(active_feats, lin.coef_):
            reg_rows.append(
                {
                    "algorithm": alg,
                    "feature": feat,
                    "std_coef": float(coef),
                    "abs_std_coef": float(abs(coef)),
                    "intercept": float(lin.intercept_),
                    "n_instances": int(n_inst),
                }
            )
        loocv_rows.append(
            {
                "algorithm": alg,
                "model": "linear_standardized",
                "n_instances": int(n_inst),
                "n_features": int(len(active_feats)),
                "train_r2": lin_train_r2,
                "loocv_r2": lin_loo_r2,
                "loocv_rmse": lin_loo_rmse,
                "note": SMALL_SAMPLE_NOTE,
            }
        )

        # 2. Árvore de decisão (max_depth=3) + LOOCV
        tree = DecisionTreeRegressor(max_depth=3, random_state=42)
        tree.fit(X, y)
        y_hat_tree = tree.predict(X)
        loo_tree = _loocv_predictions(X, y, "tree")

        tree_train_r2 = float(r2_score(y, y_hat_tree)) if np.std(y) > 0 else 1.0
        tree_loo_r2 = float(r2_score(y, loo_tree)) if np.std(y) > 0 else 0.0
        tree_loo_rmse = float(np.sqrt(np.mean((y - loo_tree) ** 2)))

        for feat, imp in zip(active_feats, tree.feature_importances_):
            tree_rows.append(
                {
                    "algorithm": alg,
                    "feature": feat,
                    "importance": float(imp),
                    "n_instances": int(n_inst),
                }
            )
        loocv_rows.append(
            {
                "algorithm": alg,
                "model": "decision_tree_d3",
                "n_instances": int(n_inst),
                "n_features": int(len(active_feats)),
                "train_r2": tree_train_r2,
                "loocv_r2": tree_loo_r2,
                "loocv_rmse": tree_loo_rmse,
                "note": SMALL_SAMPLE_NOTE,
            }
        )

        _plot_tree_and_importance(
            tree=tree,
            active_feats=active_feats,
            alg=alg,
            loocv_r2=tree_loo_r2,
            loocv_rmse=tree_loo_rmse,
            n_inst=n_inst,
            out_path=fig_dir / f"sensitivity_tree_{alg}.png",
        )

    reg_df = pd.DataFrame(reg_rows)
    tree_df = pd.DataFrame(tree_rows)
    loocv_df = pd.DataFrame(loocv_rows)

    spearman_df.to_csv(tab_dir / "sensitivity_spearman.csv", index=False)
    reg_df.to_csv(tab_dir / "sensitivity_regression.csv", index=False)
    tree_df.to_csv(tab_dir / "sensitivity_tree_importance.csv", index=False)
    loocv_df.to_csv(tab_dir / "sensitivity_loocv.csv", index=False)

    _plot_spearman_heatmap(spearman_df, fig_dir / "sensitivity_spearman_heatmap.png")
    _plot_linear_coefs(reg_df, fig_dir / "sensitivity_linear_coefs.png")

    return {
        "spearman": spearman_df,
        "regression": reg_df,
        "tree_importance": tree_df,
        "loocv": loocv_df,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Análise de sensibilidade do gap às features das instâncias")
    ap.add_argument("--runs", type=Path, default=Path("results/runs.jsonl"))
    ap.add_argument("--features", type=Path, default=Path("results/features.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    runs_df = load_table(args.runs)
    features_df = load_table(args.features)
    outputs = analyze_sensitivity(runs_df, features_df, args.out)
    loocv = outputs["loocv"]
    if not loocv.empty:
        print(loocv.drop(columns=["note"]).to_string(index=False))
        print(f"\nNota metodológica: {SMALL_SAMPLE_NOTE}")
    else:
        print("[sensitivity] Dados insuficientes para ajuste de modelos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
