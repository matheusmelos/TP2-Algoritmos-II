"""Consolidador automático de dados e insights experimentais do TP2.

Lê todos os artefatos coletados em ``<dir>`` (``runs.jsonl``, ``profile.jsonl``,
``features.csv``, ``tables/*.csv``, ``audit/*.csv``, ``supplementary/``,
``experiment_metadata.json`` e ``tables/validation_report.json``) e grava:

  * ``<dir>/insights.json`` – resumo numérico estruturado de todas as descobertas.
  * ``<dir>/INSIGHTS.md``   – dossiê analítico organizado pelas seções do artigo
                              SBC e pontos-chave para a arguição oral.

Uso:
    python -m src.generate_insights --dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tsp.common import STATUS_OK, STATUS_TIMEOUT, read_jsonl


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return None
    return None


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if path.exists() and path.stat().st_size > 0:
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _safe_float(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    return float(v)


def _fmt(v: float | None, spec: str = ".4f", suffix: str = "") -> str:
    if v is None:
        return "NA"
    return f"{v:{spec}}{suffix}"


def _df_to_md(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    if df.empty:
        return "_Sem dados disponíveis._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if pd.isna(val):
                cells.append("NA")
            elif isinstance(val, (float, np.floating)):
                cells.append(f"{float(val):{float_fmt}}")
            elif isinstance(val, (int, np.integer)):
                cells.append(str(int(val)))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_insights(results_dir: str | Path) -> dict[str, Any]:
    """Consolida todos os achados de ``results_dir`` em ``insights.json`` e ``INSIGHTS.md``."""
    base = Path(results_dir)
    tab_dir = base / "tables"
    audit_dir = base / "audit"
    supp_tab_dir = base / "supplementary" / "tables"

    meta = _read_json_if_exists(base / "experiment_metadata.json") or {}
    val_report = _read_json_if_exists(tab_dir / "validation_report.json")
    runs_df = pd.DataFrame(read_jsonl(base / "runs.jsonl")) if (base / "runs.jsonl").exists() else pd.DataFrame()
    prof_df = pd.DataFrame(read_jsonl(base / "profile.jsonl")) if (base / "profile.jsonl").exists() else pd.DataFrame()
    feats_df = _read_csv_if_exists(base / "features.csv")

    comp_alg = _read_csv_if_exists(tab_dir / "complexity_algorithms.csv")
    comp_sub = _read_csv_if_exists(tab_dir / "complexity_subroutines.csv")
    ws_summary = _read_csv_if_exists(tab_dir / "warmstart_summary.csv")
    ws_collapse = _read_csv_if_exists(tab_dir / "warmstart_collapse.csv")
    sens_spear = _read_csv_if_exists(tab_dir / "sensitivity_spearman.csv")
    sens_reg = _read_csv_if_exists(tab_dir / "sensitivity_regression.csv")
    sens_tree = _read_csv_if_exists(tab_dir / "sensitivity_tree_importance.csv")
    sens_loocv = _read_csv_if_exists(tab_dir / "sensitivity_loocv.csv")

    audit_dlist = _read_csv_if_exists(audit_dir / "audit_dlist_vs_numpy.csv")
    audit_sb = _read_csv_if_exists(audit_dir / "audit_search_bounds.csv")
    audit_ablation = _read_csv_if_exists(audit_dir / "audit_christofides_ablation.csv")
    audit_mem_gc = _read_csv_if_exists(audit_dir / "audit_memory_gc.csv")
    supp_comp_alg = _read_csv_if_exists(supp_tab_dir / "complexity_algorithms.csv")
    supp_comp_sub = _read_csv_if_exists(supp_tab_dir / "complexity_subroutines.csv")

    insights: dict[str, Any] = {
        "metadata": meta,
        "overview": {},
        "audit": {},
        "profiling_and_complexity": {},
        "warmstart": {},
        "sensitivity_and_adversarial": {},
        "validation": val_report,
    }

    md: list[str] = [
        "# Insights Experimentais Consolidados – TP2 (TSP Geométrico)",
        "",
        "Documento gerado automaticamente a partir de todos os dados coletados em `"
        + str(base)
        + "` para subsidiar o artigo SBC (`report/`) e a arguição oral.",
        "",
    ]

    # ------------------------------------------------------------------ #
    # 1. Contexto e Cobertura
    # ------------------------------------------------------------------ #
    md.append("## 1. Contexto, Hardware e Cobertura dos Dados")
    if meta:
        hw = meta.get("hardware", {})
        sw = meta.get("software", {})
        cfg = meta.get("config", {})
        md.append(
            f"- **Ambiente:** {hw.get('os', '?')} | CPU: `{hw.get('cpu_model', '?')}` "
            f"({hw.get('cpu_count', '?')} núcleos) | RAM: {hw.get('ram_total_gb', '?')} GiB."
        )
        md.append(
            f"- **Software:** Python `{sw.get('python', '?')}` | NumPy `{sw.get('numpy', '?')}` | "
            f"SciPy `{sw.get('scipy', '?')}` | NetworkX `{sw.get('networkx', '?')}` | "
            f"Scikit-learn `{sw.get('sklearn', '?')}`."
        )
        md.append(
            f"- **Execução:** commit `{meta.get('git_commit', '?')}` | "
            f"B&B=`bnb_{cfg.get('search', '?')}_{cfg.get('bound', '?')}` | "
            f"time_limit=`{cfg.get('time_limit_s', '?')}s` | reps=`{cfg.get('reps', '?')}` | "
            f"workers=`{cfg.get('workers', '?')}` | duração total=`{_fmt(_safe_float(meta.get('wall_time_s')), '.1f', 's')}`."
        )

    if not runs_df.empty:
        n_inst = int(runs_df["instance"].nunique())
        n_opt = int(runs_df.dropna(subset=["optimal_ref"])["instance"].nunique())
        bnb_rows = runs_df[runs_df["algorithm"].astype(str).str.startswith("bnb")]
        n_ok = int((bnb_rows["status"] == STATUS_OK).sum())
        n_to = int((bnb_rows["status"] == STATUS_TIMEOUT).sum())
        insights["overview"] = {
            "total_run_rows": int(len(runs_df)),
            "total_profile_rows": int(len(prof_df)),
            "instances_evaluated": n_inst,
            "instances_with_optimal_ref": n_opt,
            "bnb_jobs_total": int(len(bnb_rows)),
            "bnb_jobs_ok": n_ok,
            "bnb_jobs_timeout": n_to,
        }
        md.append(
            f"- **Cobertura:** {n_inst} instâncias ({n_opt} com referência ótima conhecida), "
            f"{len(runs_df)} linhas em `runs.jsonl`, {len(prof_df)} linhas em `profile.jsonl`, "
            f"B&B: {n_ok} `OK` / {n_to} `TIMEOUT`."
        )
    md.append("")

    # ------------------------------------------------------------------ #
    # 2. Auditoria de Implementação e Estruturas de Dados
    # ------------------------------------------------------------------ #
    md.append("## 2. Auditoria de Implementação, Estruturas de Dados e Busca")
    if audit_dlist is not None and not audit_dlist.empty:
        med_speedup = float(audit_dlist["speedup_dlist_vs_numpy"].median())
        max_speedup = float(audit_dlist["speedup_dlist_vs_numpy"].max())
        insights["audit"]["dlist_vs_numpy"] = {
            "median_speedup": med_speedup,
            "max_speedup": max_speedup,
            "rows": audit_dlist.to_dict(orient="records"),
        }
        md.append(
            f"### 2.1 Refatoração `D_list` (listas Python) vs `inst.D[i, j]` (NumPy)\n"
            f"- Speedup mediano no inner loop do B&B: **{med_speedup:.2f}×** (máximo: **{max_speedup:.2f}×**), "
            "evidenciando o overhead de boxing escalar do NumPy em loops Python puros."
        )
        md.append("")
        md.append(_df_to_md(audit_dlist[["instance", "n", "bound", "time_dlist_s", "time_numpy_s", "speedup_dlist_vs_numpy"]]))
        md.append("")

    if audit_sb is not None and not audit_sb.empty:
        insights["audit"]["search_and_bounds"] = audit_sb.to_dict(orient="records")
        md.append("### 2.2 Estratégia de Busca (`dfs` vs `best_first`) e Lower Bound (`two_min` vs `one_tree`)")
        # Destaca maior redução de nós do one_tree e maior explosão de fronteira do best_first
        piv_t = audit_sb.pivot_table(index=["instance", "n"], columns=["search", "bound"], values="time_s")
        piv_f = audit_sb.pivot_table(index=["instance", "n"], columns=["search", "bound"], values="max_frontier")
        if ("best_first", "two_min") in piv_f.columns and ("dfs", "one_tree") in piv_f.columns:
            max_bf_front = int(piv_f[("best_first", "two_min")].max())
            max_dfs_front = int(piv_f[("dfs", "one_tree")].max())
            md.append(
                f"- **Controle de memória:** `dfs + one_tree` manteve fronteira máxima de até **{max_dfs_front}** nós "
                f"($O(n^2)$), enquanto `best_first + two_min` atingiu **{max_bf_front}** nós na fronteira "
                "(risco de estouro de memória em instâncias maiores)."
            )
        if ("dfs", "two_min") in piv_t.columns and ("dfs", "one_tree") in piv_t.columns:
            valid = piv_t[[("dfs", "two_min"), ("dfs", "one_tree")]].dropna()
            if not valid.empty:
                ratios = valid[("dfs", "two_min")] / np.maximum(valid[("dfs", "one_tree")], 1e-9)
                best_idx = ratios.idxmax()
                md.append(
                    f"- **Força do bound:** `one_tree` acelerou o DFS em até **{float(ratios.max()):.1f}×** "
                    f"(em `{best_idx[0]}`, N={best_idx[1]}), graças ao MST dos não visitados memoizado por bitmask."
                )
        # Instâncias onde ambos os bounds sofrem (útil para arguição!)
        hard_sb = audit_sb[(audit_sb["search"] == "dfs") & (audit_sb["bound"] == "one_tree")].copy()
        if not hard_sb.empty and {"root_lb", "initial_ub"} <= set(hard_sb.columns):
            hard_sb["root_gap_pct"] = 100.0 * (hard_sb["initial_ub"] - hard_sb["root_lb"]) / np.maximum(hard_sb["initial_ub"], 1)
            worst_root = hard_sb.sort_values("root_gap_pct", ascending=False).iloc[0]
            md.append(
                f"- **Caso difícil para a arguição:** em `{worst_root['instance']}` (N={int(worst_root['n'])}), "
                f"o gap na raiz `(UB0 - LB_raiz)/UB0` foi de **{worst_root['root_gap_pct']:.1f}%** "
                f"(`LB_raiz={int(worst_root['root_lb'])}` vs `UB0={int(worst_root['initial_ub'])}`, status=`{worst_root['status']}`), "
                "mostrando por que instâncias com clusters/pontes longas produzem relaxação 1-tree mais frouxa."
            )
        md.append("")
        cols_sb = [c for c in ["instance", "n", "search", "bound", "status", "time_s", "expanded", "max_frontier", "root_lb", "peak_mem_kb"] if c in audit_sb.columns]
        md.append(_df_to_md(audit_sb[cols_sb], float_fmt=".3f"))
        md.append("")

    if audit_ablation is not None and not audit_ablation.empty:
        insights["audit"]["christofides_ablation"] = audit_ablation.to_dict(orient="records")
        med_sp = float(audit_ablation["speedup_greedy_vs_exact"].median())
        max_sp = float(audit_ablation["speedup_greedy_vs_exact"].max())
        med_dg = float((audit_ablation["gap_greedy"] - audit_ablation["gap_exact"]).median() * 100.0)
        md.append(
            f"### 2.3 Ablação do Christofides (Matching Exato vs Guloso)\n"
            f"- O matching guloso acelerou o algoritmo em **{med_sp:.1f}×** (mediana) até **{max_sp:.1f}×** (em N maior), "
            f"ao custo de uma piora mediana de apenas **{med_dg:+.2f} pontos percentuais** no gap relativo."
        )
        md.append("")
        md.append(_df_to_md(audit_ablation, float_fmt=".4f"))
        md.append("")

    if audit_mem_gc is not None and not audit_mem_gc.empty:
        insights["audit"]["memory_and_gc"] = audit_mem_gc.to_dict(orient="records")
        max_mem_ratio = float(audit_mem_gc["prim_copy_mem_ratio"].max())
        med_tr_slow = float(audit_mem_gc["tracemalloc_slowdown"].median())
        med_gc_oh = float(audit_mem_gc["chris_gc_overhead_pct"].median())
        md.append(
            f"### 2.4 Distorções por Memória, Cópias de Matriz e Garbage Collection\n"
            f"- Cópias desnecessárias de matriz no Prim aumentaram o pico de memória em até **{max_mem_ratio:.1f}×**.\n"
            f"- O `tracemalloc` desacelerou o Christofides em **{med_tr_slow:.2f}×** (mediana), confirmando a necessidade de passadas separadas de tempo e memória.\n"
            f"- Impacto mediano do Garbage Collector (`gc.enable` vs `gc.disable`) no Christofides: **{med_gc_oh:+.2f}%**."
        )
        md.append("")
        md.append(_df_to_md(audit_mem_gc, float_fmt=".4f"))
        md.append("")

    # ------------------------------------------------------------------ #
    # 3. Profiling de Subrotinas e Complexidade Empírica
    # ------------------------------------------------------------------ #
    md.append("## 3. Profiling de Subrotinas e Modelagem de Complexidade")
    if not prof_df.empty:
        chris_p = prof_df[prof_df["algorithm"] == "christofides"]
        if not chris_p.empty:
            piv = chris_p.pivot_table(index="n", columns="section", values="time_s", aggfunc="median")
            if "matching" in piv.columns:
                total = piv.sum(axis=1)
                share = (piv["matching"] / np.maximum(total, 1e-12)) * 100.0
                insights["profiling_and_complexity"]["christofides_matching_share_pct"] = {
                    int(k): float(v) for k, v in share.items()
                }
                md.append(
                    f"- **Gargalo do Christofides:** a subrotina `matching` consumiu de **{share.min():.1f}%** "
                    f"(menor N={int(share.index.min())}) até **{share.max():.1f}%** "
                    f"(maior N={int(share.index.max())}) do tempo total do algoritmo."
                )

    if comp_alg is not None and not comp_alg.empty:
        best_alg = comp_alg[comp_alg["is_best"] == True].copy()
        insights["profiling_and_complexity"]["best_models_algorithms"] = best_alg.to_dict(orient="records")
        md.append("### 3.1 Melhores Modelos por Algoritmo (seleção por menor AIC)")
        md.append(_df_to_md(best_alg[["algorithm", "metric", "model", "formula", "r2", "rmse", "aic", "n_points"]]))
        md.append("")

    if comp_sub is not None and not comp_sub.empty:
        best_sub = comp_sub[comp_sub["is_best"] == True].copy()
        insights["profiling_and_complexity"]["best_models_subroutines"] = best_sub.to_dict(orient="records")
        md.append("### 3.2 Melhores Modelos por Subrotina (seleção por menor AIC)")
        md.append(_df_to_md(best_sub[["algorithm", "section", "metric", "model", "formula", "r2", "rmse", "aic"]]))
        md.append("")

    if supp_comp_alg is not None and not supp_comp_alg.empty:
        best_supp = supp_comp_alg[supp_comp_alg["is_best"] == True].copy()
        insights["profiling_and_complexity"]["supplementary_best_algorithms"] = best_supp.to_dict(orient="records")
        md.append("### 3.3 Curvas Suplementares (N até 500 – TATT e Christofides)")
        md.append(_df_to_md(best_supp[["algorithm", "metric", "model", "formula", "r2", "rmse", "aic", "n_points"]]))
        md.append("")
    if supp_comp_sub is not None and not supp_comp_sub.empty:
        best_supp_sub = supp_comp_sub[supp_comp_sub["is_best"] == True].copy()
        insights["profiling_and_complexity"]["supplementary_best_subroutines"] = best_supp_sub.to_dict(orient="records")

    # ------------------------------------------------------------------ #
    # 4. Impacto do Warm-Start e Ponto de Colapso
    # ------------------------------------------------------------------ #
    md.append("## 4. Impacto do Warm-Start na Poda e Ponto de Colapso")
    if ws_collapse is not None and not ws_collapse.empty:
        insights["warmstart"]["collapse"] = ws_collapse.to_dict(orient="records")
        md.append("### 4.1 Ponto de Colapso por Estratégia")
        md.append(_df_to_md(ws_collapse))
        md.append("")

    if ws_summary is not None and not ws_summary.empty:
        ok_ws = ws_summary[ws_summary["status"] == STATUS_OK]
        by_ws_rows = []
        for ws, grp in ok_ws.groupby("warm_start"):
            pays = grp["pays_off"].dropna()
            by_ws_rows.append(
                {
                    "warm_start": ws,
                    "ok_count": int(len(grp)),
                    "median_expanded_red_pct": _safe_float(grp["expanded_reduction_pct"].median()),
                    "median_speedup": _safe_float(grp["speedup_vs_trivial"].median()),
                    "max_speedup": _safe_float(grp["speedup_vs_trivial"].max()),
                    "median_heur_s": _safe_float(grp["time_heuristic_s"].median()),
                    "pays_off_rate_pct": float(100.0 * pays.astype(bool).mean()) if len(pays) else None,
                }
            )
        by_ws_df = pd.DataFrame(by_ws_rows)
        insights["warmstart"]["tradeoff_by_strategy"] = by_ws_rows
        md.append("### 4.2 Resumo de Poda e Trade-off (Instâncias Resolvidas com OK)")
        md.append(_df_to_md(by_ws_df, float_fmt=".3f"))
        md.append("")

    # ------------------------------------------------------------------ #
    # 5. Análise de Sensibilidade e Instâncias Adversariais
    # ------------------------------------------------------------------ #
    md.append("## 5. Análise de Sensibilidade e Instâncias Adversariais")
    if not runs_df.empty:
        const_ok = runs_df[
            (runs_df["algorithm"].isin(["nn", "hull", "tatt", "christofides", "christofides_greedy"]))
            & (runs_df["status"] == STATUS_OK)
        ].dropna(subset=["gap"])

        if not const_ok.empty:
            agg_gap = const_ok.groupby(["instance", "n", "algorithm"], as_index=False)["gap"].median()
            alg_stats = (
                agg_gap.groupby("algorithm", as_index=False)["gap"]
                .agg(n_inst="count", mean_gap="mean", median_gap="median", max_gap="max", min_gap="min")
                .sort_values("mean_gap")
            )
            insights["sensitivity_and_adversarial"]["gap_summary"] = alg_stats.to_dict(orient="records")
            md.append("### 5.1 Qualidade das Heurísticas e Aproximações (Gap Relativo)")
            md.append(_df_to_md(alg_stats, float_fmt=".4f"))
            md.append("")

            # Ranking das piores (adversariais) instâncias por algoritmo cruzadas com features
            adv_rows = []
            for alg, grp in agg_gap.groupby("algorithm"):
                worst = grp.sort_values("gap", ascending=False).iloc[0]
                best = grp.sort_values("gap", ascending=True).iloc[0]
                entry: dict[str, Any] = {
                    "algorithm": alg,
                    "worst_instance": worst["instance"],
                    "worst_gap_pct": float(100.0 * worst["gap"]),
                    "best_instance": best["instance"],
                    "best_gap_pct": float(100.0 * best["gap"]),
                }
                if feats_df is not None:
                    fmatch = feats_df[feats_df["instance"] == worst["instance"]]
                    if not fmatch.empty:
                        fr = fmatch.iloc[0]
                        entry["worst_clark_evans"] = float(fr["clark_evans"])
                        entry["worst_hull_frac"] = float(fr["hull_fraction"])
                        entry["worst_mst_edge_cv"] = float(fr["mst_edge_cv"])
                        entry["worst_aspect_ratio"] = float(fr["aspect_ratio"])
                adv_rows.append(entry)
            adv_df = pd.DataFrame(adv_rows)
            insights["sensitivity_and_adversarial"]["adversarial_extremes"] = adv_rows
            md.append("### 5.2 Instâncias Mais Adversariais vs Mais Favoráveis")
            md.append(_df_to_md(adv_df, float_fmt=".3f"))
            md.append("")

    if sens_spear is not None and not sens_spear.empty:
        top_spear = (
            sens_spear.dropna(subset=["spearman_rho"])
            .assign(abs_rho=lambda d: d["spearman_rho"].abs())
            .sort_values("abs_rho", ascending=False)
            .head(10)
            .drop(columns=["abs_rho"])
        )
        insights["sensitivity_and_adversarial"]["top_spearman"] = top_spear.to_dict(orient="records")
        md.append("### 5.3 Correlações de Spearman Mais Fortes (Features vs Gap)")
        md.append(_df_to_md(top_spear, float_fmt=".3f"))
        md.append("")

    if sens_loocv is not None and not sens_loocv.empty:
        insights["sensitivity_and_adversarial"]["loocv"] = sens_loocv.to_dict(orient="records")
        md.append("### 5.4 Validação Leave-One-Out (Regressão Linear Padronizada e Árvore max_depth=3)")
        md.append(
            "> **Limitação metodológica:** amostra pequena (≤ 16 instâncias), portanto os valores de "
            "`loocv_r2` têm alta variância e servem para diagnóstico exploratório."
        )
        md.append("")
        cols_l = [c for c in sens_loocv.columns if c != "note"]
        md.append(_df_to_md(sens_loocv[cols_l], float_fmt=".4f"))
        md.append("")

    if sens_reg is not None and not sens_reg.empty:
        top_reg = sens_reg.sort_values("abs_std_coef", ascending=False).groupby("algorithm").head(3)
        insights["sensitivity_and_adversarial"]["top_linear_coefs"] = top_reg.to_dict(orient="records")
    if sens_tree is not None and not sens_tree.empty:
        top_tree = sens_tree[sens_tree["importance"] > 0].sort_values(["algorithm", "importance"], ascending=[True, False])
        insights["sensitivity_and_adversarial"]["tree_importances"] = top_tree.to_dict(orient="records")

    # ------------------------------------------------------------------ #
    # 6. Validação e Sanidade
    # ------------------------------------------------------------------ #
    md.append("## 6. Checklist de Validação e Sanidade")
    if val_report is not None:
        md.append(f"- **Status geral:** `{'APROVADO' if val_report.get('passed') else 'FALHOU'}`")
        for k, v in val_report.get("checks", {}).items():
            if k == "figures_list":
                continue
            md.append(f"- `{k}`: `{v}`")
        if val_report.get("warnings"):
            md.append("- **Avisos:**")
            for w in val_report["warnings"]:
                md.append(f"  - {w}")
        if val_report.get("errors"):
            md.append("- **Erros:**")
            for e in val_report["errors"]:
                md.append(f"  - {e}")
    else:
        md.append("_Relatório de validação ainda não executado (`python -m src.validate_results`)._")

    # Grava insights.json e INSIGHTS.md
    base.mkdir(parents=True, exist_ok=True)
    (base / "insights.json").write_text(
        json.dumps(insights, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (base / "INSIGHTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return insights


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Consolida dados e gera INSIGHTS.md + insights.json")
    ap.add_argument("--dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    generate_insights(args.dir)
    print(f"[generate_insights] gravados {args.dir / 'insights.json'} e {args.dir / 'INSIGHTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
