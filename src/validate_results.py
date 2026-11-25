"""Validador automático do checklist experimental do TP2.

Audita integridade de esquema, consistência de ótimos, tratamento de TIMEOUT,
limites teóricos das aproximações (2× e 1.5×), critérios de sanidade do
PLANO.md e existência das tabelas e figuras esperadas.

Grava ``<dir>/tables/validation_report.json``.

Uso:
    python -m src.validate_results --dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.tsp.common import (
    OPTIMA,
    PROFILE_COLUMNS,
    RUNS_COLUMNS,
    STATUS_OK,
    STATUS_TIMEOUT,
    read_jsonl,
)
from src.tsp.features import FEATURE_NAMES


def validate_results_dir(results_dir: str | Path) -> dict[str, Any]:
    """Executa todas as verificações sobre ``results_dir`` e devolve o relatório."""
    base = Path(results_dir)
    runs_path = base / "runs.jsonl"
    prof_path = base / "profile.jsonl"
    feats_path = base / "features.csv"
    tab_dir = base / "tables"
    fig_dir = base / "figures"

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    if not runs_path.exists():
        errors.append(f"Arquivo ausente: {runs_path}")
        return {"passed": False, "errors": errors, "warnings": warnings, "checks": checks}

    runs = read_jsonl(runs_path)
    runs_df = pd.DataFrame(runs)

    # 1. Esquema de runs.jsonl
    schema_runs_ok = bool(runs) and all(list(r.keys()) == RUNS_COLUMNS for r in runs)
    checks["runs_schema_ok"] = schema_runs_ok
    checks["runs_total_rows"] = len(runs)
    if not schema_runs_ok:
        errors.append("Uma ou mais linhas de runs.jsonl não seguem RUNS_COLUMNS.")

    # 2. Esquema de profile.jsonl
    if prof_path.exists():
        profs = read_jsonl(prof_path)
        schema_prof_ok = bool(profs) and all(list(p.keys()) == PROFILE_COLUMNS for p in profs)
        checks["profile_schema_ok"] = schema_prof_ok
        checks["profile_total_rows"] = len(profs)
        if not schema_prof_ok:
            errors.append("Uma ou mais linhas de profile.jsonl não seguem PROFILE_COLUMNS.")
    else:
        checks["profile_schema_ok"] = False
        warnings.append("profile.jsonl ausente.")

    # 3. Esquema de features.csv
    if feats_path.exists():
        feats_df = pd.read_csv(feats_path)
        expected_cols = ["instance", *FEATURE_NAMES]
        schema_feats_ok = list(feats_df.columns) == expected_cols and not feats_df.isna().any().any()
        checks["features_schema_ok"] = schema_feats_ok
        checks["features_instances"] = int(len(feats_df))
        if not schema_feats_ok:
            errors.append("features.csv possui colunas incorretas ou valores NaN.")
    else:
        checks["features_schema_ok"] = False
        warnings.append("features.csv ausente.")

    # 4. Consistência de ótimos do B&B e comparação com OPTIMA
    bnb_mask = runs_df["algorithm"].astype(str).str.startswith("bnb")
    bnb_df = runs_df[bnb_mask]
    bnb_ok = bnb_df[bnb_df["status"] == STATUS_OK]
    opt_divergences: list[str] = []
    for inst_name, grp in bnb_ok.groupby("instance"):
        costs = sorted(grp["cost"].dropna().astype(int).unique().tolist())
        if len(costs) > 1:
            opt_divergences.append(f"{inst_name}: custos B&B divergentes {costs}")
        elif len(costs) == 1 and inst_name in OPTIMA and costs[0] != OPTIMA[inst_name]:
            opt_divergences.append(f"{inst_name}: B&B={costs[0]} != OPTIMA={OPTIMA[inst_name]}")
    checks["optima_consistent"] = len(opt_divergences) == 0
    checks["bnb_ok_instances"] = int(bnb_ok["instance"].nunique())
    if opt_divergences:
        errors.extend(opt_divergences)

    # 5. Linhas TIMEOUT: cost/gap devem ser NA (None) e contadores parciais preenchidos
    to_df = bnb_df[bnb_df["status"] == STATUS_TIMEOUT]
    to_bad_cost = int(to_df["cost"].notna().sum()) if not to_df.empty else 0
    to_bad_gap = int(to_df["gap"].notna().sum()) if not to_df.empty else 0
    to_missing_exp = int((to_df["expanded"].isna() | (to_df["expanded"] <= 0)).sum()) if not to_df.empty else 0
    checks["timeout_rows"] = int(len(to_df))
    checks["timeout_na_cost_gap_ok"] = (to_bad_cost == 0) and (to_bad_gap == 0)
    checks["timeout_partial_counters_ok"] = to_missing_exp == 0
    if to_bad_cost or to_bad_gap:
        errors.append(f"Linhas TIMEOUT com cost/gap preenchidos: cost={to_bad_cost}, gap={to_bad_gap}")
    if to_missing_exp:
        warnings.append(f"{to_missing_exp} linhas TIMEOUT sem expanded > 0 (ex.: kill externo).")

    # 6. Garantias teóricas de aproximação (TATT <= 2*OPT, Christofides <= 1.5*OPT)
    ratio_violations: list[str] = []
    for alg, limit in (("tatt", 2.0), ("christofides", 1.5)):
        sub = runs_df[(runs_df["algorithm"] == alg) & (runs_df["status"] == STATUS_OK)]
        sub = sub.dropna(subset=["cost", "optimal_ref"])
        for _, row in sub.drop_duplicates(subset=["instance"]).iterrows():
            ratio = float(row["cost"]) / float(row["optimal_ref"])
            if ratio > limit + 1e-6:
                ratio_violations.append(
                    f"{alg} em {row['instance']}: ratio={ratio:.4f} > {limit}"
                )
    checks["approx_bounds_ok"] = len(ratio_violations) == 0
    if ratio_violations:
        errors.extend(ratio_violations)

    # 7. Sanidade do PLANO.md:
    #    (a) expanded(christofides) <= expanded(trivial) nas instâncias resolvidas (OK em ambos)
    if not bnb_ok.empty:
        piv_exp = bnb_ok.pivot_table(index="instance", columns="warm_start", values="expanded", aggfunc="median")
        if {"christofides", "trivial"} <= set(piv_exp.columns):
            both = piv_exp[["christofides", "trivial"]].dropna()
            le_count = int((both["christofides"] <= both["trivial"]).sum())
            checks["sanity_chris_le_trivial_expanded"] = {
                "passed_instances": le_count,
                "total_comparable": int(len(both)),
                "ratio": float(le_count / len(both)) if len(both) else None,
            }
            if len(both) > 0 and le_count < len(both):
                warnings.append(
                    f"Christofides expandiu mais nós que trivial em {len(both) - le_count}/{len(both)} instâncias OK."
                )

    #    (b) gap(christofides) <= gap(tatt) na maioria das instâncias com referência
    const_ok = runs_df[(runs_df["algorithm"].isin(["tatt", "christofides"])) & (runs_df["status"] == STATUS_OK)]
    const_ok = const_ok.dropna(subset=["gap"])
    if not const_ok.empty:
        piv_gap = const_ok.pivot_table(index="instance", columns="algorithm", values="gap", aggfunc="median")
        if {"christofides", "tatt"} <= set(piv_gap.columns):
            both_g = piv_gap[["christofides", "tatt"]].dropna()
            better = int((both_g["christofides"] <= both_g["tatt"] + 1e-12).sum())
            checks["sanity_chris_le_tatt_gap"] = {
                "passed_instances": better,
                "total_comparable": int(len(both_g)),
                "ratio": float(better / len(both_g)) if len(both_g) else None,
            }
            if len(both_g) > 0 and better < len(both_g) / 2:
                warnings.append(
                    f"Christofides teve gap <= TATT em apenas {better}/{len(both_g)} instâncias."
                )

    # 8. Existência de tabelas e figuras geradas
    expected_tables = [
        "complexity_algorithms.csv",
        "complexity_subroutines.csv",
        "warmstart_summary.csv",
        "warmstart_collapse.csv",
        "sensitivity_spearman.csv",
        "sensitivity_loocv.csv",
    ]
    missing_tables = [t for t in expected_tables if not (tab_dir / t).exists()]
    checks["missing_tables"] = missing_tables
    if missing_tables:
        warnings.append(f"Tabelas ausentes em {tab_dir}: {missing_tables}")

    pngs = sorted(p.name for p in fig_dir.glob("*.png") if p.stat().st_size > 1000) if fig_dir.exists() else []
    checks["figures_count"] = len(pngs)
    checks["figures_list"] = pngs
    if not pngs:
        warnings.append(f"Nenhuma figura PNG encontrada em {fig_dir}.")

    report = {
        "results_dir": str(base),
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }

    tab_dir.mkdir(parents=True, exist_ok=True)
    out_json = tab_dir / "validation_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Valida resultados e checklist experimental do TP2")
    ap.add_argument("--dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    report = validate_results_dir(args.dir)
    status = "OK" if report["passed"] else "FALHOU"
    print(f"[validate_results] {status} ({args.dir})")
    print(f"  runs={report['checks'].get('runs_total_rows')} | bnb_ok_inst={report['checks'].get('bnb_ok_instances')} | "
          f"timeouts={report['checks'].get('timeout_rows')} | figuras={report['checks'].get('figures_count')}")
    for w in report["warnings"]:
        print(f"  [AVISO] {w}")
    for e in report["errors"]:
        print(f"  [ERRO]  {e}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
