import json
from pathlib import Path

import pytest

from src.analysis.complexity import analyze_complexity
from src.analysis.sensitivity import analyze_sensitivity
from src.analysis.warmstart import analyze_warmstart
from src.generate_insights import generate_insights, main as insights_main
from src.tsp.common import STATUS_OK, STATUS_TIMEOUT, ResultWriter, make_run_row, read_jsonl
from src.validate_results import main as validate_main, validate_results_dir
from tests.test_analysis import _make_mock_dataset


def test_validate_and_insights_on_valid_mock(tmp_path):
    runs_path, profile_path, features_path = _make_mock_dataset(tmp_path)
    import pandas as pd

    runs_df = pd.DataFrame(read_jsonl(runs_path))
    prof_df = pd.DataFrame(read_jsonl(profile_path))
    feats_df = pd.read_csv(features_path)

    analyze_complexity(runs_df, prof_df, tmp_path)
    analyze_warmstart(runs_df, tmp_path)
    analyze_sensitivity(runs_df, feats_df, tmp_path)

    report = validate_results_dir(tmp_path)
    assert report["passed"] is True
    assert report["checks"]["runs_schema_ok"] is True
    assert report["checks"]["profile_schema_ok"] is True
    assert report["checks"]["features_schema_ok"] is True
    assert report["checks"]["optima_consistent"] is True
    assert report["checks"]["timeout_na_cost_gap_ok"] is True
    assert report["checks"]["approx_bounds_ok"] is True
    assert (tmp_path / "tables" / "validation_report.json").exists()

    assert validate_main(["--dir", str(tmp_path)]) == 0

    insights = generate_insights(tmp_path)
    assert (tmp_path / "insights.json").exists()
    assert (tmp_path / "INSIGHTS.md").exists()

    loaded = json.loads((tmp_path / "insights.json").read_text(encoding="utf-8"))
    assert loaded["overview"]["instances_evaluated"] == 16
    assert "christofides_matching_share_pct" in loaded["profiling_and_complexity"]
    assert len(loaded["warmstart"]["collapse"]) == 5

    md = (tmp_path / "INSIGHTS.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. Contexto",
        "## 2. Auditoria",
        "## 3. Profiling",
        "## 4. Impacto do Warm-Start",
        "## 5. Análise de Sensibilidade",
        "## 6. Checklist de Validação",
    ):
        assert heading in md

    assert insights_main(["--dir", str(tmp_path)]) == 0


def test_validate_catches_contract_violations(tmp_path):
    runs_path = tmp_path / "runs.jsonl"
    with ResultWriter(runs_path) as w:
        # 1. Ótimos divergentes no B&B
        w.write(make_run_row(instance="x", n=12, algorithm="bnb_dfs_one_tree", warm_start="trivial", status=STATUS_OK, cost=100, optimal_ref=100, gap=0.0))
        w.write(make_run_row(instance="x", n=12, algorithm="bnb_dfs_one_tree", warm_start="hull", status=STATUS_OK, cost=105, optimal_ref=100, gap=0.05))
        # 2. TIMEOUT com cost preenchido (violação)
        w.write(make_run_row(instance="y", n=40, algorithm="bnb_dfs_one_tree", warm_start="trivial", status=STATUS_TIMEOUT, cost=999, expanded=100))
        # 3. Christofides acima de 1.5*OPT (violação)
        w.write(make_run_row(instance="x", n=12, algorithm="christofides", status=STATUS_OK, cost=160, optimal_ref=100, gap=0.6))

    report = validate_results_dir(tmp_path)
    assert report["passed"] is False
    assert report["checks"]["optima_consistent"] is False
    assert report["checks"]["timeout_na_cost_gap_ok"] is False
    assert report["checks"]["approx_bounds_ok"] is False
    assert len(report["errors"]) >= 3
    assert validate_main(["--dir", str(tmp_path)]) == 1
