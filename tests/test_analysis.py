from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.complexity import analyze_complexity, fit_candidates, main as complexity_main
from src.analysis.sensitivity import analyze_sensitivity, main as sensitivity_main
from src.analysis.warmstart import analyze_warmstart, main as warmstart_main
from src.tsp.common import (
    PROFILE_COLUMNS,
    STATUS_OK,
    STATUS_TIMEOUT,
    ResultWriter,
    make_run_row,
)
from src.tsp.features import FEATURE_NAMES, write_features_csv


def _make_mock_dataset(tmp_path: Path):
    """Cria runs.jsonl, profile.jsonl e features.csv sintéticos no esquema do contrato."""
    rng = np.random.default_rng(2026)
    ns = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 36, 40, 51, 52, 70]
    instances = [f"inst_{n}" for n in ns]

    feat_rows = []
    for n, name in zip(ns, instances):
        feat_rows.append(
            {
                "instance": name,
                "n": float(n),
                "aspect_ratio": float(rng.uniform(0.25, 1.0)),
                "hull_fraction": float(rng.uniform(0.15, 0.75)),
                "spatial_entropy": float(rng.uniform(0.45, 0.98)),
                "clark_evans": float(rng.uniform(0.4, 1.6)),
                "mst_max_degree": float(rng.integers(3, 7)),
                "mst_mean_degree": float(2 * (n - 1) / n),
                "mst_edge_cv": float(rng.uniform(0.2, 1.4)),
                "nn_dist_cv": float(rng.uniform(0.15, 0.9)),
            }
        )
    features_path = tmp_path / "features.csv"
    write_features_csv(feat_rows, features_path)

    runs_rows = []
    prof_rows = []

    # 1. Algoritmos construtivos / aproximados (com 3 reps para testar agregação por mediana)
    for fr in feat_rows:
        inst = fr["instance"]
        n = int(fr["n"])
        opt = 100 * n
        base_gap_signal = 0.08 * fr["mst_edge_cv"] - 0.06 * fr["hull_fraction"] + 0.04 * (1.2 - fr["clark_evans"])

        specs = {
            "nn": (2e-6 * n**2, 1.5 * n, max(0.02, 0.14 + base_gap_signal)),
            "hull": (3e-6 * n**2, 2.0 * n, max(0.01, 0.07 + 0.5 * base_gap_signal)),
            "tatt": (2.5e-6 * n**2, 1.8 * n, max(0.03, 0.22 + 1.2 * base_gap_signal)),
            "christofides": (1.5e-7 * n**3 + 2.5e-6 * n**2, 0.3 * n**2, max(0.01, 0.09 + 0.7 * base_gap_signal)),
        }
        for alg, (t_base, mem_base, g_val) in specs.items():
            cost = int(round(opt * (1.0 + g_val)))
            for rep in range(3):
                t = float(t_base * rng.uniform(0.95, 1.05))
                runs_rows.append(
                    make_run_row(
                        instance=inst,
                        n=n,
                        algorithm=alg,
                        warm_start=None,
                        rep=rep,
                        status=STATUS_OK,
                        cost=cost,
                        optimal_ref=opt,
                        gap=(cost - opt) / opt,
                        time_total_s=t,
                        time_heuristic_s=None,
                        peak_mem_kb=float(mem_base * rng.uniform(0.98, 1.02)) if rep == 0 else None,
                    )
                )
            # Subrotinas do profiler
            if alg == "tatt":
                secs = {"mst": 0.75 * t_base, "preorder": 0.25 * t_base}
            elif alg == "christofides":
                secs = {
                    "mst": 2.5e-6 * n**2,
                    "odd": 2e-7 * n,
                    "matching": 1.5e-7 * n**3,
                    "euler": 4e-7 * n,
                    "shortcut": 1e-7 * n,
                }
            else:
                secs = {alg: t_base}
            for sec, st in secs.items():
                row = {c: None for c in PROFILE_COLUMNS}
                row.update(
                    instance=inst,
                    n=n,
                    algorithm=alg,
                    rep=0,
                    section=sec,
                    time_s=float(st),
                    calls=1,
                    peak_kb=float(mem_base * (0.8 if sec == "matching" else 0.2)),
                )
                prof_rows.append(row)

    # 2. Execuções B&B com 5 warm-starts (colapso em N maiores, trivial colapsando antes)
    ws_factors = {
        "trivial": (1.0, 0.0, 30),
        "nn": (0.45, 2e-4, 33),
        "hull": (0.28, 4e-4, 36),
        "tatt": (0.55, 3e-4, 33),
        "christofides": (0.18, 2e-3, 40),
    }
    for fr in feat_rows:
        inst = fr["instance"]
        n = int(fr["n"])
        opt = 100 * n
        for ws, (mult, t_h, collapse_n) in ws_factors.items():
            timed_out = n >= collapse_n
            expanded = int(max(10, mult * (1.28 ** n)))
            generated = int(expanded * 4.5)
            pruned = int(generated * (0.72 if ws == "trivial" else 0.86))
            t_bnb = 1800.0 if timed_out else float(mult * 1e-4 * (1.28 ** n))
            runs_rows.append(
                make_run_row(
                    instance=inst,
                    n=n,
                    algorithm="bnb_dfs_one_tree",
                    warm_start=ws,
                    rep=0,
                    status=STATUS_TIMEOUT if timed_out else STATUS_OK,
                    cost=None if timed_out else opt,
                    optimal_ref=opt,
                    gap=None if timed_out else 0.0,
                    time_total_s=t_h + t_bnb,
                    time_heuristic_s=t_h,
                    peak_mem_kb=float(50.0 + 0.5 * n**2) if n <= 24 else None,
                    expanded=expanded,
                    pruned=pruned,
                    generated=generated,
                    max_depth=min(n - 1, 20),
                    max_frontier=n * 3,
                    time_to_best_s=0.3 * t_bnb,
                )
            )

    runs_path = tmp_path / "runs.jsonl"
    profile_path = tmp_path / "profile.jsonl"
    with ResultWriter(runs_path) as w:
        w.write_many(runs_rows)
    with ResultWriter(profile_path) as w:
        w.write_many(prof_rows)
    return runs_path, profile_path, features_path


# --------------------------------------------------------------------------- #
# complexity.py
# --------------------------------------------------------------------------- #


def test_fit_candidates_identifies_true_scaling():
    n = np.array([10, 15, 20, 25, 30, 40, 50, 70, 100], dtype=float)

    _, best_q = fit_candidates(n, 2e-5 * n**2 + 1e-4)
    assert best_q is not None
    assert best_q.model in {"quadratic", "power"}
    assert best_q.r2 > 0.999

    _, best_c = fit_candidates(n, 3e-7 * n**3 + 5e-4)
    assert best_c is not None
    assert best_c.model in {"cubic", "power"}
    assert best_c.r2 > 0.999

    n_exp = np.array([10, 12, 14, 16, 18, 20, 22, 24], dtype=float)
    _, best_e = fit_candidates(n_exp, 1e-4 * (1.45 ** n_exp))
    assert best_e is not None
    assert best_e.model == "exponential"
    assert best_e.r2 > 0.999
    assert best_e.params["c"] == pytest.approx(1.45, rel=1e-2)


def test_complexity_pipeline_generates_tables_and_figures(tmp_path):
    runs_path, profile_path, _ = _make_mock_dataset(tmp_path)
    out_dir = tmp_path / "out_complexity"

    assert complexity_main(["--runs", str(runs_path), "--profile", str(profile_path), "--out", str(out_dir)]) == 0

    alg_csv = out_dir / "tables" / "complexity_algorithms.csv"
    sub_csv = out_dir / "tables" / "complexity_subroutines.csv"
    assert alg_csv.exists() and sub_csv.exists()

    alg_df = pd.read_csv(alg_csv)
    assert {"algorithm", "metric", "model", "is_best", "r2", "rmse", "aic"} <= set(alg_df.columns)
    assert (alg_df["is_best"]).sum() >= 4
    assert "bnb_dfs_one_tree:christofides" in set(alg_df["algorithm"])
    assert "bnb_dfs_one_tree:trivial" in set(alg_df["algorithm"])

    sub_df = pd.read_csv(sub_csv)
    matching_best = sub_df[(sub_df["algorithm"] == "christofides") & (sub_df["section"] == "matching") & (sub_df["metric"] == "time_s") & sub_df["is_best"]]
    assert not matching_best.empty
    assert matching_best.iloc[0]["model"] in {"cubic", "power"}

    for fname in (
        "complexity_algorithms_time.png",
        "complexity_algorithms_memory.png",
        "complexity_subroutines_time.png",
        "complexity_subroutines_memory.png",
    ):
        p = out_dir / "figures" / fname
        assert p.exists() and p.stat().st_size > 1000


# --------------------------------------------------------------------------- #
# warmstart.py
# --------------------------------------------------------------------------- #


def test_warmstart_pipeline_generates_tables_and_figures(tmp_path):
    runs_path, _, _ = _make_mock_dataset(tmp_path)
    out_dir = tmp_path / "out_ws"

    assert warmstart_main(["--runs", str(runs_path), "--out", str(out_dir)]) == 0

    summary_csv = out_dir / "tables" / "warmstart_summary.csv"
    collapse_csv = out_dir / "tables" / "warmstart_collapse.csv"
    assert summary_csv.exists() and collapse_csv.exists()

    collapse = pd.read_csv(collapse_csv).set_index("warm_start")
    assert collapse.loc["trivial", "first_timeout_n"] == 30
    assert collapse.loc["christofides", "first_timeout_n"] == 40
    assert collapse.loc["christofides", "max_n_ok"] == 36

    summary = pd.read_csv(summary_csv)
    chris_28 = summary[(summary["warm_start"] == "christofides") & (summary["n"] == 28)].iloc[0]
    assert chris_28["expanded_reduction_pct"] > 50.0
    assert chris_28["speedup_vs_trivial"] > 1.0

    for fname in ("warmstart_nodes.png", "warmstart_tradeoff.png", "warmstart_collapse.png"):
        p = out_dir / "figures" / fname
        assert p.exists() and p.stat().st_size > 1000


# --------------------------------------------------------------------------- #
# sensitivity.py
# --------------------------------------------------------------------------- #


def test_sensitivity_pipeline_generates_tables_and_figures(tmp_path):
    runs_path, _, features_path = _make_mock_dataset(tmp_path)
    out_dir = tmp_path / "out_sens"

    assert sensitivity_main(["--runs", str(runs_path), "--features", str(features_path), "--out", str(out_dir)]) == 0

    for fname in (
        "sensitivity_spearman.csv",
        "sensitivity_regression.csv",
        "sensitivity_tree_importance.csv",
        "sensitivity_loocv.csv",
    ):
        assert (out_dir / "tables" / fname).exists()

    loocv = pd.read_csv(out_dir / "tables" / "sensitivity_loocv.csv")
    assert set(loocv["algorithm"]) == {"tatt", "christofides", "nn", "hull"}
    assert set(loocv["model"]) == {"linear_standardized", "decision_tree_d3"}
    assert loocv["note"].str.contains("Amostra pequena").all()

    spearman = pd.read_csv(out_dir / "tables" / "sensitivity_spearman.csv")
    assert set(spearman["feature"]) == set(FEATURE_NAMES)

    for fname in (
        "sensitivity_spearman_heatmap.png",
        "sensitivity_linear_coefs.png",
        "sensitivity_tree_tatt.png",
        "sensitivity_tree_christofides.png",
        "sensitivity_tree_nn.png",
        "sensitivity_tree_hull.png",
    ):
        p = out_dir / "figures" / fname
        assert p.exists() and p.stat().st_size > 1000
