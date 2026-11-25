from pathlib import Path

import pandas as pd
import pytest

from src.run_experiments import (
    CONSTRUCTIVE_ALGORITHMS,
    WARM_START_ORDER,
    _run_bnb_isolated,
    backfill_optima_and_gaps,
    main as run_experiments_main,
    run_all,
)
from src.tsp.bnb import WARM_STARTS, bnb, main as bnb_main
from src.tsp.common import (
    PROFILE_COLUMNS,
    RUNS_COLUMNS,
    STATUS_OK,
    STATUS_TIMEOUT,
    Profiler,
    ResultWriter,
    instance_from_coords,
    load_tsplib,
    make_run_row,
    read_jsonl,
    write_tsplib,
)

INSTANCES_DIR = Path(__file__).resolve().parents[1] / "data" / "instances"
TSPLIB_DIR = Path(__file__).resolve().parents[1] / "data" / "tsplib"


def test_warm_starts_registry_complete():
    assert set(WARM_STARTS) == {"trivial", "nn", "hull", "tatt", "christofides"}
    assert set(WARM_START_ORDER) == set(WARM_STARTS)
    assert set(WARM_STARTS) <= set(CONSTRUCTIVE_ALGORITHMS)


@pytest.mark.parametrize("ws", ["tatt", "christofides"])
def test_bnb_cli_accepts_dev_b_warm_starts(ws, tmp_path, capsys):
    inst = instance_from_coords("sq5", [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
    p = tmp_path / "sq5.tsp"
    write_tsplib(inst, p)
    rc = bnb_main([str(p), "--warm", ws, "--bound", "one_tree", "--time", "10"])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"warm={ws}" in out
    assert "status=OK" in out


def test_end_to_end_pipeline_and_resume(tmp_path):
    out_dir = tmp_path / "results_e2e"
    summary = run_all(
        instances_dir=INSTANCES_DIR,
        tsplib_dir=TSPLIB_DIR,
        out_dir=out_dir,
        time_limit_s=15.0,
        hard_grace_s=10.0,
        reps=2,
        search="dfs",
        bound="one_tree",
        workers=2,
        mem_bnb_max_n=14,
        max_n=16,
        run_analysis=True,
    )

    # 3 instâncias em data/instances com N <= 16: N=12, 14, 16
    assert summary["instances"] == 3
    assert summary["constructive_rows"] == 3 * len(CONSTRUCTIVE_ALGORITHMS) * 2
    assert summary["bnb_rows"] == 3 * len(WARM_START_ORDER)

    runs = read_jsonl(out_dir / "runs.jsonl")
    profs = read_jsonl(out_dir / "profile.jsonl")
    assert all(list(r.keys()) == RUNS_COLUMNS for r in runs)
    assert all(list(p.keys()) == PROFILE_COLUMNS for p in profs)

    df = pd.DataFrame(runs)
    # Todos os B&B nessas 3 instâncias pequenas devem resolver com OK e mesmo custo ótimo
    bnb_df = df[df["algorithm"] == "bnb_dfs_one_tree"]
    assert (bnb_df["status"] == STATUS_OK).all()
    for inst_name, grp in bnb_df.groupby("instance"):
        assert grp["cost"].nunique() == 1, f"ótimos divergentes em {inst_name}"

    # Backfill deve ter preenchido optimal_ref e gap para as construtivas
    const_df = df[df["algorithm"].isin(["tatt", "christofides", "nn", "hull"])]
    assert const_df["optimal_ref"].notna().all()
    assert const_df["gap"].notna().all()
    assert (const_df["gap"] >= -1e-12).all()

    # Passada de memória separada deve ter preenchido peak_mem_kb em rep==0 das construtivas
    rep0_const = const_df[const_df["rep"] == 0]
    assert rep0_const["peak_mem_kb"].notna().all()
    assert (rep0_const["peak_mem_kb"] > 0).all()

    # Passada de memória separada no B&B para N <= 14
    small_bnb = bnb_df[bnb_df["n"] <= 14]
    assert small_bnb["peak_mem_kb"].notna().all()

    # Features e tabelas/figuras de análise geradas
    assert (out_dir / "features.csv").exists()
    assert (out_dir / "tables" / "complexity_algorithms.csv").exists()
    assert (out_dir / "tables" / "warmstart_summary.csv").exists()
    assert (out_dir / "figures" / "warmstart_nodes.png").exists()

    # Resume não deve reexecutar jobs já concluídos
    n_before = len(runs)
    summary2 = run_all(
        instances_dir=INSTANCES_DIR,
        tsplib_dir=TSPLIB_DIR,
        out_dir=out_dir,
        time_limit_s=15.0,
        reps=2,
        workers=2,
        max_n=16,
        resume=True,
    )
    assert summary2["constructive_rows"] == 0
    assert summary2["bnb_rows"] == 0
    assert len(read_jsonl(out_dir / "runs.jsonl")) == n_before


def test_backfill_preserves_timeout_na(tmp_path):
    p = tmp_path / "runs.jsonl"
    with ResultWriter(p) as w:
        w.write(make_run_row(instance="x", n=20, algorithm="tatt", status=STATUS_OK, cost=110))
        w.write(make_run_row(instance="x", n=20, algorithm="bnb_dfs_one_tree", warm_start="hull", status=STATUS_OK, cost=100))
        w.write(make_run_row(instance="x", n=20, algorithm="bnb_dfs_one_tree", warm_start="trivial", status=STATUS_TIMEOUT, cost=None, expanded=5000))

    optima = backfill_optima_and_gaps(p, {})
    assert optima["x"] == 100
    rows = read_jsonl(p)
    assert rows[0]["optimal_ref"] == 100 and rows[0]["gap"] == pytest.approx(0.1)
    assert rows[1]["gap"] == 0.0
    assert rows[2]["status"] == STATUS_TIMEOUT
    assert rows[2]["cost"] is None and rows[2]["gap"] is None
    assert rows[2]["optimal_ref"] == 100 and rows[2]["expanded"] == 5000


def test_isolated_bnb_handles_timeout():
    p = INSTANCES_DIR / "eil51.tsp"
    if not p.exists():
        pytest.skip("eil51.tsp não gerado em data/instances/")
    row, _ = _run_bnb_isolated(
        inst_path=p,
        inst_name="eil51",
        inst_n=51,
        warm_name="trivial",
        time_limit_s=0.3,
        hard_grace_s=10.0,
        search="dfs",
        bound="two_min",
        optimal_ref=426,
        mem_bnb_max_n=0,
    )
    assert row["status"] == STATUS_TIMEOUT
    assert row["cost"] is None and row["gap"] is None
    assert row["expanded"] > 0


def test_cli_help_and_small_run(tmp_path):
    out_dir = tmp_path / "cli_out"
    rc = run_experiments_main([
        "--instances", str(INSTANCES_DIR),
        "--out", str(out_dir),
        "--max-n", "12",
        "--reps", "1",
        "--time-limit", "10",
        "--workers", "1",
        "--skip-bnb",
    ])
    assert rc == 0
    assert (out_dir / "runs.jsonl").exists()
    assert (out_dir / "features.csv").exists()
