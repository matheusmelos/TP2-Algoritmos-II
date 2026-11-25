from pathlib import Path

from src.benchmark_audit import (
    benchmark_christofides_ablation,
    benchmark_dlist_vs_numpy,
    benchmark_memory_and_gc,
    benchmark_search_and_bounds,
)

INSTANCES_DIR = Path(__file__).resolve().parents[1] / "data" / "instances"
TSPLIB_DIR = Path(__file__).resolve().parents[1] / "data" / "tsplib"


def test_audit_benchmarks_generate_csvs_and_figures(tmp_path):
    df_d = benchmark_dlist_vs_numpy(
        INSTANCES_DIR,
        tmp_path,
        instance_names=("eil51_random12",),
        time_limit_s=5.0,
    )
    assert len(df_d) == 2
    assert (df_d["expanded_dlist"] == df_d["expanded_numpy"]).all()
    assert (tmp_path / "audit_dlist_vs_numpy.csv").exists()
    assert (tmp_path / "audit_dlist_vs_numpy.png").exists()

    df_sb = benchmark_search_and_bounds(
        INSTANCES_DIR,
        tmp_path,
        instance_names=("eil51_random12",),
        time_limit_s=5.0,
    )
    assert len(df_sb) == 4
    assert df_sb["best_cost"].nunique() == 1
    assert (tmp_path / "audit_search_bounds.csv").exists()
    assert (tmp_path / "audit_search_bounds.png").exists()

    df_ab = benchmark_christofides_ablation(TSPLIB_DIR, tmp_path, reps=1)
    assert not df_ab.empty
    assert (df_ab["gap_exact"] <= 0.5).all()
    assert (tmp_path / "audit_christofides_ablation.csv").exists()
    assert (tmp_path / "audit_christofides_ablation.png").exists()

    df_mg = benchmark_memory_and_gc(
        TSPLIB_DIR,
        tmp_path,
        names=("eil51", "st70"),
        reps=1,
    )
    assert len(df_mg) == 2
    assert (df_mg["prim_copy_mem_kb"] > df_mg["prim_clean_mem_kb"]).all()
    assert (df_mg["tracemalloc_slowdown"] > 1.0).all()
    assert (tmp_path / "audit_memory_gc.csv").exists()
    assert (tmp_path / "audit_memory_gc.png").exists()
