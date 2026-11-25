from pathlib import Path

import numpy as np
import pytest

from src.tsp.common import instance_from_coords, load_tsplib
from src.tsp.features import (
    FEATURE_NAMES,
    extract_features,
    extract_features_table,
    main as features_main,
    write_features_csv,
)

TSPLIB_DIR = Path(__file__).resolve().parents[1] / "data" / "tsplib"


def test_square_features():
    inst = instance_from_coords("sq", [(0, 0), (10, 0), (10, 10), (0, 10)])
    feats = extract_features(inst)
    assert tuple(feats.keys()) == FEATURE_NAMES
    assert feats["n"] == 4.0
    assert feats["aspect_ratio"] == pytest.approx(1.0)
    assert feats["hull_fraction"] == pytest.approx(1.0)
    assert feats["spatial_entropy"] == pytest.approx(1.0)
    assert feats["clark_evans"] > 1.0
    assert feats["nn_dist_cv"] == pytest.approx(0.0)
    assert feats["mst_mean_degree"] == pytest.approx(2 * 3 / 4)


def test_regular_grid_has_high_clark_evans_and_zero_nn_cv():
    coords = [(x * 10.0, y * 10.0) for x in range(5) for y in range(5)]
    inst = instance_from_coords("grid25", coords)
    feats = extract_features(inst)
    assert feats["n"] == 25.0
    assert feats["aspect_ratio"] == pytest.approx(1.0)
    assert feats["clark_evans"] > 1.0
    assert feats["nn_dist_cv"] == pytest.approx(0.0)
    assert feats["spatial_entropy"] == pytest.approx(1.0)
    assert feats["mst_edge_cv"] == pytest.approx(0.0)
    assert 0.0 < feats["hull_fraction"] < 1.0


def test_clustered_points_have_low_clark_evans():
    rng = np.random.default_rng(123)
    centers = [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0)]
    pts = np.vstack([rng.normal(loc=c, scale=2.0, size=(10, 2)) for c in centers])
    inst = instance_from_coords("clusters40", pts)
    feats = extract_features(inst)
    assert feats["clark_evans"] < 0.5
    assert feats["mst_edge_cv"] > 1.0  # arestas longas entre clusters vs curtas dentro


def test_aspect_ratio_elongated_rectangle():
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 100.0, 20)
    y = rng.uniform(0.0, 10.0, size=20)
    y[0], y[-1] = 0.0, 10.0
    inst = instance_from_coords("rect", np.c_[x, y])
    feats = extract_features(inst)
    assert feats["aspect_ratio"] == pytest.approx(0.1)


def test_star_topology_mst_max_degree():
    # Centro (0, 0) e 5 pontos em círculo de raio 100: distância centro-folha = 100,
    # distância entre folhas adjacentes = 2*100*sin(36°) ≈ 117.6 > 100 -> MST é uma estrela!
    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    leaves = np.c_[100.0 * np.cos(ang), 100.0 * np.sin(ang)]
    coords = np.vstack([[[0.0, 0.0]], leaves])
    inst = instance_from_coords("star6", coords)
    feats = extract_features(inst)
    assert feats["mst_max_degree"] == 5.0
    assert feats["mst_mean_degree"] == pytest.approx(2 * 5 / 6)


@pytest.mark.parametrize(
    "coords",
    [
        [(x * 10.0, 0.0) for x in range(6)],  # colineares
        [(4.0, 4.0)] * 5,  # coincidentes
    ],
)
def test_degenerate_inputs_are_finite(coords):
    inst = instance_from_coords("deg", coords)
    feats = extract_features(inst)
    assert all(np.isfinite(v) for v in feats.values())


def test_tsplib_features_and_csv_cli(tmp_path):
    files = [TSPLIB_DIR / name for name in ("eil51.tsp", "berlin52.tsp", "st70.tsp")]
    if not all(f.exists() for f in files):
        pytest.skip("Arquivos TSPLIB ausentes")

    insts = [load_tsplib(f) for f in files]
    rows = extract_features_table(insts)
    assert [r["instance"] for r in rows] == ["eil51", "berlin52", "st70"]
    # berlin52 é conhecida por ser mais clusterizada que eil51 (≈ uniforme)
    by_name = {r["instance"]: r for r in rows}
    assert by_name["berlin52"]["clark_evans"] < by_name["eil51"]["clark_evans"]

    out_csv = tmp_path / "features.csv"
    write_features_csv(rows, out_csv)
    assert out_csv.exists()
    header = out_csv.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(["instance", *FEATURE_NAMES])

    out_cli = tmp_path / "features_cli.csv"
    assert features_main(["--instances", str(TSPLIB_DIR), "--out", str(out_cli)]) == 0
    assert out_cli.exists()
