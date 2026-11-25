import numpy as np
import pytest

from src.tsp.common import instance_from_coords, load_tsplib, write_tsplib
from src.tsp.make_instances import (
    SELECTORS,
    SUB_SPECS,
    FULL_INSTANCES,
    SubSpec,
    generate,
    read_manifest,
    subsample,
    unique_points,
)


@pytest.fixture
def src_dir(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    rng = np.random.default_rng(0)
    pts = rng.integers(0, 1000, size=(60, 2)).astype(float)
    pts[10] = pts[3]  # uma duplicata
    write_tsplib(instance_from_coords("fakeA", pts), d / "fakeA.tsp")
    write_tsplib(instance_from_coords("eil51", rng.uniform(0, 100, size=(51, 2))), d / "eil51.tsp")
    return d


def test_default_set_respects_statement():
    sizes = [s.n for s in SUB_SPECS]
    assert len(SUB_SPECS) + len(FULL_INSTANCES) in range(12, 17)
    assert all(10 <= n <= 40 for n in sizes)
    assert len(sizes) / (len(sizes) + len(FULL_INSTANCES)) >= 0.8
    assert len({s.name for s in SUB_SPECS}) == len(SUB_SPECS)


@pytest.mark.parametrize("mode", sorted(SELECTORS))
def test_subsample_returns_distinct_subset(mode):
    rng = np.random.default_rng(5)
    pts = rng.integers(0, 50, size=(80, 2)).astype(float)  # com várias duplicatas
    sub = subsample(pts, 20, mode, np.random.default_rng(1))
    assert sub.shape == (20, 2)
    assert len(unique_points(sub)) == 20
    src = {tuple(p) for p in pts}
    assert all(tuple(p) in src for p in sub)


def test_subsample_errors():
    pts = np.array([[0, 0], [1, 1], [1, 1], [2, 2]], dtype=float)
    with pytest.raises(ValueError):
        subsample(pts, 4, "random", np.random.default_rng(0))  # só 3 distintos
    with pytest.raises(ValueError):
        subsample(pts, 2, "spiral", np.random.default_rng(0))


def test_generate_writes_files_and_manifest(src_dir, tmp_path):
    specs = [SubSpec(12, "fakeA", "random"), SubSpec(15, "fakeA", "knn"), SubSpec(20, "fakeA", "window")]
    out = tmp_path / "out"
    rows = generate(src_dir, out, specs=specs, full=["eil51"], seed=7)

    assert [r["n"] for r in rows] == [12, 15, 20, 51]  # ordenado por n
    for spec in specs:
        inst = load_tsplib(out / f"{spec.name}.tsp")
        assert inst.n == spec.n and inst.name == spec.name

    man = read_manifest(out)
    assert [r["name"] for r in man] == [r["name"] for r in rows]
    full = [r for r in man if r["mode"] == "full"][0]
    assert full["optimal_ref"] == 426
    assert all(r["optimal_ref"] is None for r in man if r["mode"] != "full")


def test_generate_is_deterministic(src_dir, tmp_path):
    specs = [SubSpec(14, "fakeA", "window"), SubSpec(16, "fakeA", "random")]
    generate(src_dir, tmp_path / "a", specs=specs, full=[], seed=3)
    generate(src_dir, tmp_path / "b", specs=list(reversed(specs)), full=[], seed=3)
    for spec in specs:
        fa = (tmp_path / "a" / f"{spec.name}.tsp").read_text()
        fb = (tmp_path / "b" / f"{spec.name}.tsp").read_text()
        assert fa == fb  # independe da ordem das specs

    # outra seed -> outro recorte (compara coordenadas, não o texto, que tem a seed no COMMENT)
    generate(src_dir, tmp_path / "c", specs=specs[1:], full=[], seed=4)
    ca = load_tsplib(tmp_path / "a" / f"{specs[1].name}.tsp").coords
    cc = load_tsplib(tmp_path / "c" / f"{specs[1].name}.tsp").coords
    assert not np.array_equal(ca, cc)
