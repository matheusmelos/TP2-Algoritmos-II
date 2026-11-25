import json
from pathlib import Path

import numpy as np
import pytest

from src.tsp.common import (
    OPTIMA,
    PROFILE_COLUMNS,
    RUNS_COLUMNS,
    Profiler,
    ResultWriter,
    euc2d_matrix,
    gap,
    instance_from_coords,
    load_opt_tour,
    load_tsplib,
    make_run_row,
    normalize_tour,
    read_jsonl,
    tour_cost,
    validate_tour,
    write_tsplib,
)

TSPLIB_DIR = Path(__file__).resolve().parents[1] / "data" / "tsplib"

SQUARE_TSP = """NAME : square
COMMENT : quadrado 10x10
TYPE : TSP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0 0
2 10 0
3 10 10
4 0 10
EOF
"""


@pytest.fixture
def square_file(tmp_path):
    p = tmp_path / "square.tsp"
    p.write_text(SQUARE_TSP)
    return p


def test_parser_reads_header_and_coords(square_file):
    inst = load_tsplib(square_file)
    assert inst.name == "square"
    assert inst.n == 4
    assert inst.coords.shape == (4, 2)
    assert inst.D.dtype == np.int64
    assert inst.D[0, 1] == 10
    assert inst.D[0, 2] == 14  # nint(14.142)
    assert np.array_equal(inst.D, inst.D.T)
    assert np.all(np.diag(inst.D) == 0)


def test_parser_accepts_gzip(tmp_path):
    import gzip

    p = tmp_path / "square.tsp.gz"
    p.write_bytes(gzip.compress(SQUARE_TSP.encode()))
    assert load_tsplib(p).n == 4


def test_nint_rule():
    # sqrt(2) = 1.414 -> 1 ; sqrt(2.25+0) = 1.5 -> 2 (arredonda .5 para cima) ; 2.4 -> 2
    D = euc2d_matrix(np.array([[0, 0], [1, 1], [1.5, 0], [0, 2.4]]))
    assert D[0, 1] == 1
    assert D[0, 2] == 2
    assert D[0, 3] == 2


@pytest.mark.parametrize("ewt", ["GEO", "ATT", "EXPLICIT", "CEIL_2D"])
def test_rejects_non_euc2d(tmp_path, ewt):
    p = tmp_path / "x.tsp"
    p.write_text(SQUARE_TSP.replace("EUC_2D", ewt))
    with pytest.raises(ValueError, match="EUC_2D"):
        load_tsplib(p)


def test_tour_cost_square(square_file):
    inst = load_tsplib(square_file)
    assert tour_cost(inst.D, [0, 1, 2, 3]) == 40
    assert tour_cost(inst.D, [0, 2, 1, 3]) == 14 + 10 + 14 + 10


def test_validate_tour():
    validate_tour(4, [0, 1, 2, 3])
    validate_tour(4, [0, 3, 1, 2])
    for bad in ([1, 0, 2, 3], [0, 1, 2], [0, 1, 1, 3], [0, 1, 2, 4], []):
        with pytest.raises(ValueError):
            validate_tour(4, bad)


def test_normalize_tour():
    assert normalize_tour([2, 3, 0, 1]) == [0, 1, 2, 3]
    assert normalize_tour([0, 1, 2, 0]) == [0, 1, 2]  # aceita tour fechado


def test_d_list_matches_matrix(square_file):
    inst = load_tsplib(square_file)
    assert inst.D_list == inst.D.tolist()
    assert inst.D_list is inst.D_list  # cacheado


def test_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    inst = instance_from_coords("rand20", rng.uniform(0, 1000, size=(20, 2)))
    p = tmp_path / "rand20.tsp"
    write_tsplib(inst, p, comment="teste")
    back = load_tsplib(p)
    assert back.name == "rand20"
    assert np.allclose(back.coords, inst.coords)
    assert np.array_equal(back.D, inst.D)


def test_opt_tour_parser(tmp_path):
    p = tmp_path / "sq.opt.tour"
    p.write_text("NAME : sq\nTYPE : TOUR\nDIMENSION : 4\nTOUR_SECTION\n3\n4\n1\n2\n-1\nEOF\n")
    assert load_opt_tour(p) == [0, 1, 2, 3]


def test_profiler_accumulates_time():
    prof = Profiler()
    for _ in range(3):
        with prof.section("a"):
            sum(range(1000))
    with prof.section("b"):
        pass
    assert prof.calls("a") == 3
    assert prof.total("a") > 0
    assert prof.total("missing") == 0.0
    assert prof.peak_kb("a") is None
    rows = prof.records(instance="x", n=5, algorithm="alg", rep=0)
    assert {r["section"] for r in rows} == {"a", "b"}
    assert all(set(r) == set(PROFILE_COLUMNS) for r in rows)
    prof.reset()
    assert prof.sections() == []


def test_profiler_memory_nested():
    prof = Profiler(mem=True)
    with prof.section("outer"):
        with prof.section("inner"):
            big = np.ones(1_000_000)  # ~7.8 MB
            del big
        small = np.ones(1000)
        del small
    assert prof.peak_kb("inner") > 7000
    assert prof.peak_kb("outer") >= prof.peak_kb("inner")


def test_run_row_schema():
    row = make_run_row(instance="x", n=10, algorithm="bnb", status="OK", cost=100)
    assert list(row) == RUNS_COLUMNS
    assert row["expanded"] is None
    with pytest.raises(KeyError):
        make_run_row(foo=1)


def test_gap():
    assert gap(110, 100) == pytest.approx(0.1)
    assert gap(None, 100) is None
    assert gap(100, None) is None


def test_result_writer_roundtrip(tmp_path):
    p = tmp_path / "out" / "runs.jsonl"
    with ResultWriter(p) as w:
        w.write(make_run_row(instance="a", n=np.int64(5), cost=np.float64(1.5)))
        w.write_many([make_run_row(instance="b"), make_run_row(instance="c")])
    rows = read_jsonl(p)
    assert [r["instance"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["n"] == 5
    json.dumps(rows)


@pytest.mark.parametrize("name", ["eil51", "berlin52", "st70"])
def test_known_optima(name):
    tsp, tour = TSPLIB_DIR / f"{name}.tsp", TSPLIB_DIR / f"{name}.opt.tour"
    if not (tsp.exists() and tour.exists()):
        pytest.skip(f"{name} não baixado (rode src/tsp/download_tsplib.py)")
    inst = load_tsplib(tsp)
    t = load_opt_tour(tour)
    validate_tour(inst.n, t)
    assert tour_cost(inst.D, t) == OPTIMA[name]
