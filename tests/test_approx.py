import itertools
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree

from src.tsp.approx import (
    APPROXIMATIONS,
    christofides,
    christofides_greedy_matching,
    main as approx_main,
    twice_around_the_tree,
)
from src.tsp.common import (
    OPTIMA,
    Profiler,
    instance_from_coords,
    load_tsplib,
    tour_cost,
    validate_tour,
)
from src.tsp.mst import mst_adjacency, mst_degrees, mst_edges, mst_weight, prim_mst

TSPLIB_DIR = Path(__file__).resolve().parents[1] / "data" / "tsplib"


def _rand(n: int, seed: int = 0, scale: float = 1000.0):
    rng = np.random.default_rng(seed)
    return instance_from_coords(f"r{n}_{seed}", rng.uniform(0, scale, size=(n, 2)))


def _brute_force(D: np.ndarray) -> int:
    Dl = D.tolist()
    n = len(Dl)
    best = None
    for perm in itertools.permutations(range(1, n)):
        c = Dl[0][perm[0]] + Dl[perm[-1]][0]
        for i in range(len(perm) - 1):
            c += Dl[perm[i]][perm[i + 1]]
        if best is None or c < best:
            best = c
    return best


# --------------------------------------------------------------------------- #
# Prim MST
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [3, 5, 12, 40, 80])
@pytest.mark.parametrize("seed", [0, 7])
def test_prim_mst_matches_scipy(n, seed):
    inst = _rand(n, seed=seed)
    prof = Profiler()
    parent = prim_mst(inst.D, prof)

    assert prof.calls("mst") == 1
    assert prof.total("mst") >= 0.0
    assert parent.shape == (n,)
    assert parent[0] == -1

    edges = mst_edges(parent)
    assert len(edges) == n - 1

    # Verifica conectividade e aciclicidade
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    assert nx.is_tree(G)

    # Verifica peso ótimo contra scipy
    scipy_weight = int(round(minimum_spanning_tree(inst.D).sum()))
    assert mst_weight(inst.D, parent) == scipy_weight

    # Verifica consistência de graus e adjacência
    deg = mst_degrees(parent)
    adj = mst_adjacency(parent)
    assert deg.sum() == 2 * (n - 1)
    assert [len(adj[v]) for v in range(n)] == deg.tolist()


def test_prim_mst_alternate_root_and_validation():
    inst = _rand(15, seed=3)
    p0 = prim_mst(inst.D, root=0)
    p5 = prim_mst(inst.D, root=5)
    assert p5[5] == -1
    assert mst_weight(inst.D, p0) == mst_weight(inst.D, p5)

    with pytest.raises(ValueError):
        prim_mst(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        prim_mst(inst.D, root=15)


# --------------------------------------------------------------------------- #
# Validade, profiler e casos degenerados
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(APPROXIMATIONS))
@pytest.mark.parametrize("n", [3, 4, 10, 50, 120])
def test_valid_tours(name, n):
    inst = _rand(n, seed=n + 10)
    tour = APPROXIMATIONS[name](inst, Profiler())
    validate_tour(inst.n, tour)
    assert all(isinstance(v, int) for v in tour)


@pytest.mark.parametrize("name", sorted(APPROXIMATIONS))
@pytest.mark.parametrize(
    "coords",
    [
        [(x, 2 * x) for x in range(8)],  # colineares
        [(5, 5)] * 6,  # todos coincidentes (pesos 0)
        [(0, 0), (0, 0), (10, 0), (10, 10), (0, 10), (5, 5), (5, 5)],  # duplicatas
        [(x * 10, y * 10) for x in range(3) for y in range(3)],  # grade 3x3 (muitos empates)
    ],
)
def test_degenerate_inputs(name, coords):
    inst = instance_from_coords("deg", coords)
    tour = APPROXIMATIONS[name](inst, Profiler())
    validate_tour(inst.n, tour)


def test_profiler_sections_time_and_memory():
    inst = _rand(40, seed=1)

    p_tatt = Profiler(mem=True)
    twice_around_the_tree(inst, p_tatt)
    assert set(p_tatt.sections()) == {"mst", "preorder"}
    assert all(p_tatt.peak_kb(s) is not None for s in p_tatt.sections())

    for fn in (christofides, christofides_greedy_matching):
        p = Profiler(mem=True)
        fn(inst, p)
        assert set(p.sections()) == {"mst", "odd", "matching", "euler", "shortcut"}
        assert all(p.peak_kb(s) is not None for s in p.sections())


# --------------------------------------------------------------------------- #
# Garantias de aproximação: 2·OPT (TATT) e 1.5·OPT (Christofides)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_approximation_ratios_against_brute_force(n, seed):
    inst = _rand(n, seed=seed, scale=10_000.0)
    opt = _brute_force(inst.D)

    tatt_tour = twice_around_the_tree(inst, Profiler())
    chris_tour = christofides(inst, Profiler())
    greedy_tour = christofides_greedy_matching(inst, Profiler())

    for t in (tatt_tour, chris_tour, greedy_tour):
        validate_tour(inst.n, t)

    c_tatt = tour_cost(inst.D, tatt_tour)
    c_chris = tour_cost(inst.D, chris_tour)

    assert opt <= c_tatt <= 2.0 * opt + 1e-9
    assert opt <= c_chris <= 1.5 * opt + 1e-9


@pytest.mark.parametrize("name", ["eil51", "berlin52", "st70"])
def test_approximation_ratios_on_tsplib_optima(name):
    path = TSPLIB_DIR / f"{name}.tsp"
    if not path.exists():
        pytest.skip(f"{name}.tsp ausente em data/tsplib/")

    inst = load_tsplib(path)
    opt = OPTIMA[name]

    tatt_tour = twice_around_the_tree(inst, Profiler())
    chris_tour = christofides(inst, Profiler())
    greedy_tour = christofides_greedy_matching(inst, Profiler())

    for t in (tatt_tour, chris_tour, greedy_tour):
        validate_tour(inst.n, t)

    c_tatt = tour_cost(inst.D, tatt_tour)
    c_chris = tour_cost(inst.D, chris_tour)

    assert opt <= c_tatt <= 2.0 * opt
    assert opt <= c_chris <= 1.5 * opt


def test_approx_cli(capsys):
    path = TSPLIB_DIR / "eil51.tsp"
    if not path.exists():
        pytest.skip("eil51.tsp ausente")
    rc = approx_main([str(path), "--alg", "christofides"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "matching" in out
    assert "eil51" in out
