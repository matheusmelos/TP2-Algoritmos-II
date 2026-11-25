import itertools

import numpy as np
import pytest

from src.tsp.common import Profiler, instance_from_coords, tour_cost, validate_tour
from src.tsp.heuristics import (
    HEURISTICS,
    convex_hull_insertion,
    nearest_neighbor,
    trivial,
)


def _rand(n, seed=0):
    rng = np.random.default_rng(seed)
    return instance_from_coords(f"r{n}", rng.uniform(0, 1000, size=(n, 2)))


@pytest.mark.parametrize("name", sorted(HEURISTICS))
@pytest.mark.parametrize("n", [3, 4, 10, 50, 120])
def test_valid_tours(name, n):
    inst = _rand(n, seed=n)
    tour = HEURISTICS[name](inst, Profiler())
    validate_tour(inst.n, tour)
    assert all(isinstance(v, int) for v in tour)


def test_trivial():
    inst = _rand(7)
    assert trivial(inst, Profiler()) == list(range(7))


def test_nearest_neighbor_on_line():
    # pontos numa reta em ordem embaralhada: NN a partir de 0 percorre a reta
    xs = [0, 30, 10, 20, 40]
    inst = instance_from_coords("line", [(x, 0) for x in xs])
    tour = nearest_neighbor(inst, Profiler())
    assert [xs[v] for v in tour] == [0, 10, 20, 30, 40]


def test_hull_insertion_convex_position_is_optimal():
    # pontos em posição convexa: o tour ótimo é a ordem do fecho
    # octógono com leve perturbação: ângulos bem separados evitam empates pelo nint
    rng = np.random.default_rng(1)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False) + rng.uniform(-0.1, 0.1, size=8)
    pts = np.c_[1000 * np.cos(ang), 1000 * np.sin(ang)]
    pts = pts[rng.permutation(8)]
    inst = instance_from_coords("circle", pts)
    tour = convex_hull_insertion(inst, Profiler())
    D = inst.D.tolist()
    opt = min(
        D[0][p[0]] + D[p[-1]][0] + sum(D[p[i]][p[i + 1]] for i in range(len(p) - 1))
        for p in itertools.permutations(range(1, 8))
    )
    assert tour_cost(inst.D, tour) == opt


@pytest.mark.parametrize(
    "coords",
    [
        [(x, 2 * x) for x in range(6)],             # colineares (fecho degenerado)
        [(5, 5)] * 5,                               # todos coincidentes
        [(0, 0), (0, 0), (10, 0), (10, 10), (0, 10), (5, 5), (5, 5)],  # duplicatas
        [(0, 0), (10, 0), (5, 0), (10, 10), (0, 10)],  # ponto sobre aresta do fecho
    ],
)
def test_hull_insertion_degenerate_inputs(coords):
    inst = instance_from_coords("deg", coords)
    tour = convex_hull_insertion(inst, Profiler())
    validate_tour(inst.n, tour)


def test_hull_insertion_profiler_sections():
    prof = Profiler()
    convex_hull_insertion(_rand(30), prof)
    assert {"hull", "insertion"} <= set(prof.sections())


def test_hull_insertion_beats_trivial_on_random():
    inst = _rand(80, seed=3)
    p = Profiler()
    assert tour_cost(inst.D, convex_hull_insertion(inst, p)) < tour_cost(inst.D, trivial(inst, p))


def test_hull_insertion_matches_naive_cheapest_insertion():
    """O cache incremental deve escolher as mesmas inserções que a versão O(n³)."""
    from scipy.spatial import ConvexHull

    # escala grande: custos de inserção empatados (que poderiam levar a escolhas
    # de posição diferentes, igualmente válidas) ficam improváveis
    rng = np.random.default_rng(11)
    inst = instance_from_coords("big40", rng.uniform(0, 1e6, size=(40, 2)))
    D = inst.D
    tour = [int(v) for v in ConvexHull(inst.coords).vertices]
    rest = set(range(inst.n)) - set(tour)
    while rest:
        best = None
        for k in sorted(rest):
            for pos in range(len(tour)):
                i, j = tour[pos], tour[(pos + 1) % len(tour)]
                c = D[i, k] + D[k, j] - D[i, j]
                if best is None or c < best[0]:
                    best = (c, k, pos)
        _, k, pos = best
        tour.insert(pos + 1, k)
        rest.remove(k)
    naive_cost = tour_cost(D, tour)
    assert tour_cost(D, convex_hull_insertion(inst, Profiler())) == naive_cost
