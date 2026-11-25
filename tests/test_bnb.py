import itertools

import numpy as np
import pytest

from src.tsp.bnb import BOUNDS, SEARCHES, WARM_STARTS, bnb
from src.tsp.common import (
    RUNS_COLUMNS,
    STATUS_OK,
    STATUS_TIMEOUT,
    Profiler,
    instance_from_coords,
    tour_cost,
    validate_tour,
)

CONFIGS = [(s, b) for s in SEARCHES for b in BOUNDS]


def brute_force(D) -> int:
    D = D.tolist()
    n = len(D)
    best = None
    for perm in itertools.permutations(range(1, n)):
        c = D[0][perm[0]] + D[perm[-1]][0]
        for i in range(len(perm) - 1):
            c += D[perm[i]][perm[i + 1]]
        if best is None or c < best:
            best = c
    return best


_CACHE: dict = {}


def random_instance(n: int, seed: int, scale: float = 1000.0):
    key = ("rand", n, seed, scale)
    if key not in _CACHE:
        rng = np.random.default_rng(seed)
        inst = instance_from_coords(f"r{n}_{seed}", rng.uniform(0, scale, size=(n, 2)))
        _CACHE[key] = (inst, brute_force(inst.D))
    return _CACHE[key]


@pytest.mark.parametrize("search,bound", CONFIGS)
@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_brute_force(n, seed, search, bound):
    inst, opt = random_instance(n, seed)
    res = bnb(inst, None, time_limit_s=60, search=search, bound=bound)
    assert res.status == STATUS_OK
    assert res.best_cost == opt
    validate_tour(inst.n, res.best_tour)
    assert tour_cost(inst.D, res.best_tour) == res.best_cost
    assert res.root_lb <= opt
    assert res.pruned <= res.generated


@pytest.mark.parametrize("search,bound", CONFIGS)
def test_ties_and_duplicates(search, bound):
    # grade 3x3 (muitos empates) + pontos duplicados (arestas de custo 0)
    grid = [(x, y) for x in range(3) for y in range(3)]
    for coords in (grid, grid[:6] + [grid[0], grid[4]]):
        inst = instance_from_coords("ties", np.array(coords, dtype=float) * 10)
        res = bnb(inst, None, time_limit_s=60, search=search, bound=bound)
        assert res.status == STATUS_OK
        assert res.best_cost == brute_force(inst.D)


@pytest.mark.parametrize("search,bound", CONFIGS)
@pytest.mark.parametrize("warm", sorted(WARM_STARTS))
def test_warm_starts(search, bound, warm):
    inst, opt = random_instance(9, 7)
    ub = WARM_STARTS[warm](inst, Profiler())
    res = bnb(inst, ub, time_limit_s=60, search=search, bound=bound)
    assert res.status == STATUS_OK
    assert res.best_cost == opt
    assert res.initial_ub == tour_cost(inst.D, ub)
    assert res.best_cost <= res.initial_ub


def test_optimal_warm_start_has_no_improvement():
    inst, opt = random_instance(8, 3)
    first = bnb(inst, None, time_limit_s=60)
    again = bnb(inst, first.best_tour, time_limit_s=60)
    assert again.best_cost == opt
    assert again.ub_improvements == 0
    assert again.time_to_best_s == 0.0


def test_accepts_rotated_tour_and_rejects_invalid():
    inst, opt = random_instance(6, 0)
    res = bnb(inst, [3, 4, 5, 0, 1, 2], time_limit_s=60)
    assert res.best_cost == opt
    with pytest.raises(ValueError):
        bnb(inst, [0, 1, 2, 3, 4], time_limit_s=60)
    with pytest.raises(ValueError):
        bnb(inst, None, search="bfs")
    with pytest.raises(ValueError):
        bnb(inst, None, bound="lp")


@pytest.mark.parametrize("search", SEARCHES)
def test_timeout_keeps_partial_stats(search):
    rng = np.random.default_rng(42)
    inst = instance_from_coords("big", rng.uniform(0, 1000, size=(60, 2)))
    res = bnb(inst, None, time_limit_s=0.2, search=search, bound="two_min")
    assert res.status == STATUS_TIMEOUT
    assert res.expanded > 0 and res.generated > 0
    assert res.best_cost <= res.initial_ub
    validate_tour(inst.n, res.best_tour)
    assert tour_cost(inst.D, res.best_tour) == res.best_cost

    row = res.to_row(inst, warm_start="trivial", time_heuristic_s=0.0)
    assert list(row) == RUNS_COLUMNS
    assert row["status"] == STATUS_TIMEOUT
    assert row["cost"] is None and row["gap"] is None  # NA, como pede o enunciado
    assert row["expanded"] == res.expanded


def test_to_row_ok_and_profiler_sections():
    inst, opt = random_instance(8, 1)
    prof = Profiler()
    res = bnb(inst, None, time_limit_s=60, prof=prof, bound="one_tree")
    assert {"bnb_setup", "bnb_search"} <= set(prof.sections())
    if res.mst_calls:
        assert prof.calls("bnb_mst") == res.mst_calls
    row = res.to_row(inst, warm_start="trivial", time_heuristic_s=0.01, optimal_ref=opt)
    assert row["algorithm"] == "bnb_dfs_one_tree"
    assert row["cost"] == opt and row["gap"] == 0
    assert row["time_total_s"] == pytest.approx(0.01 + res.time_s)


def test_one_tree_root_bound_is_at_least_two_min():
    inst, opt = random_instance(9, 5)
    r2 = bnb(inst, None, time_limit_s=60, bound="two_min")
    r1 = bnb(inst, None, time_limit_s=60, bound="one_tree")
    assert r2.root_lb <= r1.root_lb <= opt
