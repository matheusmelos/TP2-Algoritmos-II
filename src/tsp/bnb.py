"""Branch-and-Bound exato para o TSP simétrico (Etapa 1 – Dev A).

    bnb(inst, ub_tour, time_limit_s, prof, search="dfs", bound="two_min") -> BnBResult

Modelo da busca
---------------
Nó = caminho parcial 0 → … → u (a cidade 0 é fixa como início), bitmask de
visitados, custo acumulado do caminho e lower bound. Filhos = estender o caminho
com cada cidade não visitada.

Quebra de simetria: cada ciclo aparece em dois sentidos ([0, a, …, b] e
[0, b, …, a]). Exigimos a < b, onde a = path[1] e b = última cidade. Para podar
cedo, cada nó guarda ``big`` = nº de cidades não visitadas maiores que a; se
``big`` chega a 0 com cidades ainda por visitar, não existe última cidade válida.

Lower bounds
------------
* ``two_min`` (Levitin): cada vértice contribui com suas duas arestas no tour:
  as fixadas pelo caminho ou, se livres, as menores incidentes (m1 + m2). Assim
      2·custo(tour) ≥ 2·c(caminho) + Σ_{v∉caminho}(m1+m2) + outra(0) + outra(u)
  onde outra(v) = menor aresta de v diferente da já fixada. Atualização O(1) por
  filho (mantemos Σ_{v não visitado}(m1+m2) incrementalmente).
* ``one_tree``: max(two_min, c(caminho) + MST(U) + min d(u,U) + min d(0,U)), sendo
  U as cidades não visitadas. Qualquer completação u → U → 0 é uma aresta u–U,
  um caminho hamiltoniano em U (≥ MST(U)) e uma aresta U–0. Na raiz isso é o
  1-tree clássico. O MST é O(|U|²); como o mesmo conjunto U reaparece com ordens
  diferentes do caminho, o MST é memoizado pela bitmask (dict limitado). O two_min
  é avaliado antes (é barato) e o MST só é calculado se ele não podar.

Estratégias de busca
--------------------
* ``dfs``: profundidade com filhos ordenados pelo bound (recursiva, memória
  O(n²) no pior caso: n níveis × até n filhos pendentes). Encontra soluções
  completas cedo, o que melhora o UB rapidamente.
* ``best_first``: heap por (lb, -profundidade). Expande menos nós, mas a fronteira
  pode crescer exponencialmente. O caminho é guardado como tupla em cada nó.

Contadores (BnBResult)
----------------------
expanded   nós cujos filhos foram gerados (inclui a raiz)
generated  filhos avaliados (inclui completações de tour)
pruned     filhos descartados: lb ≥ UB (na geração ou ao sair da fronteira) ou
           por simetria (sem última cidade válida). Vale pruned ≤ generated.
max_depth  maior nº de cidades no caminho de um nó expandido
max_frontier maior nº de nós gerados e ainda não expandidos (pilha/heap)

Timeout: o relógio é checado a cada 1024 expansões. No estouro devolve
status=TIMEOUT, preservando os contadores parciais e o melhor tour encontrado.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .common import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_TIMEOUT,
    TIME_LIMIT_S,
    Instance,
    Profiler,
    gap,
    load_tsplib,
    make_run_row,
    normalize_tour,
    tour_cost,
    validate_tour,
)

SEARCHES = ("dfs", "best_first")
BOUNDS = ("two_min", "one_tree")
TIME_CHECK_MASK = 1023  # checa o relógio a cada 1024 expansões
MST_MEMO_MAX = 200_000  # limite de entradas do memo de MST (controle de memória)


class _Timeout(Exception):
    pass


@dataclass
class BnBResult:
    status: str
    best_cost: int
    best_tour: list[int]
    initial_ub: int
    root_lb: int
    search: str
    bound: str
    expanded: int = 0
    pruned: int = 0
    generated: int = 0
    max_depth: int = 0
    max_frontier: int = 0
    ub_improvements: int = 0
    time_s: float = 0.0
    time_to_best_s: float = 0.0
    mst_calls: int = 0
    mst_memo_hits: int = 0

    @property
    def algorithm(self) -> str:
        return f"bnb_{self.search}_{self.bound}"

    def to_row(
        self,
        inst: Instance,
        warm_start: str,
        time_heuristic_s: float | None = None,
        optimal_ref: int | None = None,
        rep: int = 0,
        peak_mem_kb: float | None = None,
    ) -> dict:
        """Linha no esquema RUNS_COLUMNS. Em TIMEOUT/ERROR o custo fica NA
        (o enunciado pede NA para execuções abortadas); os contadores ficam."""
        ok = self.status == STATUS_OK
        cost = self.best_cost if ok else None
        ref = optimal_ref if optimal_ref is not None else cost
        return make_run_row(
            instance=inst.name,
            n=inst.n,
            algorithm=self.algorithm,
            warm_start=warm_start,
            rep=rep,
            status=self.status,
            cost=cost,
            optimal_ref=ref,
            gap=gap(cost, ref),
            time_total_s=(time_heuristic_s or 0.0) + self.time_s,
            time_heuristic_s=time_heuristic_s,
            peak_mem_kb=peak_mem_kb,
            expanded=self.expanded,
            pruned=self.pruned,
            generated=self.generated,
            max_depth=self.max_depth,
            max_frontier=self.max_frontier,
            time_to_best_s=self.time_to_best_s,
        )


def _prim_weight(nodes: list[int], D: list[list[int]]) -> int:
    """Peso da MST (Prim denso, O(k²)) do subgrafo induzido por ``nodes``."""
    k = len(nodes)
    if k <= 1:
        return 0
    rest = nodes[1:]
    D0 = D[nodes[0]]
    key = [D0[v] for v in rest]
    total = 0
    while rest:
        i = min(range(len(key)), key=key.__getitem__)
        total += key[i]
        v = rest[i]
        last = len(rest) - 1
        rest[i] = rest[last]
        key[i] = key[last]
        rest.pop()
        key.pop()
        Dv = D[v]
        for j in range(len(rest)):
            d = Dv[rest[j]]
            if d < key[j]:
                key[j] = d
    return total


class _Search:
    def __init__(self, inst: Instance, bound: str, ub_cost: int, ub_tour: list[int], time_limit_s: float):
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + time_limit_s
        n = inst.n
        D = inst.D_list  # listas Python: acesso escalar bem mais rápido que numpy
        self.n, self.D = n, D

        # duas menores arestas incidentes de cada vértice (e o vizinho da menor)
        m1, m2, nb1 = [0] * n, [0] * n, [-1] * n
        for v in range(n):
            row = D[v]
            b1 = b2 = None
            a1 = -1
            for x in range(n):
                if x == v:
                    continue
                d = row[x]
                if b1 is None or d < b1:
                    b2, b1, a1 = b1, d, x
                elif b2 is None or d < b2:
                    b2 = d
            m1[v], m2[v], nb1[v] = b1, b2, a1
        self.m1, self.m2, self.nb1 = m1, m2, nb1
        self.free = [m1[v] + m2[v] for v in range(n)]
        self.sum_free_rest = sum(self.free) - self.free[0]  # Σ sobre não visitados na raiz
        self.use_one_tree = bound == "one_tree"

        self.best = ub_cost
        self.best_tour = list(ub_tour)

        self.expanded = self.generated = self.pruned = 0
        self.max_depth = self.max_frontier = self.frontier = 0
        self.ub_improvements = 0
        self.time_to_best = 0.0

        self.memo: dict[int, int] = {}
        self.mst_calls = self.memo_hits = 0
        self.mst_time = 0.0

        self.root_lb = (sum(self.free) + 1) // 2
        if self.use_one_tree:  # 1-tree na raiz: MST(V \ {0}) + duas menores arestas de 0
            one_tree_root = self._mst(1, list(range(1, n))) + m1[0] + m2[0]
            self.root_lb = max(self.root_lb, one_tree_root)

    # ------------------------------------------------------------------ #
    def _mst(self, visited_mask: int, nodes: list[int]) -> int:
        val = self.memo.get(visited_mask)
        if val is not None:
            self.memo_hits += 1
            return val
        t = time.perf_counter()
        val = _prim_weight(nodes, self.D)
        self.mst_time += time.perf_counter() - t
        self.mst_calls += 1
        if len(self.memo) >= MST_MEMO_MAX:
            self.memo.clear()
        self.memo[visited_mask] = val
        return val

    def _improve(self, total: int, path: Sequence[int], w: int) -> None:
        self.best = total
        self.best_tour = list(path) + [w]
        self.ub_improvements += 1
        self.time_to_best = time.perf_counter() - self.t0

    def children(self, path: Sequence[int], mask: int, cost: int, sum_u: int, big: int) -> list[tuple]:
        """Expande o nó: devolve os filhos não podados como
        (lb, w, custo, Σfree dos não visitados, big). Completações de tour
        (último vértice) atualizam o UB diretamente e não entram na lista."""
        D, n = self.D, self.n
        m1, m2, nb1, free = self.m1, self.m2, self.nb1, self.free
        u = path[-1]
        Du = D[u]
        depth = len(path)
        remaining = n - depth  # não visitados antes de acrescentar w

        self.expanded += 1
        if depth > self.max_depth:
            self.max_depth = depth

        a = path[1] if depth >= 2 else -1
        unvisited = [v for v in range(1, n) if not (mask >> v) & 1]
        out = []
        for w in unvisited:
            self.generated += 1
            c2 = cost + Du[w]

            if remaining == 1:  # w é a última cidade (big ≥ 1 garante w > a)
                total = c2 + D[w][0]
                if total < self.best:
                    self._improve(total, path, w)
                continue

            if depth == 1:  # w será path[1]: vizinho fixo de 0 e referência da simetria
                big2 = n - 1 - w
                f0 = w
            else:
                big2 = big - 1 if w > a else big
                f0 = a
            if big2 == 0:  # nenhuma última cidade válida (> a) restaria
                self.pruned += 1
                continue

            sum_u2 = sum_u - free[w]
            other0 = m1[0] if nb1[0] != f0 else m2[0]
            otherw = m1[w] if nb1[w] != u else m2[w]
            lb = (2 * c2 + sum_u2 + other0 + otherw + 1) // 2
            if lb >= self.best:
                self.pruned += 1
                continue

            if self.use_one_tree:
                rest = [x for x in unvisited if x != w]
                Dw, D0 = D[w], D[0]
                lb1 = (c2 + self._mst(mask | (1 << w), rest)
                       + min(Dw[x] for x in rest) + min(D0[x] for x in rest))
                if lb1 > lb:
                    lb = lb1
                    if lb >= self.best:
                        self.pruned += 1
                        continue

            out.append((lb, w, c2, sum_u2, big2))
        return out

    # ------------------------------------------------------------------ #
    def dfs(self, path: list[int], mask: int, cost: int, sum_u: int, big: int) -> None:
        if not (self.expanded & TIME_CHECK_MASK) and time.perf_counter() > self.deadline:
            raise _Timeout
        kids = self.children(path, mask, cost, sum_u, big)
        if not kids:
            return
        kids.sort()
        k = len(kids)
        self.frontier += k
        if self.frontier > self.max_frontier:
            self.max_frontier = self.frontier
        for i in range(k):
            lb, w, c2, s2, b2 = kids[i]
            if lb >= self.best:  # UB melhorou: este e os seguintes (ordenados) caem
                self.pruned += k - i
                self.frontier -= k - i
                return
            self.frontier -= 1
            path.append(w)
            self.dfs(path, mask | (1 << w), c2, s2, b2)
            path.pop()

    def best_first(self) -> None:
        counter = itertools.count()
        heap = [(self.root_lb, -1, next(counter), 0, 1, (0,), self.sum_free_rest, -1)]
        self.max_frontier = 1
        pops = 0
        while heap:
            if not (pops & TIME_CHECK_MASK) and time.perf_counter() > self.deadline:
                raise _Timeout
            pops += 1
            lb, negd, _, cost, mask, path, sum_u, big = heapq.heappop(heap)
            if lb >= self.best:  # heap ordenado por lb: todo o resto também cai
                self.pruned += 1 + len(heap)
                heap.clear()
                break
            for clb, w, c2, s2, b2 in self.children(path, mask, cost, sum_u, big):
                heapq.heappush(heap, (clb, negd - 1, next(counter), c2, mask | (1 << w), path + (w,), s2, b2))
            if len(heap) > self.max_frontier:
                self.max_frontier = len(heap)


def bnb(
    inst: Instance,
    ub_tour: Sequence[int] | None = None,
    time_limit_s: float = TIME_LIMIT_S,
    prof: Profiler | None = None,
    search: str = "dfs",
    bound: str = "two_min",
) -> BnBResult:
    """Resolve o TSP de forma exata. ``ub_tour`` é o warm-start (default: trivial).

    Seções no Profiler: "bnb_setup" (pré-cálculo), "bnb_search" e, com
    bound="one_tree", "bnb_mst" (tempo total e nº de MSTs efetivamente calculadas).
    """
    if search not in SEARCHES:
        raise ValueError(f"search deve ser um de {SEARCHES}")
    if bound not in BOUNDS:
        raise ValueError(f"bound deve ser um de {BOUNDS}")
    prof = prof if prof is not None else Profiler()

    with prof.section("bnb_setup"):
        tour0 = list(range(inst.n)) if ub_tour is None else normalize_tour(ub_tour)
        validate_tour(inst.n, tour0)
        ub_cost = tour_cost(inst.D, tour0)
        s = _Search(inst, bound, ub_cost, tour0, time_limit_s)

    status = STATUS_OK
    with prof.section("bnb_search"):
        if s.root_lb < s.best:  # senão o UB inicial já é ótimo (provado na raiz)
            try:
                if search == "dfs":
                    s.dfs([0], 1, 0, s.sum_free_rest, -1)
                else:
                    s.best_first()
            except _Timeout:
                status = STATUS_TIMEOUT
            except MemoryError:  # fronteira do best-first estourou a memória
                status = STATUS_ERROR
    elapsed = time.perf_counter() - s.t0
    if s.mst_calls:
        prof.add("bnb_mst", s.mst_time, s.mst_calls)

    return BnBResult(
        status=status,
        best_cost=s.best,
        best_tour=normalize_tour(s.best_tour),
        initial_ub=ub_cost,
        root_lb=s.root_lb,
        search=search,
        bound=bound,
        expanded=s.expanded,
        pruned=s.pruned,
        generated=s.generated,
        max_depth=s.max_depth,
        max_frontier=s.max_frontier,
        ub_improvements=s.ub_improvements,
        time_s=elapsed,
        time_to_best_s=s.time_to_best,
        mst_calls=s.mst_calls,
        mst_memo_hits=s.memo_hits,
    )


def _load_warm_starts():
    from .approx import christofides, twice_around_the_tree
    from .heuristics import convex_hull_insertion, nearest_neighbor, trivial

    return {
        "trivial": trivial,
        "nn": nearest_neighbor,
        "hull": convex_hull_insertion,
        "tatt": twice_around_the_tree,
        "christofides": christofides,
    }


WARM_STARTS = _load_warm_starts()


def main(argv: list[str] | None = None) -> int:
    """Execução avulsa para testes manuais:
        python -m src.tsp.bnb data/instances/eil51_random12.tsp --warm hull --time 60
    """
    ap = argparse.ArgumentParser(description="Branch-and-Bound para TSP EUC_2D")
    ap.add_argument("instance", type=Path)
    ap.add_argument("--warm", choices=list(WARM_STARTS), default="nn")
    ap.add_argument("--search", choices=SEARCHES, default="dfs")
    ap.add_argument("--bound", choices=BOUNDS, default="one_tree")
    ap.add_argument("--time", type=float, default=TIME_LIMIT_S, help="limite em segundos")
    args = ap.parse_args(argv)

    inst = load_tsplib(args.instance)
    prof = Profiler()
    t = time.perf_counter()
    ub = WARM_STARTS[args.warm](inst, prof)
    t_heur = time.perf_counter() - t

    res = bnb(inst, ub, args.time, prof, args.search, args.bound)
    print(f"instância={inst.name} n={inst.n} warm={args.warm} ({t_heur:.4f}s) "
          f"search={args.search} bound={args.bound}")
    print(f"status={res.status} custo={res.best_cost} UB0={res.initial_ub} LB_raiz={res.root_lb} "
          f"tempo={res.time_s:.3f}s melhor_em={res.time_to_best_s:.3f}s")
    print(f"expandidos={res.expanded} gerados={res.generated} podados={res.pruned} "
          f"prof_max={res.max_depth} fronteira_max={res.max_frontier} melhorias_UB={res.ub_improvements}")
    if res.mst_calls:
        print(f"MST: {res.mst_calls} calculadas, {res.mst_memo_hits} hits no memo")
    for sec in prof.sections():
        print(f"  {sec:18s} {prof.total(sec):10.4f}s  ({prof.calls(sec)} chamadas)")
    print("tour:", " ".join(map(str, res.best_tour)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
