"""Algoritmos de aproximação para o TSP métrico (Etapa 2 – Dev B).

Todos seguem o contrato ``TourBuilder``: ``fn(inst, prof) -> list[int]``
(permutação de 0..n-1 começando em 0).

  * twice_around_the_tree        – 2-aproximação via MST + pré-ordem         O(n²)
  * christofides                 – 1.5-aproximação via MST + matching ótimo  O(n³)
  * christofides_greedy_matching – ablação com matching guloso               O(n² log n)

Seções registradas no Profiler:
  * twice_around_the_tree: "mst", "preorder"
  * christofides / christofides_greedy_matching: "mst", "odd", "matching", "euler", "shortcut"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import networkx as nx
import numpy as np

from .common import Instance, Profiler, load_tsplib, tour_cost, validate_tour
from .mst import mst_degrees, mst_edges, prim_mst


def twice_around_the_tree(inst: Instance, prof: Profiler | None = None) -> list[int]:
    """2-aproximação (Twice-around-the-tree): MST + caminhamento em pré-ordem iterativo."""
    prof = prof if prof is not None else Profiler()
    parent = prim_mst(inst.D, prof, root=0)

    with prof.section("preorder"):
        n = inst.n
        children: list[list[int]] = [[] for _ in range(n)]
        for v in range(1, n):
            children[int(parent[v])].append(v)

        tour: list[int] = []
        stack = [0]
        while stack:
            u = stack.pop()
            tour.append(u)
            ch = children[u]
            if ch:
                stack.extend(reversed(ch))
        return tour


def _min_weight_perfect_matching(
    odd: list[int],
    D_list: list[list[int]],
) -> list[tuple[int, int]]:
    """Emparelhamento perfeito de peso mínimo no grafo completo dos vértices ímpares.

    Usa ``networkx.min_weight_matching`` (algoritmo Blossom, ~O(|O|³)), que é o
    gargalo dominante do algoritmo de Christofides.
    """
    m = len(odd)
    if m == 0:
        return []
    G = nx.Graph()
    G.add_weighted_edges_from(
        (odd[i], odd[j], D_list[odd[i]][odd[j]])
        for i in range(m)
        for j in range(i + 1, m)
    )
    matching = nx.min_weight_matching(G, weight="weight")
    return [(int(u), int(v)) for u, v in matching]


def _greedy_perfect_matching(
    odd: list[int],
    D: np.ndarray,
) -> list[tuple[int, int]]:
    """Ablação: emparelhamento perfeito guloso sobre os vértices de grau ímpar.

    Ordena todas as arestas entre vértices ímpares por peso crescente e casa
    pares disjuntos até cobrir todos os vértices (sempre possível porque o
    subgrafo induzido é completo K_m com m par).
    """
    m = len(odd)
    if m == 0:
        return []
    idx = np.asarray(odd, dtype=np.intp)
    subD = D[np.ix_(idx, idx)]
    iu, ju = np.triu_indices(m, k=1)
    order = np.argsort(subD[iu, ju], kind="stable")

    matched = [False] * m
    pairs: list[tuple[int, int]] = []
    remaining = m
    for pos in order.tolist():
        a = int(iu[pos])
        b = int(ju[pos])
        if not matched[a] and not matched[b]:
            matched[a] = True
            matched[b] = True
            pairs.append((odd[a], odd[b]))
            remaining -= 2
            if remaining == 0:
                break
    return pairs


def _hierholzer(n: int, edges: list[tuple[int, int]], start: int = 0) -> list[int]:
    """Circuito euleriano iterativo (Hierholzer) em multigrafo não-direcionado O(V + E).

    A lista de adjacência guarda pares ``(vizinho, edge_id)`` e um ponteiro ``ptr[u]``
    garante que cada entrada seja inspecionada no máximo uma vez, suportando
    arestas paralelas entre MST e matching sem remoções O(grau).
    """
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    used = [False] * len(edges)
    ptr = [0] * n
    stack = [start]
    circuit: list[int] = []

    while stack:
        u = stack[-1]
        au = adj[u]
        p = ptr[u]
        while p < len(au) and used[au[p][1]]:
            p += 1
        ptr[u] = p
        if p < len(au):
            v, eid = au[p]
            ptr[u] = p + 1
            used[eid] = True
            stack.append(v)
        else:
            circuit.append(stack.pop())

    circuit.reverse()
    return circuit


def _shortcut(circuit: list[int], n: int) -> list[int]:
    """Remove vértices repetidos do circuito euleriano mantendo a primeira visita."""
    seen = [False] * n
    tour: list[int] = []
    for v in circuit:
        if not seen[v]:
            seen[v] = True
            tour.append(v)
    return tour


def _christofides_core(
    inst: Instance,
    prof: Profiler | None,
    *,
    greedy_matching: bool,
) -> list[int]:
    prof = prof if prof is not None else Profiler()

    # 1. Árvore Geradora Mínima
    parent = prim_mst(inst.D, prof, root=0)

    # 2. Vértices de grau ímpar na MST
    with prof.section("odd"):
        deg = mst_degrees(parent)
        odd = [int(v) for v in np.flatnonzero(deg & 1)]

    # 3. Emparelhamento perfeito (ótimo ou guloso na ablação)
    with prof.section("matching"):
        if greedy_matching:
            matching_edges = _greedy_perfect_matching(odd, inst.D)
        else:
            matching_edges = _min_weight_perfect_matching(odd, inst.D_list)

    # 4 & 5. Multigrafo MST + matching e circuito euleriano (Hierholzer iterativo)
    with prof.section("euler"):
        edges = mst_edges(parent) + matching_edges
        circuit = _hierholzer(inst.n, edges, start=0)

    # 6. Atalhos para obter o ciclo hamiltoniano
    with prof.section("shortcut"):
        return _shortcut(circuit, inst.n)


def christofides(inst: Instance, prof: Profiler | None = None) -> list[int]:
    """1.5-aproximação de Christofides (MST + matching perfeito mínimo + Hierholzer)."""
    return _christofides_core(inst, prof, greedy_matching=False)


def christofides_greedy_matching(inst: Instance, prof: Profiler | None = None) -> list[int]:
    """Ablação do Christofides com emparelhamento guloso (troca garantia 1.5× por velocidade)."""
    return _christofides_core(inst, prof, greedy_matching=True)


APPROXIMATIONS = {
    "tatt": twice_around_the_tree,
    "christofides": christofides,
    "christofides_greedy": christofides_greedy_matching,
}


def main(argv: list[str] | None = None) -> int:
    """Execução avulsa para inspeção e profiling manual:
        python -m src.tsp.approx data/tsplib/eil51.tsp --alg christofides
    """
    ap = argparse.ArgumentParser(description="Algoritmos de aproximação para TSP EUC_2D")
    ap.add_argument("instance", type=Path)
    ap.add_argument("--alg", choices=sorted(APPROXIMATIONS), default="christofides")
    ap.add_argument("--mem", action="store_true", help="habilita tracemalloc no Profiler")
    args = ap.parse_args(argv)

    inst = load_tsplib(args.instance)
    prof = Profiler(mem=args.mem)
    t0 = time.perf_counter()
    tour = APPROXIMATIONS[args.alg](inst, prof)
    elapsed = time.perf_counter() - t0
    validate_tour(inst.n, tour)
    cost = tour_cost(inst.D, tour)

    print(f"instância={inst.name} n={inst.n} alg={args.alg} custo={cost} tempo={elapsed:.6f}s")
    for sec in prof.sections():
        kb = prof.peak_kb(sec)
        mem_str = f"  pico={kb:.1f} KB" if kb is not None else ""
        print(f"  {sec:14s} {prof.total(sec):10.6f}s  ({prof.calls(sec)} chamadas){mem_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
