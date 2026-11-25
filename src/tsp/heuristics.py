"""Heurísticas construtivas para o upper bound inicial do B&B (Etapa 1 – Dev A).

Todas seguem o contrato ``TourBuilder``: ``fn(inst, prof) -> list[int]``
(permutação de 0..n-1 começando em 0).

  * trivial               – 0 → 1 → … → n-1 (cold start)                O(n)
  * nearest_neighbor      – vizinho mais próximo a partir de 0           O(n²)
  * convex_hull_insertion – fecho convexo + inserção mais barata         O(n²) amortizado

Seções registradas no Profiler: "trivial", "nearest_neighbor", "hull", "insertion".
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from .common import Instance, Profiler, normalize_tour

_INF = np.iinfo(np.int64).max


def trivial(inst: Instance, prof: Profiler) -> list[int]:
    with prof.section("trivial"):
        return list(range(inst.n))


def nearest_neighbor(inst: Instance, prof: Profiler, start: int = 0) -> list[int]:
    """Vizinho mais próximo. Empates resolvidos pelo menor índice (argmin)."""
    with prof.section("nearest_neighbor"):
        D = inst.D
        visited = np.zeros(inst.n, dtype=bool)
        visited[start] = True
        tour = [start]
        cur = start
        for _ in range(inst.n - 1):
            row = np.where(visited, _INF, D[cur])
            cur = int(np.argmin(row))
            visited[cur] = True
            tour.append(cur)
        return normalize_tour(tour)


def _initial_hull(coords: np.ndarray, D: np.ndarray) -> list[int]:
    """Vértices do fecho convexo em ordem anti-horária. Para entradas degeneradas
    (todos os pontos colineares) cai para o par de pontos mais distante."""
    try:
        verts = [int(v) for v in ConvexHull(coords).vertices]
        if len(verts) >= 3:
            return verts
    except Exception:  # scipy QhullError: pontos colineares/coincidentes
        pass
    i, j = np.unravel_index(int(np.argmax(D)), D.shape)
    return [int(i), int(j)]


def convex_hull_insertion(inst: Instance, prof: Profiler) -> list[int]:
    """Sub-rota inicial = fecho convexo; depois insere, a cada passo, o vértice
    interno de menor custo de inserção  d(i,k) + d(k,j) - d(i,j)  (cheapest insertion).

    Estrutura: tour como vetor de sucessores ``succ``; para cada vértice fora do
    tour, cache do melhor custo de inserção (``best_cost``) e da aresta (``best_from``
    = i, aresta (i, succ[i])). Ao inserir k entre a e b, só os vértices cuja melhor
    aresta era (a, b) são recalculados por completo; os demais só comparam com as
    duas arestas novas (a, k) e (k, b). Isso evita o O(n³) da versão ingênua.
    """
    n, D = inst.n, inst.D
    if int(D.max()) == 0:  # todos os pontos coincidentes: qualquer tour custa 0
        return list(range(n))

    with prof.section("hull"):
        sub = _initial_hull(inst.coords, D)

    with prof.section("insertion"):
        succ = np.full(n, -1, dtype=np.intp)
        for a, b in zip(sub, sub[1:] + sub[:1]):
            succ[a] = b
        in_tour = succ >= 0
        remaining = ~in_tour

        rem = np.flatnonzero(remaining)
        if rem.size:
            best_cost = np.full(n, _INF, dtype=np.int64)
            best_from = np.full(n, -1, dtype=np.intp)

            A = np.asarray(sub, dtype=np.intp)
            B = succ[A]
            C = D[np.ix_(rem, A)] + D[np.ix_(rem, B)] - D[A, B]
            j = np.argmin(C, axis=1)
            best_cost[rem] = C[np.arange(rem.size), j]
            best_from[rem] = A[j]

            while True:
                R = np.flatnonzero(remaining)
                if R.size == 0:
                    break
                k = int(R[np.argmin(best_cost[R])])
                a = int(best_from[k])
                b = int(succ[a])
                succ[a] = k
                succ[k] = b
                in_tour[k] = True
                remaining[k] = False

                R = R[R != k]
                if R.size == 0:
                    break

                stale = best_from[R] == a  # a aresta (a, b) deixou de existir
                Rs = R[stale]
                if Rs.size:
                    T = np.flatnonzero(in_tour)
                    S = succ[T]
                    Cs = D[np.ix_(Rs, T)] + D[np.ix_(Rs, S)] - D[T, S]
                    js = np.argmin(Cs, axis=1)
                    best_cost[Rs] = Cs[np.arange(Rs.size), js]
                    best_from[Rs] = T[js]

                Rf = R[~stale]
                if Rf.size:
                    c_ak = D[Rf, a] + D[Rf, k] - D[a, k]
                    better = c_ak < best_cost[Rf]
                    best_cost[Rf] = np.where(better, c_ak, best_cost[Rf])
                    best_from[Rf] = np.where(better, a, best_from[Rf])

                    c_kb = D[Rf, k] + D[Rf, b] - D[k, b]
                    better = c_kb < best_cost[Rf]
                    best_cost[Rf] = np.where(better, c_kb, best_cost[Rf])
                    best_from[Rf] = np.where(better, k, best_from[Rf])

        tour = [0]
        v = int(succ[0])
        while v != 0:
            tour.append(v)
            v = int(succ[v])
        return tour


# Registro das estratégias de warm-start implementadas pelo Dev A.
# A Etapa 3 acrescenta "tatt" e "christofides" (Dev B).
HEURISTICS = {
    "trivial": trivial,
    "nn": nearest_neighbor,
    "hull": convex_hull_insertion,
}
