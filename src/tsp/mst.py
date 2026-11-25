"""Árvore Geradora Mínima via algoritmo de Prim denso O(n²) (Etapa 2 – Dev B).

Implementação própria vetorizada com NumPy (sem bibliotecas de grafos e sem
copiar a matriz de distâncias). Usada por ``approx.py`` (TATT e Christofides)
e ``features.py`` (estatísticas topológicas da MST).

Seção registrada no Profiler: ``"mst"``.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from .common import Profiler


def _prim_dense(D: np.ndarray, root: int) -> np.ndarray:
    """Prim denso O(n²) vetorizado com NumPy.

    Devolve o vetor ``parent`` de shape ``(n,)`` e dtype ``intp``, com
    ``parent[root] == -1`` e ``parent[v]`` igual ao pai de ``v`` na MST.
    """
    n = D.shape[0]
    parent = np.full(n, root, dtype=np.intp)
    parent[root] = -1
    if n <= 1:
        return parent

    in_mst = np.zeros(n, dtype=bool)
    in_mst[root] = True

    is_int = np.issubdtype(D.dtype, np.integer)
    inf = np.iinfo(np.int64).max if is_int else np.inf
    key = D[root].astype(np.int64 if is_int else np.float64, copy=True)
    key[root] = inf

    for _ in range(n - 1):
        u = int(np.argmin(key))
        in_mst[u] = True
        key[u] = inf

        row = D[u]
        better = (~in_mst) & (row < key)
        key[better] = row[better]
        parent[better] = u

    return parent


def prim_mst(D: np.ndarray, prof: Profiler | None = None, root: int = 0) -> np.ndarray:
    """Calcula a MST da matriz simétrica ``D`` usando Prim denso O(n²).

    Se ``prof`` for fornecido, instrumenta o cálculo na seção ``"mst"``.
    """
    D = np.asarray(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D deve ser matriz quadrada, recebido shape {D.shape}")
    n = D.shape[0]
    if n == 0:
        raise ValueError("D não pode ser vazia")
    if not (0 <= root < n):
        raise ValueError(f"root={root} fora de 0..{n - 1}")

    ctx = prof.section("mst") if prof is not None else nullcontext()
    with ctx:
        return _prim_dense(D, root)


def mst_edges(parent: np.ndarray) -> list[tuple[int, int]]:
    """Lista de arestas ``(pai, filho)`` da MST a partir do vetor ``parent``."""
    return [(int(parent[v]), v) for v in range(len(parent)) if parent[v] >= 0]


def mst_weight(D: np.ndarray, parent: np.ndarray) -> int | float:
    """Soma dos pesos das arestas da MST."""
    children = np.flatnonzero(parent >= 0)
    if children.size == 0:
        return 0 if np.issubdtype(D.dtype, np.integer) else 0.0
    total = D[parent[children], children].sum()
    return int(total) if np.issubdtype(D.dtype, np.integer) else float(total)


def mst_degrees(parent: np.ndarray) -> np.ndarray:
    """Grau de cada vértice na MST (vetor intp de shape ``(n,)``)."""
    n = len(parent)
    children = np.flatnonzero(parent >= 0)
    if children.size == 0:
        return np.zeros(n, dtype=np.intp)
    parents = parent[children]
    return np.bincount(children, minlength=n) + np.bincount(parents, minlength=n)


def mst_adjacency(parent: np.ndarray) -> list[list[int]]:
    """Lista de adjacência não-direcionada da MST."""
    n = len(parent)
    adj: list[list[int]] = [[] for _ in range(n)]
    for v in range(n):
        p = int(parent[v])
        if p >= 0:
            adj[p].append(v)
            adj[v].append(p)
    return adj
