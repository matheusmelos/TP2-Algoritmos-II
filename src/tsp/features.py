"""Extração de features geométricas e topológicas das instâncias (Etapa 2 – Dev B).

Usada na análise de sensibilidade (``src/analysis/sensitivity.py``) para explicar
quais características da geometria dos pontos facilitam ou degradam o gap das
heurísticas e aproximações.

Features extraídas por ``extract_features(inst)``:
  * ``n``               – número de cidades
  * ``aspect_ratio``    – razão lado menor / lado maior do bounding box (∈ [0, 1])
  * ``hull_fraction``   – fração dos pontos sobre o fecho convexo
  * ``spatial_entropy`` – entropia de Shannon normalizada na grade ⌊√n⌋ × ⌊√n⌋
  * ``clark_evans``     – índice de agregação R (< 1 cluster, ≈ 1 aleatório, > 1 regular)
  * ``mst_max_degree``  – grau máximo na MST (presença de hubs)
  * ``mst_mean_degree`` – grau médio na MST (= 2(n-1)/n)
  * ``mst_edge_cv``     – coeficiente de variação dos pesos das arestas da MST
  * ``nn_dist_cv``      – coeficiente de variação das distâncias ao vizinho mais próximo
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import ConvexHull

from .common import Instance, load_tsplib
from .mst import mst_degrees, prim_mst

FEATURE_NAMES: tuple[str, ...] = (
    "n",
    "aspect_ratio",
    "hull_fraction",
    "spatial_entropy",
    "clark_evans",
    "mst_max_degree",
    "mst_mean_degree",
    "mst_edge_cv",
    "nn_dist_cv",
)


def _euclidean_nn_distances(coords: np.ndarray) -> np.ndarray:
    """Distância euclidiana contínua de cada ponto ao seu vizinho mais próximo."""
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def _hull_fraction(coords: np.ndarray) -> float:
    n = len(coords)
    try:
        hull = ConvexHull(coords)
        return float(len(hull.vertices) / n)
    except Exception:  # QhullError: pontos colineares ou coincidentes
        uniq = np.unique(coords, axis=0)
        return float(min(len(uniq), 2) / n)


def _spatial_entropy(coords: np.ndarray) -> float:
    """Entropia de Shannon normalizada em grade b × b com b = max(2, floor(sqrt(n)))."""
    n = len(coords)
    bins = max(2, int(np.floor(np.sqrt(n))))
    hist, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=bins)
    counts = hist.ravel()
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    if p.size <= 1:
        return 0.0
    h = float(-np.sum(p * np.log(p)))
    max_cells = min(n, bins * bins)
    h_max = float(np.log(max_cells))
    return float(h / h_max) if h_max > 0 else 0.0


def _cv(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if mean <= 0.0:
        return 0.0
    return float(np.std(values) / mean)


def extract_features(inst: Instance) -> dict[str, float]:
    """Extrai o dicionário de features geométricas e topológicas de ``inst``."""
    coords = np.asarray(inst.coords, dtype=np.float64)
    n = inst.n

    # Bounding box e razão de aspecto
    span = coords.max(axis=0) - coords.min(axis=0)
    dx, dy = float(span[0]), float(span[1])
    long_side = max(dx, dy)
    short_side = min(dx, dy)
    aspect_ratio = 1.0 if long_side == 0.0 else float(short_side / long_side)

    # Fecho convexo e entropia espacial
    hull_frac = _hull_fraction(coords)
    entropy = _spatial_entropy(coords)

    # Distâncias ao vizinho mais próximo e índice de Clark-Evans
    nn_dists = _euclidean_nn_distances(coords)
    nn_mean = float(np.mean(nn_dists))
    area = dx * dy
    if area <= 0.0:
        clark_evans = 0.0
    else:
        r_expected = 0.5 * float(np.sqrt(area / n))
        clark_evans = float(nn_mean / r_expected) if r_expected > 0.0 else 0.0
    nn_cv = _cv(nn_dists)

    # Estatísticas da MST
    parent = prim_mst(inst.D, root=0)
    degrees = mst_degrees(parent)
    children = np.flatnonzero(parent >= 0)
    edge_weights = inst.D[parent[children], children].astype(np.float64)

    return {
        "n": float(n),
        "aspect_ratio": aspect_ratio,
        "hull_fraction": hull_frac,
        "spatial_entropy": entropy,
        "clark_evans": clark_evans,
        "mst_max_degree": float(degrees.max()),
        "mst_mean_degree": float(degrees.mean()),
        "mst_edge_cv": _cv(edge_weights),
        "nn_dist_cv": nn_cv,
    }


def extract_features_table(instances: Sequence[Instance]) -> list[dict]:
    """Extrai linhas ``{"instance": ..., **features}`` ordenadas por (n, instance)."""
    rows = [{"instance": inst.name, **extract_features(inst)} for inst in instances]
    rows.sort(key=lambda r: (r["n"], r["instance"]))
    return rows


def write_features_csv(rows: Sequence[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["instance", *FEATURE_NAMES]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Extrai features de todos os .tsp de um diretório e grava features.csv:
        python -m src.tsp.features --instances data/instances --out results/features.csv
    """
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Extrai features geométricas/topológicas das instâncias")
    ap.add_argument("--instances", type=Path, default=root / "data" / "instances")
    ap.add_argument("--out", type=Path, default=root / "results" / "features.csv")
    args = ap.parse_args(argv)

    tsp_files = sorted(args.instances.glob("*.tsp"))
    if not tsp_files:
        raise SystemExit(f"Nenhum .tsp encontrado em {args.instances}")

    instances = [load_tsplib(p) for p in tsp_files]
    rows = extract_features_table(instances)
    write_features_csv(rows, args.out)
    print(f"Extraídas features de {len(rows)} instâncias -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
