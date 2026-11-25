"""Gera o conjunto de teste (Etapa 1 – Dev A).

O TSPLIB não tem instâncias EUC_2D com N ≤ 40, então a faixa principal é formada
por sub-instâncias recortadas (com seed fixa) de instâncias EUC_2D reais. A faixa
limite usa as instâncias completas eil51, berlin52 e st70 (ótimos conhecidos).

Modos de recorte, para gerar diversidade nas features geométricas:
  random – amostra uniforme sem reposição dos pontos da fonte
  knn    – os k pontos mais próximos de um centro sorteado (região "circular")
  window – janela retangular de razão de aspecto sorteada em torno de um centro
           (k menores distâncias L∞ escaladas), o que varia o aspect ratio

Saída: data/instances/<nome>.tsp e data/instances/manifest.csv
(name, n, source, mode, seed, optimal_ref, file).

Uso:  python -m src.tsp.make_instances [--src data/tsplib] [--out data/instances] [--seed 2026]
"""

from __future__ import annotations

import argparse
import csv
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .common import OPTIMA, Instance, instance_from_coords, load_tsplib, write_tsplib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "data" / "tsplib"
DEFAULT_OUT = ROOT / "data" / "instances"
SEED = 2026
MANIFEST_COLUMNS = ["name", "n", "source", "mode", "seed", "optimal_ref", "file"]


@dataclass(frozen=True)
class SubSpec:
    n: int
    source: str
    mode: str

    @property
    def name(self) -> str:
        return f"{self.source}_{self.mode}{self.n}"


# 13 sub-instâncias (N ≤ 40) + 3 completas = 16 instâncias, 81% na faixa principal.
# Fontes: eil51/eil76 ≈ uniforme, berlin52 clusterizada, pr76/a280 estruturadas,
# kroA100/ch130 aleatórias.
SUB_SPECS: tuple[SubSpec, ...] = (
    SubSpec(12, "eil51", "random"),
    SubSpec(14, "berlin52", "knn"),
    SubSpec(16, "pr76", "random"),
    SubSpec(18, "eil76", "window"),
    SubSpec(20, "berlin52", "random"),
    SubSpec(22, "kroA100", "knn"),
    SubSpec(24, "a280", "window"),
    SubSpec(26, "ch130", "random"),
    SubSpec(28, "eil76", "knn"),
    SubSpec(30, "pr76", "window"),
    SubSpec(33, "berlin52", "window"),
    SubSpec(36, "kroA100", "random"),
    SubSpec(40, "a280", "knn"),
)
FULL_INSTANCES: tuple[str, ...] = ("eil51", "berlin52", "st70")


# --------------------------------------------------------------------------- #
# Seletores: recebem pontos SEM duplicatas e devolvem índices ordenados
# --------------------------------------------------------------------------- #


def _select_random(pts: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(len(pts), size=k, replace=False))


def _select_knn(pts: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    center = pts[rng.integers(len(pts))]
    d = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
    return np.sort(np.argsort(d, kind="stable")[:k])


def _select_window(pts: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    center = pts[rng.integers(len(pts))]
    ratio = rng.uniform(0.2, 1.0)
    shape = np.array([1.0, ratio]) if rng.random() < 0.5 else np.array([ratio, 1.0])
    span = pts.max(axis=0) - pts.min(axis=0)
    span[span == 0] = 1.0
    d = np.max(np.abs(pts - center) / (span * shape), axis=1)
    return np.sort(np.argsort(d, kind="stable")[:k])


SELECTORS = {"random": _select_random, "knn": _select_knn, "window": _select_window}


def unique_points(coords: np.ndarray) -> np.ndarray:
    """Remove pontos repetidos preservando a ordem original da fonte."""
    _, first = np.unique(coords, axis=0, return_index=True)
    return coords[np.sort(first)]


def subsample(coords: np.ndarray, k: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode not in SELECTORS:
        raise ValueError(f"modo desconhecido: {mode} (use {sorted(SELECTORS)})")
    pts = unique_points(np.asarray(coords, dtype=np.float64))
    if k > len(pts):
        raise ValueError(f"pedidos {k} pontos, a fonte tem {len(pts)} distintos")
    return pts[SELECTORS[mode](pts, k, rng)]


def spec_rng(seed: int, name: str) -> np.random.Generator:
    """RNG por instância, derivado do nome: reordenar SUB_SPECS não muda os recortes."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


# --------------------------------------------------------------------------- #


def generate(
    src_dir: str | Path = DEFAULT_SRC,
    out_dir: str | Path = DEFAULT_OUT,
    specs: Iterable[SubSpec] = SUB_SPECS,
    full: Iterable[str] = FULL_INSTANCES,
    seed: int = SEED,
) -> list[dict]:
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Instance] = {}

    def source(name: str) -> Instance:
        if name not in cache:
            cache[name] = load_tsplib(src_dir / f"{name}.tsp")
        return cache[name]

    rows: list[dict] = []
    for spec in specs:
        coords = subsample(source(spec.source).coords, spec.n, spec.mode, spec_rng(seed, spec.name))
        inst = instance_from_coords(spec.name, coords)
        path = out_dir / f"{spec.name}.tsp"
        write_tsplib(inst, path, comment=f"sub-instancia de {spec.source} modo={spec.mode} seed={seed}")
        rows.append(dict(name=spec.name, n=inst.n, source=spec.source, mode=spec.mode,
                         seed=seed, optimal_ref="", file=path.name))

    for name in full:
        inst = source(name)
        path = out_dir / f"{name}.tsp"
        write_tsplib(inst, path, comment=f"instancia completa TSPLIB {name}")
        rows.append(dict(name=name, n=inst.n, source=name, mode="full",
                         seed="", optimal_ref=OPTIMA.get(name, ""), file=path.name))

    rows.sort(key=lambda r: (r["n"], r["name"]))
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_manifest(out_dir: str | Path = DEFAULT_OUT) -> list[dict]:
    """Lê o manifest; ``optimal_ref`` vira int ou None."""
    with open(Path(out_dir) / "manifest.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["n"] = int(r["n"])
        r["optimal_ref"] = int(r["optimal_ref"]) if r["optimal_ref"] else None
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gera as instâncias de teste do TP2")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    rows = generate(args.src, args.out, seed=args.seed)
    small = sum(r["n"] <= 40 for r in rows)
    for r in rows:
        print(f"{r['name']:22s} n={r['n']:3d}  fonte={r['source']:9s} modo={r['mode']}")
    print(f"{len(rows)} instâncias, {small} com N ≤ 40 ({100 * small / len(rows):.0f}%) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
