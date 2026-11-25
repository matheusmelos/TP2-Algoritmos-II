"""Contrato comum do TP2 (TSP geométrico).

Tudo que as Etapas 1 (B&B + heurísticas) e 2 (aproximações + features + análise)
compartilham vive aqui: representação de instância, parser TSPLIB EUC_2D, custo e
validação de tours, profiler por subrotina e esquema dos arquivos de resultados.

Convenções:
  * Vértices são 0-based (o TSPLIB é 1-based; a conversão acontece só no I/O).
  * Um tour é uma permutação de 0..n-1 começando em 0, SEM repetir o 0 no final.
  * Toda heurística/aproximação segue a assinatura ``TourBuilder``:
        fn(inst: Instance, prof: Profiler) -> list[int]
"""

from __future__ import annotations

import gzip
import json
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Instância
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class Instance:
    """Instância do TSP geométrico.

    coords: array float64 (n, 2).
    D:      matriz de distâncias int64 (n, n), simétrica, diagonal zero,
            calculada pela regra nint do TSPLIB para EUC_2D.
    """

    name: str
    n: int
    coords: np.ndarray
    D: np.ndarray

    @cached_property
    def D_list(self) -> list[list[int]]:
        """Matriz como listas Python: acesso escalar ~10x mais rápido que
        ``D[i, j]`` em NumPy dentro de loops Python puros (ex.: B&B)."""
        return self.D.tolist()


def euc2d_matrix(coords: np.ndarray) -> np.ndarray:
    """Distâncias EUC_2D do TSPLIB: nint(sqrt(dx² + dy²)) = floor(d + 0.5)."""
    c = np.asarray(coords, dtype=np.float64)
    diff = c[:, None, :] - c[None, :, :]
    d = np.hypot(diff[..., 0], diff[..., 1])
    return np.floor(d + 0.5).astype(np.int64)


def instance_from_coords(name: str, coords: Sequence[Sequence[float]] | np.ndarray) -> Instance:
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"coords deve ter shape (n, 2), recebido {c.shape}")
    if c.shape[0] < 3:
        raise ValueError("instância precisa de pelo menos 3 vértices")
    return Instance(name=name, n=c.shape[0], coords=c, D=euc2d_matrix(c))


# --------------------------------------------------------------------------- #
# I/O TSPLIB
# --------------------------------------------------------------------------- #


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _parse_header_line(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    key, _, value = line.partition(":")
    return key.strip().upper(), value.strip()


def load_tsplib(path: str | Path) -> Instance:
    """Lê um arquivo .tsp (ou .tsp.gz) do TSPLIB com EDGE_WEIGHT_TYPE EUC_2D."""
    path = Path(path)
    header: dict[str, str] = {}
    coords: dict[int, tuple[float, float]] = {}
    in_coords = False

    with _open_text(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "EOF":
                break
            if upper.startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue
            if in_coords:
                parts = line.split()
                if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
                    # fim da seção de coordenadas (outra seção começou)
                    in_coords = False
                else:
                    coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
                    continue
            kv = _parse_header_line(line)
            if kv:
                header[kv[0]] = kv[1]

    ewt = header.get("EDGE_WEIGHT_TYPE", "").upper()
    if ewt != "EUC_2D":
        raise ValueError(f"{path.name}: EDGE_WEIGHT_TYPE={ewt or '?'}; apenas EUC_2D é suportado")
    if not coords:
        raise ValueError(f"{path.name}: NODE_COORD_SECTION ausente ou vazia")

    dim = int(header.get("DIMENSION", len(coords)))
    ids = sorted(coords)
    if len(ids) != dim or ids != list(range(1, dim + 1)):
        raise ValueError(f"{path.name}: esperados nós 1..{dim}, lidos {len(ids)}")

    name = header.get("NAME", path.name.split(".")[0])
    return instance_from_coords(name, [coords[i] for i in ids])


def write_tsplib(inst: Instance, path: str | Path, comment: str = "") -> None:
    """Grava a instância como .tsp EUC_2D válido (usado para as sub-instâncias)."""
    path = Path(path)
    lines = [
        f"NAME : {inst.name}",
        f"COMMENT : {comment}" if comment else None,
        "TYPE : TSP",
        f"DIMENSION : {inst.n}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    out = [ln for ln in lines if ln is not None]
    for i, (x, y) in enumerate(inst.coords, start=1):
        out.append(f"{i} {x:.10g} {y:.10g}")
    out.append("EOF")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def load_opt_tour(path: str | Path) -> list[int]:
    """Lê um .opt.tour do TSPLIB e devolve o tour 0-based normalizado (começa em 0)."""
    path = Path(path)
    tour: list[int] = []
    in_tour = False
    with _open_text(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("TOUR_SECTION"):
                in_tour = True
                continue
            if not in_tour:
                continue
            for tok in line.split():
                v = int(tok)
                if v == -1:
                    in_tour = False
                    break
                tour.append(v - 1)
            if not in_tour:
                break
    if not tour:
        raise ValueError(f"{path.name}: TOUR_SECTION vazia")
    return normalize_tour(tour)


# --------------------------------------------------------------------------- #
# Tours
# --------------------------------------------------------------------------- #

TourBuilder = Callable[[Instance, "Profiler"], list[int]]


def tour_cost(D: np.ndarray, tour: Sequence[int]) -> int:
    """Custo do ciclo hamiltoniano (inclui a aresta de volta ao início)."""
    t = np.asarray(tour, dtype=np.intp)
    return int(D[t, np.roll(t, -1)].sum())


def normalize_tour(tour: Sequence[int]) -> list[int]:
    """Rotaciona o tour para começar no vértice 0."""
    t = list(tour)
    if t and t[-1] == t[0] and len(t) > 1:
        t = t[:-1]  # aceita tour "fechado" (0 repetido no fim)
    k = t.index(0)
    return t[k:] + t[:k]


def validate_tour(n: int, tour: Sequence[int]) -> None:
    """Levanta ValueError se ``tour`` não for permutação de 0..n-1 começando em 0."""
    if len(tour) != n:
        raise ValueError(f"tour com {len(tour)} vértices, esperado {n}")
    if len(tour) == 0 or tour[0] != 0:
        raise ValueError("tour deve começar no vértice 0")
    if sorted(tour) != list(range(n)):
        raise ValueError("tour não é uma permutação de 0..n-1")


# --------------------------------------------------------------------------- #
# Profiler por subrotina
# --------------------------------------------------------------------------- #


@dataclass
class _SectionStats:
    time_s: float = 0.0
    calls: int = 0
    peak_kb: float | None = None


@dataclass
class Profiler:
    """Profiler leve por subrotina.

    Uso:
        prof = Profiler()            # só tempo
        with prof.section("mst"):
            ...
        prof.total("mst")

    Com ``mem=True`` registra o pico de memória alocada (tracemalloc) dentro de
    cada seção. O tracemalloc deixa o código bem mais lento, então meça tempo e
    memória em passadas separadas. Seções aninhadas são permitidas; no modo
    memória o pico de cada seção é relativo à memória viva na entrada dela e
    o pico de uma seção externa inclui o das internas.
    """

    mem: bool = False
    _stats: dict[str, _SectionStats] = field(default_factory=dict)
    _owns_tracemalloc: bool = False
    _mem_stack: list[tuple[int, int]] = field(default_factory=list)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        st = self._stats.setdefault(name, _SectionStats())
        if self.mem:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._owns_tracemalloc = True
            base, peak_so_far = tracemalloc.get_traced_memory()
            if self._mem_stack:
                # salva o pico da seção externa antes de reset_peak() apagá-lo
                pbase, ppeak = self._mem_stack[-1]
                self._mem_stack[-1] = (pbase, max(ppeak, peak_so_far - pbase))
            tracemalloc.reset_peak()
            self._mem_stack.append((base, 0))
        t0 = time.perf_counter()
        try:
            yield
        finally:
            st.time_s += time.perf_counter() - t0
            st.calls += 1
            if self.mem:
                _, peak = tracemalloc.get_traced_memory()
                base, own_peak = self._mem_stack.pop()
                peak_bytes = max(peak - base, own_peak)
                peak_kb = peak_bytes / 1024.0
                st.peak_kb = peak_kb if st.peak_kb is None else max(st.peak_kb, peak_kb)
                if self._mem_stack:  # propaga para a seção externa (em bytes, relativo à base dela)
                    pbase, ppeak = self._mem_stack[-1]
                    self._mem_stack[-1] = (pbase, max(ppeak, base - pbase + peak_bytes))
                else:
                    if self._owns_tracemalloc:
                        tracemalloc.stop()
                        self._owns_tracemalloc = False

    def add(self, name: str, time_s: float, calls: int = 1, peak_kb: float | None = None) -> None:
        """Registra tempo medido externamente (ex.: acumulado com perf_counter
        num hot loop, onde um context manager por chamada distorceria a medida)."""
        st = self._stats.setdefault(name, _SectionStats())
        st.time_s += time_s
        st.calls += calls
        if peak_kb is not None:
            st.peak_kb = peak_kb if st.peak_kb is None else max(st.peak_kb, peak_kb)

    def total(self, name: str) -> float:
        st = self._stats.get(name)
        return st.time_s if st else 0.0

    def calls(self, name: str) -> int:
        st = self._stats.get(name)
        return st.calls if st else 0

    def peak_kb(self, name: str) -> float | None:
        st = self._stats.get(name)
        return st.peak_kb if st else None

    def sections(self) -> list[str]:
        return list(self._stats)

    def reset(self) -> None:
        self._stats.clear()

    def records(self, **ctx) -> list[dict]:
        """Linhas no formato de PROFILE_COLUMNS. ``ctx`` fornece instance, n,
        algorithm, rep; colunas ausentes ficam None."""
        rows = []
        for name, st in self._stats.items():
            row = {c: ctx.get(c) for c in PROFILE_COLUMNS}
            row.update(section=name, time_s=st.time_s, calls=st.calls, peak_kb=st.peak_kb)
            rows.append(row)
        return rows


# --------------------------------------------------------------------------- #
# Esquema de resultados
# --------------------------------------------------------------------------- #

STATUS_OK = "OK"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_ERROR = "ERROR"

TIME_LIMIT_S = 30 * 60  # limite do enunciado por execução

RUNS_COLUMNS = [
    "instance", "n", "algorithm", "warm_start", "rep", "status",
    "cost", "optimal_ref", "gap",
    "time_total_s", "time_heuristic_s", "peak_mem_kb",
    "expanded", "pruned", "generated", "max_depth", "max_frontier", "time_to_best_s",
]

PROFILE_COLUMNS = ["instance", "n", "algorithm", "rep", "section", "time_s", "calls", "peak_kb"]

# Ótimos conhecidos (TSPLIB) das instâncias-fonte EUC_2D.
OPTIMA: dict[str, int] = {
    "eil51": 426,
    "berlin52": 7542,
    "st70": 675,
    "eil76": 538,
    "pr76": 108159,
    "kroA100": 21282,
    "ch130": 6110,
    "a280": 2579,
}


def gap(cost: float | None, optimal: float | None) -> float | None:
    """Gap relativo (cost - opt) / opt; None se algum lado for desconhecido."""
    if cost is None or optimal is None or optimal == 0:
        return None
    return (cost - optimal) / optimal


def make_run_row(**values) -> dict:
    """Linha de runs com todas as colunas (ausentes = None). Rejeita colunas
    desconhecidas para manter o esquema estável entre os desenvolvedores."""
    unknown = set(values) - set(RUNS_COLUMNS)
    if unknown:
        raise KeyError(f"colunas fora do esquema RUNS_COLUMNS: {sorted(unknown)}")
    return {c: values.get(c) for c in RUNS_COLUMNS}


class ResultWriter:
    """Writer JSONL append-only, com flush por linha: se um experimento longo
    morrer no meio, o que já terminou está no arquivo. Converta para CSV
    (pandas.read_json(path, lines=True)) na hora da análise."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, default=_json_default) + "\n")
        self._fh.flush()

    def write_many(self, rows: Sequence[dict]) -> None:
        for r in rows:
            self.write(r)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"não serializável: {type(o)}")
