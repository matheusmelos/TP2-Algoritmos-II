"""Microbenchmarks da Seção de Auditoria do Relatório SBC (Etapa 3).

Gera dados reprodutíveis (CSVs) e gráficos (PNGs) em ``<out>`` (padrão:
``results/audit``) cobrindo os três pontos exigidos no enunciado e no PLANO.md:

  1. Refatoração de abstração ineficiente: ``inst.D_list`` (listas Python) vs
     ``inst.D`` (indexação NumPy) no inner loop do Branch-and-Bound.
  2. Estratégia de busca (``dfs`` vs ``best_first``) e lower bound
     (``two_min`` vs ``one_tree``): tempo, nós expandidos, fronteira máxima e
     pico de memória.
  3. Ablação do Christofides: emparelhamento perfeito mínimo (Blossom/NetworkX)
     vs emparelhamento guloso (tempo, fração no ``matching``, memória e gap).

Uso:
    python -m src.benchmark_audit --out results/audit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.tsp.approx import christofides, christofides_greedy_matching
from src.tsp.bnb import BOUNDS, SEARCHES, WARM_STARTS, bnb
from src.tsp.common import (
    OPTIMA,
    STATUS_OK,
    Instance,
    Profiler,
    gap,
    load_tsplib,
    tour_cost,
)
from src.tsp.make_instances import DEFAULT_OUT as DEFAULT_INSTANCES_DIR
from src.tsp.make_instances import DEFAULT_SRC as DEFAULT_TSPLIB_DIR

ROOT = Path(__file__).resolve().parents[1]


class _NumpyDProxy:
    """Proxy mínimo de ``Instance`` que expõe ``inst.D`` (ndarray) no lugar de
    ``D_list``, permitindo medir exatamente o custo da indexação NumPy no inner
    loop do B&B sem alterar ``bnb.py`` ou ``common.py``."""

    def __init__(self, inst: Instance):
        self.name = inst.name
        self.n = inst.n
        self.coords = inst.coords
        self.D = inst.D
        self.D_list = inst.D


def benchmark_dlist_vs_numpy(
    instances_dir: Path,
    out_dir: Path,
    instance_names: tuple[str, ...] = (
        "eil51_random12",
        "berlin52_knn14",
        "pr76_random16",
        "eil76_window18",
    ),
    time_limit_s: float = 10.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    for name in instance_names:
        p = instances_dir / f"{name}.tsp"
        if not p.exists():
            continue
        inst = load_tsplib(p)
        ub = WARM_STARTS["hull"](inst, Profiler())
        proxy = _NumpyDProxy(inst)

        for bound in ("two_min", "one_tree"):
            r_list = bnb(inst, ub_tour=ub, time_limit_s=time_limit_s, search="dfs", bound=bound)
            r_np = bnb(proxy, ub_tour=ub, time_limit_s=time_limit_s, search="dfs", bound=bound)  # type: ignore[arg-type]

            nps_list = r_list.expanded / max(r_list.time_s, 1e-9)
            nps_np = r_np.expanded / max(r_np.time_s, 1e-9)
            # Se ambos completaram a mesma árvore, speedup por tempo; senão por nós/s
            if r_list.status == STATUS_OK and r_np.status == STATUS_OK and r_list.time_s > 0:
                speedup = r_np.time_s / r_list.time_s
            else:
                speedup = nps_list / max(nps_np, 1e-9)

            rows.append(
                {
                    "instance": inst.name,
                    "n": inst.n,
                    "bound": bound,
                    "status_dlist": r_list.status,
                    "status_numpy": r_np.status,
                    "expanded_dlist": r_list.expanded,
                    "expanded_numpy": r_np.expanded,
                    "time_dlist_s": r_list.time_s,
                    "time_numpy_s": r_np.time_s,
                    "nodes_per_s_dlist": nps_list,
                    "nodes_per_s_numpy": nps_np,
                    "speedup_dlist_vs_numpy": speedup,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "audit_dlist_vs_numpy.csv", index=False)

    if not df.empty:
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        for bound, color in (("two_min", "#1f77b4"), ("one_tree", "#2ca02c")):
            sub = df[df["bound"] == bound].sort_values("n")
            ax.plot(
                sub["n"],
                sub["speedup_dlist_vs_numpy"],
                marker="o",
                linewidth=2.0,
                color=color,
                label=f"bound={bound}",
            )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel("N (cidades)")
        ax.set_ylabel("Speedup (D_list / NumPy D[i, j])")
        ax.set_title("Auditoria: Impacto da Refatoração D_list vs Indexação NumPy no B&B")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "audit_dlist_vs_numpy.png", dpi=150)
        plt.close(fig)

    return df


def benchmark_search_and_bounds(
    instances_dir: Path,
    out_dir: Path,
    instance_names: tuple[str, ...] = (
        "eil51_random12",
        "berlin52_knn14",
        "pr76_random16",
        "eil76_window18",
        "berlin52_random20",
        "kroA100_knn22",
    ),
    time_limit_s: float = 15.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    for name in instance_names:
        p = instances_dir / f"{name}.tsp"
        if not p.exists():
            continue
        inst = load_tsplib(p)
        ub = WARM_STARTS["hull"](inst, Profiler())

        for search in SEARCHES:
            for bound in BOUNDS:
                prof = Profiler(mem=False)
                res = bnb(inst, ub_tour=ub, time_limit_s=time_limit_s, prof=prof, search=search, bound=bound)

                peak_kb = None
                if inst.n <= 18 and res.status == STATUS_OK and res.time_s <= 2.0:
                    mprof = Profiler(mem=True)
                    with mprof.section("__total__"):
                        bnb(inst, ub_tour=ub, time_limit_s=time_limit_s, prof=mprof, search=search, bound=bound)
                    peak_kb = mprof.peak_kb("__total__")

                rows.append(
                    {
                        "instance": inst.name,
                        "n": inst.n,
                        "search": search,
                        "bound": bound,
                        "status": res.status,
                        "initial_ub": res.initial_ub,
                        "root_lb": res.root_lb,
                        "best_cost": res.best_cost,
                        "time_s": res.time_s,
                        "expanded": res.expanded,
                        "pruned": res.pruned,
                        "generated": res.generated,
                        "max_frontier": res.max_frontier,
                        "mst_calls": res.mst_calls,
                        "mst_memo_hits": res.mst_memo_hits,
                        "peak_mem_kb": peak_kb,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "audit_search_bounds.csv", index=False)

    if not df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        styles = {
            ("dfs", "two_min"): ("#1f77b4", "--"),
            ("dfs", "one_tree"): ("#2ca02c", "-"),
            ("best_first", "two_min"): ("#d62728", "--"),
            ("best_first", "one_tree"): ("#ff7f0e", "-"),
        }
        for (search, bound), (color, ls) in styles.items():
            sub = df[(df["search"] == search) & (df["bound"] == bound)].sort_values("n")
            lbl = f"{search} + {bound}"
            axes[0].plot(sub["n"], np.maximum(sub["expanded"], 1), marker="o", linestyle=ls, color=color, label=lbl)
            axes[1].plot(sub["n"], np.maximum(sub["max_frontier"], 1), marker="o", linestyle=ls, color=color, label=lbl)

        axes[0].set_yscale("log")
        axes[0].set_title("Nós Expandidos por Estratégia e Bound")
        axes[0].set_xlabel("N (cidades)")
        axes[0].set_ylabel("Nós expandidos (log)")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=8.5)

        axes[1].set_yscale("log")
        axes[1].set_title("Fronteira Máxima (Pressão de Memória)")
        axes[1].set_xlabel("N (cidades)")
        axes[1].set_ylabel("max_frontier (log)")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(fontsize=8.5)

        fig.suptitle("Auditoria: DFS vs Best-First e two_min vs one_tree", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "audit_search_bounds.png", dpi=150)
        plt.close(fig)

    return df


def benchmark_christofides_ablation(
    tsplib_dir: Path,
    out_dir: Path,
    reps: int = 5,
) -> pd.DataFrame:
    names = ["eil51", "berlin52", "st70", "eil76", "pr76", "kroA100", "ch130", "a280"]
    rows: list[dict] = []
    for name in names:
        p = tsplib_dir / f"{name}.tsp"
        if not p.exists():
            continue
        inst = load_tsplib(p)
        opt = OPTIMA[name]

        # Exato
        t_ex, m_ex = [], []
        cost_ex = 0
        for _ in range(reps):
            pr = Profiler(mem=False)
            t0 = time.perf_counter()
            tour = christofides(inst, pr)
            t_ex.append(time.perf_counter() - t0)
            m_ex.append(pr.total("matching"))
            cost_ex = tour_cost(inst.D, tour)
        mp_ex = Profiler(mem=True)
        with mp_ex.section("__total__"):
            christofides(inst, mp_ex)

        # Guloso
        t_gr, m_gr = [], []
        cost_gr = 0
        for _ in range(reps):
            pr = Profiler(mem=False)
            t0 = time.perf_counter()
            tour = christofides_greedy_matching(inst, pr)
            t_gr.append(time.perf_counter() - t0)
            m_gr.append(pr.total("matching"))
            cost_gr = tour_cost(inst.D, tour)
        mp_gr = Profiler(mem=True)
        with mp_gr.section("__total__"):
            christofides_greedy_matching(inst, mp_gr)

        med_ex = float(np.median(t_ex))
        med_gr = float(np.median(t_gr))
        med_m_ex = float(np.median(m_ex))
        med_m_gr = float(np.median(m_gr))

        rows.append(
            {
                "instance": name,
                "n": inst.n,
                "optimal": opt,
                "cost_exact": cost_ex,
                "cost_greedy": cost_gr,
                "gap_exact": gap(cost_ex, opt),
                "gap_greedy": gap(cost_gr, opt),
                "time_exact_s": med_ex,
                "time_greedy_s": med_gr,
                "matching_exact_s": med_m_ex,
                "matching_greedy_s": med_m_gr,
                "matching_share_exact_pct": 100.0 * med_m_ex / max(med_ex, 1e-12),
                "speedup_greedy_vs_exact": med_ex / max(med_gr, 1e-12),
                "peak_mem_exact_kb": mp_ex.peak_kb("__total__"),
                "peak_mem_greedy_kb": mp_gr.peak_kb("__total__"),
            }
        )

    df = pd.DataFrame(rows).sort_values("n")
    df.to_csv(out_dir / "audit_christofides_ablation.csv", index=False)

    if not df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
        axes[0].plot(df["n"], df["time_exact_s"], marker="o", label="Christofides (Blossom exato)", color="#d62728")
        axes[0].plot(df["n"], df["matching_exact_s"], marker="^", linestyle="--", label="Subrotina matching (exato)", color="#ff7f0e")
        axes[0].plot(df["n"], df["time_greedy_s"], marker="s", label="Christofides (matching guloso)", color="#2ca02c")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("N (cidades)")
        axes[0].set_ylabel("Tempo mediano (s, log)")
        axes[0].set_title("Tempo: Gargalo do Matching Exato vs Ablação Gulosa")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=8.5)

        x = np.arange(len(df))
        w = 0.36
        axes[1].bar(x - w / 2, df["gap_exact"] * 100.0, width=w, label="Matching exato", color="#d62728")
        axes[1].bar(x + w / 2, df["gap_greedy"] * 100.0, width=w, label="Matching guloso", color="#2ca02c")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df["instance"], rotation=30, ha="right", fontsize=8.5)
        axes[1].set_ylabel("Gap relativo (%)")
        axes[1].set_title("Qualidade: Matching Exato vs Guloso (TSPLIB)")
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].legend(fontsize=8.5)

        fig.suptitle("Ablação do Emparelhamento no Algoritmo de Christofides", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "audit_christofides_ablation.png", dpi=150)
        plt.close(fig)

    return df


def _prim_with_copies(D: np.ndarray) -> np.ndarray:
    """Versão propositalmente ineficiente do Prim: copia a matriz D e aloca
    ``np.where`` a cada iteração, para quantificar o impacto de cópias
    desnecessárias de matrizes exigido no enunciado."""
    Dc = np.array(D, dtype=np.int64, copy=True)
    n = Dc.shape[0]
    parent = np.full(n, 0, dtype=np.intp)
    parent[0] = -1
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    inf = np.iinfo(np.int64).max
    key = Dc[0].copy()
    for _ in range(n - 1):
        masked = np.where(in_mst, inf, key)
        u = int(np.argmin(masked))
        in_mst[u] = True
        row = Dc[u].copy()
        better = (~in_mst) & (row < key)
        key = np.where(better, row, key)
        parent = np.where(better, u, parent)
    return parent


def benchmark_memory_and_gc(
    tsplib_dir: Path,
    out_dir: Path,
    names: tuple[str, ...] = ("eil51", "st70", "kroA100", "ch130", "a280"),
    reps: int = 5,
) -> pd.DataFrame:
    """Quantifica distorções práticas de complexidade causadas por memória/GC:
      1. Prim sem cópia vs Prim com cópias desnecessárias de matriz/vetores.
      2. Christofides (NetworkX) com GC habilitado vs GC desabilitado.
      3. Overhead do ``tracemalloc`` (``Profiler(mem=True)`` vs ``mem=False``).
    """
    import gc
    from src.tsp.mst import prim_mst

    rows: list[dict] = []
    for name in names:
        p = tsplib_dir / f"{name}.tsp"
        if not p.exists():
            continue
        inst = load_tsplib(p)

        # 1. Prim sem cópia vs com cópia de matriz
        t_clean, t_copy = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            prim_mst(inst.D)
            t_clean.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            _prim_with_copies(inst.D)
            t_copy.append(time.perf_counter() - t0)

        mp_clean = Profiler(mem=True)
        with mp_clean.section("__total__"):
            prim_mst(inst.D, mp_clean)
        mp_copy = Profiler(mem=True)
        with mp_copy.section("__total__"):
            _prim_with_copies(inst.D)

        # 2. Christofides com GC ligado vs desligado
        t_gc_on, t_gc_off = [], []
        for _ in range(reps):
            gc.collect()
            t0 = time.perf_counter()
            christofides(inst, Profiler(mem=False))
            t_gc_on.append(time.perf_counter() - t0)

            gc.collect()
            gc.disable()
            try:
                t0 = time.perf_counter()
                christofides(inst, Profiler(mem=False))
                t_gc_off.append(time.perf_counter() - t0)
            finally:
                gc.enable()

        # 3. Overhead do tracemalloc no Christofides
        t0 = time.perf_counter()
        mp_tr = Profiler(mem=True)
        with mp_tr.section("__total__"):
            christofides(inst, mp_tr)
        t_tracemalloc = time.perf_counter() - t0

        med_clean = float(np.median(t_clean))
        med_copy = float(np.median(t_copy))
        med_gc_on = float(np.median(t_gc_on))
        med_gc_off = float(np.median(t_gc_off))

        rows.append(
            {
                "instance": name,
                "n": inst.n,
                "prim_clean_s": med_clean,
                "prim_copy_s": med_copy,
                "prim_copy_slowdown": med_copy / max(med_clean, 1e-12),
                "prim_clean_mem_kb": mp_clean.peak_kb("__total__"),
                "prim_copy_mem_kb": mp_copy.peak_kb("__total__"),
                "prim_copy_mem_ratio": (mp_copy.peak_kb("__total__") or 0.0) / max(mp_clean.peak_kb("__total__") or 1e-9, 1e-9),
                "chris_gc_on_s": med_gc_on,
                "chris_gc_off_s": med_gc_off,
                "chris_gc_overhead_pct": 100.0 * (med_gc_on - med_gc_off) / max(med_gc_off, 1e-12),
                "chris_tracemalloc_s": t_tracemalloc,
                "tracemalloc_slowdown": t_tracemalloc / max(med_gc_on, 1e-12),
            }
        )

    df = pd.DataFrame(rows).sort_values("n")
    df.to_csv(out_dir / "audit_memory_gc.csv", index=False)

    if not df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
        axes[0].plot(df["n"], df["prim_clean_mem_kb"], marker="o", label="Prim sem cópia (KB)", color="#2ca02c")
        axes[0].plot(df["n"], df["prim_copy_mem_kb"], marker="s", label="Prim com D.copy() + np.where (KB)", color="#d62728")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("N (cidades)")
        axes[0].set_ylabel("Pico de memória (KB, log)")
        axes[0].set_title("Impacto de Cópias Desnecessárias de Matriz no Prim")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=8.5)

        axes[1].plot(df["n"], df["chris_gc_on_s"], marker="o", label="Christofides (GC ligado)", color="#1f77b4")
        axes[1].plot(df["n"], df["chris_gc_off_s"], marker="^", linestyle="--", label="Christofides (GC desligado)", color="#2ca02c")
        axes[1].plot(df["n"], df["chris_tracemalloc_s"], marker="s", label="Christofides (com tracemalloc)", color="#d62728")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("N (cidades)")
        axes[1].set_ylabel("Tempo (s, log)")
        axes[1].set_title("Overhead de GC e tracemalloc no Christofides")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(fontsize=8.5)

        fig.suptitle("Auditoria: Distorções de Memória, Cópias de Matriz e GC", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "audit_memory_gc.png", dpi=150)
        plt.close(fig)

    return df


def run_audit(
    instances_dir: Path = DEFAULT_INSTANCES_DIR,
    tsplib_dir: Path = DEFAULT_TSPLIB_DIR,
    out_dir: Path = ROOT / "results" / "audit",
    time_limit_s: float = 15.0,
) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[audit] 1/4 D_list vs NumPy D[i, j]...")
    df_d = benchmark_dlist_vs_numpy(instances_dir, out_dir, time_limit_s=min(time_limit_s, 10.0))
    print("[audit] 2/4 DFS vs Best-First e two_min vs one_tree...")
    df_sb = benchmark_search_and_bounds(instances_dir, out_dir, time_limit_s=time_limit_s)
    print("[audit] 3/4 Ablação Christofides (exato vs guloso)...")
    df_ab = benchmark_christofides_ablation(tsplib_dir, out_dir)
    print("[audit] 4/4 Gargalos de memória, cópias de matriz e GC...")
    df_mg = benchmark_memory_and_gc(tsplib_dir, out_dir)
    print(f"[audit] concluído -> {out_dir}")
    return {"dlist_vs_numpy": df_d, "search_bounds": df_sb, "ablation": df_ab, "memory_gc": df_mg}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Microbenchmarks de auditoria do TP2")
    ap.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES_DIR)
    ap.add_argument("--tsplib", type=Path, default=DEFAULT_TSPLIB_DIR)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "audit")
    ap.add_argument("--time-limit", type=float, default=15.0)
    args = ap.parse_args(argv)

    run_audit(args.instances, args.tsplib, args.out, time_limit_s=args.time_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
