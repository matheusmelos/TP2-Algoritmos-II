"""Orquestrador dos experimentos do TP2 (Etapa 3 – A + B).

Executa o pipeline completo de avaliação empírica:
  1. Carrega as instâncias de ``data/instances/manifest.csv`` (gera se ausente).
  2. Extrai ``results/features.csv`` para todas as instâncias.
  3. Executa construções e aproximações (``trivial``, ``nn``, ``hull``, ``tatt``,
     ``christofides`` e, por padrão, a ablação ``christofides_greedy``), com
     ``--reps`` repetições cronometradas (``Profiler(mem=False)``) e uma passada
     separada de memória (``Profiler(mem=True)``) sem distorcer os tempos.
  4. Executa os 80 jobs de B&B (instância × 5 warm-starts), do menor N para o
     maior, com ``nucleos - 1`` workers e um processo isolado por job. Garante
     limite interno (``--time-limit``, padrão 30 min) e kill externo após
     ``time_limit + grace`` (padrão +60 s = 31 min).
  5. Preenche ``optimal_ref`` e ``gap`` em ``runs.jsonl`` usando ``OPTIMA`` e os
     ótimos exatos encontrados pelo B&B.
  6. Opcionalmente roda tamanhos suplementares (``--extra-sizes``) para TATT e
     Christofides (gravados separadamente em ``*_supplementary.jsonl``) e gera
     todas as tabelas e figuras via ``src/analysis/`` (``--analyze``).

Uso típico:
    # Piloto de 60 s validando o pipeline inteiro até as figuras:
    .venv/bin/python -m src.run_experiments --pilot --out results_pilot

    # Rodada real de 30 min:
    .venv/bin/python -m src.run_experiments --out results --analyze
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from src.tsp.approx import christofides, christofides_greedy_matching, twice_around_the_tree
from src.tsp.bnb import BOUNDS, SEARCHES, WARM_STARTS, bnb
from src.tsp.common import (
    OPTIMA,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_TIMEOUT,
    TIME_LIMIT_S,
    Instance,
    Profiler,
    ResultWriter,
    gap,
    instance_from_coords,
    load_tsplib,
    make_run_row,
    read_jsonl,
    tour_cost,
    validate_tour,
)
from src.tsp.features import extract_features_table, write_features_csv
from src.tsp.heuristics import convex_hull_insertion, nearest_neighbor, trivial
from src.tsp.make_instances import DEFAULT_OUT as DEFAULT_INSTANCES_DIR
from src.tsp.make_instances import DEFAULT_SRC as DEFAULT_TSPLIB_DIR
from src.tsp.make_instances import generate as generate_instances
from src.tsp.make_instances import read_manifest, unique_points

ROOT = Path(__file__).resolve().parents[1]

WARM_START_ORDER: tuple[str, ...] = ("trivial", "nn", "hull", "tatt", "christofides")

CONSTRUCTIVE_ALGORITHMS = {
    "trivial": trivial,
    "nn": nearest_neighbor,
    "hull": convex_hull_insertion,
    "tatt": twice_around_the_tree,
    "christofides": christofides,
    "christofides_greedy": christofides_greedy_matching,
}

DEFAULT_EXTRA_SIZES: tuple[int, ...] = (100, 130, 180, 230, 280, 350, 420, 500)


# --------------------------------------------------------------------------- #
# Workers (top-level para pickling no multiprocessing / spawn)
# --------------------------------------------------------------------------- #


def _merge_mem_into_profile(time_records: list[dict], mem_prof: Profiler | None) -> list[dict]:
    if mem_prof is None:
        return time_records
    out = []
    for row in time_records:
        r = dict(row)
        r["peak_kb"] = mem_prof.peak_kb(r["section"])
        out.append(r)
    return out


def _constructive_worker(
    inst_path: str,
    alg_name: str,
    reps: int,
    optimal_ref: int | None,
    measure_mem: bool,
) -> tuple[list[dict], list[dict]]:
    """Executa ``reps`` repetições cronometradas + 1 passada separada de memória."""
    inst = load_tsplib(inst_path)
    fn = CONSTRUCTIVE_ALGORITHMS[alg_name]

    run_rows: list[dict] = []
    prof_rows: list[dict] = []

    peak_mem_kb: float | None = None
    mem_prof: Profiler | None = None
    if measure_mem:
        mem_prof = Profiler(mem=True)
        with mem_prof.section("__total__"):
            fn(inst, mem_prof)
        peak_mem_kb = mem_prof.peak_kb("__total__")

    for rep in range(reps):
        prof = Profiler(mem=False)
        t0 = time.perf_counter()
        tour = fn(inst, prof)
        elapsed = time.perf_counter() - t0
        validate_tour(inst.n, tour)
        cost = tour_cost(inst.D, tour)

        run_rows.append(
            make_run_row(
                instance=inst.name,
                n=inst.n,
                algorithm=alg_name,
                warm_start=None,
                rep=rep,
                status=STATUS_OK,
                cost=cost,
                optimal_ref=optimal_ref,
                gap=gap(cost, optimal_ref),
                time_total_s=elapsed,
                time_heuristic_s=None,
                peak_mem_kb=peak_mem_kb if rep == 0 else None,
            )
        )
        recs = prof.records(instance=inst.name, n=inst.n, algorithm=alg_name, rep=rep)
        if rep == 0:
            recs = _merge_mem_into_profile(recs, mem_prof)
        prof_rows.extend(recs)

    return run_rows, prof_rows


def _bnb_child_target(
    queue: Any,
    inst_path: str,
    warm_name: str,
    time_limit_s: float,
    search: str,
    bound: str,
    optimal_ref: int | None,
    mem_bnb_max_n: int,
) -> None:
    """Alvo executado em processo isolado para um único job de B&B."""
    try:
        inst = load_tsplib(inst_path)
        warm_fn = WARM_STARTS[warm_name]

        # 1. Warm-start cronometrado separadamente
        warm_prof = Profiler(mem=False)
        t_h0 = time.perf_counter()
        ub_tour = warm_fn(inst, warm_prof)
        t_heur = time.perf_counter() - t_h0

        # 2. Busca B&B cronometrada (sem tracemalloc)
        bnb_prof = Profiler(mem=False)
        res = bnb(
            inst,
            ub_tour=ub_tour,
            time_limit_s=time_limit_s,
            prof=bnb_prof,
            search=search,
            bound=bound,
        )

        # 3. Passada separada de memória apenas para instâncias pequenas que terminaram rápido
        peak_mem_kb: float | None = None
        mem_prof: Profiler | None = None
        if (
            mem_bnb_max_n > 0
            and inst.n <= mem_bnb_max_n
            and res.status == STATUS_OK
            and res.time_s <= max(2.0, time_limit_s * 0.2)
        ):
            mem_prof = Profiler(mem=True)
            with mem_prof.section("__total__"):
                bnb(
                    inst,
                    ub_tour=ub_tour,
                    time_limit_s=max(10.0, time_limit_s * 0.5),
                    prof=mem_prof,
                    search=search,
                    bound=bound,
                )
            peak_mem_kb = mem_prof.peak_kb("__total__")

        run_row = res.to_row(
            inst,
            warm_start=warm_name,
            time_heuristic_s=t_heur,
            optimal_ref=optimal_ref,
            rep=0,
            peak_mem_kb=peak_mem_kb,
        )
        alg_label = f"{res.algorithm}:{warm_name}"
        prof_rows = bnb_prof.records(
            instance=inst.name,
            n=inst.n,
            algorithm=alg_label,
            rep=0,
        )
        prof_rows = _merge_mem_into_profile(prof_rows, mem_prof)
        queue.put(("ok", run_row, prof_rows))
    except Exception as exc:
        queue.put(("err", repr(exc), []))


def _run_bnb_isolated(
    inst_path: Path,
    inst_name: str,
    inst_n: int,
    warm_name: str,
    time_limit_s: float,
    hard_grace_s: float,
    search: str,
    bound: str,
    optimal_ref: int | None,
    mem_bnb_max_n: int,
) -> tuple[dict, list[dict]]:
    """Dispara 1 job de B&B em processo dedicado com kill externo garantido."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_bnb_child_target,
        args=(
            q,
            str(inst_path),
            warm_name,
            time_limit_s,
            search,
            bound,
            optimal_ref,
            mem_bnb_max_n,
        ),
    )
    t_wall0 = time.perf_counter()
    proc.start()
    hard_limit = time_limit_s + hard_grace_s
    proc.join(timeout=hard_limit)

    alg_name = f"bnb_{search}_{bound}"
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)
        elapsed = time.perf_counter() - t_wall0
        row = make_run_row(
            instance=inst_name,
            n=inst_n,
            algorithm=alg_name,
            warm_start=warm_name,
            rep=0,
            status=STATUS_TIMEOUT,
            cost=None,
            optimal_ref=optimal_ref,
            gap=None,
            time_total_s=elapsed,
            time_heuristic_s=None,
        )
        return row, []

    if not q.empty():
        tag, payload, prof_rows = q.get_nowait()
        if tag == "ok":
            return payload, prof_rows

    elapsed = time.perf_counter() - t_wall0
    row = make_run_row(
        instance=inst_name,
        n=inst_n,
        algorithm=alg_name,
        warm_start=warm_name,
        rep=0,
        status=STATUS_ERROR,
        cost=None,
        optimal_ref=optimal_ref,
        gap=None,
        time_total_s=elapsed,
    )
    return row, []


# --------------------------------------------------------------------------- #
# Pós-processamento de ótimos e gaps
# --------------------------------------------------------------------------- #


def backfill_optima_and_gaps(
    runs_path: Path,
    manifest_optima: dict[str, int],
) -> dict[str, int]:
    """Atualiza ``optimal_ref`` e ``gap`` em ``runs.jsonl`` com ótimos do B&B + OPTIMA."""
    if not runs_path.exists():
        return dict(manifest_optima)

    rows = read_jsonl(runs_path)
    optima: dict[str, int] = dict(OPTIMA)
    optima.update({k: v for k, v in manifest_optima.items() if v is not None})

    # Coleta ótimos exatos provados pelo B&B
    for r in rows:
        alg = str(r.get("algorithm") or "")
        if alg.startswith("bnb") and r.get("status") == STATUS_OK and r.get("cost") is not None:
            inst = r["instance"]
            cost = int(r["cost"])
            prev = optima.get(inst)
            if prev is not None and prev != cost:
                raise ValueError(
                    f"Inconsistência de ótimo em {inst}: referência={prev}, B&B={cost}"
                )
            optima[inst] = cost

    updated: list[dict] = []
    for r in rows:
        nr = dict(r)
        ref = optima.get(nr["instance"])
        if ref is not None:
            nr["optimal_ref"] = ref
            if nr.get("status") == STATUS_OK and nr.get("cost") is not None:
                nr["gap"] = gap(nr["cost"], ref)
            else:
                nr["gap"] = None
        updated.append(nr)

    tmp = runs_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in updated:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, runs_path)
    return optima


# --------------------------------------------------------------------------- #
# Instâncias suplementares para curvas de complexidade (opcional)
# --------------------------------------------------------------------------- #


def _build_supplementary_instances(
    tsplib_dir: Path,
    sizes: tuple[int, ...] = DEFAULT_EXTRA_SIZES,
    seed: int = 2026,
) -> list[Instance]:
    """Gera instâncias suplementares (até ~500) a partir de kroA100, ch130 e a280."""
    sources: dict[str, np.ndarray] = {}
    for name in ("kroA100", "ch130", "a280"):
        p = tsplib_dir / f"{name}.tsp"
        if p.exists():
            sources[name] = unique_points(load_tsplib(p).coords)
    if not sources:
        return []

    rng = np.random.default_rng(seed)
    pool_parts = list(sources.values())
    combined = np.vstack(pool_parts)
    # Remove duplicatas exatas preservando ordem
    combined = unique_points(combined)

    insts: list[Instance] = []
    for n in sorted(sizes):
        if n == 100 and "kroA100" in sources:
            insts.append(instance_from_coords("supp_kroA100", sources["kroA100"]))
        elif n == 130 and "ch130" in sources:
            insts.append(instance_from_coords("supp_ch130", sources["ch130"]))
        elif n <= len(sources.get("a280", [])) and "a280" in sources:
            pts = sources["a280"]
            if n == len(pts):
                chosen = pts
            else:
                idx = np.sort(rng.choice(len(pts), size=n, replace=False))
                chosen = pts[idx]
            insts.append(instance_from_coords(f"supp_a280_{n}", chosen))
        else:
            # Para N > 280, combina a280 + ch130 + kroA100 e completa com jitter leve se preciso
            if n <= len(combined):
                idx = np.sort(rng.choice(len(combined), size=n, replace=False))
                chosen = combined[idx]
            else:
                extra = n - len(combined)
                base_idx = rng.integers(len(combined), size=extra)
                jitter = rng.normal(0.0, 15.0, size=(extra, 2))
                chosen = np.vstack([combined, combined[base_idx] + jitter])
            insts.append(instance_from_coords(f"supp_mix_{n}", chosen))
    return insts


def run_supplementary_scaling(
    tsplib_dir: Path,
    out_dir: Path,
    reps: int = 5,
    sizes: tuple[int, ...] = DEFAULT_EXTRA_SIZES,
) -> None:
    """Executa TATT, Christofides e Christofides guloso nas instâncias suplementares."""
    insts = _build_supplementary_instances(tsplib_dir, sizes=sizes)
    if not insts:
        return
    runs_path = out_dir / "runs_supplementary.jsonl"
    prof_path = out_dir / "profile_supplementary.jsonl"
    algs = ("tatt", "christofides", "christofides_greedy")

    with ResultWriter(runs_path) as rw, ResultWriter(prof_path) as pw:
        for inst in insts:
            for alg in algs:
                fn = CONSTRUCTIVE_ALGORITHMS[alg]
                mem_prof = Profiler(mem=True)
                with mem_prof.section("__total__"):
                    fn(inst, mem_prof)
                peak_kb = mem_prof.peak_kb("__total__")

                for rep in range(reps):
                    prof = Profiler(mem=False)
                    t0 = time.perf_counter()
                    tour = fn(inst, prof)
                    elapsed = time.perf_counter() - t0
                    validate_tour(inst.n, tour)
                    cost = tour_cost(inst.D, tour)
                    rw.write(
                        make_run_row(
                            instance=inst.name,
                            n=inst.n,
                            algorithm=alg,
                            warm_start=None,
                            rep=rep,
                            status=STATUS_OK,
                            cost=cost,
                            optimal_ref=None,
                            gap=None,
                            time_total_s=elapsed,
                            peak_mem_kb=peak_kb if rep == 0 else None,
                        )
                    )
                    recs = prof.records(instance=inst.name, n=inst.n, algorithm=alg, rep=rep)
                    if rep == 0:
                        recs = _merge_mem_into_profile(recs, mem_prof)
                    pw.write_many(recs)


def _collect_metadata(
    config: dict[str, Any],
    started_at_utc: str,
    wall_time_s: float,
) -> dict[str, Any]:
    """Coleta metadados de hardware, software e git para reprodutibilidade."""
    import platform
    import subprocess
    import sys

    cpu_model = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    ram_gb: float | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                ram_gb = round(kb / (1024 * 1024), 2)
                break
    except Exception:
        pass

    git_commit = "unknown"
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            git_commit = out
    except Exception:
        pass

    def _pkg_ver(mod_name: str) -> str:
        try:
            mod = __import__(mod_name)
            return str(getattr(mod, "__version__", "?"))
        except Exception:
            return "?"

    return {
        "started_at_utc": started_at_utc,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_s": round(wall_time_s, 3),
        "git_commit": git_commit,
        "hardware": {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "cpu_model": cpu_model,
            "cpu_count": os.cpu_count(),
            "ram_total_gb": ram_gb,
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": _pkg_ver("numpy"),
            "scipy": _pkg_ver("scipy"),
            "networkx": _pkg_ver("networkx"),
            "sklearn": _pkg_ver("sklearn"),
            "pandas": _pkg_ver("pandas"),
            "matplotlib": _pkg_ver("matplotlib"),
        },
        "config": config,
    }


def _export_jsonl_to_csv(jsonl_path: Path) -> Path | None:
    """Salva um espelho ``.csv`` de ``jsonl_path`` preservando todas as colunas."""
    if not jsonl_path.exists():
        return None
    rows = read_jsonl(jsonl_path)
    if not rows:
        return None
    import csv

    csv_path = jsonl_path.with_suffix(".csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# --------------------------------------------------------------------------- #
# Orquestração principal
# --------------------------------------------------------------------------- #


def _existing_keys(runs_path: Path) -> set[tuple]:
    if not runs_path.exists():
        return set()
    keys: set[tuple] = set()
    for r in read_jsonl(runs_path):
        keys.add((r.get("instance"), r.get("algorithm"), r.get("warm_start"), r.get("rep")))
    return keys


def run_all(
    instances_dir: Path = DEFAULT_INSTANCES_DIR,
    tsplib_dir: Path = DEFAULT_TSPLIB_DIR,
    out_dir: Path = ROOT / "results",
    time_limit_s: float = TIME_LIMIT_S,
    hard_grace_s: float = 60.0,
    reps: int = 10,
    search: str = "dfs",
    bound: str = "one_tree",
    workers: int | None = None,
    mem_bnb_max_n: int = 18,
    include_ablation: bool = True,
    max_n: int | None = None,
    instances_subset: set[str] | None = None,
    resume: bool = False,
    skip_constructive: bool = False,
    skip_bnb: bool = False,
    extra_sizes: bool = False,
    run_analysis: bool = False,
) -> dict[str, Any]:
    """Executa o pipeline experimental completo e devolve um resumo."""
    t_start = time.perf_counter()
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (instances_dir / "manifest.csv").exists():
        generate_instances(src_dir=tsplib_dir, out_dir=instances_dir)

    manifest = read_manifest(instances_dir)
    if max_n is not None:
        manifest = [m for m in manifest if m["n"] <= max_n]
    if instances_subset:
        manifest = [m for m in manifest if m["name"] in instances_subset]
    manifest.sort(key=lambda r: (r["n"], r["name"]))

    n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    runs_path = out_dir / "runs.jsonl"
    prof_path = out_dir / "profile.jsonl"
    feats_path = out_dir / "features.csv"

    if not resume:
        for p in (runs_path, prof_path):
            if p.exists():
                p.unlink()
    done_keys = _existing_keys(runs_path) if resume else set()

    # 1. Features geométricas/topológicas
    inst_objs = [load_tsplib(instances_dir / m["file"]) for m in manifest]
    feat_rows = extract_features_table(inst_objs)
    write_features_csv(feat_rows, feats_path)
    print(f"[features] {len(feat_rows)} instâncias -> {feats_path}")

    manifest_optima = {m["name"]: m["optimal_ref"] for m in manifest if m["optimal_ref"] is not None}

    # 2. Construções e aproximações (tempo + memória separada)
    constructive_count = 0
    if not skip_constructive:
        algs = list(WARM_START_ORDER)
        if include_ablation:
            algs.append("christofides_greedy")

        jobs = []
        for m in manifest:
            for alg in algs:
                if resume and all((m["name"], alg, None, r) in done_keys for r in range(reps)):
                    continue
                jobs.append((str(instances_dir / m["file"]), alg, reps, m["optimal_ref"], True))

        print(f"[constructive] {len(jobs)} jobs (×{reps} reps) com {n_workers} workers")
        with ResultWriter(runs_path) as rw, ResultWriter(prof_path) as pw:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futs = [pool.submit(_constructive_worker, *job) for job in jobs]
                for fut in as_completed(futs):
                    r_rows, p_rows = fut.result()
                    rw.write_many(r_rows)
                    pw.write_many(p_rows)
                    constructive_count += len(r_rows)

    # 3. Branch-and-Bound (menor N -> maior N, 1 processo isolado por job)
    bnb_count = 0
    if not skip_bnb:
        bnb_alg = f"bnb_{search}_{bound}"
        bnb_jobs = []
        for m in manifest:
            for ws in WARM_START_ORDER:
                if resume and (m["name"], bnb_alg, ws, 0) in done_keys:
                    continue
                bnb_jobs.append((instances_dir / m["file"], m["name"], m["n"], ws, m["optimal_ref"]))

        print(
            f"[bnb] {len(bnb_jobs)} jobs ({bnb_alg}, limit={time_limit_s:.0f}s, "
            f"kill=+{hard_grace_s:.0f}s) com {n_workers} workers"
        )
        with ResultWriter(runs_path) as rw, ResultWriter(prof_path) as pw:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                fut_to_meta = {
                    pool.submit(
                        _run_bnb_isolated,
                        path,
                        name,
                        n,
                        ws,
                        time_limit_s,
                        hard_grace_s,
                        search,
                        bound,
                        opt_ref,
                        mem_bnb_max_n,
                    ): (name, n, ws)
                    for (path, name, n, ws, opt_ref) in bnb_jobs
                }
                for fut in as_completed(fut_to_meta):
                    name, n, ws = fut_to_meta[fut]
                    run_row, prof_rows = fut.result()
                    rw.write(run_row)
                    pw.write_many(prof_rows)
                    bnb_count += 1
                    print(
                        f"  [{bnb_count:2d}/{len(bnb_jobs)}] {name:20s} n={n:2d} ws={ws:12s} "
                        f"status={run_row['status']:7s} t={run_row['time_total_s']:.2f}s "
                        f"exp={run_row['expanded']}"
                    )

    # 4. Backfill de optimal_ref e gap + espelhos CSV
    optima = backfill_optima_and_gaps(runs_path, manifest_optima)
    _export_jsonl_to_csv(runs_path)
    _export_jsonl_to_csv(prof_path)

    # 5. Instâncias suplementares (opcional)
    supp_runs = out_dir / "runs_supplementary.jsonl"
    supp_prof = out_dir / "profile_supplementary.jsonl"
    if extra_sizes:
        for p in (supp_runs, supp_prof):
            if p.exists():
                p.unlink()
        print("[supplementary] executando tamanhos extras para TATT e Christofides...")
        run_supplementary_scaling(tsplib_dir, out_dir, reps=min(reps, 5))
        _export_jsonl_to_csv(supp_runs)
        _export_jsonl_to_csv(supp_prof)

    # 6. Metadados do experimento (gravados antes do generate_insights para inclusão no dossiê)
    wall_time_s = time.perf_counter() - t_start
    meta = _collect_metadata(
        config={
            "time_limit_s": time_limit_s,
            "hard_grace_s": hard_grace_s,
            "reps": reps,
            "search": search,
            "bound": bound,
            "workers": n_workers,
            "mem_bnb_max_n": mem_bnb_max_n,
            "include_ablation": include_ablation,
            "max_n": max_n,
            "instances_subset": sorted(instances_subset) if instances_subset else None,
            "resume": resume,
            "extra_sizes": extra_sizes,
            "instances_count": len(manifest),
        },
        started_at_utc=started_at_utc,
        wall_time_s=wall_time_s,
    )
    meta_path = out_dir / "experiment_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 7. Pipeline de análise, validação e consolidação de insights (opcional)
    if run_analysis and runs_path.exists():
        import pandas as pd

        from src.analysis.complexity import analyze_complexity
        from src.analysis.sensitivity import analyze_sensitivity
        from src.analysis.warmstart import analyze_warmstart
        from src.generate_insights import generate_insights
        from src.validate_results import validate_results_dir

        runs_df = pd.DataFrame(read_jsonl(runs_path))
        prof_df = pd.DataFrame(read_jsonl(prof_path)) if prof_path.exists() else None
        feats_df = pd.read_csv(feats_path)
        analyze_complexity(runs_df, prof_df, out_dir)
        analyze_warmstart(runs_df, out_dir)
        analyze_sensitivity(runs_df, feats_df, out_dir)

        if supp_runs.exists():
            sr_df = pd.DataFrame(read_jsonl(supp_runs))
            sp_df = pd.DataFrame(read_jsonl(supp_prof)) if supp_prof.exists() else None
            analyze_complexity(sr_df, sp_df, out_dir / "supplementary")

        validate_results_dir(out_dir)
        generate_insights(out_dir)
        print(f"[analysis] tabelas, figuras, validação e INSIGHTS.md gerados em {out_dir}")

    return {
        "instances": len(manifest),
        "constructive_rows": constructive_count,
        "bnb_rows": bnb_count,
        "optima_known": len(optima),
        "runs_path": runs_path,
        "profile_path": prof_path,
        "features_path": feats_path,
        "metadata_path": meta_path,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Orquestrador de experimentos do TP2 (Etapa 3)")
    ap.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES_DIR)
    ap.add_argument("--tsplib", type=Path, default=DEFAULT_TSPLIB_DIR)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT_S, help="limite interno do B&B em segundos")
    ap.add_argument("--hard-grace", type=float, default=60.0, help="margem até kill externo (s)")
    ap.add_argument("--reps", type=int, default=10, help="repetições por heurística/aproximação")
    ap.add_argument("--search", choices=SEARCHES, default="dfs")
    ap.add_argument("--bound", choices=BOUNDS, default="one_tree")
    ap.add_argument("--workers", type=int, default=None, help="padrão: núcleos - 1")
    ap.add_argument("--mem-bnb-max-n", type=int, default=18, help="N máximo para passada de memória do B&B")
    ap.add_argument("--max-n", type=int, default=None, help="filtra instâncias com N <= max-n")
    ap.add_argument("--instances-subset", type=str, default=None, help="lista separada por vírgula de nomes de instâncias")
    ap.add_argument("--no-ablation", action="store_true", help="omite christofides_greedy")
    ap.add_argument("--resume", action="store_true", help="pula jobs já presentes em runs.jsonl")
    ap.add_argument("--skip-constructive", action="store_true")
    ap.add_argument("--skip-bnb", action="store_true")
    ap.add_argument("--extra-sizes", action="store_true", help="roda tamanhos suplementares até ~500")
    ap.add_argument("--analyze", action="store_true", help="gera tabelas, figuras, validação e INSIGHTS.md ao final")
    ap.add_argument("--pilot", action="store_true", help="preset piloto: time-limit=60, reps=5, analyze")
    args = ap.parse_args(argv)

    time_limit = 60.0 if args.pilot and args.time_limit == TIME_LIMIT_S else args.time_limit
    reps = 5 if args.pilot and args.reps == 10 else args.reps
    analyze = True if args.pilot else args.analyze
    subset = (
        {s.strip() for s in args.instances_subset.split(",") if s.strip()}
        if args.instances_subset
        else None
    )

    summary = run_all(
        instances_dir=args.instances,
        tsplib_dir=args.tsplib,
        out_dir=args.out,
        time_limit_s=time_limit,
        hard_grace_s=args.hard_grace,
        reps=reps,
        search=args.search,
        bound=args.bound,
        workers=args.workers,
        mem_bnb_max_n=args.mem_bnb_max_n,
        include_ablation=not args.no_ablation,
        max_n=args.max_n,
        instances_subset=subset,
        resume=args.resume,
        skip_constructive=args.skip_constructive,
        skip_bnb=args.skip_bnb,
        extra_sizes=args.extra_sizes,
        run_analysis=analyze,
    )
    print(
        f"Concluído: {summary['instances']} instâncias, "
        f"{summary['constructive_rows']} linhas construtivas, "
        f"{summary['bnb_rows']} linhas B&B, "
        f"{summary['optima_known']} ótimos conhecidos."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
