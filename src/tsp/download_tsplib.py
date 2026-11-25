"""Baixa as instâncias-fonte EUC_2D do TSPLIB (e os .opt.tour disponíveis).

Usa só a biblioteca padrão, então roda até sem o venv:
    python3 src/tsp/download_tsplib.py [--out data/tsplib]

Os arquivos baixados são tratados como dados: só o parser de common.py os lê.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
INSTANCES = ["eil51", "berlin52", "st70", "eil76", "pr76", "kroA100", "ch130", "a280"]
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "tsplib"


def _fetch(url: str, timeout: float = 30.0) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "tsplib-fetch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def download(name: str, ext: str, out: Path, force: bool = False) -> bool:
    """Baixa ``name + ext`` (tenta .gz e depois sem compressão)."""
    dest = out / f"{name}{ext}"
    if dest.exists() and not force:
        return True
    for suffix in (".gz", ""):
        data = _fetch(f"{BASE_URL}{name}{ext}{suffix}")
        if data is None:
            continue
        if suffix == ".gz" or data[:2] == b"\x1f\x8b":
            try:
                data = gzip.decompress(data)
            except OSError:
                continue
        if b"<html" in data[:512].lower():  # página de erro servida com 200
            continue
        dest.write_bytes(data)
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="baixa de novo mesmo se já existir")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    missing_tsp = []
    for name in INSTANCES:
        ok_tsp = download(name, ".tsp", args.out, args.force)
        ok_tour = download(name, ".opt.tour", args.out, args.force)
        print(f"{name:10s} tsp={'ok' if ok_tsp else 'FALHOU'}  opt.tour={'ok' if ok_tour else '-'}")
        if not ok_tsp:
            missing_tsp.append(name)

    if missing_tsp:
        print(f"AVISO: não foi possível baixar: {', '.join(missing_tsp)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
