"""Utilitários de leitura de resultados e configuração de figuras (Etapa 2 – Dev B)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from src.tsp.common import read_jsonl


def load_table(path: str | Path) -> pd.DataFrame:
    """Lê uma tabela de resultados em ``.jsonl`` ou ``.csv`` como DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = read_jsonl(path)
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def ensure_output_dirs(out_dir: str | Path) -> tuple[Path, Path]:
    """Garante ``<out_dir>/figures`` e ``<out_dir>/tables`` e devolve ambos."""
    base = Path(out_dir)
    fig_dir = base / "figures"
    tab_dir = base / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, tab_dir
