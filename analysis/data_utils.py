from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import math
import shutil

from utils.files import OUTPUT_DIR, ROOT_DIR

METRIC_EXTENSION = ".dat"
PLOTS_DIR = ROOT_DIR / "plots"


def list_experiments() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    experiments = [
        path.name
        for path in OUTPUT_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    return sorted(experiments)


def run_directories(experiment: str) -> list[Path]:
    base_dir = OUTPUT_DIR / experiment
    if not base_dir.exists():
        return []
    runs = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    return sorted(runs, key=lambda p: int(p.name))


def average_directory(experiment: str) -> Path:
    return OUTPUT_DIR / experiment / "average"


def ensure_clean_directory(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_dat_file(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            values: list[float] = []
            for token in tokens:
                try:
                    values.append(float(token))
                except ValueError:
                    # ignore invalid tokens but keep alignment with NaN placeholder
                    values.append(math.nan)
            rows.append(values)
    return rows


def write_dat_file(path: Path, matrix: Sequence[Sequence[float]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as stream:
        for row in matrix:
            stream.write("\t".join(f"{value:.8f}" for value in row) + "\n")


def matrix_dimensions(matrix: Sequence[Sequence[float]]) -> tuple[int, int]:
    rows = len(matrix)
    cols = max((len(row) for row in matrix), default=0)
    return rows, cols


def split_columns(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    row_count, col_count = matrix_dimensions(matrix)
    if row_count == 0 or col_count == 0:
        return []
    columns = [[math.nan] * row_count for _ in range(col_count)]
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            columns[c][r] = value
    return columns


@dataclass(frozen=True)
class MetricPlotConfig:
    name: str
    title: str
    ylabel: str


DEFAULT_METRICS: dict[str, MetricPlotConfig] = {
    "loss": MetricPlotConfig(name="loss", title="Loss de Treino", ylabel="Loss"),
    "accuracy": MetricPlotConfig(name="accuracy", title="Acuracia", ylabel="Acuracia"),
    "time": MetricPlotConfig(name="time", title="Tempo de Execucao", ylabel="Tempo (s)"),
    "size": MetricPlotConfig(name="size", title="Tamanho de Payload", ylabel="Tamanho (bytes)"),
}


def load_metric_series(experiment: str, metric_name: str) -> list[float]:
    """Return the aggregated series for a metric stored in the average folder."""
    data_file = average_directory(experiment) / f"{metric_name}{METRIC_EXTENSION}"
    matrix = read_dat_file(data_file)
    if not matrix:
        return []
    columns = split_columns(matrix)
    if not columns:
        return []
    return columns[-1]
