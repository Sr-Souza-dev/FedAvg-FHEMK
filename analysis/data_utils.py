from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import math
import shutil

from utils.files import current_output_root, current_plots_root

METRIC_EXTENSION = ".dat"
def output_dir() -> Path:
    return current_output_root()


def plots_dir() -> Path:
    return current_plots_root()


def list_experiments() -> list[str]:
    base = output_dir()
    if not base.exists():
        return []
    experiments = [
        path.name
        for path in base.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    return sorted(experiments)


def run_directories(experiment: str) -> list[Path]:
    base_dir = output_dir() / experiment
    if not base_dir.exists():
        return []
    runs = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    return sorted(runs, key=lambda p: int(p.name))


def average_directory(experiment: str) -> Path:
    return output_dir() / experiment / "average"


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


@dataclass(frozen=True)
class ExperimentStyle:
    display_name: str
    color: str


DEFAULT_METRICS: dict[str, MetricPlotConfig] = {
    "loss": MetricPlotConfig(name="loss", title="Loss de Treino", ylabel="Loss"),
    "accuracy": MetricPlotConfig(name="accuracy", title="Acuracia", ylabel="Acuracia"),
    "client_execution_time": MetricPlotConfig(name="client_execution_time", title="Tempo Total de Execucao (Cliente)", ylabel="Tempo (s)"),
    "client_train_time": MetricPlotConfig(name="client_train_time", title="Tempo de Treinamento (Cliente)", ylabel="Tempo (s)"),
    "client_encrypt_time": MetricPlotConfig(name="client_encrypt_time", title="Tempo de Cifragem (Cliente)", ylabel="Tempo (s)"),
    "client_decrypt_time": MetricPlotConfig(name="client_decrypt_time", title="Tempo de Decifragem (Cliente)", ylabel="Tempo (s)"),
    "server_execution_time": MetricPlotConfig(name="server_execution_time", title="Tempo Total de Execucao (Servidor)", ylabel="Tempo (s)"),
    "server_aggregation_time": MetricPlotConfig(name="server_aggregation_time", title="Tempo de Agregacao (Servidor)", ylabel="Tempo (s)"),
    "server_decrypt_time": MetricPlotConfig(name="server_decrypt_time", title="Tempo de Decifragem (Servidor)", ylabel="Tempo (s)"),
    "client_size": MetricPlotConfig(name="client_size", title="Tamanho de Payload (Cliente)", ylabel="Tamanho (bytes)"),
    "setup_time": MetricPlotConfig(name="setup_time", title="Tempo de Setup (Fase 1)", ylabel="Tempo (s)"),
}


EXPERIMENT_STYLES: dict[str, ExperimentStyle] = {
    "baseline-fl": ExperimentStyle("Baseline", "#1f77b4"),
    "new_ckks-fl": ExperimentStyle("FedAvg-CCHMC", "#ff7f0e"),
    "full_ckks-fl": ExperimentStyle("CCH", "#2ca02c"),
    "selective_ckks-fl-10": ExperimentStyle("M-CCH 10%", "#d62728"),
    "selective_ckks-fl-20": ExperimentStyle("M-CCH 20%", "#9467bd"),
    "selective_ckks-fl-40": ExperimentStyle("M-CCH 40%", "#8c564b"),
    "selective_ckks-fl-80": ExperimentStyle("M-CCH 80%", "#e377c2"),
    # Escalabilidade
    "baseline-fl-p5": ExperimentStyle("Baseline P=5", "#aec7e8"),
    "baseline-fl-p20": ExperimentStyle("Baseline P=20", "#1f77b4"),
    "new_ckks-fl-p5": ExperimentStyle("NEWCKKS P=5", "#ffbb78"),
    "new_ckks-fl-p10": ExperimentStyle("NEWCKKS P=10", "#ff7f0e"),
    "new_ckks-fl-p20": ExperimentStyle("NEWCKKS P=20", "#d62728"),
}

DEFAULT_EXPERIMENT_STYLE = ExperimentStyle("Experimento", "#7f7f7f")


def get_experiment_style(experiment: str) -> ExperimentStyle:
    return EXPERIMENT_STYLES.get(experiment, DEFAULT_EXPERIMENT_STYLE)


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
