from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence

from .data_utils import (
    METRIC_EXTENSION,
    average_directory,
    ensure_clean_directory,
    list_experiments,
    read_dat_file,
    run_directories,
    write_dat_file,
)


def _collect_metric_files(run_dir: Path) -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    for path in run_dir.rglob(f"*{METRIC_EXTENSION}"):
        if path.is_file():
            relative = path.relative_to(run_dir)
            files[relative] = path
    return files


def _average_matrices(matrices: Sequence[Sequence[Sequence[float]]]) -> list[list[float]]:
    sums: Dict[tuple[int, int], float] = defaultdict(float)
    counts: Dict[tuple[int, int], int] = defaultdict(int)
    max_row = -1
    for matrix in matrices:
        for r, row in enumerate(matrix):
            max_row = max(max_row, r)
            for c, value in enumerate(row):
                key = (r, c)
                sums[key] += value
                counts[key] += 1

    averaged: list[list[float]] = []
    for r in range(max_row + 1):
        cols = [c for (row_idx, c) in sums.keys() if row_idx == r]
        if not cols:
            continue
        row_values = []
        for c in sorted(cols):
            key = (r, c)
            row_values.append(sums[key] / counts[key])
        averaged.append(row_values)
    return averaged


def compute_average_for_experiment(experiment: str) -> Path | None:
    runs = run_directories(experiment)
    if len(runs) < 1:
        return None

    file_map: Dict[Path, list[Path]] = defaultdict(list)
    for run in runs:
        for relative, absolute in _collect_metric_files(run).items():
            file_map[relative].append(absolute)

    if not file_map:
        return None

    avg_dir = ensure_clean_directory(average_directory(experiment))
    for relative, sources in file_map.items():
        matrices = [read_dat_file(path) for path in sources]
        matrices = [matrix for matrix in matrices if matrix]
        if not matrices:
            continue
        averaged = _average_matrices(matrices)
        target_path = avg_dir / relative
        write_dat_file(target_path, averaged)
    return avg_dir


def compute_all_averages(experiments: Iterable[str] | None = None) -> dict[str, bool]:
    experiment_names = list(experiments) if experiments is not None else list_experiments()
    results: dict[str, bool] = {}
    for experiment in experiment_names:
        results[experiment] = compute_average_for_experiment(experiment) is not None
    return results
