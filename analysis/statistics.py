from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict

import numpy as np

from .data_utils import DEFAULT_METRICS, PLOTS_DIR, ensure_directory, load_metric_series

ANALYSIS_DIR = PLOTS_DIR / "analysis"
SUMMARY_CSV = ANALYSIS_DIR / "metrics_summary.csv"


def _safe_array(values: list[float]) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr[np.isfinite(arr)]


def _compute_trend(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    x = np.arange(1, values.size + 1, dtype=float)
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def compute_metric_statistics(series: list[float]) -> dict[str, float]:
    arr = _safe_array(series)
    if arr.size == 0:
        return {}
    stats = {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "final": float(arr[-1]),
        "delta": float(arr[-1] - arr[0]),
        "auc": float(np.trapz(arr, dx=1.0)),
    }
    trend = _compute_trend(arr)
    if trend is not None:
        stats["trend"] = trend
    return stats


def compute_experiment_statistics(experiment: str) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in DEFAULT_METRICS:
        series = load_metric_series(experiment, metric)
        metrics_stats = compute_metric_statistics(series)
        if metrics_stats:
            summary[metric] = metrics_stats
    return summary


def compute_all_statistics(experiments: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    return {experiment: compute_experiment_statistics(experiment) for experiment in experiments}


def save_statistics_table(
    statistics: dict[str, dict[str, dict[str, float]]],
    output_csv: Path = SUMMARY_CSV,
) -> Path:
    ensure_directory(output_csv.parent)
    fieldnames = [
        "experiment",
        "metric",
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "final",
        "delta",
        "trend",
        "auc",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for experiment, metrics in statistics.items():
            for metric_name, stats in metrics.items():
                row = {"experiment": experiment, "metric": metric_name}
                for key in fieldnames[2:]:
                    value = stats.get(key)
                    if value is None or (isinstance(value, float) and math.isnan(value)):
                        continue
                    row[key] = value
                writer.writerow(row)
    return output_csv
