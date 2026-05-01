from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict

import numpy as np

from .data_utils import DEFAULT_METRICS, ensure_directory, load_metric_series, plots_dir


def analysis_dir() -> Path:
    return plots_dir() / "analysis"


def summary_csv_path() -> Path:
    return analysis_dir() / "metrics_summary.csv"


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
    try:
        auc_value = float(np.trapz(arr, dx=1.0))
    except AttributeError:
        auc_value = float(np.trapezoid(arr, dx=1.0))
    stats = {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "final": float(arr[-1]),
        "delta": float(arr[-1] - arr[0]),
        "auc": auc_value,
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
    output_csv: Path | None = None,
) -> Path:
    target = output_csv or summary_csv_path()
    ensure_directory(target.parent)
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
    with target.open("w", newline="", encoding="utf-8") as stream:
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
    return target


def comparative_table_path() -> Path:
    return analysis_dir() / "comparative_table.csv"


def comparative_table_latex_path() -> Path:
    return analysis_dir() / "comparative_table.tex"


def save_comparative_table(experiments: list[str]) -> tuple[Path | None, Path | None]:
    """
    Generate a comparative table with all experiments showing final metrics.
    Returns both CSV and LaTeX table paths.
    """
    from .data_utils import average_directory, read_dat_file, METRIC_EXTENSION, split_columns, get_experiment_style
    
    # Collect final metrics for each experiment
    experiment_data = []
    
    for experiment in experiments:
        avg_dir = average_directory(experiment)
        if not avg_dir.exists():
            continue
        
        # Load metrics with new prefixed names
        metrics = {}
        metric_names = [
            "accuracy", "client_train_time", "client_encrypt_time", "client_decrypt_time",
            "server_aggregation_time", "server_decrypt_time", "server_execution_time", "client_size"
        ]
        
        for metric in metric_names:
            data_file = avg_dir / f"{metric}{METRIC_EXTENSION}"
            if not data_file.exists():
                continue
            
            matrix = read_dat_file(data_file)
            if not matrix:
                continue
            
            columns = split_columns(matrix)
            if columns and columns[-1]:
                # Get final value (last round) or average for time metrics
                values = columns[-1]
                valid_values = [v for v in values if not (math.isnan(v) or math.isinf(v))]
                
                if not valid_values:
                    continue
                
                if metric == "accuracy":
                    metrics[metric] = valid_values[-1]  # Final accuracy
                elif metric == "client_size":
                    # Convert to MB and get average
                    avg_bytes = sum(valid_values) / len(valid_values)
                    metrics[metric] = avg_bytes / (1024 * 1024)  # Convert to MB
                elif metric in ["server_aggregation_time", "server_decrypt_time", "server_execution_time"]:
                    # Server-side times - single value per round
                    metrics[metric] = valid_values[0] if valid_values else 0.0
                else:
                    # Client-side times - average across clients
                    metrics[metric] = sum(valid_values) / len(valid_values)
        
        if metrics:
            style = get_experiment_style(experiment)
            experiment_data.append({
                "experiment": experiment,
                "display_name": style.display_name,
                "final_accuracy": metrics.get("accuracy", 0.0),
                "avg_train_time": metrics.get("client_train_time", 0.0),
                "avg_encrypt_time": metrics.get("client_encrypt_time", 0.0),
                "avg_decrypt_time": metrics.get("client_decrypt_time", 0.0),
                "avg_aggregation_time": metrics.get("server_aggregation_time", 0.0),
                "avg_server_decrypt_time": metrics.get("server_decrypt_time", 0.0),
                "avg_server_execution_time": metrics.get("server_execution_time", 0.0),
                "avg_size_mb": metrics.get("client_size", 0.0),
            })
    
    if not experiment_data:
        return None, None
    
    ensure_directory(analysis_dir())
    
    # Save CSV
    csv_path = comparative_table_path()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Baseline", "Final Accuracy", 
            "Avg Train Time (s)", "Avg Encrypt Time (s)", "Avg Decrypt Time (s)",
            "Avg Aggregation Time (s)", "Avg Server Decrypt Time (s)", "Avg Server Total Time (s)",
            "Avg Communication per Round (MB)"
        ])
        writer.writeheader()
        for data in experiment_data:
            writer.writerow({
                "Baseline": data["display_name"],
                "Final Accuracy": f"{data['final_accuracy']:.4f}",
                "Avg Train Time (s)": f"{data['avg_train_time']:.4f}",
                "Avg Encrypt Time (s)": f"{data['avg_encrypt_time']:.4f}",
                "Avg Decrypt Time (s)": f"{data['avg_decrypt_time']:.4f}",
                "Avg Aggregation Time (s)": f"{data['avg_aggregation_time']:.4f}",
                "Avg Server Decrypt Time (s)": f"{data['avg_server_decrypt_time']:.4f}",
                "Avg Server Total Time (s)": f"{data['avg_server_execution_time']:.4f}",
                "Avg Communication per Round (MB)": f"{data['avg_size_mb']:.2f}",
            })
    
    # Save LaTeX
    latex_path = comparative_table_latex_path()
    with latex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\t\\centering\n")
        f.write("\t\\caption{Resultados experimentais}\n")
        f.write("\t\\label{tab:results_comparison}\n")
        f.write("\t\\begin{tabular}{|\n")
        f.write("\t\t\tp{2.3cm}|\n")  # Baseline
        f.write("\t\t\t>{\\centering\\arraybackslash}p{1.6cm}|\n")  # Accuracy
        f.write("\t\t\t>{\\centering\\arraybackslash}p{1.4cm}|\n")  # Train Time
        f.write("\t\t\t>{\\centering\\arraybackslash}p{1.4cm}|\n")  # Encrypt Time
        f.write("\t\t\t>{\\centering\\arraybackslash}p{1.4cm}|\n")  # Decrypt Time
        f.write("\t\t\t>{\\centering\\arraybackslash}p{1.5cm}|\n")  # Aggregation Time
        f.write("\t\t\t>{\\centering\\arraybackslash}p{1.7cm}|\n")  # Server Decrypt Time
        f.write("\t\t\t>{\\centering\\arraybackslash}p{1.5cm}|\n")  # Server Total Time
        f.write("\t\t\t>{\\centering\\arraybackslash}p{2.0cm}|}\n")  # Communication
        f.write("\t\t\\hline\n")
        f.write("\t\tBaseline & Acurácia & Treino (s) & Cifrar (s) & Decifrar (s) & Agregar (s) & Decif. Serv. (s) & Total Serv. (s) & Comunic. (MB) \\\\ \\hline\n")
        
        for data in experiment_data:
            f.write(f"\t\t{data['display_name']} & ")
            f.write(f"{data['final_accuracy']:.4f} & ")
            f.write(f"{data['avg_train_time']:.4f} & ")
            f.write(f"{data['avg_encrypt_time']:.4f} & ")
            f.write(f"{data['avg_decrypt_time']:.4f} & ")
            f.write(f"{data['avg_aggregation_time']:.4f} & ")
            f.write(f"{data['avg_server_decrypt_time']:.4f} & ")
            f.write(f"{data['avg_server_execution_time']:.4f} & ")
            f.write(f"{data['avg_size_mb']:.2f} \\\\ \\hline\n")
        
        f.write("\t\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    return csv_path, latex_path
