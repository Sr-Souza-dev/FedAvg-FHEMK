from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_utils import (
    DEFAULT_METRICS,
    METRIC_EXTENSION,
    average_directory,
    ensure_directory,
    get_experiment_style,
    list_experiments,
    load_metric_series,
    plots_dir,
    read_dat_file,
    split_columns,
)


def _metric_config(metric_name: str):
    if metric_name in DEFAULT_METRICS:
        return DEFAULT_METRICS[metric_name]
    return DEFAULT_METRICS.get("loss", DEFAULT_METRICS[next(iter(DEFAULT_METRICS))])


def _round_axis(length: int) -> list[int]:
    return list(range(1, length + 1))


def _plot_metric_lines(
    experiment: str,
    metric_name: str,
    data: Sequence[Sequence[float]],
    output_path: Path,
) -> None:
    if not data:
        return
    config = _metric_config(metric_name)
    style = get_experiment_style(experiment)
    columns = split_columns(data)
    if not columns:
        return
    aggregated_series = columns[-1]
    per_client_columns = columns[:-1]
    x_axis = _round_axis(len(data))

    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_client_label = False
    for series in per_client_columns:
        if all(math.isnan(value) for value in series):
            continue
        ax.plot(
            x_axis,
            series,
            linestyle="--",
            linewidth=1,
            alpha=0.35,
            color=style.color,
            label="Clientes" if not plotted_client_label else None,
        )
        plotted_client_label = True

    ax.plot(
        x_axis,
        aggregated_series,
        label=style.display_name,
        linewidth=2.4,
        color=style.color,
    )
    ax.set_title(f"{config.title} – {style.display_name}")
    ax.set_xlabel("Rodada")
    ax.set_ylabel(config.ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    if plotted_client_label or aggregated_series:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_experiment_plots(experiment: str) -> list[Path]:
    avg_dir = average_directory(experiment)
    if not avg_dir.exists():
        return []
    output_dir = ensure_directory(plots_dir() / experiment)
    created_files: list[Path] = []
    for data_file in avg_dir.rglob(f"*{METRIC_EXTENSION}"):
        if not data_file.is_file():
            continue
        matrix = read_dat_file(data_file)
        if not matrix:
            continue
        metric_name = data_file.stem
        target = output_dir / f"{metric_name}.png"
        _plot_metric_lines(experiment, metric_name, matrix, target)
        created_files.append(target)
    return created_files


def generate_all_experiment_plots(experiments: Iterable[str] | None = None) -> dict[str, int]:
    experiment_names = list(experiments) if experiments is not None else list_experiments()
    generated: dict[str, int] = {}
    for experiment in experiment_names:
        files = generate_experiment_plots(experiment)
        generated[experiment] = len(files)
    return generated


def generate_comparison_plots(experiments: Iterable[str] | None = None) -> dict[str, Path | None]:
    experiment_names = list(experiments) if experiments is not None else list_experiments()
    ensure_directory(plots_dir() / "analysis")
    outputs: dict[str, Path | None] = {}
    for metric_name, config in DEFAULT_METRICS.items():
        series_map = {
            experiment: load_metric_series(experiment, metric_name)
            for experiment in experiment_names
        }
        series_map = {k: v for k, v in series_map.items() if v}
        if len(series_map) < 2:
            outputs[metric_name] = None
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for experiment, series in series_map.items():
            rounds = _round_axis(len(series))
            style = get_experiment_style(experiment)
            ax.plot(
                rounds,
                series,
                label=style.display_name,
                linewidth=2,
                color=style.color,
            )
        ax.set_title(f"Comparacao de {config.title}")
        ax.set_xlabel("Rodada")
        ax.set_ylabel(config.ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        target = plots_dir() / "analysis" / f"{metric_name}_comparison.png"
        fig.savefig(target, dpi=200)
        plt.close(fig)
        outputs[metric_name] = target
    return outputs


def generate_metric_boxplots(experiments: Iterable[str]) -> dict[str, Path | None]:
    ensure_directory(plots_dir() / "analysis")
    outputs: dict[str, Path | None] = {}
    experiment_list = list(experiments)
    for metric_name, config in DEFAULT_METRICS.items():
        data = {
            experiment: load_metric_series(experiment, metric_name)
            for experiment in experiment_list
        }
        cleaned = {exp: series for exp, series in data.items() if len(series) >= 2}
        if len(cleaned) < 2:
            outputs[metric_name] = None
            continue
        labels = [get_experiment_style(exp).display_name for exp in cleaned.keys()]
        colors = [get_experiment_style(exp).color for exp in cleaned.keys()]
        fig, ax = plt.subplots(figsize=(8, 5))
        box = ax.boxplot(cleaned.values(), labels=labels, patch_artist=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(f"Distribuicao por rodada – {config.title}")
        ax.set_ylabel(config.ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        target = plots_dir() / "analysis" / f"{metric_name}_boxplot.png"
        fig.savefig(target, dpi=200)
        plt.close(fig)
        outputs[metric_name] = target
    return outputs


def generate_metric_barplots(
    statistics: dict[str, dict[str, dict[str, float]]],
    value_key: str = "final",
) -> dict[str, Path | None]:
    ensure_directory(plots_dir() / "analysis")
    outputs: dict[str, Path | None] = {}
    experiments = list(statistics.keys())
    if not experiments:
        return outputs

    for metric_name, config in DEFAULT_METRICS.items():
        values = []
        labels = []
        colors = []
        for experiment in experiments:
            metric_stats = statistics.get(experiment, {}).get(metric_name)
            if not metric_stats:
                continue
            value = metric_stats.get(value_key)
            if value is None or math.isnan(value):
                continue
            values.append(value)
            style = get_experiment_style(experiment)
            labels.append(style.display_name)
            colors.append(style.color)
        if not values:
            outputs[metric_name] = None
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, values, color=colors or "#1f77b4")
        ax.set_title(f"Comparacao ({value_key}) – {config.title}")
        ax.set_ylabel(config.ylabel)
        ax.set_xlabel("Experimento")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        target = plots_dir() / "analysis" / f"{metric_name}_bar_{value_key}.png"
        fig.savefig(target, dpi=200)
        plt.close(fig)
        outputs[metric_name] = target
    return outputs


def generate_accuracy_time_tradeoff(
    statistics: dict[str, dict[str, dict[str, float]]],
) -> Path | None:
    ensure_directory(plots_dir() / "analysis")
    points: list[tuple[float, float]] = []
    styles = []
    for experiment, metrics in statistics.items():
        acc = metrics.get("accuracy", {}).get("final")
        time_metric = metrics.get("time", {}).get("mean")
        if acc is None or time_metric is None or math.isnan(acc) or math.isnan(time_metric):
            continue
        points.append((time_metric, acc))
        styles.append(get_experiment_style(experiment))
    if len(points) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for (x, y), style in zip(points, styles):
        ax.scatter([x], [y], s=80, color=style.color)
        ax.annotate(style.display_name, (x, y), textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_xlabel("Tempo medio por rodada (s)")
    ax.set_ylabel("Acuracia final")
    ax.set_title("Trade-off Tempo vs. Acuracia")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    target = plots_dir() / "analysis" / "accuracy_time_tradeoff.png"
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target
