from __future__ import annotations

from typing import Iterable

from .averager import compute_all_averages
from .data_utils import list_experiments
from .plotters import (
    generate_all_experiment_plots,
    generate_accuracy_time_tradeoff,
    generate_comparison_plots,
    generate_metric_barplots,
    generate_metric_boxplots,
)
from .statistics import compute_all_statistics, save_statistics_table


def run_full_analysis(experiments: Iterable[str] | None = None) -> dict[str, object]:
    experiment_names = list(experiments) if experiments is not None else list_experiments()
    averages = compute_all_averages(experiment_names)
    experiment_plots = generate_all_experiment_plots(experiment_names)
    comparisons = generate_comparison_plots(experiment_names)
    statistics = compute_all_statistics(experiment_names)
    stats_table = save_statistics_table(statistics) if statistics else None
    boxplots = generate_metric_boxplots(experiment_names)
    barplots = generate_metric_barplots(statistics)
    tradeoff = generate_accuracy_time_tradeoff(statistics)
    return {
        "experiments": experiment_names,
        "averages": averages,
        "experiment_plots": experiment_plots,
        "comparisons": comparisons,
        "statistics": {
            "data": statistics,
            "table": stats_table,
        },
        "boxplots": boxplots,
        "barplots": barplots,
        "tradeoff": tradeoff,
    }
