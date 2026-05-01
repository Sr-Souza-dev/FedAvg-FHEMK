from __future__ import annotations

"""Baseline Flower server app without encryption."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from experiment_config import get_experiment_config
from fl_simulation.model import Net, get_weights
from fl_simulation.strategies.timed_fed_avg import TimedFedAvg
from utils.files import (
    current_logs_dir,
    delete_directory_files,
    experiment_output_dir,
    next_run_id,
    write_numbers_to_file,
)

EXPERIMENT_NAME = "baseline-fl"
execution_id = ""
current_encrypted = False
current_strategy = None  # Reference to strategy for accessing timing metrics
EXPERIMENT_CONFIG = get_experiment_config(EXPERIMENT_NAME)


def _output_path() -> str:
    if not execution_id:
        raise RuntimeError("execution_id should be initialised before aggregating metrics")
    return str(experiment_output_dir(EXPERIMENT_NAME, current_encrypted, execution_id))


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"train_loss": 0.0}
    base_path = _output_path()

    def _aggregate(key: str) -> float:
        return sum(num_examples * m.get(key, 0.0) for num_examples, m in metrics) / total_examples

    def _series(key: str) -> list[float]:
        return [m.get(key, 0.0) for _, m in metrics] + [_aggregate(key)]

    # Server-side metrics
    server_aggregation_time = current_strategy.last_aggregation_time if current_strategy else 0.0
    server_decrypt_time = 0.0  # No decryption in baseline
    server_execution_time = server_aggregation_time + server_decrypt_time

    # Write all metrics with proper prefixes
    write_numbers_to_file("loss", [_series("train_loss")], base_path=base_path)
    write_numbers_to_file("client_train_time", [_series("train_time")], base_path=base_path)
    write_numbers_to_file("client_encrypt_time", [_series("encrypt_time")], base_path=base_path)
    write_numbers_to_file("client_decrypt_time", [_series("decrypt_time")], base_path=base_path)
    write_numbers_to_file("client_execution_time", [_series("execution_time")], base_path=base_path)
    write_numbers_to_file("client_size", [_series("size")], base_path=base_path)
    write_numbers_to_file("server_aggregation_time", [[server_aggregation_time]], base_path=base_path)
    write_numbers_to_file("server_decrypt_time", [[server_decrypt_time]], base_path=base_path)
    write_numbers_to_file("server_execution_time", [[server_execution_time]], base_path=base_path)
    return {"train_loss": _aggregate("train_loss")}


def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}
    base_path = _output_path()
    weighted = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics)
    accuracy = weighted / total_examples
    per_round = [m.get("accuracy", 0.0) for _, m in metrics] + [accuracy]
    write_numbers_to_file("accuracy", [per_round], base_path=base_path)
    return {"accuracy": accuracy}


def server_fn(context: Context):
    global execution_id, current_encrypted, current_strategy

    run_cfg = context.run_config
    current_encrypted = run_cfg.get("is-encrypted", 0) == 1
    execution_id = next_run_id(EXPERIMENT_NAME)

    initial_parameters = ndarrays_to_parameters(get_weights(Net()))

    logs_dir = current_logs_dir()
    if logs_dir.exists():
        delete_directory_files(logs_dir)

    def _fit_config(server_round: int) -> dict[str, float | int]:
        return {"server_round": server_round}

    def _eval_config(server_round: int) -> dict[str, float | int]:
        return {"server_round": server_round}

    strategy = TimedFedAvg(
        fraction_fit=run_cfg["fraction-fit"],
        fraction_evaluate=run_cfg["fraction-evaluate"],
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        on_fit_config_fn=_fit_config,
        on_evaluate_config_fn=_eval_config,
    )
    current_strategy = strategy  # Store reference for metrics access
    server_config = ServerConfig(num_rounds=EXPERIMENT_CONFIG.num_rounds)
    return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)
