from __future__ import annotations

from typing import Dict, List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from experiment_config import get_experiment_config
from fl_simulation.crypto.ckks_context import build_shared_context
from fl_simulation.model.model import Net, get_weights
from fl_simulation.strategies.fed_avg_ckks import HomomorphicFedAvg
from utils.files import experiment_output_dir, next_run_id, write_numbers_to_file


EXPERIMENT_NAME = "full_ckks-fl"
execution_id = ""
current_encrypted = True
current_strategy: HomomorphicFedAvg | None = None  # Store strategy reference
EXPERIMENT_CONFIG = get_experiment_config(EXPERIMENT_NAME)


def _base_output_path() -> str:
    if not execution_id:
        raise RuntimeError("execution_id not set; ensure server_fn initialized the run.")
    return str(experiment_output_dir(EXPERIMENT_NAME, current_encrypted, execution_id))


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"train_loss": 0.0, "execution_time": 0.0}
    base_path = _base_output_path()

    def _aggregate(key: str) -> float:
        return sum(num_examples * m.get(key, 0.0) for num_examples, m in metrics) / total_examples

    def _series(key: str) -> list[float]:
        return [m.get(key, 0.0) for _, m in metrics] + [_aggregate(key)]

    # Get server-side timing from strategy
    server_aggregation_time = current_strategy.last_aggregation_time if current_strategy else 0.0
    server_decrypt_time = current_strategy.last_server_decrypt_time if current_strategy else 0.0
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

    return {
        "train_loss": _aggregate("train_loss"),
        "execution_time": _aggregate("execution_time"),
    }


def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}
    base_path = _base_output_path()
    weighted = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics)
    accuracy = weighted / total_examples
    per_round = [m.get("accuracy", 0.0) for _, m in metrics] + [accuracy]
    write_numbers_to_file("accuracy", [per_round], base_path=base_path)
    return {"accuracy": accuracy}


def server_fn(context: Context) -> ServerAppComponents:
    global execution_id, current_encrypted, current_strategy
    run_cfg = context.run_config
    encrypted = run_cfg["is-encrypted"] == 1
    current_encrypted = encrypted
    execution_id = next_run_id(EXPERIMENT_NAME)

    ckks_context = build_shared_context()
    if encrypted:
        ckks_context.ensure_keys()

    init_weights = get_weights(Net())
    initial_parameters = ndarrays_to_parameters(init_weights)

    strategy = HomomorphicFedAvg(
        encrypted=encrypted,
        fraction_fit=run_cfg["fraction-fit"],
        fraction_evaluate=run_cfg["fraction-evaluate"],
        min_available_clients=2,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )
    current_strategy = strategy  # Store reference for timing access

    server_config = ServerConfig(num_rounds=EXPERIMENT_CONFIG.num_rounds)
    return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)
