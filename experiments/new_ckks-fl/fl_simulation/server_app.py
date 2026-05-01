from __future__ import annotations

"""fl-simulation: A Flower / PyTorch app."""

import time
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from experiment_config import get_experiment_config
from fl_simulation.ckks_instance import ckks
from fl_simulation.model.model import Net, get_weights
from fl_simulation.strategies.fedAvg import FedAvg
from utils.files import (
    current_logs_dir,
    delete_directory_files,
    experiment_output_dir,
    next_run_id,
    write_numbers_to_file,
)
from utils.flatten import flatten

EXPERIMENT_NAME = "new_ckks-fl"
execution_id = ""
is_flattened = True
current_strategy: FedAvg | None = None  # Store strategy reference
EXPERIMENT_CONFIG = get_experiment_config(EXPERIMENT_NAME)


def _metrics_base_path() -> str:
    if not execution_id:
        raise RuntimeError("execution_id not set; ensure server_fn initialized the run.")
    return str(experiment_output_dir(EXPERIMENT_NAME, is_flattened, execution_id))


def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global is_flattened
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}
    base_path = _metrics_base_path()
    weighted = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics)
    accuracy = weighted / total_examples
    per_round = [m.get("accuracy", 0.0) for _, m in metrics] + [accuracy]
    write_numbers_to_file("accuracy", [per_round], base_path=base_path)
    return {"accuracy": accuracy}


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global is_flattened
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"train_loss": 0.0}
    base_path = _metrics_base_path()

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

    return {"train_loss": _aggregate("train_loss")}


def server_fn(context: Context):
    global ckks, is_flattened, execution_id, current_strategy

    num_rounds = EXPERIMENT_CONFIG.num_rounds
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    is_flattened = context.run_config["is-encrypted"] == 1
    execution_id = next_run_id(EXPERIMENT_NAME)

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    array_params = flatten(ndarrays)

    print(f"Parameters_size: {len(array_params)}")
    delete_directory_files("keys/")
    delete_directory_files(current_logs_dir())
    delete_directory_files("public/")

    # Measure Phase 1 (setup) time: key generation for the server aggregated key.
    # The server-side setup consists of generating the initial CRS and the fixed 'a' polynomial.
    setup_start = time.time()
    ckks.gen_new_fixed_a()
    setup_time = time.time() - setup_start
    base_path = str(experiment_output_dir(EXPERIMENT_NAME, is_flattened, execution_id))
    write_numbers_to_file("setup_time", [[setup_time]], base_path=base_path)
    print(f"Setup (Phase 1) time: {setup_time:.4f}s")

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        is_flattened=is_flattened,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
    )
    current_strategy = strategy  # Store reference for timing access
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
