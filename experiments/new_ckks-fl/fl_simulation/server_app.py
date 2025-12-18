from __future__ import annotations

"""fl-simulation: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

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

    write_numbers_to_file("loss", [_series("train_loss")], base_path=base_path)
    write_numbers_to_file("time", [_series("execution_time")], base_path=base_path)
    write_numbers_to_file("size", [_series("size")], base_path=base_path)

    return {"train_loss": _aggregate("train_loss")}


def server_fn(context: Context):
    global ckks, is_flattened, execution_id

    num_rounds = context.run_config["num-server-rounds"]
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

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        is_flattened=is_flattened,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
