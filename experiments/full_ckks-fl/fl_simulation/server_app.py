from __future__ import annotations

from typing import Dict, List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fl_simulation.crypto.ckks_context import build_shared_context
from fl_simulation.model.model import Net, get_weights
from fl_simulation.strategies.fed_avg_ckks import HomomorphicFedAvg
from utils.files import experiment_output_dir, write_numbers_to_file
from utils.uuid import get_uid_per_minute


EXPERIMENT_NAME = "full_ckks-fl"
execution_id = get_uid_per_minute()
current_encrypted = True


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"train_loss": 0.0, "execution_time": 0.0}
    base_path = str(experiment_output_dir(EXPERIMENT_NAME, current_encrypted, execution_id))

    def _aggregate(key: str) -> float:
        return sum(num_examples * m.get(key, 0.0) for num_examples, m in metrics) / total_examples

    def _series(key: str) -> list[float]:
        return [m.get(key, 0.0) for _, m in metrics] + [_aggregate(key)]

    write_numbers_to_file("loss", [_series("train_loss")], base_path=base_path)
    write_numbers_to_file("time", [_series("execution_time")], base_path=base_path)
    write_numbers_to_file("size", [_series("size")], base_path=base_path)

    return {
        "train_loss": _aggregate("train_loss"),
        "execution_time": _aggregate("execution_time"),
    }


def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}
    base_path = str(experiment_output_dir(EXPERIMENT_NAME, current_encrypted, execution_id))
    weighted = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics)
    accuracy = weighted / total_examples
    per_round = [m.get("accuracy", 0.0) for _, m in metrics] + [accuracy]
    write_numbers_to_file("accuracy", [per_round], base_path=base_path)
    return {"accuracy": accuracy}


def server_fn(context: Context) -> ServerAppComponents:
    global execution_id, current_encrypted
    run_cfg = context.run_config
    encrypted = run_cfg["is-encrypted"] == 1
    current_encrypted = encrypted
    execution_id = get_uid_per_minute()

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

    server_config = ServerConfig(num_rounds=run_cfg["num-server-rounds"])
    return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)
