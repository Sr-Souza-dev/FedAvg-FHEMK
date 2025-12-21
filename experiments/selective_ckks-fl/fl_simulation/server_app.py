from __future__ import annotations

from typing import Dict, List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from experiment_config import get_experiment_config
from model.model import Net, get_weights
from fl_simulation.strategies.selective_fed_avg import SelectiveHomomorphicFedAvg
from utils.files import experiment_output_dir, next_run_id, write_numbers_to_file

EXPERIMENT_NAME = "selective_ckks-fl"
execution_id = ""
current_encrypted = True
EXPERIMENT_CONFIG = get_experiment_config(EXPERIMENT_NAME)


def _base_output_path() -> str:
    if not execution_id:
        raise RuntimeError("execution_id not initialised")
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
    base_path = _base_output_path()
    weighted = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics)
    accuracy = weighted / total_examples
    per_round = [m.get("accuracy", 0.0) for _, m in metrics] + [accuracy]
    write_numbers_to_file("accuracy", [per_round], base_path=base_path)
    return {"accuracy": accuracy}


def server_fn(context: Context) -> ServerAppComponents:
    global execution_id, current_encrypted
    run_cfg = context.run_config
    current_encrypted = run_cfg["is-encrypted"] == 1
    execution_id = next_run_id(EXPERIMENT_NAME)

    mask_ratio = float(run_cfg.get("mask-ratio", 0.15))
    proposal_multiplier = float(run_cfg.get("mask-proposal-multiplier", 3.0))

    init_weights = get_weights(Net())
    initial_parameters = ndarrays_to_parameters(init_weights)

    strategy = SelectiveHomomorphicFedAvg(
        encrypted=current_encrypted,
        mask_ratio=mask_ratio,
        proposal_multiplier=proposal_multiplier,
        fraction_fit=run_cfg["fraction-fit"],
        fraction_evaluate=run_cfg["fraction-evaluate"],
        min_available_clients=2,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )
    server_config = ServerConfig(num_rounds=EXPERIMENT_CONFIG.num_rounds)
    return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)
