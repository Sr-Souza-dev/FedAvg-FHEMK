from __future__ import annotations

from logging import WARNING
from typing import Callable, Optional, Union

from utils.files import logging_enabled, register_logs

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from ckks.cryptogram.main import Cryptogram
from fl_simulation.ckks_instance import MODEL_STRUCTURE, ckks
from utils.flatten import unflatten

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


def aggregate_ndarrays(array: list[list[Cryptogram]]) -> list[Cryptogram]:
    """Aggregate a list of ciphertext arrays using weighted addition."""
    aggregated_weights: list[Cryptogram] = []
    for columns in zip(*array):
        acc = Cryptogram(
            c0=ckks.zero_polynomial(),
            c1=ckks.zero_polynomial(),
            q=ckks.params.qs,
        )
        for ciphertext in columns:
            acc = acc + ciphertext
        aggregated_weights.append(acc)
    return aggregated_weights


class FedAvg(Strategy):
    def __init__(
        self,
        *,
        is_flattened: bool = False,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.is_flattened = is_flattened

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        ckks.gen_new_fixed_a()

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        config = {
            "server_round": server_round,
            "is_flattened": self.is_flattened,
            "clients_qtd": len(clients),
        }

        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        config = {
            "server_round": server_round,
            "is_flattened": self.is_flattened,
            "clients_qtd": len(clients),
        }

        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results or (not self.accept_failures and failures):
            return None, {}

        log_enabled = logging_enabled()
        if log_enabled:
            register_logs(
                file_name="server",
                title=f"\n ------------- Round {server_round} -------------- \n",
                value=f"clients={len(results)}",
            )

        if self.is_flattened:
            sk = ckks.zero_polynomial()
            weights_cypher = []
            total_examples = 0
            for client, res in results:
                model_enc = ckks.construct_cryptograms(parameters_to_ndarrays(res.parameters))

                num_examples = res.num_examples
                total_examples += num_examples
                model_enc = [c * num_examples for c in model_enc]
                weights_cypher.append(model_enc)

                sk_cl = ckks.load_key(prefix=str(client.node_id))
                sk += sk_cl
                if log_enabled:
                    register_logs(
                        file_name="server",
                        title=f"\n Client Model ({client.node_id}):",
                        value=f"ciphertexts={len(model_enc)} examples={num_examples}",
                    )

            aggregated_ndarrays = aggregate_ndarrays(weights_cypher)
            aggregated_vector = ckks.decrypt_batch(sk=sk, ciphertexts=aggregated_ndarrays)

            if total_examples > 0:
                aggregated_vector = aggregated_vector / total_examples

            if log_enabled:
                register_logs(
                    file_name="server",
                    title="\n Model decrypted vector:",
                    value=f"vector_len={aggregated_vector.size}",
                )

            structure = ckks.model_structure or MODEL_STRUCTURE
            aggregated_weights = unflatten(aggregated_vector, structure)
            parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        else:
            weights_results = [
                (parameters_to_ndarrays(res.parameters), res.num_examples)
                for _, res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)
            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        if log_enabled:
            register_logs(
                file_name="server",
                title="\n Model parameters broadcast:",
                value=f"tensor_count={len(parameters_to_ndarrays(parameters_aggregated))}",
            )

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
