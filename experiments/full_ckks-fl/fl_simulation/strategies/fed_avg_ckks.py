from __future__ import annotations

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
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

from fl_simulation.crypto.ckks_context import build_shared_context
from utils.weights import clone_template, unflatten_weights

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = (
    "Setting `min_available_clients` lower than `min_fit_clients` or `min_evaluate_clients` "
    "can cause the server to fail when there are too few clients connected to the server."
)


class HomomorphicFedAvg(Strategy):
    def __init__(
        self,
        *,
        encrypted: bool,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()
        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.encrypted = encrypted
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
        self.weight_template: Optional[List[np.ndarray]] = None
        self.ckks_context = build_shared_context() if encrypted else None
        self.he = self.ckks_context.build_he(with_secret=True) if encrypted else None

    def __repr__(self) -> str:
        return "HomomorphicFedAvg()"

    # ------------------------------------------------------------------
    # Strategy API
    # ------------------------------------------------------------------
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        params = self.initial_parameters
        if params is not None:
            self.weight_template = clone_template(parameters_to_ndarrays(params))
        self.initial_parameters = None
        return params

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None
        ndarrays = parameters_to_ndarrays(parameters)
        result = self.evaluate_fn(server_round, ndarrays, {})
        if result is None:
            return None
        loss, metrics = result
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        if self.encrypted and self.weight_template is None:
            self.weight_template = clone_template(parameters_to_ndarrays(parameters))
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(sample_size, min_num_clients)
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "is_encrypted": int(self.encrypted),
        }
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0:
            return []
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(sample_size, min_num_clients)
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "is_encrypted": 0,
        }
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results or (not self.accept_failures and failures):
            return None, {}

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.encrypted:
            aggregated_params = self._aggregate_encrypted(results)
        else:
            weights_results = [
                (parameters_to_ndarrays(res.parameters), res.num_examples)
                for _, res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)
            aggregated_params = ndarrays_to_parameters(aggregated_ndarrays)

        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return aggregated_params, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results or (not self.accept_failures and failures):
            return None, {}
        loss = weighted_loss_avg(
            [(res.num_examples, res.loss) for _, res in results]
        )
        metrics: Dict[str, Scalar] = {}
        if self.evaluate_metrics_aggregation_fn is not None:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss, metrics

    # ------------------------------------------------------------------
    # Internal aggregation helpers
    # ------------------------------------------------------------------
    def _aggregate_encrypted(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> Parameters:
        if not self.ckks_context or not self.he:
            raise RuntimeError("CKKS context not initialised")
        total_examples = 0
        aggregated_ciphertexts: List = []
        vector_len: Optional[int] = None
        for _, res in results:
            payload = parameters_to_ndarrays(res.parameters)
            ciphertexts, current_len = self.ckks_context.deserialize_ciphertexts(
                payload, self.he
            )
            if vector_len is None:
                vector_len = current_len
            scaled = self.ckks_context.scale_ciphertexts(
                ciphertexts, res.num_examples
            )
            aggregated_ciphertexts = (
                scaled
                if not aggregated_ciphertexts
                else self.ckks_context.add_ciphertext_lists(
                    aggregated_ciphertexts, scaled
                )
            )
            total_examples += res.num_examples

        scaling = 1.0 / max(total_examples, 1)
        averaged_ciphertexts = self.ckks_context.scale_ciphertexts(
            aggregated_ciphertexts, scaling
        )
        decrypted = self.ckks_context.decrypt_vector(
            self.he, averaged_ciphertexts, vector_len or 0
        )
        if self.weight_template is None:
            raise RuntimeError("Weight template missing for unflattening")
        weights = unflatten_weights(decrypted, self.weight_template)
        return ndarrays_to_parameters(weights)
