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
from fl_simulation.masking import (
    boolean_mask,
    decode_mask_scores,
    encode_mask,
    mask_size_from_ratio,
)
from utils.weights import clone_template, flatten_weights, unflatten_weights


class SelectiveHomomorphicFedAvg(Strategy):
    def __init__(
        self,
        *,
        encrypted: bool,
        mask_ratio: float,
        proposal_multiplier: float,
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
            log(
                WARNING,
                "min_available_clients must be >= min_fit_clients and min_evaluate_clients",
            )
        self.encrypted = encrypted
        self.mask_ratio = mask_ratio
        self.proposal_multiplier = proposal_multiplier
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
        self.vector_length: int = 0
        self.current_mask: np.ndarray = np.array([], dtype=np.int64)
        self.mask_version: int = 0

        self.ckks_context = build_shared_context() if encrypted else None
        self.he = self.ckks_context.build_he(with_secret=True) if encrypted else None

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
            self.vector_length = flatten_weights(self.weight_template).size
            self.current_mask = np.array([], dtype=np.int64)
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
        if self.weight_template is None:
            self.weight_template = clone_template(parameters_to_ndarrays(parameters))
            self.vector_length = flatten_weights(self.weight_template).size
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(sample_size, min_num_clients)
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "is_encrypted": int(self.encrypted),
            "mask-ratio": self.mask_ratio,
            "mask-positions": encode_mask(self.current_mask.tolist()),
            "mask-version": self.mask_version,
            "mask-proposal-multiplier": self.proposal_multiplier,
        }
        if self.on_fit_config_fn is not None:
            config.update(self.on_fit_config_fn(server_round))
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(sample_size, min_num_clients)
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "is_encrypted": 0,
            "mask-ratio": 0.0,
            "mask-positions": "",
        }
        if self.on_evaluate_config_fn is not None:
            config.update(self.on_evaluate_config_fn(server_round))
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

        aggregated_params: Optional[Parameters]
        metrics_aggregated: Dict[str, Scalar] = {}

        if self.encrypted and self.he and self.ckks_context:
            aggregated_params = self._aggregate_selective_encrypted(results)
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

        self._update_mask(results)
        self.mask_version += 1

        return aggregated_params, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results or (not self.accept_failures and failures):
            return None, {}
        loss = weighted_loss_avg([(res.num_examples, res.loss) for _, res in results])
        metrics: Dict[str, Scalar] = {}
        if self.evaluate_metrics_aggregation_fn is not None:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss, metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _aggregate_selective_encrypted(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> Parameters:
        if self.weight_template is None or self.vector_length == 0:
            raise RuntimeError("Weight template not initialised")

        plain_accumulator = np.zeros(self.vector_length, dtype=np.float64)
        total_examples = 0

        encrypted_accumulator = None
        encrypted_examples = 0
        reference_mask: np.ndarray | None = None

        for _, res in results:
            payload = parameters_to_ndarrays(res.parameters)
            meta = np.asarray(payload[0], dtype=np.int64)
            total_len = int(meta[0])
            mask_len = int(meta[1])
            mask_indices = np.asarray(payload[1], dtype=np.int64)
            plain_values = np.asarray(payload[2], dtype=np.float64)

            if total_len != self.vector_length:
                raise ValueError("Inconsistent vector length in payload")

            mask_bool = boolean_mask(mask_indices, total_len)
            complement_indices = np.where(~mask_bool)[0]
            vector = np.zeros(total_len, dtype=np.float64)
            vector[complement_indices] = plain_values
            plain_accumulator += vector * res.num_examples
            total_examples += res.num_examples

            if mask_len == 0:
                continue

            if reference_mask is None:
                reference_mask = mask_indices
            elif reference_mask.size != mask_indices.size or not np.array_equal(reference_mask, mask_indices):
                raise ValueError("Mask mismatch between clients; masks must be synchronised")

            enc_payload = payload[3:]
            ciphertexts, original_length = self.ckks_context.deserialize_ciphertexts(enc_payload, self.he)
            if original_length != mask_len:
                raise ValueError("Encrypted payload does not match mask length")
            scaled = self.ckks_context.scale_ciphertexts(ciphertexts, res.num_examples)
            encrypted_accumulator = (
                scaled
                if encrypted_accumulator is None
                else self.ckks_context.add_ciphertext_lists(encrypted_accumulator, scaled)
            )
            encrypted_examples += res.num_examples

        if total_examples == 0:
            return ndarrays_to_parameters(self.weight_template)

        averaged = plain_accumulator / total_examples

        if reference_mask is not None and reference_mask.size > 0:
            if encrypted_accumulator is None or encrypted_examples == 0:
                raise RuntimeError("Missing encrypted aggregates for masked weights")
            averaged_ciphertexts = self.ckks_context.scale_ciphertexts(
                encrypted_accumulator, 1.0 / encrypted_examples
            )
            decrypted = self.ckks_context.decrypt_vector(self.he, averaged_ciphertexts, reference_mask.size)
            averaged[reference_mask] = decrypted

        weights = unflatten_weights(averaged, self.weight_template)
        return ndarrays_to_parameters(weights)

    def _update_mask(self, results: List[Tuple[ClientProxy, FitRes]]) -> None:
        if self.vector_length == 0:
            self.current_mask = np.array([], dtype=np.int64)
            return

        target_size = mask_size_from_ratio(self.vector_length, self.mask_ratio)
        if target_size == 0:
            self.current_mask = np.array([], dtype=np.int64)
            return

        aggregate_scores: Dict[int, float] = {}
        for _, res in results:
            metrics = res.metrics
            version = int(metrics.get("mask_version", -1))
            if version != self.mask_version:
                continue
            serialized = metrics.get("mask_scores", "")
            for idx, score in decode_mask_scores(serialized):
                aggregate_scores[idx] = aggregate_scores.get(idx, 0.0) + score

        if not aggregate_scores:
            return

        sorted_indices = sorted(
            aggregate_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        next_mask = [idx for idx, _ in sorted_indices[:target_size]]
        self.current_mask = np.array(sorted(next_mask), dtype=np.int64)
