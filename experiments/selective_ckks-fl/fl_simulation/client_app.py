from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from experiment_config import get_experiment_config
from fl_simulation.crypto.ckks_context import build_shared_context
from fl_simulation.masking import (
    boolean_mask,
    decode_mask,
    encode_mask_scores,
    finalize_exposed_after_training,
    gradient_snapshot,
    mask_proposal_scores,
    mask_size_from_ratio,
    prepare_exposed_before_training,
)
from model.data_loader import load_data
from model.model import Net, get_weights, set_weights, test, train
from utils.files import logging_enabled, register_logs
from utils.weights import flatten_weights

EXPERIMENT_NAME = "selective_ckks-fl"
EXPERIMENT_CONFIG = get_experiment_config(EXPERIMENT_NAME)


class SelectiveClient(NumPyClient):
    def __init__(
        self,
        *,
        net: Net,
        trainloader,
        valloader,
        local_epochs: int,
        encrypted_updates: bool,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.encrypted_updates = encrypted_updates
        self.ckks_context = build_shared_context() if encrypted_updates else None
        self.he = self.ckks_context.build_he(with_secret=True) if encrypted_updates else None
        self.exposed_vector: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Flower NumPyClient API
    # ------------------------------------------------------------------
    def fit(self, parameters: List[Any], config: Dict[str, Any]):
        if parameters:
            set_weights(self.net, parameters)

        start_time = time.time()
        logging = logging_enabled()
        if logging:
            register_logs(
                file_name="client",
                title=f"Round {config.get('server_round', 'na')} - starting fit",
                value=f"Mask version={config.get('mask-version', 0)}",
            )

        global_weights = get_weights(self.net)
        flat_global = flatten_weights(global_weights)
        mask_indices = np.sort(decode_mask(str(config.get("mask-positions", ""))))
        mask_bool = boolean_mask(mask_indices, flat_global.size)
        exposed_before = prepare_exposed_before_training(self.exposed_vector, flat_global, mask_bool)

        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        trained_weights = get_weights(self.net)
        flat_trained = flatten_weights(trained_weights)

        gradients = gradient_snapshot(self.net, self.trainloader, self.device, max_batches=1)
        delta = exposed_before - flat_trained
        importance = gradients * delta

        mask_ratio = float(config.get("mask-ratio", 0.0))
        mask_target = mask_size_from_ratio(flat_trained.size, mask_ratio)
        proposal_multiplier = max(1.0, float(config.get("mask-proposal-multiplier", 3.0)))
        proposal_limit = max(mask_target, 1) * int(round(proposal_multiplier))
        proposal_payload = encode_mask_scores(mask_proposal_scores(importance, proposal_limit))

        payload = self._build_payload(flat_trained, mask_indices)
        payload_size = sum(block.nbytes for block in payload)

        self.exposed_vector = finalize_exposed_after_training(exposed_before, flat_trained, mask_bool)

        metrics = {
            "train_loss": float(train_loss),
            "execution_time": float(time.time() - start_time),
            "size": float(payload_size),
            "mask_scores": proposal_payload,
            "mask_version": int(config.get("mask-version", 0)),
        }
        if logging:
            register_logs(
                file_name="client",
                title="Mask proposal payload",
                value=f"indices_length={len(mask_indices)}, proposal_length={proposal_limit}",
            )
        return payload, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: List[Any], config: Dict[str, Any]):
        if parameters:
            set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_payload(self, flat_weights: np.ndarray, mask_indices: np.ndarray) -> List[np.ndarray]:
        mask_indices = np.asarray(mask_indices, dtype=np.int64)
        mask_len = mask_indices.size
        total_len = flat_weights.size
        mask_bool = boolean_mask(mask_indices, total_len)
        plain_indices = np.where(~mask_bool)[0]
        plain_values = flat_weights[plain_indices].astype(np.float64, copy=True)
        masked_values = flat_weights[mask_bool].astype(np.float64, copy=True)

        payload: List[np.ndarray] = [
            np.array([total_len, mask_len], dtype=np.int64),
            mask_indices,
            plain_values,
        ]

        if mask_len == 0:
            return payload

        if not self.encrypted_updates or self.ckks_context is None or self.he is None:
            raise RuntimeError("Encryption disabled but mask is non-empty; selective CKKS requires encryption.")

        encrypted_payload = self.ckks_context.encrypt_vector(self.he, masked_values)
        payload.extend(encrypted_payload)
        return payload


def client_fn(context: Context):
    trainloader, valloader = load_data(
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
    )
    net = Net()
    return SelectiveClient(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        local_epochs=EXPERIMENT_CONFIG.epochs,
        encrypted_updates=context.run_config["is-encrypted"] == 1,
    ).to_client()


app = ClientApp(client_fn)
