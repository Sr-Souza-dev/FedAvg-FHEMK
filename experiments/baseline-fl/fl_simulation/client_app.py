"""Baseline FL experiment client implementation."""

import time
from typing import Sequence

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from experiment_config import get_experiment_config
from fl_simulation.model import (
    Net,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
)
from utils.files import logging_enabled, register_logs

EXPERIMENT_NAME = "baseline-fl"
EXPERIMENT_CONFIG = get_experiment_config(EXPERIMENT_NAME)


class BaselineClient(NumPyClient):
    def __init__(
        self,
        net: Net,
        trainloader,
        valloader,
        local_epochs: int,
        partition_id: int,
        client_id: str,
    ) -> None:
        super().__init__()
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.client_id = client_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def _log(self, title: str, value: str) -> None:
        if logging_enabled():
            register_logs(file_name=self.client_id, title=title, value=value)

    def fit(self, parameters: Sequence[np.ndarray], config):
        self._log(
            title=f"\n --- Round (fit) {config['server_round']} ---",
            value=f"Partition {self.partition_id} received {len(parameters)} tensors",
        )
        set_weights(self.net, parameters)
        
        # Training time
        train_start = time.time()
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        train_time = time.time() - train_start

        weights = get_weights(self.net)
        payload_size = float(sum(np.asarray(w).nbytes for w in weights))

        metrics = {
            "train_loss": train_loss,
            "train_time": train_time,
            "encrypt_time": 0.0,  # No encryption in baseline
            "decrypt_time": 0.0,  # No decryption in baseline
            "execution_time": train_time,  # Total time (backward compatibility)
            "size": payload_size,
        }
        self._log(title="Updated metrics", value=repr(metrics))
        return weights, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: Sequence[np.ndarray], config):
        self._log(
            title=f"\n --- Round (evaluate) {config['server_round']} ---",
            value=f"Partition {self.partition_id} evaluating",
        )
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        metrics = {"accuracy": accuracy}
        self._log(title="Validation", value=f"loss={loss:.4f}, acc={accuracy:.4f}")
        return loss, len(self.valloader.dataset), metrics


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = EXPERIMENT_CONFIG.epochs

    return BaselineClient(
        net=Net(),
        trainloader=trainloader,
        valloader=valloader,
        local_epochs=local_epochs,
        partition_id=partition_id,
        client_id=str(context.node_id),
    ).to_client()


app = ClientApp(client_fn)
