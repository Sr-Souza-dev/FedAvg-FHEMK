from __future__ import annotations

import time
from typing import Any, Dict, List

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from experiment_config import get_experiment_config
from fl_simulation.model.data_loader import load_data
from fl_simulation.model.model import Net, get_weights, set_weights, test, train
from fl_simulation.crypto.ckks_context import build_shared_context
import numpy as np

from utils.weights import flatten_weights

EXPERIMENT_NAME = "full_ckks-fl"
EXPERIMENT_CONFIG = get_experiment_config(EXPERIMENT_NAME)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        net: Net,
        trainloader,
        valloader,
        local_epochs: int,
        client_id: str,
        encrypted_updates: bool,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.client_id = client_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.encrypted_updates = encrypted_updates
        self.ckks_context = build_shared_context() if encrypted_updates else None
        self.he = self.ckks_context.build_he(with_secret=True) if encrypted_updates else None

    def fit(self, parameters: List[Any], config: Dict[str, Any]):
        if parameters:
            set_weights(self.net, parameters)
        start = time.time()
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        updated_weights = get_weights(self.net)

        payload_size = 0
        if self.encrypted_updates and self.he:
            flat = flatten_weights(updated_weights)
            payload = self.ckks_context.encrypt_vector(self.he, flat)
            weights_payload = payload
            payload_size = sum(block.nbytes for block in payload)
        else:
            weights_payload = updated_weights
            payload_size = sum(np.asarray(weight).nbytes for weight in updated_weights)

        metrics = {
            "train_loss": train_loss,
            "execution_time": time.time() - start,
            "size": float(payload_size),
        }
        return weights_payload, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: List[Any], config: Dict[str, Any]):
        if parameters:
            set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    net = Net()
    local_epochs = EXPERIMENT_CONFIG.epochs
    encrypted_updates = context.run_config["is-encrypted"] == 1

    return FlowerClient(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        local_epochs=local_epochs,
        client_id=str(context.node_id),
        encrypted_updates=encrypted_updates,
    ).to_client()


app = ClientApp(client_fn)
