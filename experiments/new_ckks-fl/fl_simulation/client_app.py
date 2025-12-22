"""fl-simulation: A Flower / PyTorch app."""

import time

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from experiment_config import get_experiment_config
from fl_simulation.ckks_instance import MODEL_STRUCTURE, ckks
from fl_simulation.model.data_loader import load_data
from fl_simulation.model.model import Net, get_weights, set_weights, test, train
from utils.flatten import flatten, unflatten
from utils.files import logging_enabled, register_logs

EXPERIMENT_NAME = "new_ckks-fl"
EXPERIMENT_CONFIG = get_experiment_config(EXPERIMENT_NAME)

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id, is_flattened=False, clientId=0):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.partition_id = partition_id
        self.structure = MODEL_STRUCTURE
        self.size_flattened = ckks.model_size
        self.is_flattened = is_flattened
        self.client_id = str(clientId)
        self.sk = ckks.load_key(prefix=str(clientId))

    def fit(self, parameters, config):
        start = time.time()
        log_enabled = logging_enabled()
        if log_enabled:
            register_logs(
                file_name=self.client_id,
                title=f"\n ------------- Round (fit) {config['server_round']} -------------- \n",
                value=f"received_tensors={len(parameters)}",
            )

        model_parameters = parameters
        if (
            self.is_flattened
            and config.get("is_flattened")
            and len(parameters) == 1
            and parameters[0].ndim == 1
            and parameters[0].size == self.size_flattened
        ):
            model_parameters = unflatten(parameters[0], self.structure)
            if log_enabled:
                register_logs(
                    file_name=self.client_id,
                    title="\nModel Received (unflattened)",
                    value=f"num_tensors={len(model_parameters)}",
                )

        set_weights(self.net, model_parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        weights = get_weights(self.net)
        payload_size = float(sum(np.asarray(w).nbytes for w in weights))

        if self.is_flattened:
            flat_weights = flatten(weights)
            if log_enabled:
                register_logs(
                    file_name=self.client_id,
                    title="\n Model flatten details:",
                    value=f"vector_len={flat_weights.size}",
                )
            ciphertexts = ckks.encrypt_batch(sk=self.sk, plaintext=flat_weights)
            weights, encrypted_bytes = ckks.serialize_ciphertexts(ciphertexts)
            payload_size = float(encrypted_bytes)
            if log_enabled:
                register_logs(
                    file_name=self.client_id,
                    title="\n Encryption summary:",
                    value=f"ciphertexts={len(weights)} payload_bytes={payload_size}",
                )

        end = time.time()
        return (
            weights,
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "execution_time": end - start,
                "size": payload_size,
            },
        )

    def evaluate(self, parameters, config):
        log_enabled = logging_enabled()
        if log_enabled:
            register_logs(
                file_name=self.client_id,
                title=f"\n ------------- Round (evaluate) {config['server_round']} -------------- \n",
                value=f"received_tensors={len(parameters)}",
            )

        model_parameters = parameters
        if (
            self.is_flattened
            and config.get("is_flattened")
            and len(parameters) == 1
            and parameters[0].ndim == 1
            and parameters[0].size == self.size_flattened
        ):
            model_parameters = unflatten(parameters[0], self.structure)

        set_weights(self.net, model_parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    global clientId, createdNodesId

    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = EXPERIMENT_CONFIG.epochs
    is_flattened = True if context.run_config["is-encrypted"] == 1 else False

    # Return Client instance
    return FlowerClient(
        net, 
        trainloader, 
        valloader, 
        local_epochs, 
        partition_id, 
        is_flattened=is_flattened,
        clientId=context.node_id
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
