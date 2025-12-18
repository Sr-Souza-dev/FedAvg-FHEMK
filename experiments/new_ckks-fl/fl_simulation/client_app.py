"""fl-simulation: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_simulation.model.model import Net, get_weights, set_weights, test, train
from fl_simulation.model.data_loader import load_data

from utils.flatten import flatten, get_structure, unflatten
from utils.files import logging_enabled, register_logs
from fl_simulation.ckks_instance import ckks

import time

import numpy as np

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
        self.structure = get_structure(get_weights(self.net))
        self.size_flattened = len(flatten(get_weights(self.net)))
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
                value="",
            )
            register_logs(
                file_name=self.client_id,
                title=f"\nModel Received Start: {type(parameters)}",
                value=repr(parameters),
            )
        if self.is_flattened and config.get("is_flattened") and config["server_round"] > 1:
            if log_enabled:
                register_logs(
                    file_name=self.client_id,
                    title=f"\nModel Received: {type(parameters[0])}",
                    value=repr(parameters[0]),
                )
            parameters = parameters[0]
            parameters = unflatten(parameters, self.structure)
            if log_enabled:
                register_logs(
                    file_name=self.client_id,
                    title="\nModel Received _after:",
                    value=repr(parameters),
                )

        
        set_weights(self.net, parameters)
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
            payload_size = float(flat_weights.nbytes)
            if log_enabled:
                register_logs(
                    file_name=self.client_id,
                    title="\n Model after flatten:",
                    value=repr(flat_weights),
                )
            ciphertexts = ckks.encrypt_batch(sk=self.sk, plaintext=flat_weights)
            weights = ckks.extract_vector(ciphertexts)
            if log_enabled:
                register_logs(
                    file_name=self.client_id,
                    title="\n Model encripted vector:",
                    value=repr(weights),
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
                value="",
            )
            register_logs(
                file_name=self.client_id,
                title="\nModel Received:",
                value=repr(parameters[0]),
            )

        if self.is_flattened and config.get("is_flattened"):
            parameters = parameters[0]
            parameters = unflatten(parameters, self.structure)

        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    global clientId, createdNodesId

    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
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
