from __future__ import annotations

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.common.training import evaluate_classifier, train_classifier
from models.common.transforms import MNIST_TRANSFORM, apply_vision_transform
from models.common.weights import get_weights, set_weights

MODEL_NAME = "mlp-mnist"
MODEL_LABEL = "MLP - MNIST (IID)"
_fds_cache: dict[int, FederatedDataset] = {}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _federated_dataset(num_partitions: int) -> FederatedDataset:
    dataset = _fds_cache.get(num_partitions)
    if dataset is None:
        dataset = FederatedDataset(
            dataset="tanganke/kmnist",
            partitioners={"train": IidPartitioner(num_partitions=num_partitions)},
        )
        _fds_cache[num_partitions] = dataset
    return dataset


def load_data(partition_id: int, num_partitions: int) -> tuple[DataLoader, DataLoader]:
    dataset = _federated_dataset(num_partitions)
    partition = dataset.load_partition(partition_id)
    split = partition.train_test_split(test_size=0.2, seed=42)

    train_ds = split["train"].with_transform(lambda batch: apply_vision_transform(batch, MNIST_TRANSFORM))
    test_ds = split["test"].with_transform(lambda batch: apply_vision_transform(batch, MNIST_TRANSFORM))

    trainloader = DataLoader(train_ds, batch_size=32, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=32)
    return trainloader, testloader


def train(net: nn.Module, trainloader: DataLoader, epochs: int, device):
    return train_classifier(net, trainloader, epochs=epochs, device=device, lr=0.02)


def test(net: nn.Module, testloader: DataLoader, device):
    return evaluate_classifier(net, testloader, device=device)
