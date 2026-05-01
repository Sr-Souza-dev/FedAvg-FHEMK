from __future__ import annotations

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader

from models.common.transforms import (
    CIFAR10_TEST_TRANSFORM,
    CIFAR10_TRAIN_TRANSFORM,
    apply_vision_transform,
)
from models.common.weights import get_weights, set_weights
from models.resnet20_cifar10.shared import ResNet20 as Net, evaluate as test, train

MODEL_NAME = "resnet20-cifar10-noniid"
MODEL_LABEL = "ResNet-20 - CIFAR10 non-IID"
_fds_cache: dict[int, FederatedDataset] = {}


def _build_dataset(num_partitions: int) -> FederatedDataset:
    dataset = _fds_cache.get(num_partitions)
    if dataset is None:
        dataset = FederatedDataset(
            dataset="cifar10",
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=num_partitions,
                    alpha=0.3,
                    partition_by="label",
                )
            },
        )
        _fds_cache[num_partitions] = dataset
    return dataset


def load_data(partition_id: int, num_partitions: int) -> tuple[DataLoader, DataLoader]:
    dataset = _build_dataset(num_partitions)
    partition = dataset.load_partition(partition_id)
    split = partition.train_test_split(test_size=0.1, seed=123)

    train_ds = split["train"].with_transform(
        lambda batch: apply_vision_transform(batch, CIFAR10_TRAIN_TRANSFORM)
    )
    test_ds = split["test"].with_transform(
        lambda batch: apply_vision_transform(batch, CIFAR10_TEST_TRANSFORM)
    )

    trainloader = DataLoader(train_ds, batch_size=64, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=64)
    return trainloader, testloader
