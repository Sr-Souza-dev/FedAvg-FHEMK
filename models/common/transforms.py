from __future__ import annotations

from typing import Iterable, MutableMapping

from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomCrop, ToTensor


MNIST_TRANSFORM = Compose(
    [
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ]
)

CIFAR10_TRAIN_TRANSFORM = Compose(
    [
        ToTensor(),
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

CIFAR10_TEST_TRANSFORM = Compose(
    [
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

IMAGE_KEYS: tuple[str, ...] = ("image", "img", "images", "pixel_values")


def apply_vision_transform(
    batch: MutableMapping[str, Iterable],
    transform,
) -> MutableMapping[str, Iterable]:
    """Normalize image column names and apply torchvision transforms."""
    source_key = next((key for key in IMAGE_KEYS if key in batch), None)
    if source_key is None:
        raise KeyError(f"Nenhuma chave de imagem conhecida encontrada no batch: {list(batch.keys())}")
    images = batch[source_key]
    batch["image"] = [transform(img) for img in images]
    if source_key != "image":
        batch.pop(source_key, None)
    return batch
