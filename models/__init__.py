"""Unified registry for all available model backends."""

from . import mlp_mnist  # noqa: F401
from . import resnet20_cifar10_iid  # noqa: F401
from . import resnet20_cifar10_noniid  # noqa: F401
from .data_loader import load_data
from .model import Net, get_weights, set_weights, test, train
from .registry import (
    MODEL_ENV_VAR,
    ModelSpec,
    available_model_names,
    get_model_spec,
    list_models,
)

__all__ = [
    "ModelSpec",
    "MODEL_ENV_VAR",
    "Net",
    "available_model_names",
    "get_model_spec",
    "get_weights",
    "list_models",
    "load_data",
    "mlp_mnist",
    "resnet20_cifar10_iid",
    "resnet20_cifar10_noniid",
    "set_weights",
    "test",
    "train",
]
