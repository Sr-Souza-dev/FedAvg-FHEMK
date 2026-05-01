from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

MODEL_ENV_VAR = "AQUIPLACA_MODEL_NAME"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    module: str


AVAILABLE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("mlp-mnist", "MLP - MNIST (IID)", "models.mlp_mnist"),
    ModelSpec("resnet20-cifar10-iid", "ResNet-20 - CIFAR10 IID", "models.resnet20_cifar10_iid"),
    ModelSpec("resnet20-cifar10-noniid", "ResNet-20 - CIFAR10 non-IID", "models.resnet20_cifar10_noniid"),
)

_MODEL_LOOKUP = {spec.name: spec for spec in AVAILABLE_MODELS}
_ALIASES: dict[str, str] = {}
for spec in AVAILABLE_MODELS:
    sanitized = spec.name.replace("_", "-")
    _ALIASES[sanitized] = spec.name
    _ALIASES[spec.name.replace("-", "_")] = spec.name
DEFAULT_MODEL = AVAILABLE_MODELS[0]


def list_models() -> Iterable[ModelSpec]:
    return AVAILABLE_MODELS


def available_model_names() -> list[str]:
    return [spec.name for spec in AVAILABLE_MODELS]


def _normalize_candidate(candidate: str) -> str:
    value = candidate.strip().lower()
    value = value.replace(" ", "-")
    value = value.replace("__", "-").replace("_", "-")
    return value


def resolve_model_name(explicit: str | None = None) -> str:
    candidate = explicit or os.environ.get(MODEL_ENV_VAR, "")
    normalized = _normalize_candidate(candidate) if candidate else ""
    preferred = _ALIASES.get(normalized, normalized)
    if preferred in _MODEL_LOOKUP:
        return preferred
    return DEFAULT_MODEL.name


def get_model_spec(name: str | None = None) -> ModelSpec:
    resolved = resolve_model_name(name)
    return _MODEL_LOOKUP.get(resolved, DEFAULT_MODEL)
