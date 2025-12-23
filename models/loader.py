from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType

from .registry import get_model_spec, resolve_model_name


@lru_cache(maxsize=None)
def _load_backend(name: str) -> ModuleType:
    spec = get_model_spec(name)
    return importlib.import_module(spec.module)


def get_backend(name: str | None = None) -> ModuleType:
    resolved = resolve_model_name(name)
    return _load_backend(resolved)


def current_model_name() -> str:
    return resolve_model_name(None)
