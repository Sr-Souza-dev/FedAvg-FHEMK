from __future__ import annotations

from types import ModuleType

from .loader import get_backend

_backend: ModuleType = get_backend()

Net = _backend.Net
train = _backend.train
test = _backend.test
get_weights = _backend.get_weights
set_weights = _backend.set_weights
