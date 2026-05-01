from __future__ import annotations

from types import ModuleType

from .loader import get_backend

_backend: ModuleType = get_backend()
load_data = _backend.load_data
