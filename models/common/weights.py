from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

import torch


def get_weights(model: torch.nn.Module) -> list:
    """Return model weights as CPU numpy arrays."""
    return [param.detach().cpu().numpy() for _, param in model.state_dict().items()]


def set_weights(model: torch.nn.Module, parameters: Sequence[Iterable]) -> None:
    """Load a list of numpy arrays back into the model state dict."""
    pairs = zip(model.state_dict().keys(), parameters)
    state = OrderedDict({name: torch.tensor(value) for name, value in pairs})
    model.load_state_dict(state, strict=True)
