from __future__ import annotations

from typing import Iterable, List
import numpy as np


def flatten_weights(weights: Iterable[np.ndarray]) -> np.ndarray:
    """Flatten a list of tensors/arrays into a single 1D numpy array."""
    arrays: List[np.ndarray] = [np.asarray(w).astype(np.float64) for w in weights]
    if not arrays:
        return np.array([], dtype=np.float64)
    return np.concatenate([arr.reshape(-1) for arr in arrays])


def unflatten_weights(flat: np.ndarray, template: Iterable[np.ndarray]) -> List[np.ndarray]:
    """Rebuild tensors based on the template shapes."""
    rebuilt: List[np.ndarray] = []
    idx = 0
    flat = np.asarray(flat, dtype=np.float64)
    for ref in template:
        ref_arr = np.asarray(ref)
        size = ref_arr.size
        chunk = flat[idx: idx + size]
        rebuilt.append(chunk.reshape(ref_arr.shape))
        idx += size
    return rebuilt


def clone_template(weights: Iterable[np.ndarray]) -> List[np.ndarray]:
    """Deep copy the template so shapes are preserved for future unflatten calls."""
    return [np.array(w, copy=True) for w in weights]
