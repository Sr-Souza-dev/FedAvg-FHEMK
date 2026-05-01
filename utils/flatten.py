from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

ArrayStructure = Tuple[Tuple[int, ...], np.dtype]


def _ensure_array(weight) -> np.ndarray:
    """Convert model weights to a NumPy array without forcing a copy when possible."""
    arr = np.asarray(weight)
    if arr.ndim == 0:
        # Ensure scalar parameters still behave like 1-element arrays
        arr = arr.reshape(1)
    return arr


def get_structure(weights: Sequence) -> Tuple[ArrayStructure, ...]:
    """
    Capture the shape/dtype metadata for every NumPy array in `weights`.

    This structure can later be used to rebuild the original tensors from a
    flattened 1-D vector via :func:`unflatten`.
    """
    structure: list[ArrayStructure] = []
    for weight in weights:
        arr = _ensure_array(weight)
        structure.append((arr.shape, arr.dtype))
    return tuple(structure)


def flatten(weights: Sequence) -> np.ndarray:
    """
    Flatten a list of arrays into a single contiguous float64 vector.

    The function avoids repeated reallocations by pre-allocating the final
    buffer and copying each view only once.
    """
    arrays: list[np.ndarray] = []
    total_size = 0
    for weight in weights:
        arr = np.asarray(_ensure_array(weight), dtype=np.float64)
        arrays.append(arr.reshape(-1))
        total_size += arr.size

    flat = np.empty(total_size, dtype=np.float64)
    offset = 0
    for arr in arrays:
        size = arr.size
        flat[offset : offset + size] = arr
        offset += size
    return flat


def unflatten(flat: Iterable[float], structure: Sequence[ArrayStructure]) -> list[np.ndarray]:
    """
    Restore the original list of arrays from a flattened vector and its structure.
    """
    flat_arr = np.asarray(flat, dtype=np.float64).reshape(-1)
    restored: list[np.ndarray] = []
    offset = 0
    total = flat_arr.size

    for shape, dtype in structure:
        size = int(np.prod(shape, dtype=int)) if shape else 1
        if offset + size > total:
            raise ValueError("Flat array is too small to match the provided structure.")
        segment = flat_arr[offset : offset + size]
        restored.append(segment.reshape(shape).astype(dtype, copy=False))
        offset += size

    if offset != total:
        raise ValueError("Flat array contains extra data for the provided structure.")
    return restored
