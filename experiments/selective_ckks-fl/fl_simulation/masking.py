from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from utils.weights import flatten_weights


def mask_size_from_ratio(vector_length: int, ratio: float) -> int:
    if vector_length <= 0 or ratio <= 0:
        return 0
    return int(max(1, min(vector_length, round(vector_length * ratio))))


def encode_mask(indices: Sequence[int]) -> str:
    if not indices:
        return ""
    return ",".join(str(int(idx)) for idx in indices)


def decode_mask(serialized: str) -> np.ndarray:
    tokens = [token.strip() for token in serialized.split(",") if token.strip()]
    if not tokens:
        return np.array([], dtype=np.int64)
    return np.array([int(token) for token in tokens], dtype=np.int64)


def boolean_mask(indices: np.ndarray, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if indices.size == 0:
        return mask
    mask[np.clip(indices, 0, length - 1)] = True
    return mask


def prepare_exposed_before_training(
    previous_exposed: np.ndarray | None,
    incoming_weights: np.ndarray,
    encrypted_mask: np.ndarray,
) -> np.ndarray:
    if previous_exposed is None or previous_exposed.shape != incoming_weights.shape:
        return incoming_weights.copy()
    updated = previous_exposed.copy()
    updated[~encrypted_mask] = incoming_weights[~encrypted_mask]
    return updated


def finalize_exposed_after_training(
    exposed_before: np.ndarray,
    trained_weights: np.ndarray,
    encrypted_mask: np.ndarray,
) -> np.ndarray:
    final = exposed_before.copy()
    final[~encrypted_mask] = trained_weights[~encrypted_mask]
    return final


def gradient_snapshot(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 1,
) -> np.ndarray:
    criterion = torch.nn.CrossEntropyLoss().to(device)
    grads_accumulator: np.ndarray | None = None
    batches_processed = 0
    model.eval()
    data_iter = iter(dataloader)
    state_names = list(model.state_dict().keys())
    while batches_processed < max_batches:
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        grad_map = {
            name: (param.grad.detach().cpu().numpy() if param.grad is not None else np.zeros_like(param.detach().cpu().numpy()))
            for name, param in model.named_parameters()
        }
        aligned_grads = []
        for name in state_names:
            if name in grad_map:
                aligned_grads.append(grad_map[name])
            else:
                tensor = model.state_dict()[name]
                aligned_grads.append(torch.zeros_like(tensor).cpu().numpy())
        grad_vector = flatten_weights(aligned_grads)
        grads_accumulator = (
            grad_vector.copy()
            if grads_accumulator is None
            else grads_accumulator + grad_vector
        )
        batches_processed += 1
    if grads_accumulator is None:
        total_params = sum(param.numel() for param in model.parameters())
        return np.zeros(total_params, dtype=np.float64)
    return grads_accumulator / max(1, batches_processed)


def mask_proposal_scores(importance: np.ndarray, limit: int) -> list[list[float]]:
    if limit <= 0 or importance.size == 0:
        return []
    indices = np.argsort(importance)[::-1][:limit]
    return [[int(idx), float(importance[idx])] for idx in indices]


def encode_mask_scores(pairs: list[list[float]]) -> str:
    if not pairs:
        return ""
    return json.dumps(pairs)


def decode_mask_scores(serialized: str) -> list[tuple[int, float]]:
    if not serialized:
        return []
    try:
        raw = json.loads(serialized)
    except json.JSONDecodeError:
        return []
    result: list[tuple[int, float]] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            try:
                idx = int(entry[0])
                score = float(entry[1])
            except (TypeError, ValueError):
                continue
            result.append((idx, score))
    return result
