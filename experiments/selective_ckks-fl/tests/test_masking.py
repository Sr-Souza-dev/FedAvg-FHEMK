from __future__ import annotations

import numpy as np

from fl_simulation.masking import (
    boolean_mask,
    decode_mask_scores,
    encode_mask_scores,
    finalize_exposed_after_training,
    mask_proposal_scores,
    prepare_exposed_before_training,
)


def test_prepare_exposed_before_training_respects_mask():
    previous = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    incoming = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    mask = boolean_mask(np.array([1], dtype=np.int64), length=3)
    result = prepare_exposed_before_training(previous, incoming, mask)
    assert np.allclose(result, np.array([10.0, 2.0, 12.0]))


def test_finalize_exposed_after_training_updates_unmasked_entries():
    exposed_before = np.array([10.0, 2.0, 12.0], dtype=np.float64)
    trained = np.array([20.0, 21.0, 22.0], dtype=np.float64)
    mask = boolean_mask(np.array([1], dtype=np.int64), length=3)
    result = finalize_exposed_after_training(exposed_before, trained, mask)
    assert np.allclose(result, np.array([20.0, 2.0, 22.0]))


def test_mask_proposal_scores_sorted_descending():
    importance = np.array([0.1, 0.9, -0.2, 0.5], dtype=np.float64)
    payload = mask_proposal_scores(importance, limit=2)
    assert payload == [[1, importance[1]], [3, importance[3]]]


def test_encode_decode_mask_scores_roundtrip():
    pairs = [[1, 0.5], [7, 1.2]]
    encoded = encode_mask_scores(pairs)
    decoded = decode_mask_scores(encoded)
    assert decoded == [(1, 0.5), (7, 1.2)]
