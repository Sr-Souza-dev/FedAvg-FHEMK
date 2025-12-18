from __future__ import annotations

import numpy as np

from utils.weights import clone_template, flatten_weights, unflatten_weights


def test_flatten_and_unflatten_roundtrip():
    weights = [
        np.random.randn(2, 3).astype(np.float64),
        np.random.randn(4).astype(np.float64),
        np.random.randn(1, 2, 2).astype(np.float64),
    ]
    flat = flatten_weights(weights)
    rebuilt = unflatten_weights(flat, weights)

    assert len(rebuilt) == len(weights)
    for original, restored in zip(weights, rebuilt):
        np.testing.assert_allclose(restored, original)


def test_clone_template_produces_independent_copy():
    weights = [
        np.ones((2, 2), dtype=np.float64),
        np.full((3,), 5.0, dtype=np.float64),
    ]
    cloned = clone_template(weights)

    cloned[0][0, 0] = 123.0
    cloned[1][1] = -5.0

    assert not np.array_equal(cloned[0], weights[0])
    assert not np.array_equal(cloned[1], weights[1])
    np.testing.assert_array_equal(weights[0], np.ones((2, 2)))
    np.testing.assert_array_equal(weights[1], np.full((3,), 5.0))
