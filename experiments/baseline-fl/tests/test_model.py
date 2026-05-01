import numpy as np
import torch

from fl_simulation.model import Net, get_weights, set_weights


def test_forward_pass_produces_logits():
    net = Net()
    sample = torch.randn(1, 1, 28, 28)
    logits = net(sample)
    assert logits.shape == (1, 10)
    assert torch.isfinite(logits).all()


def test_weight_roundtrip_after_update():
    net = Net()
    original = get_weights(net)
    updated = [w.copy() for w in original]
    updated[0] = updated[0] + np.ones_like(updated[0]) * 0.5
    set_weights(net, updated)
    roundtrip = get_weights(net)
    assert np.allclose(updated[0], roundtrip[0])
    for src, dst in zip(updated[1:], roundtrip[1:]):
        assert np.allclose(src, dst)
