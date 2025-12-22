import numpy as np

from tests.utils import generate_message_error
from utils.flatten import flatten, get_structure, unflatten


def get_message_wrapper(params: dict, result: dict, function: str) -> str:
    return generate_message_error(params, result, "ND_TO_1D", function)


def _random_weight_list() -> list[np.ndarray]:
    tensors: list[np.ndarray] = []
    for _ in range(np.random.randint(1, 8)):
        dims = np.random.randint(1, 4)
        shape = tuple(np.random.randint(1, 5) for _ in range(dims))
        dtype = np.random.choice([np.float32, np.float64])
        tensors.append(np.random.randn(*shape).astype(dtype))
    return tensors


class TestNDTo1D:
    def test_flatten_unflatten(self):
        for _ in range(100):
            weights = _random_weight_list()
            structure = get_structure(weights)
            flat = flatten(weights)
            rebuilt = unflatten(flat, structure)

            for original, recovered in zip(weights, rebuilt):
                assert original.shape == recovered.shape, get_message_wrapper(
                    {"shape": original.shape}, {"restored_shape": recovered.shape}, "test_flatten_unflatten"
                )
                assert original.dtype == recovered.dtype, get_message_wrapper(
                    {"dtype": original.dtype}, {"restored_dtype": recovered.dtype}, "test_flatten_unflatten"
                )
                assert np.allclose(original, recovered, atol=1e-8), get_message_wrapper(
                    {"original": original}, {"restored": recovered}, "test_flatten_unflatten"
                )
    
