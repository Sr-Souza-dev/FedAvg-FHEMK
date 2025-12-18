from tests.utils import generate_message_error
from utils.flatten import flatten, unflatten, get_structure
import numpy as np

def get_message_wrapper(params:dict, result:dict, function:str) -> str:
    return generate_message_error(params, result, "ND_TO_1D", function)

def generate_random_structure() -> list:
    """
        Generate a random structure of nested lists.
    """

    def _generate_structure(depth: int) -> list:
        if depth == 0:
            size = np.random.randint(2, 9)
            return np.random.randint(-1000, 1000, size)
        else:
            new_structure = [_generate_structure(depth - 1) for _ in range(np.random.randint(1, 5))]
            size = np.random.randint(1, 15)
            for _ in range(size):
                new_structure.append(np.random.randint(-1000, 1000))
            return new_structure
        
    depth = np.random.randint(1, 5)
    structure = _generate_structure(depth)
    return structure

def is_structure_equal(structure1: list, structure2: list) -> bool:
    """
        Check if two structures are equal.
    """
    def _is_structure_equal(s1: list, s2: list) -> bool:
        if len(s1) != len(s2):
            print(f"Different lengths: {len(s1)} != {len(s2)}")
            return False
        for i in range(len(s1)):
            if isinstance(s1[i], (list, np.ndarray)) and isinstance(s2[i], (list, np.ndarray)):
                if not _is_structure_equal(s1[i], s2[i]):
                    print(f"Different structures: {s1[i]} != {s2[i]}")
                    return False
            elif isinstance(s1[i], (int, float, np.number)) and isinstance(s2[i], (int, float, np.number)):
                if s1[i] != s2[i]:
                    print(f"Different values: {s1[i]} != {s2[i]}")
                    return False
            else:
                return False
        return True
    return _is_structure_equal(structure1, structure2)

class TestNDTo1D:
    def test_flatten_unflatten(self):
        for _ in range(100):
            data = generate_random_structure()
            structure = get_structure(data=data)
            flat = flatten(data=data)
            unflat = unflatten(flat=flat, structure=structure)
            assert is_structure_equal(unflat, data), get_message_wrapper(
                {"structure": structure, "data": data},
                {"flat": flat, "unflat": unflat},
                "test_flatten_unflatten"
            )
    
