import numpy as np

def get_structure(data):
    """
    Get the structure of a nested list.
    """
    structure = []

    def _get_structure(x):
        if isinstance(x, (list, np.ndarray)):
            sub_structure = []
            for item in x:
                sub_structure.append(_get_structure(item))
            return sub_structure
        else:
            return None  # marca uma folha

    structure = _get_structure(data)
    return structure

def flatten(data):
    """
    Flatten a nested list into a 1D list and return the structure.
    """
    flat = []

    def _flatten(x):
        if isinstance(x, (list, np.ndarray)):
            for item in x:
                _flatten(item)
        elif isinstance(x, (int, float, np.number)):
            flat.append(x)
        else: 
            raise ValueError(f"Unsupported type on Flatten: {type(x)}")

    _flatten(data)
    return np.array(flat, dtype=np.float64)


def unflatten(flat, structure):
    it = iter(flat)

    def _unflatten(struct):
        if struct is None:
            return next(it)
            # return 0.5
        return [_unflatten(s) for s in struct]

    return _unflatten(structure)
