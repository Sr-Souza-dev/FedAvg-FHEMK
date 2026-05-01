import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Tuple

def reconstruct_polynomial(pontos: List[Tuple[float, float]] ):
    """
    Recebe uma lista de pontos (x, y) e retorna um objeto Polynomial
    que interpola exatamente esses pontos.
    """
    xs, ys = zip(*pontos)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    # Monta a matriz de Vandermonde
    V = np.vander(xs, increasing=True)

    # Resolve o sistema V * coef = ys
    coef = np.linalg.solve(V, ys)

    # Retorna polinômio
    return Polynomial(coef)