import numpy as np
from ckks.polynomials.main import Polynomials

class Encoder:

    def __init__(self, N, scale=2**16):
        self.N = N
        self.vec_size = N
        self.scale = scale

    def encode(self, v: np.ndarray) -> Polynomials:
        v = np.array(v, dtype=np.float64)
        if len(v) < self.vec_size:
            v = np.concatenate((v, np.zeros(self.vec_size - len(v), dtype=np.float64)))
        elif len(v) > self.vec_size:
            raise ValueError(f"Vector size {len(v)} is greater than {self.vec_size}")

        v_scaled = v * self.scale
        v_scaled = np.nan_to_num(v_scaled, nan=0.0, posinf=2**62, neginf=-2**62)
        v_int = np.clip(np.round(np.real(v_scaled)), -2**62, 2**62).astype(np.int64)
        return Polynomials(v_int)

    def decode(self, p: Polynomials) -> np.ndarray:
        coeffs = p.coefficients.astype(np.float64) / self.scale
        if len(coeffs) < self.vec_size:
            coeffs = np.pad(coeffs, (0, self.vec_size - len(coeffs)))
        return coeffs
