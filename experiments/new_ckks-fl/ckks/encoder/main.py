import numpy as np
from ckks.polynomials.main import Polynomials

class Encoder:

    def __init__(self, N, scale=2**16):
        """
            Set the parameters for the encoder
        """
        self.N = N
        self.vec_size = N
        self.scale = scale
    
    def encode(self, v: np.ndarray) -> Polynomials:
        v = np.array(v, dtype=np.float64)
        if len(v) < self.vec_size:
            v = np.concatenate((v, np.zeros(self.vec_size - len(v), dtype=np.float64)))
        elif len(v) > self.vec_size:
            raise ValueError(f"Vector size {len(v)} is greater than {self.vec_size}")

        # Multiplica pelo scale
        v_scaled = v * self.scale
        
        # Remove NaN e Inf antes de processar
        v_scaled = np.nan_to_num(v_scaled, nan=0.0, posinf=2**62, neginf=-2**62)
        
        # Arredonda e converte para int64 (suporta valores maiores)
        # Usa clip para evitar overflow em valores extremos
        v_int = np.round(np.real(v_scaled))
        v_int = np.clip(v_int, -2**62, 2**62)  # Limita para evitar overflow
        
        p = Polynomials(v_int.astype(np.int64).tolist())
        return p

    def decode(self, p: Polynomials) -> np.ndarray:
        v = p / self.scale
        v = v.coefficients

        # Pad with zeros if necessary
        if len(v) < self.vec_size:
            v = v + [0] * (self.vec_size - len(v))

        return np.array(v)