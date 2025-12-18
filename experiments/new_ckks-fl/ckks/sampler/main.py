import numpy as np
from ckks.polynomials.main import Polynomials
from ckks.polynomials.ntt import NTT

def get_random_uniform_polynomial(N: int, bound: int, seed=42) -> Polynomials:
    """
    Generates a random polynomial with coefficients centered around 0.
    :param N: Degree of the polynomial
    :param desvio: Standard deviation for the coefficients
    :return: Random polynomial
    """
    rgn = np.random.default_rng(seed)
    return Polynomials(rgn.integers(-bound, bound, N, endpoint=True))

def get_random_normal_polynomial(N: int, bound: int, seed=42) -> Polynomials:
    """
    Generates a random polynomial with positive coefficients.
    :param N: Degree of the polynomial
    :param desvio: Standard deviation for the coefficients
    :return: Random polynomial
    """

    rgn = np.random.default_rng(seed)
    return Polynomials(rgn.normal(0, bound, N))

def round_coordinates(coordinates):
    """Gives the integral rest."""
    coordinates = coordinates - np.floor(coordinates)
    return coordinates

def coordinate_wise_random_rounding(coordinates):
    """Rounds coordinates randonmly."""
    r = round_coordinates(coordinates)
    f = np.array([np.random.choice([c, c-1], 1, p=[1-c, c]) for c in r]).reshape(-1)
    
    rounded_coordinates = coordinates - f
    rounded_coordinates = [int(coeff) for coeff in rounded_coordinates]
    return rounded_coordinates

class Sampler:
    # Scale padrão para codificação
    # Ajustado para 2^16 (~65K) para melhor compatibilidade com gradientes ML
    # que tipicamente são pequenos (0.001 - 1.0)
    DEFAULT_SCALE = 2**16  # ~16 bits de precisão decimal (~5 dígitos)

    def __init__(self,
        N: int,
        sigma: float,
        seed: float = 42):

        """
        :param N: Polynomial degree (number of coefficients/slots)
        :param sigma: Standard deviation for noise generation
        :param seed: Random seed for reproducibility
        
        Note: q (qs) is set to NTT.DEFAULT_PRIME for optimal NTT performance
              scale is fixed at 2^30 for ~30 bits of decimal precision
        """

        self.N = N
        self.seed = seed
        self.sigma = sigma
        
        # q é o primo NTT para garantir consistência e performance
        # Importado diretamente da classe NTT
        self.qs = NTT.DEFAULT_PRIME
        
        # Scale fixo em 2^20 para boa precisão em aplicações ML
        # Permite ~6 dígitos de precisão, ideal para gradientes (0.001 - 1.0)
        self.scale = self.DEFAULT_SCALE
        
        Polynomials.N = N

    def generate_error(self):
        poly_coeff = coordinate_wise_random_rounding(get_random_normal_polynomial(self.N, self.sigma, self.seed).coefficients)
        return Polynomials(poly_coeff)
    
    def generate_a(self):
        return get_random_uniform_polynomial(self.N, self.qs//2, self.seed)
    
    def generate_sk(self):
        seed = int(
            np.random.default_rng().integers(
                0, np.iinfo(np.uint32).max, dtype=np.uint32
            )
        )
        return get_random_uniform_polynomial(self.N, 1, seed)
    
    def generate_keys(self):
        sk = self.generate_sk()
        return sk

    def __str__(self):
        return f"LWESampler(N={self.N}, sigma={self.sigma}, level={self.level}) \nscale={self.scale} \np_scale={self.p_scale} \nqs={self.qs},"
