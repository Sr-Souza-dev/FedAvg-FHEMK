from __future__ import annotations

import numpy as np

from ckks.polynomials.main import Polynomials
from ckks.polynomials.ntt import NTT


def get_random_uniform_polynomial(N: int, bound: int, rng: np.random.Generator) -> Polynomials:
    """Generates a random polynomial with coefficients centered around 0."""
    return Polynomials(rng.integers(-bound, bound + 1, N))


def get_random_normal_polynomial(N: int, bound: float, rng: np.random.Generator) -> Polynomials:
    """Generates a random polynomial drawn from N(0, bound^2)."""
    return Polynomials(rng.normal(0, bound, N))


def coordinate_wise_random_rounding(coordinates, rng: np.random.Generator):
    """Rounds coordinates randomly using stochastic rounding."""
    coords = np.asarray(coordinates, dtype=np.float64)
    fractional = coords - np.floor(coords)
    samples = rng.random(len(coords))
    rounded = np.where(samples < fractional, np.ceil(coords), np.floor(coords))
    return rounded.astype(int).tolist()


class Sampler:
    # Scale padrão para codificação
    DEFAULT_SCALE = 2**16

    def __init__(self, N: int, sigma: float, seed: float = 42):
        """
        :param N: Polynomial degree (number of coefficients/slots)
        :param sigma: Standard deviation for noise generation
        :param seed: Random seed for reproducibility
        """

        self.N = N
        self.seed = seed
        self.sigma = sigma

        # q é o primo NTT para garantir consistência e performance
        self.qs = NTT.DEFAULT_PRIME

        # Scale fixo em 2^16 para boa precisão em aplicações ML
        self.scale = self.DEFAULT_SCALE

        Polynomials.N = N
        self._rng_uniform = np.random.default_rng(seed)
        self._rng_error = np.random.default_rng(seed + 1)

    def generate_error(self):
        gaussian = get_random_normal_polynomial(self.N, self.sigma, self._rng_error)
        poly_coeff = coordinate_wise_random_rounding(gaussian.coefficients, self._rng_error)
        return Polynomials(poly_coeff)

    def generate_a(self):
        return get_random_uniform_polynomial(self.N, self.qs // 2, self._rng_uniform)

    def generate_sk(self):
        seed = int(
            np.random.default_rng().integers(
                0,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
        )
        return get_random_uniform_polynomial(self.N, 1, np.random.default_rng(seed))

    def generate_keys(self):
        sk = self.generate_sk()
        return sk

    def __str__(self):
        return (
            f"LWESampler(N={self.N}, sigma={self.sigma}) "
            f"\nscale={self.scale} \nqs={self.qs}"
        )
