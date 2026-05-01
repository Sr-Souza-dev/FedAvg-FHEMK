from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from ckks.cryptogram.main import Cryptogram
from ckks.encoder.main import Encoder
from ckks.polynomials.main import Polynomials
from ckks.polynomials.ntt import NTT
from ckks.sampler.main import Sampler
from utils.files import delete_directory_files, load_numbers_file, write_numbers_to_file


class CKKS:
    @property
    def NTT_PRIME(self):
        """Retorna o primo NTT usado (definido na classe NTT)."""
        return NTT.DEFAULT_PRIME

    def __init__(
        self,
        N: int,
        sigma: float,
        model_size: int,
        fix_a: bool = False,
        structure: Sequence[Tuple[Tuple[int, ...], np.dtype]] | None = None,
    ):
        """
        Inicializa o esquema CKKS com parâmetros otimizados.

        :param N: Polynomial degree (number of coefficients/slots)
        :param sigma: Standard deviation for noise generation
        :param model_size: Size of the model to encrypt
        :param fix_a: Whether to use a fixed 'a' polynomial for encryption

        Note: q is automatically set to NTT.DEFAULT_PRIME (61 bits) for optimal performance.
              scale is fixed at 2^30 for ~30 bits of decimal precision.
        """
        Polynomials.N = N
        Cryptogram.fix_a = fix_a
        self.params = Sampler(N, sigma)
        self.encoder = Encoder(self.params.N, self.params.scale)
        self.model_size = model_size
        self.model_structure = tuple(structure or [])
        self._key_cache: dict[str, Polynomials] = {}
        self._fixed_a_cache: Polynomials | None = None

    def get_vector_size(self) -> int:
        """Returns the encoder slot count."""
        return self.encoder.vec_size

    def get_cryptogram_quantity(self) -> int:
        if self.model_size <= self.get_vector_size():
            return 1
        return math.ceil(self.model_size / self.get_vector_size())

    def generate_keys(self):
        return self.params.generate_keys()

    def save_key(self, sk: Polynomials, prefix: str = "def"):
        self._key_cache[prefix] = sk
        write_numbers_to_file(
            open_mode="w",
            filename=prefix + "_sk",
            values=[sk.coefficients.tolist()],
            basePath="keys/",
            type=".dat",
        )

    def agg_key(self, sk: Polynomials, prefix: str = "server"):
        cached = self._key_cache.get(prefix)
        if cached is None:
            sk_coefficients = load_numbers_file(prefix + "_sk", basePath="keys/", type=".dat")
            cached = Polynomials(sk_coefficients) if len(sk_coefficients) != 0 else Polynomials([0])
        aggregated = cached + sk
        self.save_key(prefix=prefix, sk=aggregated)

    def load_key(self, prefix: str = "def"):
        if prefix in self._key_cache:
            return self._key_cache[prefix]
        sk_coefficients = load_numbers_file(prefix + "_sk", basePath="keys/", type=".dat")
        if len(sk_coefficients) == 0:
            sk = self.generate_keys()
            self.agg_key(sk=sk)
            self.save_key(prefix=prefix, sk=sk)
        else:
            sk = Polynomials(sk_coefficients)
            self._key_cache[prefix] = sk
        return sk

    @staticmethod
    def zero_polynomial() -> Polynomials:
        return Polynomials([0])

    def encrypt(self, sk: Polynomials, plaintext: np.ndarray) -> Cryptogram:
        """Encrypts a plaintext polynomial using the secret key."""
        c = self.encrypt_phase1(sk)
        return self.encrypt_phase2(c, plaintext)

    def gen_new_fixed_a(self):
        delete_directory_files(dir="public/")
        a = self.params.generate_a()
        write_numbers_to_file(
            open_mode="w",
            filename="fixed_a",
            values=[a.coefficients.tolist()],
            basePath="public/",
            type=".dat",
        )
        self._fixed_a_cache = a
        return a

    def encrypt_phase1(self, sk: Polynomials) -> Cryptogram:
        if Cryptogram.fix_a:
            if self._fixed_a_cache is not None:
                a = self._fixed_a_cache
            else:
                a_coefficients = load_numbers_file("fixed_a", basePath="public/", type=".dat")
                if len(a_coefficients) == 0:
                    a = self.gen_new_fixed_a()
                else:
                    a = Polynomials([int(val) for val in a_coefficients])
                    self._fixed_a_cache = a
        else:
            a = self.params.generate_a()
        e = self.params.generate_error()

        # Use FFT-based ring multiplication (sk has small coefficients)
        neg_a = -a
        c0 = (neg_a.ring_mul_small_mod(sk, self.params.qs) + e) % self.params.qs
        c1 = a
        return Cryptogram(c0, c1, self.params.qs)

    def encrypt_phase2(self, c: Cryptogram, plaintext: np.ndarray) -> Cryptogram:
        m = self.encoder.encode(plaintext)
        c0 = (c.c0 + m) % self.params.qs
        return Cryptogram(c0, c.c1, self.params.qs)

    def encrypt_batch(self, sk: Polynomials, plaintext: np.ndarray) -> list[Cryptogram]:
        """Divides the plaintext into batches and encrypts each batch using the secret key."""
        cryptograms = self.encrypt_batch_phase1(sk)
        return self.encrypt_batch_phase2(cryptograms, plaintext)

    def encrypt_batch_phase1(self, sk: Polynomials) -> list[Cryptogram]:
        quantity = self.get_cryptogram_quantity()
        cryptograms = []
        for _ in range(quantity):
            cryptograms.append(self.encrypt_phase1(sk))
        return cryptograms

    def encrypt_batch_phase2(self, cryptograms: list[Cryptogram], plaintext: np.ndarray):
        batch_size = self.get_vector_size()
        batches = [plaintext[i : min(i + batch_size, len(plaintext))] for i in range(0, len(plaintext), batch_size)]
        ciphertexts = []

        for c, m in zip(cryptograms, batches):
            ciphertext = self.encrypt_phase2(c, m)
            ciphertexts.append(ciphertext)

        return ciphertexts

    def decrypt(self, sk: Polynomials, ciphertext: Cryptogram) -> np.ndarray:
        """Decrypts a ciphertext polynomial using the secret key."""
        # Use FFT-based ring multiplication (sk has small coefficients)
        sk_times_c1 = ciphertext.c1.ring_mul_small_mod(sk, ciphertext.q)
        decrypted = (ciphertext.c0 + sk_times_c1) % ciphertext.q
        decrypted = self.encoder.decode(decrypted)
        return decrypted

    def decrypt_batch(self, sk: Polynomials, ciphertexts: list[Cryptogram]) -> np.ndarray:
        """Decrypts a list of ciphertexts using the secret key."""

        decrypted = []

        for ciphertext in ciphertexts:
            decrypted.append(self.decrypt(sk, ciphertext))

        return np.concatenate(decrypted)[: self.model_size]

    def _pack_cryptogram(self, cryptogram: Cryptogram) -> np.ndarray:
        c0 = np.asarray(cryptogram.c0.coefficients, dtype=np.int64)
        c1 = np.asarray(cryptogram.c1.coefficients, dtype=np.int64)
        max_len = max(len(c0), len(c1))
        packed = np.zeros((2, max_len), dtype=np.int64)
        packed[0, : len(c0)] = c0
        packed[1, : len(c1)] = c1
        return packed

    def serialize_ciphertexts(self, cryptograms: list[Cryptogram]) -> tuple[list[np.ndarray], int]:
        packed: list[np.ndarray] = []
        total_bytes = 0
        for cryptogram in cryptograms:
            blob = self._pack_cryptogram(cryptogram)
            total_bytes += blob.nbytes
            packed.append(blob)
        return packed, total_bytes

    def extract_vector(self, cryptograms: list[Cryptogram]) -> list[np.ndarray]:
        packed, _ = self.serialize_ciphertexts(cryptograms)
        return packed

    def construct_cryptograms(self, data: list[np.ndarray]) -> list[Cryptogram]:
        cryptograms: list[Cryptogram] = []
        for block in data:
            arr = np.asarray(block, dtype=np.int64)
            if arr.ndim != 2 or arr.shape[0] != 2:
                raise ValueError("Unexpected ciphertext shape when reconstructing cryptograms.")
            c0 = Polynomials(arr[0].tolist())
            c1 = Polynomials(arr[1].tolist())
            cryptograms.append(Cryptogram(c0, c1, self.params.qs))
        return cryptograms
