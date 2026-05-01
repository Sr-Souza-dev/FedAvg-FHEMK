import numpy as np
from scipy.signal import fftconvolve

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ---------- numba-accelerated helpers ----------

if HAS_NUMBA:
    @numba.njit
    def _mulmod(a, b, m):
        """Modular multiplication (a*b) % m without int64 overflow."""
        result = numba.int64(0)
        a = a % m
        while b > 0:
            if b & 1:
                result = (result + a) % m
            a = (a + a) % m
            b >>= 1
        return result

    @numba.njit
    def _reconstruct_mod_numba(hi_ring, lo_ring, shift, q, half_q, N):
        """Reconstruct hi*2^31 + lo, mod q, with centered reduction."""
        result = np.empty(N, dtype=np.int64)
        for i in range(N):
            prod = _mulmod(hi_ring[i] % q, shift, q)
            val = (prod + lo_ring[i] % q) % q
            if val > half_q:
                val -= q
            result[i] = val
        return result

    # Trigger JIT compilation at import time with small inputs
    _dummy_hi = np.zeros(2, dtype=np.int64)
    _dummy_lo = np.zeros(2, dtype=np.int64)
    _reconstruct_mod_numba(_dummy_hi, _dummy_lo, np.int64(1), np.int64(7), np.int64(3), 2)


def _reconstruct_mod_python(hi_ring, lo_ring, shift, q, half_q, N):
    """Pure Python fallback for reconstruction."""
    result = np.empty(N, dtype=np.int64)
    for i in range(N):
        val = (int(hi_ring[i]) * shift + int(lo_ring[i])) % q
        if val > half_q:
            val -= q
        result[i] = val
    return result


def _reconstruct_mod(hi_ring, lo_ring, shift, q, half_q, N):
    if HAS_NUMBA:
        return _reconstruct_mod_numba(hi_ring, lo_ring, np.int64(shift), np.int64(q), np.int64(half_q), N)
    return _reconstruct_mod_python(hi_ring, lo_ring, shift, q, half_q, N)


# ---------- FFT-based small-coefficient ring multiplication ----------

def fft_ring_mul_mod(a_np, b_small_np, N, q):
    """
    Multiply polynomial `a` (large coefficients) by `b` (small coefficients)
    in Z_q[x]/(x^N + 1) using FFT-based convolution.

    This avoids the NTT bottleneck by splitting `a` into two 31-bit halves
    and using scipy's FFT (C implementation) for the convolution.

    Requirements:
    - |b[i]| should be small enough that N * 2^31 * max|b[i]| < 2^52
      (float64 mantissa). For N=8192 and |b[i]| <= 128, this gives
      8192 * 2^31 * 128 = 2^50 < 2^52. Safe.
    - a[i] can be any int64 value.

    Returns: numpy int64 array of length N with centered mod q values.
    """
    half_q = q // 2

    # Ensure a is in [0, q) range
    a_pos = a_np % q

    # Split a into two 31-bit halves
    a_lo = (a_pos & 0x7FFFFFFF).astype(np.float64)  # lower 31 bits
    a_hi = (a_pos >> 31).astype(np.float64)          # upper ~30 bits
    b_f = b_small_np.astype(np.float64)

    # FFT-based convolution (O(N log N) in C)
    conv_lo = np.round(fftconvolve(a_lo, b_f)).astype(np.int64)
    conv_hi = np.round(fftconvolve(a_hi, b_f)).astype(np.int64)

    # Ring reduction: x^N ≡ -1 mod (x^N + 1)
    lo_ring = conv_lo[:N].copy()
    hi_ring = conv_hi[:N].copy()
    conv_len = len(conv_lo)
    if conv_len > N:
        excess = min(conv_len - N, N)
        lo_ring[:excess] -= conv_lo[N:N + excess]
        hi_ring[:excess] -= conv_hi[N:N + excess]

    # Reconstruct: result = hi * 2^31 + lo, mod q
    shift = pow(2, 31, q)  # = 2^31 since q > 2^31
    return _reconstruct_mod(hi_ring, lo_ring, shift, q, half_q, N)


# ---------- Pure Python NTT (fallback for general multiplication) ----------

def _ntt_core_python(a_list, twiddles_list, p, inverse, n_inv):
    """Pure Python NTT using native ints (fast for big-int multiplication)."""
    n = len(a_list)
    a = a_list

    # Bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Iterative NTT (Cooley-Tukey) with pre-computed twiddle factors
    level = 0
    length = 2
    while length <= n:
        half = length >> 1
        wlen = twiddles_list[level]
        for start in range(0, n, length):
            w = 1
            for i in range(start, start + half):
                j_idx = i + half
                u = a[i]
                v = a[j_idx] * w % p
                a[i] = (u + v) % p
                a[j_idx] = (u - v) % p
                w = w * wlen % p
        level += 1
        length <<= 1

    if inverse:
        for i in range(n):
            a[i] = a[i] * n_inv % p

    return a


# ---------- NTT class ----------

class NTT:
    """
    Number Theoretic Transform otimizada para multiplicação polinomial eficiente.
    """

    DEFAULT_PRIME = 4179340454199820289  # 61 bits
    DEFAULT_ROOT = 3

    _cache: dict[int, 'NTT'] = {}

    @classmethod
    def get_instance(cls, n: int) -> 'NTT':
        if n not in cls._cache:
            cls._cache[n] = NTT(n)
        return cls._cache[n]

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()

    def __init__(self, n: int, p: int = None, root: int = None):
        if not (n & (n - 1)) == 0:
            raise ValueError(f"n deve ser potência de 2, recebido: {n}")

        self.n = n
        self.p = p if p is not None else self.DEFAULT_PRIME
        g = root if root is not None else self.DEFAULT_ROOT

        self.w = pow(g, (self.p - 1) // self.n, self.p)
        self.w_inv = pow(self.w, self.p - 2, self.p)
        self.n_inv = pow(self.n, self.p - 2, self.p)

        self._fwd_twiddles = self._compute_twiddles(self.w)
        self._inv_twiddles = self._compute_twiddles(self.w_inv)

    def _compute_twiddles(self, w_base):
        twiddles = []
        length = 2
        while length <= self.n:
            angle = self.n // length
            wlen = pow(w_base, angle, self.p)
            twiddles.append(wlen)
            length <<= 1
        return twiddles

    def _ntt_transform(self, coeffs_list, inverse=False):
        twiddles = self._inv_twiddles if inverse else self._fwd_twiddles
        return _ntt_core_python(coeffs_list, twiddles, self.p, inverse, self.n_inv)

    def _centered_mod(self, x: int) -> int:
        x = x % self.p
        if x > self.p // 2:
            x -= self.p
        return x

    def multiply(self, a, b):
        """General polynomial multiplication using NTT. O(n log n)."""
        if not isinstance(a, np.ndarray):
            a = np.asarray(a, dtype=np.int64)
        if not isinstance(b, np.ndarray):
            b = np.asarray(b, dtype=np.int64)

        result_size = len(a) + len(b) - 1
        n = 1
        while n < result_size:
            n <<= 1

        if n != self.n:
            return NTT.get_instance(n).multiply(a, b)

        p = self.p

        # Convert to Python int lists for NTT
        a_list = [0] * n
        b_list = [0] * n
        for i in range(len(a)):
            a_list[i] = int(a[i]) % p
        for i in range(len(b)):
            b_list[i] = int(b[i]) % p

        a_ntt = self._ntt_transform(a_list, inverse=False)
        b_ntt = self._ntt_transform(b_list, inverse=False)

        c_ntt = [0] * n
        for i in range(n):
            c_ntt[i] = (a_ntt[i] * b_ntt[i]) % p

        c = self._ntt_transform(c_ntt, inverse=True)

        half_p = p // 2
        result = np.empty(result_size, dtype=np.int64)
        for i in range(result_size):
            val = c[i] % p
            if val > half_p:
                val -= p
            result[i] = val

        return result
