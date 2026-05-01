from ckks.polynomials.ntt import NTT, fft_ring_mul_mod
import numpy as np


class Polynomials:
    N = 2**8  # Considerando o polinomio de redução x^N + 1

    def __init__(self, coefficients):
        if isinstance(coefficients, np.ndarray):
            if coefficients.dtype in (np.int64, np.float64):
                arr = coefficients
            else:
                arr = coefficients.astype(np.int64)
        else:
            arr = np.asarray(coefficients, dtype=np.int64)
        # Trim trailing zeros (keep at least 1 element)
        idx = len(arr) - 1
        while idx > 0 and arr[idx] == 0:
            idx -= 1
        self.coefficients = arr[:idx + 1]

    @property
    def degree(self):
        """Degree of the polynomial (index of last non-zero coefficient)."""
        c = self.coefficients
        idx = len(c) - 1
        while idx > 0 and c[idx] == 0:
            idx -= 1
        return idx

    def _trimmed_coefficients(self):
        """Return coefficients with trailing zeros removed (for display/comparison only)."""
        c = self.coefficients
        idx = len(c) - 1
        while idx > 0 and c[idx] == 0:
            idx -= 1
        return c[:idx + 1]

    def __str__(self):
        coeffs = self._trimmed_coefficients()
        termos = []
        for i in reversed(range(len(coeffs))):
            c = int(coeffs[i])
            if c == 0:
                continue
            termo = f"{'' if c == 1 and i != 0 else '-' if c == -1 and i != 0 else c}"
            if i == 1:
                termo += "x"
            elif i != 0:
                termo += f"x^{i}"
            termos.append(termo)
        return " + ".join(termos).replace("+ -", "- ")

    def __eq__(self, other):
        a = self._trimmed_coefficients()
        b = other._trimmed_coefficients()
        return np.array_equal(a, b)

    def __neg__(self):
        return Polynomials(-self.coefficients)

    def __hash__(self):
        return hash(tuple(self.coefficients.tolist()))

    def __call__(self, x: int) -> int:
        """Avalia o polinômio no ponto x."""
        resultado = 0
        for coef in reversed(self.coefficients):
            resultado = resultado * x + int(coef)
        return resultado

    def __add__(self, other):
        if isinstance(other, Polynomials):
            a = self.coefficients
            b = other.coefficients
            if len(a) >= len(b):
                result = a.copy()
                result[:len(b)] += b
            else:
                result = b.copy()
                result[:len(a)] += a
            return Polynomials(result)
        elif isinstance(other, (int, np.integer)):
            result = self.coefficients.copy()
            result[0] += other
            return Polynomials(result)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, np.integer)):
            result = self.coefficients.copy()
            result[0] += other
            return Polynomials(result)
        elif isinstance(other, Polynomials):
            return self + other
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, np.integer)):
            result = self.coefficients.copy()
            result[0] -= other
            return Polynomials(result)
        elif isinstance(other, Polynomials):
            a = self.coefficients
            b = other.coefficients
            max_len = max(len(a), len(b))
            result = np.zeros(max_len, dtype=np.int64)
            result[:len(a)] += a
            result[:len(b)] -= b
            return Polynomials(result)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, np.integer)):
            result = -self.coefficients.copy()
            result[0] += other
            return Polynomials(result)
        elif isinstance(other, Polynomials):
            return other - self
        else:
            return NotImplemented

    def __mul__(self, other):
        """
        Multiplicação rápida usando NTT otimizada.
        Complexidade: O(n log n) em vez de O(n^2)
        """
        if isinstance(other, (int, np.integer)):
            return Polynomials(self.coefficients * int(other))
        elif isinstance(other, Polynomials):
            # Determina tamanho necessário
            result_size = len(self.coefficients) + len(other.coefficients) - 1
            n = 1
            while n < result_size:
                n <<= 1

            ntt = NTT.get_instance(n)
            result = ntt.multiply(self.coefficients, other.coefficients)
            return Polynomials(result)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other  # permite constante * polinômio

    def __truediv__(self, constant):
        """Divisão que permite coeficientes reais."""
        if not isinstance(constant, (int, float, np.integer, np.floating)):
            raise TypeError("Divisão de polinômio apenas por escalar numérico (int ou float).")
        if constant == 0:
            raise ZeroDivisionError("Divisão por zero.")
        result = self.coefficients / constant
        return Polynomials(result)

    def __floordiv__(self, constant: int):
        """Divisão inteira por escalar, exige que todos os coeficientes sejam divisíveis."""
        if not isinstance(constant, (int, np.integer)):
            raise TypeError("Divisão inteira de polinômio apenas por escalar inteiro.")
        if constant == 0:
            raise ZeroDivisionError("Divisão por zero.")
        return Polynomials(self.coefficients // constant)

    def ring_mul_small_mod(self, small_poly: 'Polynomials', q: int) -> 'Polynomials':
        """
        Fast ring multiplication in Z_q[x]/(x^N+1) when `small_poly` has
        small coefficients (e.g., secret key with coeffs in {-1,0,1}).
        Uses FFT-based convolution — ~50x faster than NTT for this case.
        """
        N = Polynomials.N
        a = self.coefficients
        b = small_poly.coefficients
        # Pad to N
        if len(a) < N:
            a = np.pad(a, (0, N - len(a)))
        if len(b) < N:
            b = np.pad(b, (0, N - len(b)))
        return Polynomials(fft_ring_mul_mod(a[:N], b[:N], N, q))

    @staticmethod
    def centered_mod(x: int, q: int) -> int:
        r = x % q
        if r > q // 2:
            r -= q
        return r

    def __mod__(self, q: int):
        """Reduz o polinômio em R_q = Z_q[x]/(x^N + 1) ->> Considera o modulo centrado [-q/2, q/2]"""
        N = Polynomials.N
        coeffs = self.coefficients
        n_coeffs = len(coeffs)

        # Use Python ints for mod to avoid int64 overflow with large q
        q_int = int(q)
        half_q = q_int // 2

        if n_coeffs <= N:
            # Simple case: no ring reduction needed, just centered mod
            result = np.empty(N, dtype=np.int64)
            # Mod with Python ints to handle large values
            for i in range(min(n_coeffs, N)):
                val = int(coeffs[i]) % q_int
                if val > half_q:
                    val -= q_int
                result[i] = val
            for i in range(n_coeffs, N):
                result[i] = 0
            return Polynomials(result)

        # Ring reduction: x^N ≡ -1 (mod x^N + 1)
        # Accumulate blocks of N with alternating signs
        result_py = [0] * N
        for i in range(n_coeffs):
            block_idx = i // N
            pos = i % N
            sign = -1 if (block_idx % 2 == 1) else 1
            result_py[pos] += sign * int(coeffs[i])

        # Apply centered mod
        result = np.empty(N, dtype=np.int64)
        for i in range(N):
            val = result_py[i] % q_int
            if val > half_q:
                val -= q_int
            result[i] = val
        return Polynomials(result)
