from ckks.polynomials.ntt import NTT
import numpy as np


class Polynomials:
    N = 2**8  # Considerando o polinomio de redução x^N + 1

    def __init__(self, coefficients: list[int]):
        coefficients = list(coefficients)
        self.coefficients = Polynomials._remove_zeros(coefficients)
        self.degree = len(self.coefficients) - 1
    
    @classmethod
    def _remove_zeros(cls, c: list[int]):
        # Remove zeros da cauda
        while len(c) > 1 and c[-1] == 0:
            c.pop()
        return cls.normalize_python_types(c)

    @staticmethod
    def normalize_python_types(arr):
        """
        Converte todos os elementos de um array/lista para tipos
        primitivos do Python (int, float, bool).
        """
        result = []

        for x in arr:
            # Se for tipo numpy (int32, int64, float32, float64, etc.)
            if isinstance(x, (np.generic,)):
                result.append(x.item())  # converte para tipo Python nativo
            else:
                result.append(x)

        return result


    def __str__(self):
        termos = []
        for i in reversed(range(len(self.coefficients))):
            c = self.coefficients[i]
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
        return self.coefficients == other.coefficients
    
    def __neg__(self):
        return Polynomials([-c for c in self.coefficients])


    def __hash__(self):
        return hash(self.coefficients)  # usa o hash da tupla de coefficients
    
    def __call__(self, x: int) -> int:
        """Avalia o polinômio no ponto x."""
        resultado = 0
        for coef in reversed(self.coefficients):
            resultado = resultado * x + coef
        return resultado


    def __add__(self, other):
        if isinstance(other, Polynomials):
            max_len = max(len(self.coefficients), len(other.coefficients))
            result = [0] * max_len
            for i in range(max_len):
                a = self.coefficients[i] if i < len(self.coefficients) else 0
                b = other.coefficients[i] if i < len(other.coefficients) else 0
                result[i] = a + b
            return Polynomials(result)
        elif isinstance(other, int):
            result = self.coefficients.copy()
            result[0] += other
            return Polynomials(result)
        else:
            return NotImplemented
        
    def __radd__(self, other):
        if isinstance(other, Polynomials):
            return self + other
        elif isinstance(other, int):
            result = self.coefficients.copy()
            result[0] += other
            return Polynomials(result)
        else:
            return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, int):
            result = self.coefficients.copy()
            result[0] -= other
            return Polynomials(result)
        elif isinstance(other, Polynomials):
            max_len = max(len(self.coefficients), len(other.coefficients))
            result = [0] * max_len
            for i in range(max_len):
                a = self.coefficients[i] if i < len(self.coefficients) else 0
                b = other.coefficients[i] if i < len(other.coefficients) else 0
                result[i] = a - b
            return Polynomials(result)
        else:
            return NotImplemented
        
    def __rsub__(self, other):
        if isinstance(other, int):
            result = self.coefficients.copy()
            result[0] = other - result[0]
            return Polynomials(result)
        elif isinstance(other, Polynomials):
            return other - self
        else:
            return NotImplemented
        
    
    def __mul1__(self, other):
        if isinstance(other, int):
            result = [a * other for a in self.coefficients]
            return Polynomials(result)
        elif isinstance(other, Polynomials):
            result = [0] * (self.degree + other.degree + 1)
            for i, a in enumerate(self.coefficients):
                for j, b in enumerate(other.coefficients):
                    result[i + j] += a * b
            return Polynomials(result)
        else:
            return NotImplemented
    

    def __mul__(self, other):        
        """
        Multiplicação rápida usando NTT otimizada.
        Complexidade: O(n log n) em vez de O(n^2)
        
        Usa um único primo grande (61 bits) adequado para coeficientes de até 64 bits.
        """
        if isinstance(other, int):
            result = [a * other for a in self.coefficients]
            return Polynomials(result)
        elif isinstance(other, Polynomials):
            # Determina tamanho necessário
            result_size = len(self.coefficients) + len(other.coefficients) - 1
            n = 1
            while n < result_size:
                n <<= 1
            
            # Inicializa NTT com tamanho adequado
            # Usa o primo padrão que suporta coeficientes de até 61 bits
            ntt = NTT(n)
            
            # Multiplica usando NTT otimizada
            result = ntt.multiply(self.coefficients, other.coefficients)
            
            return Polynomials(result)
        else:
            return NotImplemented
        

    def __rmul__(self, other):
        return self * other  # permite constante * polinômio

    def __truediv__(self, constant: int | float):
        """Divisão que permite coeficientes reais."""
        if not isinstance(constant, (int, float)):
            raise TypeError("Divisão de polinômio apenas por escalar numérico (int ou float).")
        if constant == 0:
            raise ZeroDivisionError("Divisão por zero.")
        result = [a / constant for a in self.coefficients]
        return Polynomials(result)

    def __floordiv__(self, constant: int):
        """Divisão inteira por escalar, exige que todos os coeficientes sejam divisíveis."""
        if not isinstance(constant, int):
            raise TypeError("Divisão inteira de polinômio apenas por escalar inteiro.")
        if constant == 0:
            raise ZeroDivisionError("Divisão por zero.")
        result = [a // constant for a in self.coefficients]
        return Polynomials(result)
 
    @classmethod
    def centered_mod(clc, x: int, q: int) -> int:
        r = x % q
        if r > q // 2:
            r -= q
        return r

    
    def __mod__(self, q: int):
        """Reduz o polinômio em R_q = Z_q[x]/(x^N + 1) ->> Considera o modulo centrado [-q/2, q/2]"""
        res = [0] * Polynomials.N
        for i, a in enumerate(self.coefficients):
            a_mod = int(a) % int(q)
            # print(f"{a} % {q} = {a_mod}")
            if i < Polynomials.N:
                res[i] = Polynomials.centered_mod((res[i] + a_mod), q)
            else:
                idx = i % Polynomials.N
                sinal = -1 if ((i // Polynomials.N) % 2 == 1) else 1
                res[idx] = Polynomials.centered_mod((res[idx] + sinal * a_mod), q)
        return Polynomials(res)