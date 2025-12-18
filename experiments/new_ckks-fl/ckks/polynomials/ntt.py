class NTT:
    """
    Number Theoretic Transform otimizada para multiplicação polinomial eficiente.
    Usa um único primo grande adequado para coeficientes de até 64 bits.
    """
    
    # Primos da forma p = k*2^n + 1 (adequados para NTT)
    # Selecionamos primos grandes o suficiente para suportar coeficientes CKKS
    PRIMES = {
        # Formato: tamanho_máximo_n: (primo, raiz_primitiva)
        256: (4179340454199820289, 3),      # 61 bits, suporta até N=2^56
        512: (4179340454199820289, 3),      # 61 bits, suporta até N=2^56
        1024: (4179340454199820289, 3),     # 61 bits, suporta até N=2^56
        2048: (4179340454199820289, 3),     # 61 bits, suporta até N=2^56
        4096: (4179340454199820289, 3),     # 61 bits, suporta até N=2^56
        8192: (4179340454199820289, 3),     # 61 bits, suporta até N=2^56
    }
    
    # Primo padrão para casos gerais
    DEFAULT_PRIME = 4179340454199820289  # 61 bits
    DEFAULT_ROOT = 3
    
    def __init__(self, n: int, p: int = None, root: int = None):
        """
        Inicializa NTT para polinômios de grau até n-1.
        
        Args:
            n: Tamanho da transformada (deve ser potência de 2)
            p: Primo para usar (None = auto-select)
            root: Raiz primitiva do primo (None = auto-select)
        """
        if not (n & (n - 1)) == 0:
            raise ValueError(f"n deve ser potência de 2, recebido: {n}")
        
        self.n = n
        
        # Seleciona primo adequado
        if p is None:
            if n in self.PRIMES:
                self.p, g = self.PRIMES[n]
            else:
                self.p, g = self.DEFAULT_PRIME, self.DEFAULT_ROOT
        else:
            self.p = p
            g = root if root is not None else self.DEFAULT_ROOT
        
        # Pré-calcula raízes da unidade
        self._precompute_roots(g)
    
    def _precompute_roots(self, g: int):
        """Pré-calcula raízes da unidade para este primo."""
        # w = g^((p-1)/n) é a raiz n-ésima primitiva da unidade mod p
        self.w = pow(g, (self.p - 1) // self.n, self.p)
        self.w_inv = pow(self.w, self.p - 2, self.p)  # Inverso modular
        self.n_inv = pow(self.n, self.p - 2, self.p)  # Inverso de n mod p
    
    def _ntt_transform(self, a: list[int], inverse: bool = False) -> list[int]:
        """
        Aplica NTT iterativa com bit-reversal.
        Otimizada para um único primo.
        
        Args:
            a: Coeficientes do polinômio
            inverse: Se True, aplica NTT inversa
            
        Returns:
            Transformada NTT dos coeficientes
        """
        n = len(a)
        if n == 1:
            return a.copy()
        
        a = a.copy()
        w_base = self.w_inv if inverse else self.w
        
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
        
        # Iterative NTT (Cooley-Tukey)
        length = 2
        while length <= n:
            half = length >> 1
            
            # Calcula raiz para este nível: w^(n/length)
            angle = n // length
            wlen = pow(w_base, angle, self.p)
            
            for start in range(0, n, length):
                w = 1
                for i in range(start, start + half):
                    j = i + half
                    
                    u = a[i]
                    v = (a[j] * w) % self.p
                    
                    a[i] = (u + v) % self.p
                    a[j] = (u - v) % self.p
                    
                    w = (w * wlen) % self.p
            
            length <<= 1
        
        # Se for inversa, divide por n
        if inverse:
            for i in range(n):
                a[i] = (a[i] * self.n_inv) % self.p
        
        return a
    
    def _centered_mod(self, x: int) -> int:
        """Converte x mod p para o intervalo centrado [-p/2, p/2]."""
        x = x % self.p
        if x > self.p // 2:
            x -= self.p
        return x
    
    def multiply(self, a: list[int], b: list[int]) -> list[int]:
        """
        Multiplica dois polinômios usando NTT otimizada.
        Complexidade: O(n log n)
        
        Args:
            a, b: Coeficientes dos polinômios
            
        Returns:
            Coeficientes do produto
        """
        # Determina tamanho necessário (próxima potência de 2)
        result_size = len(a) + len(b) - 1
        n = 1
        while n < result_size:
            n <<= 1
        
        # Ajusta NTT se necessário
        if n != self.n:
            self.__init__(n, self.p, self.DEFAULT_ROOT)
        
        # Padding com zeros
        a_padded = a + [0] * (n - len(a))
        b_padded = b + [0] * (n - len(b))
        
        # Reduz coeficientes mod p
        a_mod = [x % self.p for x in a_padded]
        b_mod = [x % self.p for x in b_padded]
        
        # NTT direta
        a_ntt = self._ntt_transform(a_mod, inverse=False)
        b_ntt = self._ntt_transform(b_mod, inverse=False)
        
        # Multiplicação ponto a ponto no domínio da frequência
        c_ntt = [(a_ntt[i] * b_ntt[i]) % self.p for i in range(n)]
        
        # NTT inversa
        c = self._ntt_transform(c_ntt, inverse=True)
        
        # Converte para módulo centrado e remove padding
        result = [self._centered_mod(c[i]) for i in range(result_size)]
        
        return result
