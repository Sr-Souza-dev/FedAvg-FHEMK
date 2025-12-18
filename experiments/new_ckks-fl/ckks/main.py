from ckks.polynomials.main import Polynomials
from ckks.encoder.main import Encoder
from ckks.sampler.main import Sampler
from ckks.cryptogram.main import Cryptogram
from ckks.polynomials.ntt import NTT
import numpy as np
import math
from utils.files import write_numbers_to_file, load_numbers_file, delete_directory_files

class CKKS:
    @property
    def NTT_PRIME(self):
        """Retorna o primo NTT usado (definido na classe NTT)."""
        return NTT.DEFAULT_PRIME
    
    def __init__(self, N: int, sigma: float, model_size: int, fix_a: bool = False):
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


    def get_vector_size(self) -> int:
        """
        Returns the size of the vector.
        """
        return self.encoder.vec_size
    
    def get_cryptogram_quantity(self) -> int:
        if(self.model_size <= self.get_vector_size()):
            return 1
        return  math.ceil(self.model_size / self.get_vector_size())
    

    def generate_keys(self):
        return self.params.generate_keys()


    def save_key(self, sk: Polynomials, prefix: str = "def"):
        write_numbers_to_file(
            open_mode="w",
            filename=prefix+"_sk",
            values=[sk.coefficients],
            basePath="keys/",
            type=".dat"
        )
    
    def agg_key(self, sk: Polynomials, prefix:str = "server"):
        sk_coefficients = load_numbers_file(prefix+"_sk", basePath="keys/", type=".dat")
        if(len(sk_coefficients) == 0):
            self.save_key(prefix=prefix, sk=sk)
        else:
            sk = Polynomials(sk_coefficients) + sk
            self.save_key(prefix=prefix, sk=sk)

    def load_key(self, prefix: str = "def"):
        sk_coefficients = load_numbers_file(prefix+"_sk", basePath="keys/", type=".dat")
        if(len(sk_coefficients) == 0):
            sk = self.generate_keys()
            self.agg_key(sk=sk)
            self.save_key(prefix=prefix, sk=sk)
        else:
            sk = Polynomials(sk_coefficients)
        return sk
    
    def encrypt(self, sk: Polynomials, plaintext: np.ndarray) -> Cryptogram:
        """
        Encrypts a plaintext polynomial using the secret key.
        """
        c = self.encrypt_phase1(sk)
        return self.encrypt_phase2(c, plaintext)
    
    def gen_new_fixed_a(self):
        delete_directory_files(dir="public/")
        a = self.params.generate_a()
        write_numbers_to_file(
            open_mode="w",
            filename="fixed_a",
            values=[a.coefficients],
            basePath="public/",
            type=".dat"
        )
        return a
    

    def encrypt_phase1(self, sk: Polynomials) -> Cryptogram:
        if Cryptogram.fix_a:
            a_coefficients = [int(val) for val in load_numbers_file("fixed_a", basePath="public/", type=".dat")]
            if(len(a_coefficients) == 0):
                a = self.gen_new_fixed_a()
            else:
                a = Polynomials(a_coefficients)
        else:
            a = self.params.generate_a()
        e = self.params.generate_error()
        
        c0 = ((-a*sk) + e) % self.params.qs
        c1 = a
        return Cryptogram(c0, c1, self.params.qs)
    

    def encrypt_phase2(self, c: Cryptogram, plaintext: np.ndarray) -> Cryptogram:
        m = self.encoder.encode(plaintext)

        c0 = (c.c0 + m) % self.params.qs
        return Cryptogram(c0, c.c1, self.params.qs)


    def encrypt_batch(self, sk: Polynomials, plaintext: np.ndarray) -> list[Cryptogram]:
        """
            Divides the plaintext into batches and encrypts each batch using the secret key.
        """
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
        batches = [plaintext[i:min(i + batch_size, len(plaintext))] for i in range(0, len(plaintext), batch_size)]
        ciphertexts = []

        for c, m in zip(cryptograms, batches):
            ciphertext = self.encrypt_phase2(c, m)
            ciphertexts.append(ciphertext)
        
        return ciphertexts

    
    def decrypt(self, sk: Polynomials, ciphertext: Cryptogram) -> np.ndarray:
        """
        Decrypts a ciphertext polynomial using the secret key.
        """
        decrypted = (ciphertext.c0 + (sk*ciphertext.c1)) % ciphertext.q
        decrypted = self.encoder.decode(decrypted)
        return decrypted
    

    def decrypt_batch(self, sk: Polynomials, ciphertexts: list[Cryptogram]) -> np.ndarray:
        """
            Decrypts a list of ciphertexts using the secret key.
        """

        decrypted = []

        for ciphertext in ciphertexts:
            decrypted.append(self.decrypt(sk, ciphertext))       

        return np.concatenate(decrypted)[:self.model_size]
    
    def extract_vector(self, cryptograms: list[Cryptogram]) -> list[list[list]]: 
        return [[c.c0.coefficients, c.c1.coefficients] for c in cryptograms]
    
    def construct_cryptograms(self, data: list[list[list]]) -> list[Cryptogram]:
        return [Cryptogram(Polynomials(c[0]), Polynomials(c[1]), self.params.qs) for c in data]