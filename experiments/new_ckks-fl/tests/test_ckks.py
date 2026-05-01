from ckks.main import CKKS
import numpy as np

from tests.utils import generate_message_error

MAX_TESTS = 100

def get_message_wrapper(params:dict, result:dict, function:str) -> str:
    return generate_message_error(params, result, "CKKS", function)

def get_random_vector(N:int) -> np.array:
    """
        Generate a random vector of size N
    """
    return np.random.random(N).astype(np.float64)

def is_vector_iquals(v1:np.ndarray, v2:np.ndarray) -> bool:
    """
        Check if two vectors are equal
    """
    if len(v1) != len(v2):
        return False
    return np.allclose(v1, v2, atol=1e-1)

class TestCKKS:
    def test_save_load_keys(self):
        for _ in range(MAX_TESTS):
            N = 2**np.random.randint(2, 8)
            ckks =  CKKS(
                N=N,
                sigma=3,
                model_size=N
            )

            sk= ckks.generate_keys()
            ckks.save_key(sk, prefix="test")
            sk_loaded = ckks.load_key(prefix="test")

            assert is_vector_iquals(sk.coefficients, sk_loaded.coefficients), get_message_wrapper(
                {"N": N},
                {"sk": sk.coefficients, "sk_loaded": sk_loaded.coefficients},
                "test_save_load_keys_sk"
            )

    def test_encryption_decryption_sk(self):
        for _ in range(MAX_TESTS):
            N = 2**np.random.randint(2, 3)
            ckks =  CKKS(
                N=N,
                sigma=3,
                model_size=N
            )

            plaintext = get_random_vector(N)
            sk = ckks.generate_keys()
            ciphertext = ckks.encrypt(sk, plaintext)
            decrypted = ckks.decrypt(sk, ciphertext)

            assert is_vector_iquals(plaintext, decrypted), get_message_wrapper(
                {"N": N, "plaintext": plaintext},
                {"ciphertext": ciphertext, "decrypted": decrypted},
                "test_encryption_decryption_sk"
            )

    def test_batch_encryption_decryption_sk(self):
        for _ in range(MAX_TESTS):
            N = 2**np.random.randint(2, 8)
            factor = np.random.randint(1, 10, 1)[0]
            ckks =  CKKS(
                N=N,
                sigma=3, 
                model_size=N*factor
            )

            plaintext = get_random_vector(N*factor)
            sk = ckks.generate_keys()
            ciphertext = ckks.encrypt_batch(sk, plaintext)
            decrypted = ckks.decrypt_batch(sk, ciphertext)

            assert is_vector_iquals(plaintext, decrypted), get_message_wrapper(
                {"N": N, "Factor": factor, "plaintext": plaintext},
                {"ciphertext": ciphertext, "decrypted": decrypted},
                "test_batch_encryption_decryption_sk"
            )

    def test_sum_encryption_decryption_sk(self):
        for _ in range(MAX_TESTS):
            N = 2**np.random.randint(2, 8)
            ckks =  CKKS(
                N=N,
                sigma=3, 
                model_size=N,
            )

            plaintext1 = get_random_vector(N)
            plaintext2 = get_random_vector(N)
            sk = ckks.generate_keys()
            ciphertext1 = ckks.encrypt(sk, plaintext1)
            ciphertext2 = ckks.encrypt(sk, plaintext2)
            decrypted = ckks.decrypt(sk, ciphertext1 + ciphertext2)

            assert is_vector_iquals(plaintext1 + plaintext2, decrypted), get_message_wrapper(
                {"N": N, "plaintext1": plaintext1, "plaintext2": plaintext2},
                {"ciphertext1": ciphertext1, "ciphertext2": ciphertext2, "decrypted": decrypted},
                "test_sum_encryption_decryption_sk"
           )

    def test_agg_key_operations(self):
        for _ in range(MAX_TESTS):
            N = 2**1
            model_size = 2
            ckks =  CKKS(
                N=N,
                sigma=3, 
                model_size=model_size,
                fix_a=True
            )

            plaintext1 = get_random_vector(model_size)
            plaintext2 = get_random_vector(model_size)

            sk1 = ckks.generate_keys()
            sk2 = ckks.generate_keys()

            ciphertext1 = ckks.encrypt_batch(sk1, plaintext1)
            ciphertext2 = ckks.encrypt_batch(sk2, plaintext2)

            c_sum = [c1+c2 for c1, c2 in zip(ciphertext1, ciphertext2)]
            decrypted = ckks.decrypt_batch(sk1+sk2, c_sum)

            assert is_vector_iquals(plaintext1 + plaintext2, decrypted), get_message_wrapper(
                {"N": N, "plaintext1": plaintext1, "plaintext2": plaintext2},
                {"ciphertext1": ciphertext1, "ciphertext2": ciphertext2, "decrypted": decrypted, "expected":plaintext1+plaintext2},
                "test_agg_key_operations"
            )
    
    # def test_mult_constant_encryption_decryption_sk(self):
    #     for _ in range(MAX_TESTS):
    #         N = 2**np.random.randint(2, 8)
    #         ckks =  CKKS(
    #             N=N, 
    #             int_precision= 32, 
    #             decimal_precision= 12, 
    #             sigma=3, 
    #         )

    #         c_var = int(np.random.randint(-10, 10, 1)[0])
    #         plaintext1 = get_random_vector(N//2)
    #         sk, pk, rlk = ckks.generate_keys()
    #         c = ckks.encrypt_sk(sk, plaintext1)
    #         decrypted = ckks.decrypt(sk, c * c_var)
    #         np_res = plaintext1 * c_var

    #         assert is_vector_iquals(np_res, decrypted), get_message_wrapper(
    #             {"N": N, "Const_Var": c_var, "plaintext1": plaintext1, "np_res": np_res},
    #             {"ciphertext1": c, "decrypted": decrypted},
    #             "test_mult_constant_encryption_decryption_sk"
    #         )