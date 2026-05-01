from ckks.polynomials.main import Polynomials
from numpy.polynomial import Polynomial
from tests.utils import generate_message_error
import numpy as np
import random

MAX_TESTS = 100

def remove_trailing_zeros(vec):
    i = len(vec) - 1
    while i >= 0 and vec[i] == 0:
        i -= 1
    return vec[:i+1]

def get_random_polynomial(coef_t = 2**2) -> tuple[Polynomials, Polynomial]:
    """Gera um polinômio aleatório de grau `degree` com coeficientes entre -10 e 10."""
    coefficients = [random.randint(-coef_t, coef_t) for _ in range(random.randint(2**2, 2**3))]
    coefficients = remove_trailing_zeros(coefficients)
    if not coefficients:
        coefficients = [random.randint(1, coef_t) or 1]
    return Polynomials(coefficients), Polynomial(coefficients)

def compary_array(a, b):
    """Compara dois arrays de coeficientes."""
    if len(a) != len(b):
        print("Diff from len: ",len(a), len(b))
        return False
    for i in range(len(a)):
        if np.float32(a[i]) != np.float32(b[i]):
            print("Diff from value: ", np.float32(a[i]), np.float32(b[i]))
            return False
    return True

def iquals(a, b):

    """Verifica se dois polinômios são iguais."""
    if isinstance(a, Polynomials) and isinstance(b, Polynomial):
        return a.degree == b.degree() and compary_array(a.coefficients, b.coef)
    elif isinstance(a, Polynomial) and isinstance(b, Polynomials):
        return a.degree() == b.degree and compary_array(a.coef, b.coefficients)
    print("another type")
    return False

def get_mod_polynomial():
    degree = random.randint(7, 10)
    coefficients = [0] * (degree+1)
    coefficients[0] = 1
    coefficients[-1] = 1
    Polynomials.N = degree

    return Polynomial(coefficients)

def get_message_wrapper(params:dict, result:dict, function:str) -> str:
    return generate_message_error(params, result, "Polynomials", function)

Polynomials.N = 7
class TestPolynomials:
    def test_declaration(self):
        for _ in range(MAX_TESTS):
            poly, np_poly = get_random_polynomial()

            assert iquals(poly, np_poly), get_message_wrapper(
                {"poly": poly, "np_poly": np_poly},
                {"poly": poly, "np_poly": np_poly},
                "test_declaration"
            )


    def test_addition(self):
        for _ in range(MAX_TESTS):
            poly1, np_poly1 = get_random_polynomial()
            poly2, np_poly2 = get_random_polynomial()

            result = poly1 + poly2
            np_result = np_poly1 + np_poly2

            assert iquals(result, np_result), get_message_wrapper({
                "poly1": poly1,
                "poly2": poly2,
                "np_poly1": np_poly1,
                "np_poly2": np_poly2
            }, {
                "result": result,
                "np_result": np_result
            }, "addition")

    def test_subtraction(self):
        for _ in range(MAX_TESTS):
            poly1, np_poly1 = get_random_polynomial()
            poly2, np_poly2 = get_random_polynomial()

            result = poly1 - poly2
            np_result = np_poly1 - np_poly2

            assert iquals(result, np_result), get_message_wrapper({
                "poly1": poly1,
                "poly2": poly2,
                "np_poly1": np_poly1,
                "np_poly2": np_poly2
            }, {
                "result": result,
                "np_result": np_result
            }, "subtraction")


    def test_multiplication(self):
        for _ in range(MAX_TESTS):
            poly1, np_poly1 = get_random_polynomial(coef_t=2)
            poly2, np_poly2 = get_random_polynomial(coef_t=2**35)

            result = poly1 * poly2
            np_result = np_poly1 * np_poly2

            print(poly1)
            print(poly2)
            print("REs: ", result)
            print("np_res: ", np_result)

            assert iquals(result, np_result), get_message_wrapper({
                "poly1": poly1,
                "poly2": poly2,
                "np_poly1": np_poly1,
                "np_poly2": np_poly2
            }, {
                "result": result,
                "np_result": np_result
            }, "multiplication")

    def test_true_division_by_constant(self):
        for _ in range(MAX_TESTS):
            poly1, np_poly1 = get_random_polynomial()
            poly2, np_poly2 = get_random_polynomial()
            div = random.randint(1, 100)

            result = poly1 / div
            np_result = np_poly1 / div

            assert iquals(result, np_result), get_message_wrapper({
                "poly1": poly1,
                "np_poly1": np_poly1,
                "np_poly2": np_poly2
            }, {
                "result": result,
                "np_result": np_result
            }, "true_division_by_constant")

    def test_integer_division_by_constant(self):
        for _ in range(MAX_TESTS):
            poly1, np_poly1 = get_random_polynomial()
            div = random.randint(1, 100)
            result = poly1 // div 
            coefs = np.floor(np_poly1.coef / div).astype(int)
            # remove 0 on the right of coefs
            while len(coefs) > 1 and coefs[-1] == 0:
                coefs = coefs[:-1]

            # print(coefs)
            np_result = Polynomial(coefs)

            assert iquals(result, np_result), get_message_wrapper({
                "poly1": poly1,
                "np_poly1": np_poly1,
            }, {
                "result": result,
                "np_result": np_result
            }, "integer_division_by_constant")

    def test_modulus(self):
        for _ in range(MAX_TESTS):
            poly1, np_poly1 = get_random_polynomial()
            np_poly2 = get_mod_polynomial()

            result = poly1 % (2**33)
            np_result = np_poly1 % np_poly2

            assert iquals(result, np_result), get_message_wrapper({
                "poly1": poly1,
                "np_poly1": np_poly1,
                "np_poly2": np_poly2,
                "mod_degree": Polynomials.N
            }, {
                "result": result,
                "np_result": np_result
            }, "modulus")
