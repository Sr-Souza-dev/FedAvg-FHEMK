from ckks.polynomials.main import Polynomials

class Cryptogram:
    fix_a = False
    def __init__(self, c0: Polynomials, c1: Polynomials, q: int):
        self.c0 = c0
        self.c1 = c1
        self.q = q
    
    def __str__(self):
        return f"Cryptogram:\n  q={self.q},\n  c0: {self.c0},\n  c1:{self.c1}."

    def __add__(self, other):
        if isinstance(other, Cryptogram):
            c0 = self.c0 + other.c0
            if not Cryptogram.fix_a:
                c1 = self.c1 + other.c1
            else: 
                c1 = self.c1 if self.c1.degree > 2 else other.c1
        elif isinstance(other, int):
            c0 = self.c0 + other
            c1 = self.c1
        else:
            raise TypeError("Unsupported type for addition.")
        return Cryptogram(c0, c1, self.q)
    
    def __sub__(self, other):
        if isinstance(other, Cryptogram):
            c0 = (self.c0 - other.c0) % self.q
            c1 = (self.c1 - other.c1) % self.q
        elif isinstance(other, int):
            c0 = (self.c0 - other) % self.q
            c1 = self.c1
        else:
            raise TypeError("Unsupported type for subtraction.")
        return Cryptogram(c0, c1, self.q)
    
    def __mul__(self, other):
        if isinstance(other, int):
            c0 = (self.c0 * other) % self.q
            c1 = (self.c1 * other) % self.q
        else:
            raise TypeError("Unsupported type for multiplication.")
        return Cryptogram(c0, c1, self.q)
    
    def __truediv__(self, other):
        if isinstance(other, int):
            c0 = (self.c0 // other) % self.q
            c1 = (self.c1 // other) % self.q
            return Cryptogram(c0, c1, self.q)
        else:
            raise TypeError("Unsupported type for division.")
