def next_power_of_two(n):
    """
    Find the next power of two greater than or equal to n.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    return 1 << (n - 1).bit_length()

def is_power_of_two(n):
    """
    Check if n is a power of two.
    """
    return n > 0 and (n & (n - 1)) == 0