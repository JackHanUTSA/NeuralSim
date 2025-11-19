import math
import numpy as np


def f_1(x, check_domain: bool = False):
    """Compute f_1(x) = e^{-2x} cos(5π x) + x for x in [0,1].

    Supports scalar (int/float), list/tuple, and numpy.ndarray.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        x_arr = np.asarray(x, dtype=float)
        if check_domain:
            if np.any((x_arr < 0) | (x_arr > 1)):
                raise ValueError("f_1 input outside domain [0,1]")
        return np.exp(-2.0 * x_arr) * np.cos(5.0 * np.pi * x_arr) + x_arr

    x_f = float(x)
    if check_domain and not (0.0 <= x_f <= 1.0):
        raise ValueError("f_1 input outside domain [0,1]")
    return math.exp(-2.0 * x_f) * math.cos(5.0 * math.pi * x_f) + x_f


if __name__ == '__main__':
    print("Testing standalone f_1")
    print("f_1(0) =", f_1(0.0))
    print("f_1(1) =", f_1(1.0))
    xs = np.linspace(0.0, 1.0, 5)
    print("f_1 on linspace ->", f_1(xs))
