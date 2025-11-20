import math
import numpy as np

def check_arrays_close(a, b) -> bool:
    atol, rtol = 1e-4, 1e-4
    try:
        np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), atol=atol, rtol=rtol
        )
        return True
    except AssertionError:
        return False

def check_values_close(a, b):
    atol, rtol = 1e-4, 1e-4
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)