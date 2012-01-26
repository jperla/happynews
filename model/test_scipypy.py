import scipypy
import scipy.special

def test_psi():
    assert check_functions_match(scipypy.psi, scipy.special.psi)

def test_gammaln():
    assert check_functions_match(scipypy.gammaln, scipy.special.gammaln)

def check_functions_match(f, g):
    n = 100000
    n = 100
    for i in xrange(-100, n, 1):
        f = i / 100.0
        a = scipypy.psi(f)
        b = scipy.special.psi(f)
        if a == float('inf'):
            if b != float('inf'):
                return False
        elif a == float('inf'):
            if b != float('-inf'):
                return False
        else:
            if abs(a - b) >= 0.0000000001:
                return False
    else:
        return True
