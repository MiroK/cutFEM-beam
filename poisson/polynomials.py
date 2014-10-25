from sympy import sin, pi, sqrt, symbols
from itertools import product
import numpy as np


def sine_basis(N):
    '''
    Return functions sqrt(2)*sin(k*pi*x) for k in Ns (or range(1, N)).
    These are normalized eigenfunctions of Laplace and biharmonic operators
    over [0, 1] so that their L^2(0, 1) inner product is 1.
    '''
    x = symbols('x')
    try:
        return [sin(k*pi*x)*sqrt(2) for k in N]
    except TypeError:
        return sine_basis(range(1, N))


def equidistant_points(domain, N):
    '''
    [-1, -1], [[-1, 1], [-1, 1]]
    '''
    try:
        a = domain[0]
        b = domain[1]
        return np.linspace(a, b, N)
    except TypeError:
        return np.array([point
                         for point in product(*[equidistant_points(sub, n)
                                                for sub, n in zip(domain, N)])])

def chebyshev_points(domain, N):
    pass

def gauss_lobatto_points(domain, N):
    pass

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import integrate

    for p in equidistant_points([[-1, 1], [3, 4]], [4, 3]):
        print p


    exit()
    # Make sure that the basis is orthonormal
    x = symbols('x')
    for i, si in enumerate(sine_basis(3)):
        for j, sj in enumerate(sine_basis(3)):
            if i == j:
                assert abs(integrate(si*sj, (x, 0, 1)) - 1) < 1E-15
            else:
                assert abs(integrate(si*sj, (x, 0, 1))) < 1E-15

