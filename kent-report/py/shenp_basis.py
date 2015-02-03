from __future__ import division
from sympy import legendre, Symbol, lambdify, sqrt
from sympy.mpmath import quad
import numpy as np

x = Symbol('x')

def shenp_basis(n):
    '''
    Yield first n basis function due to Shen - combinations of Legendre
    polynomials that have zeros at -1, 1 and yield sparse mass and stiffness
    matrices.
    '''
    x = Symbol('x')
    k = 0
    while k < n:
        weight = 1/sqrt(4*k + 6)
        yield weight*(legendre(k+2, x) - legendre(k, x))
        k += 1

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    x = Symbol('x')
    n = 10
    sym_basis = list(shenp_basis(n))

    # Boundary values should be 0
    for v in sym_basis:
        # Check the boundary value at -1
        assert v.subs(x, -1) == 0
        # Check the boundary value at 1
        assert v.subs(x, 1) == 0

    # Mass matrix should be tridiagonal with entries from weight 
    # Symbolic
    basis = [lambdify(x, v) for v in sym_basis]
    M = np.zeros((n, n))
    for i, v in enumerate(basis):
        M[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            M[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            M[j, i] = M[i, j]

    # Numeric
    weight = lambda k: float(1/sqrt(4*k + 6))
    MM = np.zeros_like(M)
    for i in range(n):
        MM[i, i] = weight(i)*weight(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, n):
            MM[i, j] = -weight(i)*weight(j)*(2./(2*j+1)) if i+2 == j else 0
            MM[j, i] = MM[i, j]

    assert np.allclose(M, MM, 1E-13)

    # Stiffness matrix of the Poisson problem should be identity
    basis = [lambdify(x, v.diff(x, 1)) for v in sym_basis]
    A = np.zeros((n, n))
    for i, v in enumerate(basis):
        A[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            A[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            A[j, i] = A[i, j]

    assert np.allclose(A, np.eye(n), 1E-13)

    print 'OK'

