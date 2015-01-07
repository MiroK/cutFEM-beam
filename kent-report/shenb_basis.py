from __future__ import division
from sympy import legendre, Symbol, lambdify, sqrt
from sympy.mpmath import quad
import numpy as np

def shenb_basis(n):
    '''
    Yield first n basis function due to Shen - combinations of Legendre
    polynomials that have zeros at -1, 1 for u and du and yield sparse mass
    matrix and stiffness for biharmonic and laplace operators.
    '''
    x = Symbol('x')
    k = 0
    while k < n:
        weight = 1/sqrt(2*(2*k+3)**2*(2*k+5))
        yield weight*(legendre(k, x) -
                      2*(2*k + 5)/(2*k + 7)*legendre(k+2, x) +
                      (2*k + 3)/(2*k + 7)*legendre(k+4, x))
        k += 1

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    x = Symbol('x')
    n = 10
    sym_basis = list(shenb_basis(n))

    # Boundary values for u and du should be 0
    for v in sym_basis:
        # Check the boundary value of u at -1
        assert v.subs(x, -1) == 0
        # Check the boundary value of u at 1
        assert v.subs(x, 1) == 0
        # Check the boundary value of du at -1
        assert v.diff(x, 1).subs(x, -1) == 0
        # Check the boundary value of du at 1
        assert v.diff(x, 1).subs(x, 1) == 0


    # Coeficients for entries in M, C
    _d = lambda k: 1/sqrt(2*(2*k+3)**2*(2*k+5))
    _e = lambda k: 2/(2*k + 1)
    _g = lambda k: (2*k + 3)/(2*k + 7)
    _h = lambda k: -(1 + _g(k))

    def mass_matrix(n):
        'Tabulated mass matrix'
        M = np.zeros((n, n))
        for k in range(n):
            M[k, k] = _d(k)**2*(_e(k) + _h(k)**2*_e(k+2) + _g(k)**2*_e(k+4))
            if k + 2 < n:
                M[k, k+2] = _d(k)*_d(k+2)*(_h(k)*_e(k+2) + _g(k)*_h(k+2)*_e(k+4))
                M[k+2, k] = M[k, k+2]
                if k + 4 < n:
                    M[k, k+4] = _d(k)*_d(k+4)*_g(k)*_e(k+4)
                    M[k+4, k] = M[k, k+4]
        return M

    # Mass matrix should be sparse with tabulated values
    basis = [lambdify(x, v) for v in sym_basis]
    M = np.zeros((n, n))
    for i, v in enumerate(basis):
        M[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            M[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            M[j, i] = M[i, j]

    MM = mass_matrix(n)
    assert np.allclose(M - MM, np.zeros_like(M), 1E-13)

    def laplace_matrix(n):
        'Tabulated stiffness matrix of laplacian'
        C = np.zeros((n, n))
        for k in range(n):
            C[k, k] = -2*(2*k + 3)*_d(k)**2*_h(k)
            if k + 2 < n:
                C[k, k+2] = -2*(2*k + 3)*_d(k)*_d(k+2)
                C[k+2, k] = C[k, k+2]
        return C

    # Stiffness matrix of the laplacian should be sparse with tabulated values 
    basis = [lambdify(x, v.diff(x, 1)) for v in sym_basis]
    C = np.zeros((n, n))
    for i, v in enumerate(basis):
        C[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            C[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            C[j, i] = C[i, j]

    CC = laplace_matrix(n)
    assert np.allclose(C - CC, np.zeros_like(C), 1E-13)

    # Stiffness matrix of the biharmonic operator should be identity
    basis = [lambdify(x, v.diff(x, 2)) for v in sym_basis]
    A = np.zeros((n, n))
    for i, v in enumerate(basis):
        A[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            A[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            A[j, i] = A[i, j]

    assert np.allclose(A, np.eye(n), 1E-13)

    print 'OK'
