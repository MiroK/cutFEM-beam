from __future__ import division
from sympy import sin, cos, pi, Symbol, lambdify
from sympy.mpmath import quad
import numpy as np


def eigen_basis(n):
    'Yield first n eigenfunctions of laplacian over (-1, 1) with Dirichlet bcs'
    x = Symbol('x')
    k = 0
    while k < n:
        alpha = pi/2 + k*pi/2
        if k % 2 == 0:
            yield cos(alpha*x)
        else:
            yield sin(alpha*x)
        k += 1

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    x = Symbol('x')
    n = 10
    # Check that these are indeed eigenfunctions
    for k, uk in enumerate(eigen_basis(n)):
        # Check the boundary value at -1
        assert uk.subs(x, -1) == 0
        # Check the boundary value at 1
        assert uk.subs(x, 1) == 0
        # Check that uk is an eigenfunction of laplacian
        assert ((-uk.diff(x, 2)/uk).simplify() - (pi/2 + k*pi/2)**2) == 0

    # Check mass matrix, or orthogonality of eigenfunctions
    basis = [lambdify(x, v) for v in list(eigen_basis(n))]
    assert len(basis) == n
    M = np.zeros((n, n))
    for i, v in enumerate(basis):
        M[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            M[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            M[j, i] = M[i, j]
    assert np.allclose(M, np.eye(n), 1E-13)

    # Check stiffness matrix, or A-orthogonality of eigenfunctions
    basis = [lambdify(x, v.diff(x, 1)) for v in list(eigen_basis(n))]
    eigenvalues = [float((pi/2 + k*pi/2)**2) for k in range(n)]
    A = np.zeros((n, n))
    for i, v in enumerate(basis):
        A[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            A[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            A[j, i] = A[i, j]
    assert np.allclose(A, np.diag(eigenvalues), 1E-13)

    print 'OK'
