from __future__ import division
from sympy import sin, cos, pi, Symbol, lambdify, sqrt
from sympy.mpmath import quad
import numpy as np

# FIXME: eigen_basis_dirichlet?
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


def eigen_basis_neumann(n):
    '''
    Eigenfunctions of -u`` = lmbda * u on (-1, 1) with homg. Neumann conditions 
    and constraint mean(u) = 0.
    '''
    # Note that this is then positive definite operator. Symmetry.
    # The functions here have zero mean - we don't have lambda 0 for which the
    # eigenfunction is constant.
    x = Symbol('x')
    k = 0
    while k < n:
        alpha = pi/2 + k*pi/2
        if k % 2 == 0:
            yield sin(alpha*x)
        else:
            yield cos(alpha*x)
        k += 1


def eigen_basis_mixed(n):
    'Eigenfunctions of -u`` = lmbda * u on (-1, 1) with u(-1) = u`(1) = 0.'
    # TODO how is this this self adjoint?
    x = Symbol('x')
    k = 0
    # The functions are normalized to be orthonormal in L2 inner product
    while k < n:
        alpha = pi/4 + k*pi/2
        if k % 2 == 0:
            yield (sin(alpha*x) + cos(alpha*x))/sqrt(2)
        else:
            yield (-sin(alpha*x) + cos(alpha*x))/sqrt(2)
        k += 1

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import integrate
    x = Symbol('x')
    n = 10
    # Dirichlet: verify properties 
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

    # This basis also has functions that are eigenfunctions of
    # u'''' = lmbda u on (-1, 1), with u=u''= 0 {-1, 1}
    # Verify these props
    for k, uk in enumerate(eigen_basis(n)):
        # We already know that the value is correct
        dd_uk = uk.diff(x, 2)
        # Check derivative value at -1
        assert uk.subs(x, -1) == 0
        # Check derivative value at 1
        assert uk.subs(x, 1) == 0
        # Check that uk is an eigenfunction of laplacian
        assert ((uk.diff(x, 4)/uk).simplify() - (pi/2 + k*pi/2)**4) == 0

    # Check the stiffness matrix of the biharmonic operator
    basis = [lambdify(x, v.diff(x, 2)) for v in list(eigen_basis(n))]
    eigenvalues = [float((pi/2 + k*pi/2)**4) for k in range(n)]
    A = np.zeros((n, n))
    for i, v in enumerate(basis):
        A[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            A[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            A[j, i] = A[i, j]
    assert np.allclose(A, np.diag(eigenvalues), 1E-13)

    # Neumann: verify properties
    for k, uk in enumerate(eigen_basis_neumann(n)):
        # u`(-1)
        assert uk.diff(x, 1).subs(x, -1) == 0
        # u`(1)
        assert uk.diff(x, 1).subs(x, 1) == 0
        # eigenvalues
        assert ((-uk.diff(x, 2)/uk).simplify() - (pi/2 + k*pi/2)**2) == 0
        # mean
        assert integrate(uk, (x, -1, 1)) == 0

    # Check mass matrix, or orthogonality of eigenfunctions
    basis = [lambdify(x, v) for v in list(eigen_basis_neumann(n))]
    assert len(basis) == n
    M = np.zeros((n, n))
    for i, v in enumerate(basis):
        M[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            M[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            M[j, i] = M[i, j]
    assert np.allclose(M, np.eye(n), 1E-13)

    # Check stiffness matrix, or A-orthogonality of eigenfunctions
    basis = [lambdify(x, v.diff(x, 1)) for v in list(eigen_basis_neumann(n))]
    eigenvalues = [float((pi/2 + k*pi/2)**2) for k in range(n)]
    A = np.zeros((n, n))
    for i, v in enumerate(basis):
        A[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            A[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            A[j, i] = A[i, j]
    assert np.allclose(A, np.diag(eigenvalues), 1E-13)

    # Mixed: verify properties
    for k, uk in enumerate(eigen_basis_mixed(n)):
        # u(-1)
        assert uk.subs(x, -1) == 0
        # u`(1)
        assert uk.diff(x, 1).subs(x, 1) == 0
        # eigenvalues
        assert ((-uk.diff(x, 2)/uk).simplify() - (pi/4 + k*pi/2)**2) == 0

    # Check mass matrix, or orthogonality of eigenfunctions
    basis = [lambdify(x, v) for v in list(eigen_basis_mixed(n))]
    assert len(basis) == n
    M = np.zeros((n, n))
    for i, v in enumerate(basis):
        M[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            M[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            M[j, i] = M[i, j]
    assert np.allclose(M, np.eye(n), 1E-13)

    # Check stiffness matrix, or A-orthogonality of eigenfunctions
    basis = [lambdify(x, v.diff(x, 1)) for v in list(eigen_basis_mixed(n))]
    eigenvalues = [float((pi/4 + k*pi/2)**2) for k in range(n)]
    A = np.zeros((n, n))
    for i, v in enumerate(basis):
        A[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(basis[(i+1):], i+1):
            A[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            A[j, i] = A[i, j]
    assert np.allclose(A, np.diag(eigenvalues), 1E-13)

    print 'OK'
