from sympy.mpmath import legendre, quad
import numpy as np
from functools import partial
import scipy.linalg as la
from math import sqrt, pi, sin

'''
Eigenvalue problem
     -u`` = lmbda u  in [0, 1]
     u(0) = u(1) = 0
'''


def c(k):
    'Shen weights'
    return sqrt(4*k + 6)**-1


def shen_basis(m):
    'Basis in from Legendre polynomials that have 0 at [-1, 1]'
    return [lambda x, i=i:
            c(i)*(partial(legendre, n=i)(x=x) - partial(legendre, n=i+2)(x=x))
            for i in range(m-2)]


def mass_matrix(n):
    'Mass matrix assembled from Shen basis'
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = c(i)*c(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, n):
            M[i, j] = -c(i)*c(j)*(2./(2*j+1)) if i+2 == j else 0
            M[j, i] = M[i, j]
    return M


def stiffness_matrix(n):
    'Stiffness matrix assembled from Shen basis'
    return np.eye(n)


def assemble_uh(basis, coefs):
    'Return u, such that u(x) = sum_i b_i(x)*U_i'
    assert len(basis) == len(coefs)
    return lambda x: sum(coef*b(x) for coef, b in zip(coefs, basis))

# -----------------------------------------------------------------------------

# How large m do I need to get first (target) eigenvalues with (tol) accuracy
target = 15
tol = 1E-10
m = 18  # starting value
not_converged = True

while not_converged:
    print 'm=%d' % m

    basis = shen_basis(m)
    n = len(basis)
    assert n > target
    M = mass_matrix(n)
    A = stiffness_matrix(n)

    eigenvalues, eigenvectors = la.eigh(A, M)

    # I want the problem on [0, 1]
    eigenvectors = eigenvectors.T[:target]
    lmbda_errors = []
    for i, (lmbda, vec) in enumerate(zip(eigenvalues[:], eigenvectors[:]), 1):
        # Assemble, defined on [-1, 1], maybe the signs came out wrong
        uh_p = assemble_uh(basis, vec)
        uh_m = assemble_uh(basis, -vec)
        # L2 norm of eigenvector error
        vec_errors = []
        for uh in [uh_p, uh_m]:
            # Note that u is mapped to work on [0, 1]
            e = quad(lambda x: (uh(2*x - 1) - sin(i*pi*x))**2, [0, 1])
            e = sqrt(e)
            vec_errors.append(e)
        # Error in eigenvalues
        numeric = lmbda*4  # Eigenvalues need to be scaled from the orig [-1, 1]
        exact = (i*pi)**2
        lmbda_error = abs(numeric-exact)

        print '\ti = %d lambda error = %.2E, vec error = (%.4E, %.4E)' % \
            (i, lmbda_error, vec_errors[0], vec_errors[1])
        if lmbda_error > tol:
            m += 1
            break
