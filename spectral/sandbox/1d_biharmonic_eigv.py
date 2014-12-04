from __future__ import division
from sympy.mpmath import legendre, quad, sqrt
import numpy as np
from functools import partial
import scipy.linalg as la
import matplotlib.pyplot as plt

def _d(k):
    'Shen weights'
    return sqrt(2*(2*k+3)**2*(2*k+5))**-1

# Coeficients for entries in mass matrix
def _e(k):
    return 2/(2*k + 1)

def _g(k):
    return (2*k + 3)/(2*k + 7)

def _h(k):
    return -(1 + _g(k))

def shen_basis(m):
    'Basis in from Legendre polynomials that have 0 at [-1, 1]'
    return [lambda x, i=i:
            _d(i)*(partial(legendre, n=i)(x=x) -
                   2*(2*i+5)/(2*i+7)*partial(legendre, n=i+2)(x=x) +
                   (2*i+3)/(2*i+7)*partial(legendre, n=i+4)(x=x))
            for i in range(m-4)]

def mass_matrix(n):
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

def stiffness_matrix(n):
    'Stiffness matrix assembled from Shen basis'
    return np.eye(n)


def assemble_uh(basis, coefs):
    'Return u, such that u(x) = sum_i b_i(x)*U_i'
    assert len(basis) == len(coefs)
    return lambda x: sum(coef*b(x) for coef, b in zip(coefs, basis))

# -----------------------------------------------------------------------------

m = 30
basis = shen_basis(m)
n = len(basis)
M = mass_matrix(n)
A = stiffness_matrix(n)

eigenvalues, eigenvectors = la.eigh(A, M)

# I want the problem on [0, 1]
x = np.linspace(0, 1, 100)
eigenvectors = eigenvectors.T
for i, (lmbda, vec) in enumerate(zip(eigenvalues[:6], eigenvectors[:6]), 1):
    # Assemble, defined on [-1, 1], maybe the signs came out wrong
    uh_p = assemble_uh(basis, vec)
    uh_m = assemble_uh(basis, -vec)

    # Eigs
    # Eigenvalues need to be scaled from the orig [-1, 1]
    numeric = lmbda*(2**4)
    print numeric

    # TODO exact?
    #
    #

    plt.figure()
    plt.plot(x, map(uh_p, 2*x - 1), label=str(i))
    plt.legend(loc='best')

plt.show()
