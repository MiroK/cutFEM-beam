from __future__ import division
from sympy.mpmath import legendre, quad, diff
from functools import partial
import numpy.linalg as la
import numpy as np

# Hat matrices are matrices whose coefficients are defined using basis functions
# of simple Legendre polynomials

m = 15

def legendre_basis(m):
    return [lambda x, i=i: partial(legendre, n=i)(x=x)
            for i in range(m)]

basis = legendre_basis(m)
d_basis = [lambda x, f=f: diff(f, x) for f in basis]
dd_basis = [lambda x, f=f: diff(f, x, 2) for f in basis]

assert len(basis) == m


# Mass matrix computed from definition (Li, Lj)
M = np.zeros((m, m))
for i, bi in enumerate(basis):
    M[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
    for j, bj in enumerate(basis[i+1:], i+1):
        M[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
        M[j, i] = M[i, j]

# Hand computed mass matrix -- Legendre polynomials are orthogonal
M_ = np.zeros((m, m))
for i in range(m):
    M_[i, i] = 2/(2*i + 1)
assert la.norm(M - M_)/m < 1E-15

# Penalty matrix from definitions Li*Lj(1) + Li*Lj(-1)
P = np.zeros((m, m))
for i, bi in enumerate(basis):
    P[i, i] = bi(1)*bi(1) + bi(-1)*bi(-1)
    for j, bj in enumerate(basis[i+1:], i+1):
        P[i, j] = bi(1)*bj(1) + bi(-1)*bj(-1)
        P[j, i] = P[i, j]

# Hand computed
P_ = np.zeros((m, m))
for i in range(m):
    P_[i, i] = 1 + (-1)**(2*i)
    for j in range(i+1, m):
        P_[i, j] = 1 + (-1)**(i+j)
        P_[j, i] = P_[i, j]
assert la.norm(P-P_)/m < 1E-15

#  Consisitency matrix, Li*Lj`(1) - Li*Lj`(-1)
G = np.zeros((m, m))
for i, bi in enumerate(basis):
    for j, dbj in enumerate(d_basis):
        G[i, j] = bi(1)*dbj(1) - bi(-1)*dbj(-1)

# Hand computed
G_ = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        G_[i, j] = 0.5*j*(j+1)*(1 + (-1)**(i+j))
assert la.norm(G-G_)/m < 1E-15

# Stiffness matrix
A = np.zeros((m, m))
for i, dbi in enumerate(d_basis):
    for j, dbj in enumerate(d_basis):
        A[i, j] = quad(lambda x: dbi(x)*dbj(x), [-1, 1])

A_ = G
for i in range(m):
    for j in range(m):
        A_[i, j] -= j*(j+1) - i*(i+1) if (i < j-1) and ((i+j) % 2) == 0 else 0

assert la.norm(A - A_)/m < 1E-15
