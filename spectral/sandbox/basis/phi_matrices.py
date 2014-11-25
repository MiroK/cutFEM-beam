from __future__ import division
from sympy.mpmath import legendre, quad, diff
from functools import partial
import numpy.linalg as la
import numpy as np

n = 15

# Basis that shen claims yields dense mass and stiffness.
# tha boundary values should be 0 so P and G = 0
def phi_basis(n):
    basis = []
    for i in range(n-2):
        if i % 2 == 0:
            basis.append(lambda x, i=i: partial(legendre, n=i+2)(x=x) -
                         partial(legendre, n=0)(x=x))
        else:
            basis.append(lambda x, i=i: partial(legendre, n=i+2)(x=x) -
                         partial(legendre, n=1)(x=x))
    return basis

basis = phi_basis(n)
d_basis = [lambda x, f=f: diff(f, x) for f in basis]
n = len(basis)

M = np.zeros((n, n))
for i, bi in enumerate(basis):
    M[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
    for j, bj in enumerate(basis[i+1:], i+1):
        M[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
        M[j, i] = M[i, j]

# Penalty matrix from definitions
P = np.zeros((n, n))
for i, bi in enumerate(basis):
    P[i, i] = bi(1)*bi(1) + bi(-1)*bi(-1)
    for j, bj in enumerate(basis[i+1:], i+1):
        P[i, j] = bi(1)*bj(1) + bi(-1)*bj(-1)
        P[j, i] = P[i, j]

#  Consisitency matrix
G = np.zeros((n, n))
for i, bi in enumerate(basis):
    for j, dbj in enumerate(d_basis):
        G[i, j] = bi(1)*dbj(1) - bi(-1)*dbj(-1)

# Stiffness matrix
A = np.zeros((n, n))
for i, dbi in enumerate(d_basis):
    for j, dbj in enumerate(d_basis):
        A[i, j] = quad(lambda x: dbi(x)*dbj(x), [-1, 1])

# All matrices can be computed from hat matrices by transformation
def alpha_phi(n, m):
    alpha_phi = np.zeros((n, m))
    for i in range(alpha_phi.shape[0]):
        if (i % 2) == 0:
            alpha_phi[i, 0] = -1
        else:
            alpha_phi[i, 1] = -1
        alpha_phi[i, i+2] = 1
    return alpha_phi

m = n+2
alpha = alpha_phi(n, m)

# Hat matrices
Mhat = np.zeros((m, m))
for i in range(m):
    Mhat[i, i] = 2/(2*i + 1)

Phat = np.zeros((m, m))
for i in range(m):
    Phat[i, i] = 1 + (-1)**(2*i)
    for j in range(i+1, m):
        Phat[i, j] = 1 + (-1)**(i+j)
        Phat[j, i] = Phat[i, j]

# Hand computed
Ghat = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        Ghat[i, j] = 0.5*j*(j+1)*(1 + (-1)**(i+j))

Ahat = np.zeros_like(Ghat)
Ahat[:] = Ghat[:]
for i in range(m):
    for j in range(m):
        Ahat[i, j] -= j*(j+1) - i*(i+1) if (i < j-1) and ((i+j) % 2) == 0 else 0

# -----------------------------------------------------------------------------

for matrix, hat_matrix in zip([P, G, A, M], [Phat, Ghat, Ahat, Mhat]):
    print la.norm(matrix - (alpha.dot(hat_matrix).dot(alpha.T)))/n

# -----------------------------------------------------------------------------

# row of alpha is col of alpha T
for row in alpha:
    assert len(row) == m
    assert la.norm(Phat.dot(row))/m < 1E-15
