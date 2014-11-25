from __future__ import division
from sympy.mpmath import legendre, quad, diff, sqrt
from functools import partial
import numpy.linalg as la
import numpy as np

n = 20
def c(k):
    return sqrt(4*k + 6)**-1

# We construct a new basis that has always zero endpoint values
def shen_basis(n):
    return [lambda x, i=i: c(i)*(partial(legendre, n=i)(x=x) -
                                 partial(legendre, n=i+2)(x=x))
            for i in range(n-2)]

basis = shen_basis(n)
d_basis = [lambda x, f=f: diff(f, x) for f in basis]
n = len(basis)

P = np.zeros((n, n))
G = np.zeros((n, n))
A = np.eye(n)

M = np.zeros((n, n))
for i in range(n):
    M[i, i] = c(i)*c(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
    for j in range(i+1, n):
        M[i, j] = -c(i)*c(j)*(2./(2*j+1)) if i+2 == j else 0
        M[j, i] = M[i, j]

# Now build transformation matrix and show
def alpha_phi(n, m):
    alpha = np.zeros((n, m))
    for i in range(n):
        alpha[i, i] = -c(i)
        alpha[i, i+2] = c(i)
    return alpha

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

Ghat = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        Ghat[i, j] = 0.5*j*(j+1)*(1 + (-1)**(i+j))

Ahat = np.zeros_like(Ghat)
Ahat[:] = Ghat[:]
for i in range(m):
    for j in range(m):
        Ahat[i, j] -= j*(j+1) - i*(i+1) if (i < j-1) and ((i+j) % 2) == 0 else 0

# ------------------------------------------------------------------------------

for matrix, hat_matrix in zip([P, G, A, M], [Phat, Ghat, Ahat, Mhat]):
    assert la.norm(matrix - (alpha.dot(hat_matrix).dot(alpha.T)))/n < 1E-15

# ------------------------------------------------------------------------------

# row of alpha is col of alpha T
for row in alpha:
    assert len(row) == m
    assert la.norm(Phat.dot(row))/m < 1E-15
