from __future__ import division
import numpy as np
import numpy.linalg as la
from sympy.mpmath import legendre, sqrt
from scipy.linalg import toeplitz
from math import log as ln


def Ashen_matrix(m):
    'Stiffness matrix w.r.t to Shen basis.'
    return np.eye(m)


def Mshen_matrix(m):
    'Mass matrix w.r.t to Shen basis.'
    weight = lambda k: 1./sqrt(4*k + 6)
    M = np.zeros((m, m), dtype='float64')
    for i in range(m):
        M[i, i] = weight(i)*weight(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, m):
            M[i, j] = -weight(i)*weight(j)*(2./(2*j+1)) if i+2 == j else 0
            M[j, i] = M[i, j]

    return M

# -----------------------------------------------------------------------------

def P_matrix(n, m):
    'Takes m Shen basis function into n interior CG1 functions over -1, 1.'
    # Mesh size
    h = 2/(n+1)
    # All mesh vertices
    vertices = [-1 + h*i for i in range(n+2)]
    # Shen functions 
    Sk = lambda k, x: (legendre(k+2, x) - legendre(k, x))/sqrt(4*k + 6)
    
    P = np.zeros((n, m))
    # This loops computet integrals: i-th has * j-th shen
    for i in range(n):
        vertex_p = vertices[i]
        vertex = vertices[i+1]
        vertex_n = vertices[i+2]

        for j in range(m):
            P[i, j] = 2*Sk(j, vertex)/h
            P[i, j] -= Sk(j, vertex_p)/h
            P[i, j] -= Sk(j, vertex_n)/h

    return P

# -----------------------------------------------------------------------------

def Afem_matrix(n):
    'Stiffness matrix of H10 FEM(with n functions).'
    h = 2/(n+1)
    row = np.zeros(n)
    row[0] = 2
    row[1] = -1
    A = toeplitz(row)
    A /= h
    return A


def Mfem_matrix(n):
    'Mass matrix of H10 FEM.'
    h = 2/(n+1)
    row = np.zeros(n)
    row[0] = 4
    row[1] = 1
    M = toeplitz(row)
    M *= h/6.
    return M

# -----------------------------------------------------------------------------

n = 4
A = Afem_matrix(n)
M = Mfem_matrix(n)

print A

print M

# The claim now is that A, M which are n x n matrices can be obtained as a limit
# from Ashen, Mshen, the m x m stiffness and mass matrices in the shenbasis, and
# the transformation matrix P which is n x m and takes the shenenfunctions to
# CG1 functions.
temp = 'm=%d, |A-A_|=%.2E, A_rate=%.2f, |M-M_|=%.2E, M_rate=%.2f'

for m in [2, 4, 8, 16, 32, 64, 128, 256]:
    # Compute shen matrices and transformation
    Ashen = Ashen_matrix(m)
    Mshen = Mshen_matrix(m)
    P = P_matrix(n, m)

    # Compute A_ as P*Ashen*.Pt ans same for M
    A_ = P.dot(Ashen.dot(P.T))
    M_ = P.dot(Mshen.dot(P.T))

    A_norm = la.norm(A-A_)
    M_norm = la.norm(M-M_)

    if m != 2:
        rateA = ln(A_norm/A_norm_)/ln(m_/m)
        rateM = ln(M_norm/M_norm_)/ln(m_/m)

        print temp % (m, la.norm(A-A_), rateA, la.norm(M-M_), rateM)

    A_norm_ = A_norm
    M_norm_ = M_norm
    m_ = m
