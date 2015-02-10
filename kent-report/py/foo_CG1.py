from __future__ import division
from sympy.mpmath import quad, legendre, sqrt, pi
from scipy.linalg import toeplitz
from math import log as ln
import numpy.linalg as la
import numpy as np
import math

import matplotlib.pyplot as plt


def eigen_basis(n):
    'Eigen basis'
    k = 0
    functions = []
    while k < n:
        alpha = pi/2 + k*pi/2
        if k % 2 == 0:
            yield lambda x, alpha=alpha: math.cos(alpha*x)
        else:
            yield lambda x, alpha=alpha: math.sin(alpha*x)
        k += 1


def Aeig_matrix(m):
    'Stiffness matrix w.r.t to eigen basis.'
    return np.diag([(pi/2 + k*pi/2)**2 for k in range(m)])


def Meig_matrix(m):
    'Mass matrix w.r.t to eigen basis.'
    return np.eye(m)

# -----------------------------------------------------------------------------

def shen_basis(n):
    'Shen basis'
    for k in range(n):
        yield lambda x, k=k: (legendre(k+2, x) - legendre(k, x))/sqrt(4*k + 6)


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

def P_matrix(n, m, what):
    'Transformation matrix (n x m) interpolate n function in m xi points.'
    # Pick basis
    basis = eigen_basis(n) if what == 'eigen' else shen_basis(n)
    # Mesh size
    h = 2/(m+1)
    # All mesh vertices
    vertices = [-1 + h*i for i in range(m+2)]

    P = np.zeros((n, m))
    for i, f in enumerate(basis):
        for j, vertex in enumerate(vertices[1:-1]):
            P[i, j] = f(vertex)

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

def A_matrix(m, what):
    'Dispatcher for stiffness matrices'
    return Ashen_matrix(m) if what == 'shen' else Aeig_matrix(m)


def M_matrix(m, what):
    'Dispatcher for mass matrices'
    return Mshen_matrix(m) if what == 'shen' else Meig_matrix(m)

# -----------------------------------------------------------------------------

# Monitor the condition number of P
P_conds = {}
A_norms = {}
M_norms = {}

what = 'shen'

for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:   # shen/eige size
    # Decide target
    A = A_matrix(n, what)
    M = M_matrix(n, what)

    print '\n n=%d' % n 
    temp = 'm=%d, |A-A_|=%.2E, A_rate=%.2f, |M-M_|=%.2E, M_rate=%.2f Pcond=%.2E'
    temp1 = '\tm=%d, |A[0, 0]-A_[0, 0]|=%.2E, A_rate=%.2f, |M[0, 0]-M_[0, 0]|=%.2E, M_rate=%.2f'

    A_row, M_row, P_row = [], [], []

    for m in [2**i for i in range(1, 13)]:
        Afem = Afem_matrix(m)
        Mfem = Mfem_matrix(m)

        # Transformation
        P = P_matrix(n, m, what)
        P_cond = la.cond(P)

        # Compute A_ as P*Aeig*.Pt ans same for M
        A_ = P.dot(Afem.dot(P.T))
        M_ = P.dot(Mfem.dot(P.T))

        A_norm = la.norm(A-A_, 2)
        M_norm = la.norm(M-M_, 2)

        # Have a look at entri convergence
        Aentry = abs(A[0, 0]-A_[0, 0])
        Mentry = abs(M[0, 0]-M_[0, 0])

        if m != 2:
            rateA = ln(A_norm/A_norm_)/ln(m_/m)
            rateM = ln(M_norm/M_norm_)/ln(m_/m)
            Arate = ln(Aentry/Aentry_)/ln(m_/m)
            Mrate = ln(Mentry/Mentry_)/ln(m_/m)

            print temp % (m, la.norm(A-A_), rateA, la.norm(M-M_), rateM, P_cond)
            # print temp1 % (m, Aentry, Arate, Mentry, Mrate)

        A_norm_ = A_norm
        M_norm_ = M_norm
        m_ = m
        Aentry_ = Aentry
        Mentry_ = Mentry

        # Collect for m
        A_row.append(A_norm)
        M_row.append(M_norm)
        P_row.append(P_cond)

    # Collect for n
    P_conds[n] = P_row
    M_norms[n] = M_row
    A_norms[n] = A_row

print 'P conditioning'
print P_conds

print
print 'A norms'
print A_norms

print
print 'M norms'
print M_norms
