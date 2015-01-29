from __future__ import division
from dolfin import *
import numpy as np
import numpy.linalg as la
from scipy.linalg import toeplitz
import math 


def eigen_basis(n):
    '''
    Return first n eigenfunctions of Laplacian over biunit interval with homog.
    Dirichlet bcs. at endpoints -1, 1. Functions of x.
    '''
    k = 0
    functions = []
    while k < n:
        alpha = pi/2 + k*pi/2
        if k % 2 == 0:
            f = Expression('cos(alpha*x[0])', alpha=alpha, degree=8)
        else:
            f = Expression('sin(alpha*x[0])', alpha=alpha, degree=8)
        functions.append(f)
        k += 1
    return functions


def Aeig_matrix(m):
    'Stiffness matrix w.r.t to eigen basis.'
    return np.diag([(pi/2 + k*pi/2)**2 for k in range(m)])


def Meig_matrix(m):
    'Mass matrix w.r.t to eigen basis.'
    return np.eye(m)

# -----------------------------------------------------------------------------

def P_matrix(n, m):
    'Transformation from FEM(with n functions!) to Eigen.'
    h = 2/(n+1)
    P = np.zeros((n, m))
    for i in range(n):
        xi = -1 + (i+1)*h
        x_next = xi + h
        x_prev = xi - h
        for j in range(m):
            dd_f = lambda x, j=j: math.cos((pi/2 + j*pi/2)*x)/(pi/2 + j*pi/2)**2\
                                  if (j % 2) == 0 else \
                                  math.sin((pi/2 + j*pi/2)*x)/(pi/2 + j*pi/2)**2

            val = 2*dd_f(xi)/h - (dd_f(x_next) + dd_f(x_prev))/h
            P[i, j] = val
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
# Monitor the condition number of P
P_conds = {}
A_norms = {}
M_norms = {}
for n in [4, 8, 16, 32, 64, 128]:
    A = Afem_matrix(n)
    M = Mfem_matrix(n)
    # The claim now is that A, M which are n x n matrices can be obtained as a
    # limit from Aeig, Meig, the m x m stiffness and mass matrices in the 
    # eigenbasis, and the transformation matrix P which is n x m and takes the
    # eigenfunctions to CG1 functions.
    print '\n n=%d' % n 
    temp = 'm=%d, |A-A_|=%.2E, A_rate=%.2f, |M-M_|=%.2E, M_rate=%.2f Pcond=%.2E'

    A_row, M_row, P_row = [], [], []

    for m in [16, 32, 64, 128, 256, 512, 1024]:
        eigen_functions = eigen_basis(m)
        Aeig = Aeig_matrix(m)
        Meig = Meig_matrix(m)

        # Transformation
        P = P_matrix(n, m)
        P_cond = la.cond(P.dot(P.T))

        # Compute A_ as P*Aeig*.Pt ans same for M
        A_ = P.dot(Aeig.dot(P.T))
        M_ = P.dot(Meig.dot(P.T))

        A_norm = la.norm(A-A_, 2)
        M_norm = la.norm(M-M_, 2)
        if m != 16:
            rateA = ln(A_norm/A_norm_)/ln(m_/m)
            rateM = ln(M_norm/M_norm_)/ln(m_/m)

            print temp % (m, la.norm(A-A_), rateA, la.norm(M-M_), rateM, P_cond)

        A_norm_ = A_norm
        M_norm_ = M_norm
        m_ = m

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
