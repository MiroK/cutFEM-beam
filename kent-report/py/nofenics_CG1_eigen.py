from __future__ import division
import numpy as np
import numpy.linalg as la
from scipy.linalg import toeplitz
import math 
from math import pi, log as ln


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
for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    A = Afem_matrix(n)
    M = Mfem_matrix(n)
    # The claim now is that A, M which are n x n matrices can be obtained as a
    # limit from Aeig, Meig, the m x m stiffness and mass matrices in the 
    # eigenbasis, and the transformation matrix P which is n x m and takes the
    # eigenfunctions to CG1 functions.
    print '\n n=%d' % n 
    temp = 'm=%d, |A-A_|=%.2E, A_rate=%.2f, |M-M_|=%.2E, M_rate=%.2f Pcond=%.2E'
    temp1 = '\tm=%d, |A[0, 0]-A_[0, 0]|=%.2E, A_rate=%.2f, |M[0, 0]-M_[0, 0]|=%.2E, M_rate=%.2f'

    A_row, M_row, P_row = [], [], []

    for m in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        Aeig = Aeig_matrix(m)
        Meig = Meig_matrix(m)

        # Transformation
        P = P_matrix(n, m)
        P_cond = la.cond(P)

        # Compute A_ as P*Aeig*.Pt ans same for M
        A_ = P.dot(Aeig.dot(P.T))
        M_ = P.dot(Meig.dot(P.T))

        A_norm = la.norm(A-A_, 2)
        M_norm = la.norm(M-M_, 2)

        # Have a look at entri convergence
        Aentry = abs(A[0, 0]-A_[0, 0])
        Mentry = abs(M[0, 0]-M_[0, 0])

        if m != 16:
            rateA = ln(A_norm/A_norm_)/ln(m_/m)
            rateM = ln(M_norm/M_norm_)/ln(m_/m)
            Arate = ln(Aentry/Aentry_)/ln(m_/m)
            Mrate = ln(Mentry/Mentry_)/ln(m_/m)

            print temp % (m, la.norm(A-A_), rateA, la.norm(M-M_), rateM, P_cond)
            print temp1 % (m, Aentry, Arate, Mentry, Mrate)

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

import pickle
pickle.dump(P_conds, open('eigen_pcond.pickle', 'wb'))
pickle.dump(A_norms, open('A_norms.pickle', 'wb'))
pickle.dump(M_norms, open('M_norms.pickle', 'wb'))
