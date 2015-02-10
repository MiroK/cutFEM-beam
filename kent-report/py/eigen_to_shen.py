from __future__ import division
import numpy as np
import numpy.linalg as la
from sympy.mpmath import legendre, sqrt, pi, sin, cos, quad
from functools import partial
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
    'Takes m Eigen basis function to n Shen functions.'
    # Shen functions 
    Sk = lambda k, x: (legendre(k+2, x) - legendre(k, x))/sqrt(4*k + 6)

    # Eigenfunctions
    Ek = lambda k, x: cos((pi/2 + k*pi/2)*x)\
                        if k % 2 == 0 else\
                      sin((pi/2 + k*pi/2)*x)

    # Integrand
    integrand = lambda i, j, x: Ek(j, x)*Sk(i, x)
    
    P = np.zeros((n, m))
    # This loops computet integrals: i-th has * j-th shen
    for i in range(n):
        for j in range(m):
            P[i, j] = quad(partial(integrand, i, j), [-1, 1])

    return P

# -----------------------------------------------------------------------------

def Aeig_matrix(n):
    'Stiffness matrix w.r.t Eigen function basis.'
    return np.diag([(pi/2 + k*pi/2)**2 for k in range(n)])


def Meig_matrix(n):
    'Mass matrix w.r.t Eigen function basis.'
    return np.eye(n)

# -----------------------------------------------------------------------------

# Monitor the condition number of P
P_conds = {}
A_norms = {}
M_norms = {}
for n in [2, 4, 8, 12, 16, 20]:
    A = Ashen_matrix(n)
    M = Mshen_matrix(n)

    print '\n n=%d' % n 
    temp = 'm=%d, |A-A_|=%.2E, A_rate=%.2f, |M-M_|=%.2E, M_rate=%.2f Pcond=%.2E'

    A_row, M_row, P_row = [], [], []

    for m in [2, 4, 8, 16, 20, 32, 64, 128, 256]:
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

        if m != 2:
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

        if abs(A_norm) < 1E-14 and abs(M_norm) < 1E-14:
            break

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
pickle.dump(P_conds, open('es_pcond.pickle', 'wb'))
pickle.dump(A_norms, open('esA_norms.pickle', 'wb'))
pickle.dump(M_norms, open('esM_norms.pickle', 'wb'))
