from __future__ import division
from dolfin import *
import numpy as np
import numpy.linalg as la
import sympy as sp

def shen_basis(n):
    'Return Shen basis of H^1_0((-1, 1)).'
    x = sp.Symbol('x')
    k = 0
    functions = []
    while k < n:
        weight = 1/sp.sqrt(4*k + 6)
        f = weight*(sp.legendre(k+2, x) - sp.legendre(k, x))
        functions.append(f)
        k += 1
    return functions


def Ashen_matrix(m):
    'Stiffness matrix w.r.t to Shen basis.'
    return np.eye(m)


def Mshen_matrix(m):
    'Mass matrix w.r.t to Shen basis.'
    weight = lambda k: 1./sqrt(4*k + 6)
    M = np.zeros((m, m))
    for i in range(m):
        M[i, i] = weight(i)*weight(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, m):
            M[i, j] = -weight(i)*weight(j)*(2./(2*j+1)) if i+2 == j else 0
            M[j, i] = M[i, j]

    return M

# -----------------------------------------------------------------------------

def P_matrix(n, m):
    'Takes m Shen basis function into n interior CG1 functions over -1, 1.'
    x = sp.Symbol('x')
    # Mesh size
    h = 2/(n+1)
    # All mesh vertices
    vertices = [-1 + h*i for i in range(n+2)]
    # Shen functions
    basis = shen_basis(m)
    
    P = np.zeros((n, m))
    # This loops computet integrals: i-th has * j-th shen
    for i in range(n):
        vertex_p = vertices[i]
        vertex = vertices[i+1]
        vertex_n = vertices[i+2]

        for j, shen in enumerate(basis):
            P[i, j] = 2*shen.evalf(subs={x:vertex})/h
            P[i, j] += -shen.evalf(subs={x:vertex_p})/h
            P[i, j] += -shen.evalf(subs={x:vertex_n})/h

    return P

# -----------------------------------------------------------------------------

n = 4
mesh = IntervalMesh(n, -1, 1)

V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v))*dx
m = inner(u, v)*dx

A = assemble(a)
M = assemble(m)

# Only take the inner part of matrices - i.e. of basis functions that are in H10
A = A.array()[1:n, 1:n]
M = M.array()[1:n, 1:n]
n = A.shape[0]  # One less than the num of elements

# The claim now is that A, M which are n x n matrices can be obtained as a limit
# from Ashen, Mshen, the m x m stiffness and mass matrices in the shenbasis, and
# the transformation matrix P which is n x m and takes the shenenfunctions to
# CG1 functions.
temp = 'm=%d, |A-A_|=%.2E, A_rate=%.2f, |M-M_|=%.2E, M_rate=%.2f'

import matplotlib.pyplot as plt
plt.figure()

for m in [2, 16, 32, 64]:
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

    plt.plot(P[0, :]) 
    
    A_norm_ = A_norm
    M_norm_ = M_norm
    m_ = m

plt.show()
