from __future__ import division
from dolfin import *
import numpy as np
import numpy.linalg as la
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

n = 4
order = 1

mesh = IntervalMesh(n, -1, 1)

V = FunctionSpace(mesh, 'CG', order)
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
# from Aeig, Meig, the m x m stiffness and mass matrices in the eigenbasis, and
# the transformation matrix P which is n x m and takes the eigenfunctions to CG1
# functions.
temp = 'm=%d, |A-A_|=%.2E, A_rate=%.2f, |M-M_|=%.2E, M_rate=%.2f, lmbda=%.2E lmbda_rate=%.2f'
for m in [16, 32, 64, 128, 256]:
    eigen_functions = eigen_basis(m)
    Aeig = Aeig_matrix(m)
    Meig = Meig_matrix(m)

    # The transformation matrix has P_ij = (cg_i, eig_j). To get the integral
    # exactly we represent first each cg function on a much finner mesh. Further
    # degree of expression for eig_j should yield higher order quadrature so
    # that the resulting integral is sufficiently accurate
    mesh_fine = IntervalMesh(10000, -1, 1)
    V_fine = FunctionSpace(mesh_fine, 'CG', order)
    v = Function(V)

    P = np.zeros((n, m))
    for i in range(n):
        # Create i-th test function on V
        cg_values = np.zeros(V.dim())
        cg_values[i+1] = 1
        v.vector()[:] = cg_values
        cg = Function(V, v)


        # Now represent it in finer space
        cg = interpolate(cg, V_fine)
        
        for j, f in enumerate(eigen_functions):
            P[i, j] = assemble(inner(cg, f)*dx)

    # Compute A_ as P*Aeig*.Pt ans same for M
    A_ = P.dot(Aeig.dot(P.T))
    M_ = P.dot(Meig.dot(P.T))

    A_norm = la.norm(A-A_, 2)
    M_norm = la.norm(M-M_, 2)
    lmbda = np.sort(la.eigvals(M-M_))[-1]
    if m != 16:
        rateA = ln(A_norm/A_norm_)/ln(m_/m)
        rateM = ln(M_norm/M_norm_)/ln(m_/m)
        rate_lmbda = ln(lmbda/lmbda_)/ln(m_/m)

        print temp % (m, la.norm(A-A_), rateA, la.norm(M-M_), rateM,
                      lmbda, rate_lmbda)

    A_norm_ = A_norm
    M_norm_ = M_norm
    lmbda_ = lmbda
    m_ = m

# print la.norm(P - P_matrix(n, m))

