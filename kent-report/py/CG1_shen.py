from __future__ import division
from dolfin import *
import numpy as np
import numpy.linalg as la
import sympy as sp
from sympy.printing.ccode import CCodePrinter


class DolfinCodePrinter(CCodePrinter):
    def __init__(self, settings={}):
        CCodePrinter.__init__(self)

    def _print_Pi(self, expr):
        return 'pi'


def dolfincode(expr, assign_to=None, **settings):
    # Print scalar expression
    dolfin_xs = sp.symbols('x[0] x[1] x[2]')
    xs = sp.symbols('x y z')

    for x, dolfin_x in zip(xs, dolfin_xs):
        expr = expr.subs(x, dolfin_x)
    return DolfinCodePrinter(settings).doprint(expr, assign_to)


def shen_basis(n):
    'Return Shen basis of H^1_0((-1, 1)).'
    x = sp.Symbol('x')
    k = 0
    functions = []
    while k < n:
        weight = 1/sp.sqrt(4*k + 6)
        f = weight*(sp.legendre(k+2, x) - sp.legendre(k, x))

        # Now we turn the sympy symbolic to expression
        f = Expression(dolfincode(f), degree=8)

        functions.append(f)
        k += 1
    return functions


def Ashen_matrix(m):
    'Stiffness matrix w.r.t to Shen basis.'
    return np.eye(m)


def Mshen_matrix(m):
    'Mass matrix w.r.t to Shen basis.'
    weight = lambda k: 1/sqrt(4*k + 6)
    M = np.zeros((m, m))
    for i in range(m):
        M[i, i] = weight(i)*weight(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, m):
            M[i, j] = -weight(i)*weight(j)*(2./(2*j+1)) if i+2 == j else 0
            M[j, i] = M[i, j]

    return M

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
for m in [4, 8]:# 12, 16, 20]:
    shen_functions = shen_basis(m)
    Ashen = Ashen_matrix(m)
    Mshen = Mshen_matrix(m)

    # The transformation matrix has P_ij = (cg_i, shen_j). To get the integral
    # exactly we represent first each cg function on a much finner mesh. Further
    # degree of expression for shen_j should yield higher order quadrature so
    # that the resulting integral is sufficiently accurate
    mesh_fine = IntervalMesh(10000, -1, 1)
    V_fine = FunctionSpace(mesh_fine, 'CG', 1)
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

        for j, f in enumerate(shen_functions):
            P[i, j] = assemble(inner(cg, f)*dx)

    # Compute A_ as P*Ashen*.Pt ans same for M
    A_ = P.dot(Ashen.dot(P.T))
    M_ = P.dot(Mshen.dot(P.T))

    A_norm = la.norm(A-A_)
    M_norm = la.norm(M-M_)

    if m != 4:
        rateA = ln(A_norm/A_norm_)/ln(m_/m)
        rateM = ln(M_norm/M_norm_)/ln(m_/m)

        #print temp % (m, la.norm(A-A_), rateA, la.norm(M-M_), rateM)

    A_norm_ = A_norm
    M_norm_ = M_norm
    m_ = m

    print P
