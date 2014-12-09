from __future__ import division
import numpy as np
import numpy.linalg as la
from math import sqrt
from sympy import legendre, symbols, lambdify
from sympy.mpmath import quad

x = symbols('x')


def legendre_basis():
    i = 0
    while True:
        yield legendre(i, x)
        i += 1


def shen_basis_1():
    '''
    Shen basis that has functions that are zero on (-1, 1). Yields diagonal
    stiffness matrix and tridiag. mass matrix.
    '''
    i = 0
    while True:
        yield (legendre(i, x) - legendre(i+2, x))/sqrt(4*i + 6)
        i += 1


def shen_basis_2():
    '''
    Shen basis that has functions that are zero on (-1, 1). Leads to dense
    matrices.
    '''
    i = 0
    while True:
        yield legendre(i+2, x)-(legendre(0, x)
                                if (i % 2) == 0 else legendre(1, x))
        i += 1


def Pmat(m, test=False):
    'This is matrix of boundary values of legendre polynomilas L0, ..., Lm-1'
    M = np.zeros((m, m))
    for i in range(0, m, 2):
        for j in range(0, m, 2):
            M[i, j] = 2
    for i in range(1, m, 2):
        for j in range(1, m, 2):
            M[i, j] = 2

    # Check if correct assembly
    if test:
        for i, li in zip(range(m), legendre_basis()):
            for j, lj in zip(range(m), legendre_basis()):
                f = li*lj
                value = f.evalf(subs={x: 1}) + f.evalf(subs={x: -1})
                assert abs(value - M[i, j]) < 1E-15

    return M

# We know that for Pmat with size mxm kernel has dimension m-2
m = 10
P = Pmat(m)
dim_ker = len(np.where(np.abs(la.eigvals(P)) < 1E-13)[0])
assert dim_ker == m-2


def Vmat(m, orthogonal):
    'Matrix whose columns span the entire kernel of Pmat(m)'
    E = np.zeros((m, m-2))
    for j in range(m-2):
        E[j, j] = 1
        E[j+2, j] = -1
    E = E.T

    if orthogonal:
        for i, u in enumerate(E):
            for v in E[:i]:
                u -= u.dot(v)*v
            u /= la.norm(u)
    return E.T

orthogonality = False
V = Vmat(m, orthogonality)
# Check
assert all(la.norm(P.dot(v))/m < 1E-13 for v in V.T)
if orthogonality:
    assert la.norm(V.T.dot(V) - np.eye(dim_ker))/dim_ker < 1E-13

# Show that alpha such that alpha.T = V*C where C is dim_ker * dim_ker
# diagonal diag(1/sqrt(4*i + 6)) and V not orthogonal leads to
# {basis} = alpha*{legendre basis} where {basis} is shen 1
C = np.diag(np.array([1/sqrt(4*i + 6) for i in range(dim_ker)]))
alpha = np.transpose(V.dot(C))

leg_basis = [li for (_, li) in zip(range(m), legendre_basis())]
new_basis = [sum(col*li for col, li in zip(row, leg_basis)) for row in alpha]
shen1_basis = [si for (_, si) in zip(range(dim_ker), shen_basis_1())]
shen2_basis = [si for (_, si) in zip(range(dim_ker), shen_basis_2())]

# compare L2 norms
for shen1, new in zip(shen1_basis, new_basis):
    e = shen1 - new
    e = lambdify(x, e**2)
    assert quad(e, [-1, 1]) < 1E-13


def assemble_stiffness_matrix(basis, domain=[-1, 1]):
    'Stiffness matrix in given basis.'
    n = len(basis)
    _basis = map(lambda f: lambdify(x, f.diff(x, 1)), basis)
    A = np.zeros((n, n))
    for i, bi in enumerate(_basis):
        A[i, i] = quad(lambda x: bi(x)**2, domain)
        for j, bj in enumerate(_basis[i+1:], i+1):
            A[i, j] = quad(lambda x: bi(x)*bj(x), domain)
            A[j, i] = A[i, j]
    return A


def assemble_mass_matrix(basis, domain=[-1, 1]):
    'Mass matrix in given basis.'
    n = len(basis)
    _basis = map(lambda f: lambdify(x, f), basis)
    M = np.zeros((n, n))
    for i, bi in enumerate(_basis):
        M[i, i] = quad(lambda x: bi(x)**2, domain)
        for j, bj in enumerate(_basis[i+1:], i+1):
            M[i, j] = quad(lambda x: bi(x)*bj(x), domain)
            M[j, i] = M[i, j]
    return M

# Transformation alpha is used to map matrices of operators
Aleg = assemble_stiffness_matrix(leg_basis)
Ashen1 = assemble_stiffness_matrix(shen1_basis)
assert la.norm(Ashen1 - (alpha.dot(Aleg)).dot(alpha.T))/m < 1E-13

Mleg = assemble_mass_matrix(leg_basis)
Mshen1 = assemble_mass_matrix(shen1_basis)
assert la.norm(Mshen1 - (alpha.dot(Mleg)).dot(alpha.T))/m < 1E-13

# Okay, it works so far and we have faith in the idea
# Lets check how conditioning of A, M effects the matrix of 2d poisson problem
# on [-1, 1] x [-1, 1]
def conditioning(basis):
    '''
    Take 2 and more basis functions and create the subspace. Then cond...
    '''
    conds = []
    for n in range(2, len(basis)):
        sub_basis = basis[:n]
        A = assemble_stiffness_matrix(sub_basis)
        M = assemble_mass_matrix(sub_basis)
        # 2d laplacian
        L = np.kron(A, M) + np.kron(M, A)

        condA = la.cond(A)
        condM = la.cond(M)
        condL = la.cond(L)

        conds.append([n, condA, condM, condL])
    conds = np.array(conds)
    return np.diff(conds, axis=0)

# for row in conditioning(shen1_basis):
#     print row

# for row in conditioning(shen2_basis):
#     print row

# -----------------------------------------------------------------------------

V = Vmat(m, True)
C = np.diag(2)
alpha = np.transpose(V.dot(C))
basis = [sum(col*li for col, li in zip(row, leg_basis)) for row in alpha]
print alpha.dot(assemble_mass_matrix(leg_basis).dot(alpha.T))
#print assemble_mass_matrix(leg_basis)
