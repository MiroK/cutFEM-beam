from __future__ import division
import numpy.linalg as la
from math import sqrt
import numpy as np

# Here we assume that everything is done with Legendre polynomials
# There is a matrix P which appears when setting dirichlet bcs at (-1, 1)
# There is a matrix Pp which appears with dirichlet bcs at -1
# There is a matrix Pm which appears with dirichlet bcs at 1
# For all three matrices I know how the dimension of their kernels
# And orthogonal basis (TODO also analytically)
# As a ground work for more general bc combinations we view the case (-1, 1)
# as 'combination of -1, 1' in a sense that vectors in ker of P must be in
# both ker(Pp) and ker(Pm) [For more general case, P might not be available
# while Pp and Pm are]. So thi interest is in finding the common kernel of
# two matrices. Fortunatelly ch 6.4 in Golub has something on this topic

def Pmat(m):
    'Matrix corresponding to Nitsche penalty for Dirichlet bcs at -1, 1.'
    mat = np.zeros((m, m))

    for i in range(0, m, 2):
        for j in range(0, m, 2):
            mat[i, j] = 2

    for i in range(1, m, 2):
        for j in range(1, m, 2):
            mat[i, j] = 2

    return mat


def Ppmat(m):
    'Matrix corresponding to Nitsche penalty for Dirichlet bcs at 1.'
    mat = np.ones((m, m))

    return mat


def Pmmat(m):
    'Matrix corresponding to Nitsche penalty for Dirichlet bcs at -1.'
    mat = np.ones((m, m))

    for i in range(0, m, 2):
        for j in range(1, m, 2):
            mat[i, j] = -1

    for i in range(1, m, 2):
        for j in range(0, m, 2):
            mat[i, j] = -1

    return mat


def ker_Ppmat(m, orthogonal):
    'Matrix whose columns are basis of ker(Ppmat(n)).'
    E = np.zeros((m, m-1))
    for j in range(m-1):
        E[j, j] = 1
        E[j+1, j] = -1

    if orthogonal:
        for j in range(E.shape[1]):
            v = E[:, j]
            for jj in range(j):
                v -= v.dot(E[:, jj])*E[:, jj]
            v /= sqrt(v.dot(v))

    return E


def ker_Pmmat(m, orthogonal):
    'Matrix whose columns are basis of ker(Pmmat(n)).'
    E = np.zeros((m, m-1))
    for j in range(m-1):
        E[j, j] = 1
        E[j+1, j] = 1

    if orthogonal:
        for j in range(E.shape[1]):
            v = E[:, j]
            for jj in range(j):
                v -= v.dot(E[:, jj])*E[:, jj]
            v /= sqrt(v.dot(v))

    return E


def ker_Pmat(m, orthogonal):
    'Matrix whose columns are basis of ker(Pmat(n)).'
    E = np.zeros((m, m-2))
    for j in range(m-2):
        E[j, j] = 1
        E[j+2, j] = -1

    if orthogonal:
        for j in range(E.shape[1]):
            v = E[:, j]
            for jj in range(j):
                v -= v.dot(E[:, jj])*E[:, jj]
            v /= sqrt(v.dot(v))

    return E

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    n = 5

    P = Pmat(n)
    V = ker_Pmat(n, orthogonal=True)
    assert la.norm(P.dot(V))/n < 1E-13
    assert la.norm(V.T.dot(V) - np.eye(n-2))/n < 1E-13

    Pp = Ppmat(n)
    Vp = ker_Ppmat(n, orthogonal=True)
    assert la.norm(Pp.dot(Vp))/n < 1E-13
    assert la.norm(Vp.T.dot(Vp) - np.eye(n-1))/n < 1E-13

    Pm = Pmmat(n)
    Vm = ker_Pmmat(n, orthogonal=True)
    assert la.norm(Pm.dot(Vm))/n < 1E-13
    assert la.norm(Vm.T.dot(Vm) - np.eye(n-1))/n < 1E-13

    for v in Vm.T:
        assert not la.norm(Pp.dot(v))/n < 1E-13

    for v in Vp.T:
        assert not la.norm(Pm.dot(v))/n < 2E-13

    # Golub 329
    s = 0        # dim(ker(Pp)*ker(Pm))
    Y = None     # Columns should be orthogonal basis of ker(Pp)*ker(Pm)
    Up, Sp, Vp = la.svd(Pp)
    r = la.matrix_rank(Sp)
    if r < n:
        # This is orthogonal basis of ker(Pp)
        Vp = Vp.T[:, r:]
        la.norm(Pp.dot(Vp))/n < 1E-13

        # Next build orthogonal basis of ker(Pm*Vp)
        C = Pm.dot(Vp)
        Uc, Sc, Vc = la.svd(C)
        q = la.matrix_rank(C)
        Vc = Vc.T[:, q:(n-r)]
        # S is the dim of common kernel
        # Y has colums of that are ON and are in ker(Pp) and ker(Pm)
        s = n - q - r
        Y = Vp.dot(Vc)
        assert Y.shape == (n, s)
        assert la.norm(Y.T.dot(Y) - np.eye(s))/s < 1E-13
        assert la.norm(Pp.dot(Y))/n < 1E-13
        assert la.norm(Pm.dot(Y))/n < 1E-13

    if s:
        # We know the dim of ker P
        assert s == n-2

    from sympy import legendre, symbols
    x = symbols('x')
    # Just take C as eye(m-2) and assemble new basis from Legendre basis
    alpha = Y.T
    basis = []
    for i in range(alpha.shape[0]):
        f = 0
        for j in range(alpha.shape[1]):
            f += alpha[i, j]*legendre(j, x)
        basis.append(f)
    # Check that it yields zeros at -1, 1
    for f in basis:
        print f.evalf(subs={x: -1}), f.evalf(subs={x: 1})
