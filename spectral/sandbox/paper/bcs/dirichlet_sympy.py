from __future__ import division
from sympy.mpmath import matrix, svd
from math import sqrt
import numpy.linalg as la
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

    return matrix(mat.tolist())


def Ppmat(m):
    'Matrix corresponding to Nitsche penalty for Dirichlet bcs at 1.'
    mat = np.ones((m, m))

    return matrix(mat.tolist())


def Pmmat(m):
    'Matrix corresponding to Nitsche penalty for Dirichlet bcs at -1.'
    mat = np.ones((m, m))

    for i in range(0, m, 2):
        for j in range(1, m, 2):
            mat[i, j] = -1

    for i in range(1, m, 2):
        for j in range(0, m, 2):
            mat[i, j] = -1

    return matrix(mat.tolist())


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    n = 15

    P = Pmat(n)
    Pp = Ppmat(n)
    Pm = Pmmat(n)

    # Golub 329
    s = 0        # dim(ker(Pp)*ker(Pm))
    Y = None     # Columns should be orthogonal basis of ker(Pp)*ker(Pm)
    Up, Sp, Vp = svd(Pp)

    r = la.matrix_rank(np.array(Sp.tolist()))
    if r < n:
        # This is orthogonal basis of ker(Pp)
        Vp = Vp.T[:, r:]

        # Next build orthogonal basis of ker(Pm*Vp)
        C = Pm * Vp
        Uc, Sc, Vc = svd(C)
        q = la.matrix_rank(np.array(C.tolist()))
        Vc = Vc.T[:, q:(n-r)]
        # S is the dim of common kernel
        # Y has colums of that are ON and are in ker(Pp) and ker(Pm)
        s = n - q - r
        Y = Vp * Vc
        assert (Y.rows, Y.cols) == (n, s)

    if s:
        # We know the dim of ker P
        assert s == n-2

    from sympy import legendre, symbols
    x = symbols('x')
    # Just take C as eye(m-2) and assemble new basis from Legendre basis
    alpha = Y.T
    basis = []
    for i in range(alpha.rows):
        f = 0
        for j in range(alpha.cols):
            f += alpha[i, j]*legendre(j, x)
        basis.append(f)
    # Check that it yields zeros at -1, 1
    for f in basis:
        print f.evalf(subs={x: -1}), f.evalf(subs={x: 1})
