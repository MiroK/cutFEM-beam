from __future__ import division
from sympy import sin, cos, pi, sqrt, symbols, lambdify
from sympy.mpmath import quad
import numpy as np

x, y, s = symbols('x, y, s')

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
            functions.append(cos(alpha*x))
        else:
            functions.append(sin(alpha*x))
        k += 1
    return functions


def shen_basis(n):
    '''
    Return first n Shen basis functions. Special polynomials made of Legendre
    polynomials that have 0 values at -1, 1. Functions of x.
    '''
    k = 0
    functions = []
    while k < n:
        weight = 1/sqrt(4*k + 6)
        functions.append(weight*(legendre(k+2, x) - legendre(k, x)))
        k += 1
    return functions


def beam_restrict(A, B, u):
    '''
    Restict function(s) u of x, y to beam = {(x, y)=0.5*A*(1-s) + 0.5*B*(1+s)}.
    '''
    if isinstance(u, list):
        return [beam_restrict(A, B, v) for v in u]
    else:
        assert x in u.atoms() and y in u.atoms()
        ux = u.subs(x, A[0]/2*(1-s) + B[0]/2*(1+s))
        u = ux.subs(y, A[1]/2*(1-s) + B[1]/2*(1+s))
        return u


def L2_distance(f, g):
    'L2 norm over [-1, 1] of f-g.'
    d = f-g
    d = lambdify(s, d)
    return sqrt(quad(lambda s: d(s)**2, [-1, 1]))


def H10_distance(f, g):
    'H10 norm over [-1, 1] of f-g.'
    d = (f-g).diff(s, 1)
    d = lambdify(s, d)
    return sqrt(quad(lambda s: d(s)**2, [-1, 1]))


def distance_matrices(A, B, Vp, Vb, Q, norm):
    '''
    Given beam specified by A, B return two matrices. The first matrix has
    norm(u-q) where u are functions from Vp restricted to beam and q are
    functions from Q. The other matrix is norm(p-q) for p in Vb and Q in 
    Q.
    '''
    if norm == 'L2':
        distance = L2_distance
    elif norm == 'H10':
        distance = H10_distance
    else:
        raise ValueError

    m, n, r = len(Vp), len(Vb), len(Q)
    mat0 = np.zeros((m, r))
    # First do the restriction
    Vp = beam_restrict(A, B, Vp)
    for i, u in enumerate(Vp):
        for j, q in enumerate(Q):
            mat0[i, j] = distance(u, q)

    mat1 = np.zeros((n, r))
    for i, p in enumerate(Vb):
        for j, q in enumerate(Q):
            mat1[i, j] = distance(p, q)

    return mat0, mat1

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from itertools import product

    # Number of plate function in 1d, number of beam functions and number of
    # functions for Lagrange multiplier space
    m, n, r = 20, 20, 20

    # Vp basis - functions of x, y
    Vp = [fx*fy.subs(x, y) for fx, fy in product(eigen_basis(m), eigen_basis(m))]
    # Vb basis - functions of s
    Vb = [f.subs(x, s) for f in eigen_basis(n)]
    # Q basis - functions of s
    Q = [f.subs(x, s) for f in eigen_basis(r)]

    # Sample beam
    A = np.array([0, 0])
    B = np.array([1, 1])

    for norm in ['L2', 'H10']:
        matBp, matBb = distance_matrices(A, B, Vp, Vb, Q, norm)

        plt.figure()
        plt.title(norm)
        plt.pcolor(matBp)
        plt.xlabel('$Q$')
        plt.ylabel('$V_p$')
        plt.colorbar()

        plt.figure()
        plt.title(norm)
        plt.pcolor(matBb)
        plt.xlabel('$Q$')
        plt.ylabel('$V_b$')
        plt.colorbar()

    plt.show()
