from __future__ import division
from sympy import sin, cos, pi, sqrt, symbols, lambdify
from sympy.mpmath import quad
import numpy as np

x, y, s = symbols('x, y, s')

def eigen_basis(n):
    '''
    Return first n eigenfunctions of Laplacian over biunit interval with homog.
    Dirichlet bcs. at endpoints -1, 1.
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
    polynomials that have 0 values at -1, 1.
    '''
    k = 0
    functions = []
    while k < n:
        weight = 1/sqrt(4*k + 6)
        functions.append(weight*(legendre(k+2, x) - legendre(k, x)))
        k += 1
    return functions


def beam_restrict(A, B, u):
    'Restict function u of x, y to beam = {(x, y)=0.5*A*(1-s) + 0.5*B*(1+s)}.'
    assert x in u.atoms() and y in u.atoms()
    ux = u.subs(x, A[0]/2*(1-s) + B[0]/2*(1+s))
    u = ux.subs(y, A[1]/2*(1-s) + B[1]/2*(1+s))
    return u


def beam_L2_distance(A, B, f, g):
    'L2 norm of f restricted to beam - g.'
    f = beam_restrict(A, B, f)
    d = f-g
    d = lambdify(s, f-g)
    return sqrt(quad(lambda s: d(s)**2, [-1, 1]))


def beam_H10_distance(A, B, f, g):
    'H10 norm of f restricted to beam - g.'
    f = beam_restrict(A, B, f)
    d = (f-g).diff(s, 1)
    d = lambdify(s, f-g)
    return sqrt(quad(lambda s: d(s)**2, [-1, 1]))


def beam_distance_matrix(A, B, us, vs, norm):
    '''
    Compute distance in norm of all functions from us restricted to beam to all
    functions in vs.
    '''
    if norm == 'L2':
        distance = beam_L2_distance
    elif norm == 'H10':
        distance = beam_H10_distance
    else:
        raise ValueError

    m = len(us)
    n = len(vs)
    mat = np.zeros((m, n))
    size = m*n
    for i, f in enumerate(us):
        for j, g in enumerate(vs):
            mat[i, j] = distance(A, B, f, g)

    return mat

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from itertools import product

    # Number of plate functions in 1d, number of beam functions
    m = 10
    n = 10

    # Plate basis is a tensor product
    plate_basis = [fx*fy.subs(x, y)
                   for fx, fy in product(eigen_basis(m), eigen_basis(m))]
    # For beam make function of s
    beam_basis = [f.subs(x, s) for f in eigen_basis(n)]
    
    # Sample beam
    A = np.array([0, 0])
    B = np.array([1, 1])

    # Distance matrtices
    matL2 = beam_distance_matrix(A, B, plate_basis, beam_basis, norm='L2')
    matH10 = beam_distance_matrix(A, B, plate_basis, beam_basis, norm='H10')

    # Plotting
    plt.figure()
    plt.pcolor(matL2)
    plt.colorbar()

    plt.figure()
    plt.pcolor(matH10)
    plt.colorbar()

    plt.show()
