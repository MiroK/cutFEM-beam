from __future__ import division
import sys
sys.path.insert(0, '../')

from problems import manufacture_poisson_2d
from scipy.interpolate import interp2d
from points import gauss_legendre_points as gl_points
from sympy.mpmath import legendre
from sympy import symbols, lambdify
from quadrature import GLQuadrature
from math import sqrt, log as ln
from itertools import product
from functools import partial
import scipy.linalg as la
import numpy as np
import time


def solve_2d(f, E, domain, n):
    '''
        -E*laplace(u) = f in domain = [[ax, bx], [ay, by]]
                    u = 0 on boundary

    Following Jie Shen paper on Legendre polynomials.
    '''
    [[ax, bx], [ay, by]] = domain
    assert ax < bx and ay < by

    # The basis functions of shen are certain combinations
    # of Legendre polynomials such that the values at -1, 1 are
    # 0, the mass matrix is 3 diagonal and the stiffness matrix
    # is diagonal
    def c(k):
        return sqrt(4*k + 6)**-1

    # For n we have n functions that are quadratic ... n+2
    # degree polynomials
    # Return lambda such that lambda(x) = basis_i(x)
    basis = [lambda x, i=i: c(i)*(partial(legendre, n=i)(x=x) -
                                  partial(legendre, n=i+2)(x=x))
             for i in range(n)]
    # The 2d basis functions are bi(x)*bj(y) for bi in basis for bj in basis

    # Need scaling from [-1, 1] to [ax, bx] and [ay, by] respectively
    Lx = bx - ax
    Ly = by - ay

    # Mass matrix on [-1, 1]
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = c(i)*c(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, n):
            M[i, j] = -c(i)*c(j)*(2./(2*j+1)) if i+2 == j else 0
            M[j, i] = M[i, j]

    # To get the right hand side first map f from domain to [-1, 1]^2
    x, y = symbols('x, y')
    f_hat = f.subs({x: 0.5*ax*(1-x) + 0.5*bx*(1+x),
                    y: 0.5*ay*(1-y) + 0.5*by*(1+y)})
    f_lambda = lambdify([x, y], f_hat)
    # Further we represent f_hat as a polynomial
    # for n --> n+2 is max degree
    points = gl_points([n+3])
    values = np.array([f_lambda(x_, y_)
                       for x_, y_ in product(points, points)])
    values = values.reshape((len(points), len(points)))
    F = interp2d(points, points, values.T)

    # The right hand side over [-1, 1]^2 = B, B_ij = int_[-1, 1] F*bi(x)*bj(y)
    # The inner product is at least polynomial of degree 2*n + 2
    # Note that this is not exact
    start = time.time()
    B = np.zeros_like(M)
    _errors = []
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis):
            quad = GLQuadrature([n+3, n+3])
            B[i, j], _e = quad.eval_adapt(lambda x, y:
                                          F(float(x), float(y))[0]*bi(x)*bj(y),
                                          [[-1., 1.], [-1., 1.]],
                                          n_refs=10,
                                          eps=1E-8,
                                          error=True)
            _errors.append(_e)

    # Scale
    B *= 0.25*Lx*Ly
    print '\t\t Assembled B in %g s with error in [%.2E, %.2E]' % \
        (time.time() - start, min(_errors), max(_errors))

    # The tensor product method
    lmbda, V = la.eigh(M)
    # Map the right hand side to eigen space
    G = (V.T).dot(B.dot(V))
    U_ = np.array([[G[i, j]/(E*lmbda[i]*Ly/Lx + E*lmbda[j]*Lx/Ly)
                    for j in range(G.shape[0])]
                   for i in range(G.shape[1])])
    # Map back to physical space
    U = V.dot(U_.dot(V.T))

    # Map basis to domain
    basis = np.array([[lambda x, y, bi=bi, bj=bj:
                       bi((2.*x - (bx+ax))/Lx)*bj((2.*y - (by+ay))/Ly)
                       for bj in basis]
                      for bi in basis])

    return U, basis


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import exp, sin, pi
    from sympy.plotting import plot3d
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # Generate problem from specs
    x, y = symbols('x, y')
    [[ax, bx], [ay, by]] = [[0, 1.], [-1, 0]]
    domain = [[ax, bx], [ay, by]]
    u = exp(x)*x*(x-1)*sin(pi*y)
    E = 2

    plot = False

    problem = manufacture_poisson_2d(u=u, domain=domain, E=E)
    u = problem['u']
    f = problem['f']

    for n in range(1, 6):
        # Exact
        if plot:
            plot3d(u, (x, ax, bx), (y, ay, by), xlabel='$x$', ylabel='$y$')

        (U, basis) = solve_2d(f, domain=domain, E=E, n=n)

        # Numeric
        def uh(x, y):
            return sum(Ui*bi(x, y) for Ui, bi in zip(U.flatten(),
                                                     basis.flatten()))

        # Plot
        if plot:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            x_ = np.linspace(ax, bx, 20)
            y_ = np.linspace(ay, by, 20)
            X, Y = np.meshgrid(x_, y_)
            Z = np.zeros_like(X)
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    Z[i, j] = uh(X[i, j], Y[i, j])

            axes.plot_surface(X, Y, Z, cstride=1, rstride=1,
                              cmap=plt.get_cmap('jet'))
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.show()

        # Error in L2 norm
        u_lambda = lambdify([x, y], u)

        def e(x, y):
            return u_lambda(x, y) - uh(x, y)

        quad = GLQuadrature([n+3, n+3])
        error = quad.eval_adapt(lambda x, y: e(x, y)**2,
                                domain, n_refs=10, eps=1E-8)

        if error < 0:
            error = 0
        else:
            error = sqrt(error)

        if n > 1:
            rate = ln(error/error_)/ln(float(n_)/n)
            print 'n=%d error=%.4E rate=%.2f' % (n, error, rate)

        error_ = error
        n_ = n
