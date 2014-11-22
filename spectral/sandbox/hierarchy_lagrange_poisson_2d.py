from __future__ import division
import sys
sys.path.insert(0, '../')

from points import gauss_legendre_lobatto_points as gl_points
from functions import lagrange_basis
from quadrature import GLQuadrature
from random import choice
from sympy import symbols, lambdify
from math import sqrt, log as ln
import scipy.linalg as la
import numpy as np
import time

# We use these quadraturs to integrate everything
quad1d = GLQuadrature([40])
quad2d = GLQuadrature([40, 40])


def hierararchical_basis(n):
    # Construct spaces by taking lagrangian polynomials nodal
    # at GLL points with bdry value 0 at -1, 1
    # For degree of polynomial higher than 2 there are multiple choices
    # for the polynomial -- here we make it random
    return [lagrange_basis([gl_points([i])])[choice(range(1, i-1))]
            for i in range(3, n)]


def solve_2d(f, E, domain, n):
    '''
        -E*laplace(u) = f in domain = [[ax, bx], [ay, by]]
                    u = 0 on boundary

    Following Jie Shen paper on Legendre polynomials.
    '''
    n > 3
    [[ax, bx], [ay, by]] = domain
    assert ax < bx and ay < by

    basis = hierararchical_basis(n)
    x = symbols('x')
    d_basis = [v.diff(x, 1) for v in basis]
    # Basis and derivative is only needed as lambdas
    basis = map(lambda f: lambdify(x, f), basis)
    d_basis = map(lambda f: lambdify(x, f), d_basis)

    # Stiffness matrix
    n = len(basis)
    A = np.zeros((n, n))

    start = time.time()
    for i, dbi in enumerate(d_basis):
        A[i, i] = quad1d.eval(lambda x: dbi(x)*dbi(x), [[-1, 1]])
        for j, dbj in enumerate(d_basis[i+1:], i+1):
            A[i, j] = quad1d.eval(lambda x: dbi(x)*dbj(x), [[-1, 1]])
            A[j, i] = A[i, j]
    asmbl = time.time() - start

    # Mass matrix
    M = np.zeros_like(A)

    start = time.time()
    for i, bi in enumerate(basis):
        M[i, i] = quad1d.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(basis[i+1:], i+1):
            M[i, j] = quad1d.eval(lambda x:  bi(x)*bj(x), [[-1, 1]])
            M[j, i] = M[i, j]
    asmbl += time.time() - start

    # Need scaling from [-1, 1] to [ax, bx] and [ay, by] respectively
    Lx = bx - ax
    Ly = by - ay
    # To get the right hand side first map f from domain to [-1, 1]^2
    x, y = symbols('x, y')
    f_hat = f.subs({x: 0.5*ax*(1-x) + 0.5*bx*(1+x),
                    y: 0.5*ay*(1-y) + 0.5*by*(1+y)})
    f_lambda = lambdify([x, y], f_hat)

    # The right hand side over [-1, 1]^2 = B, B_ij = int_[-1, 1] F*bi(x)*bj(y)
    B = np.zeros_like(M)

    start = time.time()
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis):
            B[i, j] = quad2d.eval(lambda x, y: f_lambda(x, y)*bi(x)*bj(y),
                                  [[-1., 1.], [-1., 1.]])
    asmbl += time.time() - start

    # Scale
    B *= 0.25*Lx*Ly

    # The tensor product method
    lmbda, V = la.eigh(A, M)
    # Map the right hand side to eigen space
    G = (V.T).dot(B.dot(V))
    # Apply inverse
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

    return (U, basis, asmbl)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from problems import manufacture_poisson_2d
    from sympy import exp, sin, pi
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import plots

    plot = False

    # Generate problem from specs
    x, y = symbols('x, y')
    [[ax, bx], [ay, by]] = [[0, 1.], [-1, 0]]
    domain = [[ax, bx], [ay, by]]
    u = exp(x)*x*(x-1)*sin(pi*y)
    E = 2

    problem = manufacture_poisson_2d(u=u, domain=domain, E=E)
    u = problem['u']
    f = problem['f']

    if plot:
        plots.plot(u, domain)

    ns = []
    errors = []
    assembly_times = []

    for n in range(4, 20):
        # Get the solution coefs and basis functions
        U, basis, assembly_time = solve_2d(f, E=E, domain=domain, n=n)

        # Assemble solution
        def uh(x, y):
            return sum(Ui*bi(x, y) for Ui, bi in zip(U.flatten(),
                                                     basis.flatten()))
        # We are actually interested only in the error
        u_lambda = lambdify([x, y], u)

        def e(x, y):
            return u_lambda(x, y) - uh(x, y)

        # Let's compute the L2 error
        error = quad2d.eval(lambda x, y: e(x, y)**2, domain)
        if error < 0:
            error = 0
        else:
            error = sqrt(error)

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

        ns.append(n)
        errors.append(error)
        assembly_times.append(assembly_time)

        if n > 4:
            rate = ln(error/error_)/ln(float(n_)/n)
            print 'n=%d error=%.4E rate=%.2f assembly_time=%g s' %\
                (n, error, rate, assembly_time)

        error_ = error
        n_ = n

        # No need to force machine precision
        if error < 1E-15:
            break

    # Plotting n vs error ans assembly time
    fig, ax1 = plt.subplots()
    ax1.semilogy(ns, errors, 'bo-')
    ax1.set_xlabel('$n$')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('$||e||_{L^2}$', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(ns, assembly_times, 'gs-')
    ax2.set_ylabel('$s$', color='g')
    for tl in ax2.get_yticklabels():
        tl.set_color('g')
    plt.show()
