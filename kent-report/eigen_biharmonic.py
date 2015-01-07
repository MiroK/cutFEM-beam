from __future__ import division
from sympy import Symbol, lambdify, exp, pi, symbols
from sympy.mpmath import quad
import numpy as np
import numpy.linalg as la
from math import log as ln, sqrt
from eigen_basis import eigen_basis
from itertools import product
from scipy.linalg import eigh

def biharmonic_solver_1d(f, n):
    '''
    Simple solver for u(4) = f in (-1, 1) with Dirichlet bcs on u and u(2).
    Uses eigenbasis.
    '''
    # Assemble stiffness matrix A, we know about A-orthogonality
    eigenvalues = [float((pi/2 + k*pi/2)**4) for k in range(n)]
    A = np.diag(eigenvalues)

    # Assemble right hand side b
    x = Symbol('x')
    f = lambdify(x, f)
    basis = [lambdify(x, v) for v in list(eigen_basis(n))]
    b = np.zeros(n)
    for i, v in enumerate(basis):
        b[i] = quad(lambda x: v(x)*f(x), [-1, 1])

    # Solve and return solution - assembled linear combination
    U = la.solve(A, b)
    uh = lambda x: sum(Uk*v(x) for Uk, v in zip(U, basis))
    return uh

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import exp, integrate, sin, pi, simplify
    from numpy.linalg import solve, lstsq
    # Rest the poisson solvers
    # Rate in L2 norm should be 4

    # 1D
    # Exact solution computed from f
    x = Symbol('x')
    f = x*exp(x)
    # u4 = f
    u3 = integrate(f, x)
    u2 = integrate(u3, x)
    u1 = integrate(u2, x)
    u = integrate(u1, x) # + ax^3/6 + bx^/2 + cx + d

    mat = np.array([[-1/6, 1/2, -1, 1],
                    [1/6, 1/2, 1, 1],
                    [-1, 1, 0, 0],
                    [1, 1, 0, 0]])

    vec = np.array([-u.subs(x, -1),
                    -u.subs(x, 1),
                    -u2.subs(x, -1),
                    -u2.subs(x, 1)])

    a, b, c, d = la.solve(mat, vec)
    u += a*x**3/6 + b*x**2/2 + c*x + d

    # Check that it is the solution
    assert abs(u.evalf(subs={x: -1})) < 1E-15
    assert abs(u.evalf(subs={x: 1})) < 1E-15
    assert abs(u.diff(x, 2).evalf(subs={x: -1})) < 1E-15
    assert abs(u.diff(x, 2).evalf(subs={x: 1})) < 1E-15
    assert (simplify(u.diff(x, 4)) - f) == 0

    # Right hand side
    u = lambdify(x, u)
    # Numerical solution
    ns = range(2, 16)
    print '1D'

    # Colect data for least square fit of the rate
    b = []
    col0 = []
    for n in ns:
        uh = biharmonic_solver_1d(f, n)
        # L2 Error as lambda
        e = lambda x: (u(x) - uh(x))**2
        error = sqrt(quad(e, [-1, 1]))

        if n > ns[0]:
            rate = ln(error/error_)/ln(n_/n)
            print '\t', n, error, rate

        error_ = error
        n_ = n

        b.append(ln(error))
        col0.append(ln(n))

    # The rate should be e = n**(-p) + shift
    A = np.ones((len(b), 2))
    A[:, 0] = col0
    ans = lstsq(A, b)[0]
    p = -ans[0]
    print 'Least square rate %.2f' % p
