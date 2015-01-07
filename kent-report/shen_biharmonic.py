from __future__ import division
from sympy import Symbol, lambdify, exp, pi, symbols
from sympy.mpmath import quad
import numpy as np
import numpy.linalg as la
from math import log as ln, sqrt
from shenb_basis import shenb_basis
from itertools import product

def biharmonic_solver_1d(f, n):
    '''
    Solver for u```` = f in (-1, 1) with 0 bdry value for u and du.
    Using shenb_basis
    '''
    # Assemble stiffness matrix A, we know about A-orthogonality
    A = np.eye(n)

    # Assemble right hand side b
    x = Symbol('x')
    f = lambdify(x, f)
    basis = [lambdify(x, v) for v in list(shenb_basis(n))]
    b = np.zeros(n)
    for i, v in enumerate(basis):
        b[i] = quad(lambda x: v(x)*f(x), [-1, 1])

    # Solve and return solution - assembled linear combination
    U = la.solve(A, b)
    uh = lambda x: sum(Uk*v(x) for Uk, v in zip(U, basis))
    return uh

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import exp, integrate, sin, pi, simplify, cos
    from numpy.linalg import solve, lstsq
    # Test the biharmonic solvers
    # Rate in L2 norm should be exponential

    # 1D
    # Exact solution computed from f
    x = Symbol('x')
    f = sin(x)*exp(x)
    # u4 = f
    u3 = integrate(f, x)
    u2 = integrate(u3, x)
    u1 = integrate(u2, x)
    u = integrate(u1, x) # + ax^3/6 + bx^/2 + cx + d

    # Note that the first derivative is here, unlike u2 for eigen!
    mat = np.array([[-1/6, 1/2, -1, 1],
                    [1/6, 1/2, 1, 1],
                    [1/2, -1, 1, 0],
                    [1/2, 1, 1, 0]])

    vec = np.array([-u.subs(x, -1),
                    -u.subs(x, 1),
                    -u1.subs(x, -1),
                    -u1.subs(x, 1)])

    a, b, c, d = la.solve(mat, vec)
    u_sym = u + a*x**3/6 + b*x**2/2 + c*x + d

    # Check that it is the solution
    assert abs(u_sym.evalf(subs={x: -1})) < 1E-13
    assert abs(u_sym.evalf(subs={x: 1})) < 1E-13
    assert abs(u_sym.diff(x, 1).evalf(subs={x: -1})) < 1E-13
    assert abs(u_sym.diff(x, 1).evalf(subs={x: 1})) < 1E-13
    assert simplify(simplify(u_sym.diff(x, 4)) - f) == 0

    # Right hand side
    u = lambdify(x, u_sym)
    # Numerical solution
    ns = range(2, 16)
    print '1D'

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

    # 2D
    # Let's manufacture a solution
    # We shall have u which solves the 1d problem for some f and
    # v which solver the 1d problem for some g. Then u*v has correct bcs.
    # We already have u
    u = u_sym
    # Compute v
    g = cos(x)*exp(-x)
    # v4 = g
    v3 = integrate(g, x)
    v2 = integrate(v3, x)
    v1 = integrate(v2, x)
    v = integrate(v1, x) # + ax^3/6 + bx^/2 + cx + d

    # Note that the first derivative is here, unlike u2 for eigen!
    mat = np.array([[-1/6, 1/2, -1, 1],
                    [1/6, 1/2, 1, 1],
                    [1/2, -1, 1, 0],
                    [1/2, 1, 1, 0]])

    vec = np.array([-v.subs(x, -1),
                    -v.subs(x, 1),
                    -v1.subs(x, -1),
                    -v1.subs(x, 1)])

    a, b, c, d = la.solve(mat, vec)
    v += a*x**3/6 + b*x**2/2 + c*x + d

    # Check that it is the solution
    assert abs(v.evalf(subs={x: -1})) < 1E-13
    assert abs(v.evalf(subs={x: 1})) < 1E-13
    assert abs(v.diff(x, 1).evalf(subs={x: -1})) < 1E-13
    assert abs(v.diff(x, 1).evalf(subs={x: 1})) < 1E-13
    assert simplify(simplify(v.diff(x, 4)) - g) == 0
    # Finally turn it into a function of y
    y = symbols('y')
    v = v.subs(x, y)
    # The 2d function
    uv = u*v

    # Compute the rhs for 2d problem
    uv_xx = uv.diff(x, 2)
    assert simplify(uv_xx - u.diff(x, 2)*v) == 0
    uv_yy = uv.diff(y, 2)
    assert simplify(uv_yy - u*v.diff(y, 2)) == 0

    uv_xxxx = uv_xx.diff(x, 2)
    uv_xxyy = uv_xx.diff(y, 2)
    uv_yyxx = uv_yy.diff(x, 2)
    uv_yyyy = uv_yy.diff(y, 2)
    # Here it is
    fg = uv_xxxx + uv_xxyy + uv_yyxx + uv_yyyy

    # TODO there should be a relation between fg and f, g, u, v
