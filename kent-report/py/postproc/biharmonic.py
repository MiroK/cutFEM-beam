from __future__ import division
# Put py on path
import sys
sys.path.append('../')
# Solvers
from eigen_biharmonic import biharmonic_solver_1d

# Common
from sympy import Symbol, exp, lambdify, integrate, simplify
from sympy.mpmath import quad
import numpy as np
from numpy.linalg import lstsq
from math import sqrt, log as ln
import numpy.linalg as la

def biharmonic_1d(norm):
    'Convergence test for Fourier-Galerkin solver. Rate measured in norm.'
    # Exact solution computed from f, again code take from eige_biharmonic
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

    # Numerical solution
    ns = range(2, 16)
    # Data for least square
    b = []
    col0 = []
    for n in ns:
        uh = biharmonic_solver_1d(f, n, as_sym=True)
        # Symbolic error
        e = u - uh
        # Norm decide derivative
        e = e.diff(x, norm)
        # Lambdified the integrand of norm expression
        e = lambdify(x, e)
        eL = lambda x: e(x)**2
        error, integral_error = quad(eL, [-1, 1], error=True)
        error = sqrt(error)

        if n > ns[0]:
            rate = ln(error/error_)/ln(n_/n)
            print n, error, rate, integral_error

        error_ = error
        n_ = n

        b.append(ln(error))
        col0.append(ln(n))

    # The rate should be e = n**(-p) + shift
    A = np.ones((len(b), 2))
    A[:, 0] = col0
    ans = lstsq(A, b)[0]
    p = -ans[0]
    print '\tLeast square rate %.2f' % p

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from common import merge_tables
    # 1d
    #biharmonic_1d(norm=0)  # results/eig_b_1d_0
    #biharmonic_1d(norm=1)  # results/eig_b_1d_1
    #biharmonic_1d(norm=2)  # results/eig_b_1d_2
    files = ['./results/eig_b_1d_0', './results/eig_b_1d_1',
            './results/eig_b_1d_2']
    rows = [0, -1]
    columns = [[0, 1, 2, 3], [1, 2, 3], [1, 2, 3]]
    row_format = ['%d',
                  '%.2E', '%.2f', '%1.0E',
                  '%.2E', '%.2f', '%1.0E',
                  '%.2E', '%.2f', '%1.0E']
    header = [r'$n$',
            r'$e$', r'$p$', r'$E$',
            r'$e$', r'$p$', r'$E$',
            r'$e$', r'$p$', r'$E$']
    merge_tables(files, rows, columns, row_format, header)
