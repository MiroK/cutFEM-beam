from __future__ import division
# Put py on path
import sys
sys.path.append('../')
# Solvers
from eigen_biharmonic import biharmonic_solver_1d
from eigen_biharmonic import biharmonic_solver_2d

# Common
from sympy import Symbol, exp, lambdify, integrate, simplify, symbols, sin, pi
from sympy.mpmath import quad
import numpy as np
from numpy.linalg import lstsq
from math import sqrt, log as ln
import numpy.linalg as la
from collections import defaultdict

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


def biharmonic_2d():
    '''
    Convergence test for Fourier-Galerkin solver. Rate measure in 0, 1, 2 norms.
    # OUTPUT GOES TO SINGLE FILE AND IS USED TO MAKE TABLE IN THE PAPER.
    '''
    # We take the solution used in eigen_biharmonic
    x, y = symbols('x, y')
    u = (x-1)**2*(x+1)**2*sin(pi*x)*(y-1)**2*(y+1)**2*sin(-pi*y)
    # Compute f for which u is the solution
    u_xx = u.diff(x, 2)
    u_yy = u.diff(y, 2)
    u_xxxx = u_xx.diff(x, 2)
    u_xxyy = u_xx.diff(y, 2)
    u_yyxx = u_yy.diff(x, 2)
    u_yyyy = u_yy.diff(y, 2)
    f = u_xxxx + u_xxyy + u_yyxx + u_yyyy

    # Numerical solution
    ns = range(2, 16)
    # Data for least square
    # The solution is expensive so after each solve we compute errors
    bs = defaultdict(list)
    col0 = []
    for n in ns:
        uh = biharmonic_solver_2d(f, n, as_sym=True)
        # Symbolic error for L2 and H1 and H2
        e0 = u - uh
        e1 = e0.diff(x, 1)**2 + e0.diff(y, 1)**2
        e2 = e0.diff(x, 2)**2 + 2*e0.diff(x, 2)*e0.diff(y, 2) + e0.diff(y, 2)**2

        # Lambdified integrand of L2 norm expression
        e0 = lambdify([x, y], e0)
        eL0 = lambda x, y: e0(x, y)**2
        error0, integral_error0 = quad(eL0, [-1, 1], [-1, 1], error=True,
                                       maxdegree=40)
        error0 = sqrt(error0)
        
        # Lambdified integrand of H1 norm expression
        e1 = lambdify([x, y], e1)
        eL1 = lambda x, y: e1(x, y)
        error1, integral_error1 = quad(eL1, [-1, 1], [-1, 1], error=True,
                                       maxdegree=40)
        error1 = sqrt(error1)
        
        # Lambdified integrand of H1 norm expression
        e2 = lambdify([x, y], e2)
        eL2 = lambda x, y: e2(x, y)
        error2, integral_error2 = quad(eL2, [-1, 1], [-1, 1], error=True,
                                       maxdegree=40)
        error2 = sqrt(error2)

        if n > ns[0]:
            rate0 = ln(error0/error0_)/ln(n_/n)
            rate1 = ln(error1/error1_)/ln(n_/n)
            rate2 = ln(error2/error2_)/ln(n_/n)
            print n, error0, rate0, integral_error0,\
                     error1, rate1, integral_error1,\
                     error2, rate2, integral_error2\

        error0_, error1_, error2_ = error0, error1, error2
        n_ = n

        # Save different rhs
        bs[0].append(ln(error0))
        bs[1].append(ln(error1))
        bs[2].append(ln(error2))
        # Same matrix
        col0.append(ln(n))

    # The rate should be e = n**(-p) + shift
    A = np.ones((len(col0), 2))
    A[:, 0] = col0
    for norm, b in bs.items():
        ans = lstsq(A, b)[0]
        p = -ans[0]
        print '\tLeast square rate in %d norm %.2f' % (norm, p)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from common import merge_tables

    # 1d
    if False:
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

    if True:
        biharmonic_2d()

