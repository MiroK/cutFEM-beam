from __future__ import division
# Put py on path
import sys
sys.path.append('../')
# Solvers
from eigen_poisson import poisson_solver_1d
from eigen_poisson import poisson_solver_2d

# Common
from sympy import Symbol, exp, lambdify, symbols
from sympy.mpmath import quad
import numpy as np
from numpy.linalg import lstsq
from math import sqrt, log as ln
from collections import defaultdict

def poisson_1d(norm):
    '''
    Convergence test for Fourier-Galerkin solver. Rate measure in norm.
    # OUTPUT REDIRECTED TO FILES(TABLES) AND THESE ARE MERGED FOR REPORT.
    '''
    # We take the solution used in eigen_poisson
    x = Symbol('x')
    u = (x**2 - 1)*exp(x)
    # Right hand side
    f = -u.diff(x, 2)
    # Numerical solution
    ns = range(2, 16)
    # Data for least square
    b = []
    col0 = []
    for n in ns:
        uh = poisson_solver_1d(f, n, as_sym=True)
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


def poisson_2d():
    '''
    Convergence test for Fourier-Galerkin solver. Rate measure in 0, 1 norms.
    # OUTPUT GOES TO SINGLE FILE AND IS USED TO MAKE TABLE IN THE PAPER.
    '''
    # We take the solution used in eigen_poisson
    x, y = symbols('x, y')
    u = (x**2 - 1)*exp(x)*(y**2 - 1)
    # Right hand side
    f = -u.diff(x, 2) - u.diff(y, 2)
    # Numerical solution
    ns = range(2, 6)
    # Data for least square
    # The solution is expensive so after each solve we compute errors
    bs = defaultdict(list)
    col0 = []
    for n in ns:
        uh = poisson_solver_2d(f, n, as_sym=True)
        # Symbolic error for L2 and H1
        e0 = u - uh
        e1 = e0.diff(x, 1)**2 + e0.diff(y, 1)**2
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

        if n > ns[0]:
            rate0 = ln(error0/error0_)/ln(n_/n)
            rate1 = ln(error1/error1_)/ln(n_/n)
            print n, error0, rate0, integral_error0, error1, rate1, integral_error1

        error0_, error1_ = error0, error1
        n_ = n

        # Save different rhs
        bs[0].append(ln(error0))
        bs[1].append(ln(error1))
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

    if False:
        # 1d
        # poisson_1d(norm=0)  # results/eig_p_1d_0
        # poisson_1d(norm=1)  # results/eig_p_1d_1
        files = ['./results/eig_p_1d_0', './results/eig_p_1d_1']
        rows = [0, -1]
        columns = [[0, 1, 2, 3], [1, 2, 3]]
        row_format = ['%d', '%.2E', '%.2f', '%1.0E', '%.2E', '%.2f', '%1.0E']
        header = [r'$n$', r'$e$', r'$p$', r'$E$', r'$e$', r'$p$', r'$E$']
        merge_tables(files, rows, columns, row_format, header)
    
    if True:
        poisson_2d()

        #files = ['./results/eig_p_2d']
        #rows = [0, -2]
        #columns = [[0, 1, 2, 3, 4, 5, 6]]
        #row_format = ['%d', '%.2E', '%.2f', '%1.0E', '%.2E', '%.2f', '%1.0E']
        #header = [r'$n$', r'$e$', r'$p$', r'$E$', r'$e$', r'$p$', r'$E$']
        #merge_tables(files, rows, columns, row_format, header)
