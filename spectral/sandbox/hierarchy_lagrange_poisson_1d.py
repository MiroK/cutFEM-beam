import sys
sys.path.insert(0, '../')

from points import gauss_legendre_lobatto_points as gl_points
from functions import lagrange_basis
from random import choice
from sympy.mpmath import quad
from sympy import symbols, lambdify
from math import sqrt, log as ln
import numpy.linalg as la
import numpy as np
import time


def hierararchical_basis(n):
    # Construct spaces by taking lagrangian polynomials nodal
    # at GLL points with bdry value 0 at -1, 1
    # For degree of polynomial higher than 2 there are multiple choices
    # for the polynomial -- here we make it random
    return [lagrange_basis([gl_points([i])])[choice(range(1, i-1))]
            for i in range(3, n)]


def solve_1d(f, E, domain, n):
    '''
        -E*u`` = f in domain = [a, b]
          u(a) = u(b) = 0

    '''
    assert n > 3
    [ax, bx] = domain
    assert ax < bx

    # Symbolic basis and its derivative
    basis = hierararchical_basis(n)
    degree_max = basis[-1].as_poly().degree()
    x = symbols('x')
    d_basis = [v.diff(x, 1) for v in basis]
    # Basis and derivative as lambdas
    basis_lambda = map(lambda f: lambdify(x, f), basis)
    d_basis_lambda = map(lambda f: lambdify(x, f), d_basis)

    # Jacobian of transoformation between [-1, 1] and [a, b]
    jac = 0.5*(bx-ax)

    # Stiffness matrix on [a, b]
    n = len(basis)
    A = np.zeros((n, n))
    start = time.time()
    for i, dbi in enumerate(d_basis_lambda):
        A[i, i] = quad(lambda x: dbi(x)*dbi(x), [-1, 1])
        for j, dbj in enumerate(d_basis_lambda[i+1:], i+1):
            A[i, j] = quad(lambda x: dbi(x)*dbj(x), [-1, 1])
            A[j, i] = A[i, j]
    asmbl = time.time() - start
    A *= E/jac

    # To get the right hand side first map f from domain to -1, 1
    f_hat = f.subs(x, 0.5*ax*(1-x) + 0.5*bx*(1+x))
    f_lambda = lambdify(x, f_hat)
    # Get components of the rhs vector as (F, bi), [-1, 1]
    start = time.time()
    b = np.array([quad(lambda x: f_lambda(x)*bi(x), [-1, 1])
                  for bi in basis_lambda], dtype='float64')
    asmbl += time.time() - start
    # Get to [a, b]
    b *= jac

    # Solve the system to get coefficients
    U = la.solve(A, b)

    # Map the basis to domain [a, b]
    basis_lambda = [lambda x, v=v: v((2.*x - (bx+ax))/(bx-ax))
                    for v in basis_lambda]

    return (U, basis_lambda, asmbl)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import exp
    from problems import manufacture_poisson_1d
    import matplotlib.pyplot as plt

    # Generate problem from specs
    x = symbols('x')
    a, b = -1., 2.
    u = exp(x)*(x-a)*(x-b)
    domain = [a, b]
    E = 2.

    problem = manufacture_poisson_1d(u=u, a=a, b=b, E=E)
    u = problem['u']
    f = problem['f']
    u_lambda = lambdify(x, u)

    ns = []
    errors = []
    assembly_times = []

    for n in range(4, 20):
        # Get the solution coefs and basis functions
        U, basis, assembly_time = solve_1d(f, E=E, domain=domain, n=n)

        # Assemble solution
        def uh(x):
            return sum(Ui*v(x) for Ui, v in zip(U, basis))

        # We are actually interested only in the error
        def e(x):
            return u_lambda(x) - uh(x)

        # Let's compute the L2 error
        error = quad(lambda x: e(x)*e(x), [a, b])
        if error < 0:
            error = 0
        else:
            error = sqrt(error)

        ns.append(n)
        errors.append(error)
        assembly_times.append(assembly_time)

        if n > 4:
            rate = ln(error/error_)/ln(float(n_)/n)
            print 'n=%d error=%.4E rate=%.2f' % (n, error, rate)

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
