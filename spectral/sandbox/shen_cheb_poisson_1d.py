import sys
sys.path.insert(0, '../')

from sympy.mpmath import chebyt, quad, pi
from sympy import symbols, lambdify
from math import sqrt, log as ln
from functools import partial
import numpy.linalg as la
import numpy as np

# Chebyshev inner prodict weight
def w(x):
    return sqrt(1-x**2)**-1


def solve_1d(f, E, domain, n):
    '''
        -E*u`` = f in domain = [ax, bx]
          u(a) = u(b) = 0

    Following Jie Shen paper on Chebyshev polynomials.
    '''
    [ax, bx] = domain
    assert ax < bx
    # Jacobian of mappping [-1, 1] --> [ax, bx]
    jac = 0.5*(bx - ax)
    # Special basis of chebyshev polynomials with zero on the boundary
    basis = [lambda x, i=i: partial(chebyt, n=i)(x=x) -
             partial(chebyt, n=i+2)(x=x)
             for i in range(n)]

    # Stiffness matrix on [-1, 1]
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2*pi*(i+1)*(i+2)
        for j in range(i+2, n, 2):
            A[i, j] = 4*pi*(i+1)
    A *= E
    A /= jac**2

    # Need to map f to reference domain [-1, 1] by mapping F
    x = symbols('x')
    f_hat = f.subs(x, 0.5*ax*(1-x) + 0.5*bx*(1+x))
    f_lambda = lambdify(x, f_hat)
    # Get components of the rhs vector as (F, bi)_w, [-1, 1]

    b = np.array([quad(lambda x: f_lambda(x)*bi(x)*w(x), [-1, 1])
                  for bi in basis], dtype='float64')

    # Solve the system to get coefficients
    U = la.solve(A, b)
    # So now I have a solution over domain [-1, 1]. Don't push it [ax, bx]
    # Rather evaluate the error in [-1, 1] by pullback the exact solution

    return (U, basis)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import exp
    from problems import manufacture_poisson_1d
    import matplotlib.pyplot as plt

    # Generate problem from specs
    x = symbols('x')
    a, b = -1., 2.
    u = exp(x)*(x-a)*(x-b)
    # Pull back u to reference domain for error computation
    u_hat = u.subs(x, 0.5*a*(1-x) + 0.5*b*(1+x))
    domain = [a, b]
    E = 2.

    problem = manufacture_poisson_1d(u=u, a=a, b=b, E=E)
    u = problem['u']
    f = problem['f']
    u_hat_lambda = lambdify(x, u_hat)

    for n in range(1, 15):
        # Get the solution coefs and basis functions
        U, basis = solve_1d(f, E=E, domain=domain, n=n)

        # Assemble solution
        def uh(x):
            return sum(Ui*v(x) for Ui, v in zip(U, basis))

        # We are actually interested only in the error
        def e(x):
            return u_hat_lambda(x) - uh(x)

        # Let's compute the L2 error over [-1, 1]!
        error = quad(lambda x: e(x)**2*w(x), [-1, 1])
        if error < 0:
            error = 0
        else:
            error = sqrt(error)

        if n > 1:
            rate = ln(error/error_)/ln(float(n_)/n)
            print 'n=%d error=%.4E rate=%.2f' % (n, error, rate)

        error_ = error
        n_ = n

    # Plot the final solution over [-1, 1]
    x_values = np.linspace(-1, 1, 100)
    u_values = map(u_hat_lambda, x_values)
    uh_values = map(uh, x_values)
    plt.figure()
    plt.plot(x_values, u_values, label='exact')
    plt.plot(x_values, uh_values, label='numeric')
    plt.legend(loc='best')
    plt.show()
