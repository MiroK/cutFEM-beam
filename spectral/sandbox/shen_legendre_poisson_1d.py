import sys
sys.path.insert(0, '../')

from problems import manufacture_poisson_1d

from points import gauss_legendre_points as gl_points
from scipy.interpolate import BarycentricInterpolator
from sympy.mpmath import legendre, quad, diff
from sympy import symbols, lambdify
from math import sqrt, log as ln
from functools import partial
import numpy.linalg as la
import numpy as np


def solve_1d(f, E, domain, n):
    '''
        -E*u`` = f in domain = [a, b]
          u(a) = u(b) = 0

    Following Jie Shen paper on Legendre polynomials.
    '''
    [ax, bx] = domain
    assert ax < bx

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

    # Let's assemble the matrices. Note that they are on [-1, 1]
    # and must be mapped to domain with Jacobians
    jac = 0.5*(bx-ax)

    # Mass matrix on [-1, 1], not needed here
    # M = np.zeros((n, n))
    # for i in range(n):
    #     M[i, i] = c(i)*c(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
    #     for j in range(i+1, n):
    #         M[i, j] = -c(i)*c(j)*(2./(2*j+1)) if i+2 == j else 0
    #         M[j, i] = M[i, j]
    # M = jac*M

    # Stiffness matrix on [a, b]
    # A = np.zeros((n, n))
    # d_basis = [lambda x, v=v: diff(v, x) for v in basis]
    # for i, bi in enumerate(d_basis):
    #     A[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
    #     for j, bj in enumerate(d_basis[i+1:], i+1):
    #         A[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
    #         A[j, i] = A[i, j]
    # A *= E/jac
    A = np.eye(n)*E/jac

    # To get the right hand side first map f from domain to -1, 1
    x = symbols('x')
    f_hat = f.subs(x, 0.5*ax*(1-x) + 0.5*bx*(1+x))
    f_lambda = lambdify(x, f_hat)
    # Further we represent f_hat as a polynomial
    # for n --> n+2 is max degree
    points = gl_points([n+3])
    values = map(f_lambda, points)
    F = BarycentricInterpolator(points, values)
    # Get components of the rhs vector as (F, bi), [-1, 1]
    b = np.array([quad(lambda x: F(x)*bi(x), [-1, 1])
                  for bi in basis], dtype='float64')
    # Get to [a, b]
    b *= jac

    # Solve the system to get coefficients
    U = la.solve(A, b)

    # Map the basis to domain [a, b]
    basis = [lambda x, v=v: v((2.*x - (bx+ax))/(bx-ax)) for v in basis]

    return (U, basis)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import exp
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

    for n in range(1, 15):
        # Get the solution coefs and basis functions
        U, basis = solve_1d(f, E=E, domain=domain, n=n)

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

        if n > 1:
            rate = ln(error/error_)/ln(float(n_)/n)
            print 'n=%d error=%.4E rate=%.2f' % (n, error, rate)

        error_ = error
        n_ = n

