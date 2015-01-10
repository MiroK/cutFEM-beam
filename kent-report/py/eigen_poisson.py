from __future__ import division
from sympy import Symbol, lambdify, exp, pi, symbols
from sympy.mpmath import quad
import numpy as np
import numpy.linalg as la
from math import log as ln, sqrt
from eigen_basis import eigen_basis
from itertools import product

# Matrices in the basis have tabulized values
def mass_matrix(m):
    'Matrix of identity operator w.r.t to Eigen basis.'
    return np.eye(m)


def laplacian_matrix(m):
    'Matrix of Laplacian operator w.r.t to Eigen basis.'
    return np.diag([float((pi/2 + k*pi/2)**2) for k in range(m)])


def poisson_solver_1d(f, n, as_sym=False, with_coefs=False):
    '''
    Simple Poisson solver for -u`` = f in (-1, 1) with Dirichlet bcs. Uses
    eigenbasis.
    '''
    # Assemble stiffness matrix A, we know about A-orthogonality
    eigenvalues = [float((pi/2 + k*pi/2)**2) for k in range(n)]
    A = laplacian_matrix(n)

    # Assemble right hand side b
    x = Symbol('x')
    f = lambdify(x, f)
    # Symbolic basis
    sym_basis = list(eigen_basis(n))
    # Lambdified
    basis = [lambdify(x, v) for v in sym_basis]
    b = np.zeros(n)
    for i, v in enumerate(basis):
        b[i] = quad(lambda x: v(x)*f(x), [-1, 1])

    # Solve and return solution - assembled linear combination
    U = la.solve(A, b)
    # Retur either lambdified or symbolic solution. The latter is better for
    # error computation - symbolic differentiation
    if as_sym:
        uh = sum(Uk*v for Uk, v in zip(U, sym_basis))
    else:
        uh = lambda x: sum(Uk*v(x) for Uk, v in zip(U, basis))

    if with_coefs:
        return uh, U
    else:
        return uh


def poisson_solver_2d(f, n, as_sym=False):
    '''
    Simple Poisson solver -laplace(u) = f in (-1, 1)^2 with Dirichlet bcs.
    Uses tensor product of eigenbasis.
    '''
    # Use the tensor product method so only 1d matrices are needed
    # Stiffness matrix
    eigenvalues = [float((pi/2 + k*pi/2)**2) for k in range(n)]
    A = laplacian_matrix(n)
    # Mass matrix
    M = mass_matrix(n)

    # Create tensor product basis
    x, y = symbols('x, y')
    sym_basis = [vx*(vy.subs(x, y))
                 for vx, vy in product(eigen_basis(n), eigen_basis(n))]
    basis = [lambdify([x, y], v) for v in sym_basis]

    # Assemble right hand side b
    f = lambdify([x, y], f)
    b = np.zeros((n, n))
    for k, v in enumerate(basis):
        i, j = k // n, k % n
        b[i, j] = quad(lambda x, y: v(x, y)*f(x, y), [-1, 1], [-1, 1])

    # Solve by tensor product method - very easy - no mappings :)
    U = np.array([[b[i, j]/(eigenvalues[i] + eigenvalues[j]) for j in range(n)]
                   for i in range(n)])
    # Flatten the 2d representation so match the basis
    U = U.flatten()
    assert len(U) == len(basis)

    # Assemble the solution as a linear combination
    if as_sym:
        uh = sum(Uk*v for Uk, v in zip(U, sym_basis))
    else:
        uh = lambda x, y: sum(Uk*v(x, y) for Uk, v in zip(U, basis))
    return uh

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from numpy.linalg import lstsq
    # Rest the poisson solvers
    # Rate in L2 norm should be 2

    # 1D
    # Exact solution
    x = Symbol('x')
    u = (x**2 - 1)*exp(x)
    # Right hand side
    f = -u.diff(x, 2)
    u = lambdify(x, u)
    # Numerical solution
    ns = range(2, 16)
    print '1D'
    # Data for least square
    b = []
    col0 = []
    for n in ns:
        uh = poisson_solver_1d(f, n)
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
    print '\tLeast square rate %.2f' % p

    # 2D
    # Exact solution
    y = Symbol('y')
    u = (x**2 - 1)*exp(x)*(y**2 - 1)
    # Right hand side
    f = -u.diff(x, 2) - u.diff(y, 2)
    u = lambdify([x, y], u)
    # Numerical solution
    ns = range(2, 11)
    print '2D'
    # Data for least square
    b = []
    col0 = []
    for n in ns:
        uh = poisson_solver_2d(f, n)
        # L2 Error as lambda
        e = lambda x, y: (u(x, y) - uh(x, y))**2
        error, integral_error = quad(e, [-1, 1], [-1, 1],
                                     maxdegree=40,
                                     error=True)
        error = sqrt(error)

        if n > ns[0]:
            rate = ln(error/error_)/ln(n_/n)
            print '\t', n, error, rate, '[%4e]' % integral_error

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
