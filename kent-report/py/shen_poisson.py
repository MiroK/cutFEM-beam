from __future__ import division
from sympy import Symbol, lambdify, exp, pi, symbols
from sympy.mpmath import quad
import numpy as np
import numpy.linalg as la
from math import log as ln, sqrt
from shenp_basis import shenp_basis
from itertools import product
from scipy.linalg import eigh

# Basis yield to matrices whose values can be tabulized
def mass_matrix(m):
    'Mass matrix for the Shenp basis of len m.'
    weight = lambda k: float(1/sqrt(4*k + 6))
    M = np.zeros((m, m))
    for i in range(m):
        M[i, i] = weight(i)*weight(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, m):
            M[i, j] = -weight(i)*weight(j)*(2./(2*j+1)) if i+2 == j else 0
            M[j, i] = M[i, j]

    return M


def laplacian_matrix(m):
    'Matrix of Laplacian with respect to the Shenp basis of len m.'
    return np.eye(m)


def poisson_solver_1d(f, n, as_sym=False, with_coefs=False):
    '''
    Poisson solver for -u`` = f in (-1, 1) with Dirichlet bcs using shenp_basis
    '''
    # Assemble stiffness matrix A, we know about A-orthogonality
    A = laplacian_matrix(n)

    # Assemble right hand side b
    x = Symbol('x')
    f = lambdify(x, f)
    sym_basis = list(shenp_basis(n))
    basis = [lambdify(x, v) for v in sym_basis]
    b = np.zeros(n)
    for i, v in enumerate(basis):
        b[i] = quad(lambda x: v(x)*f(x), [-1, 1])

    # Solve and return solution - assembled linear combination
    U = la.solve(A, b)
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
    Poisson solver -laplace(u) = f in (-1, 1)^2 with Dirichlet bcs.
    Uses tensor product of shenp_basis.
    '''
    # Use the tensor product method so only 1d matrices are needed
    # Stiffness matrix
    A = laplacian_matrix(n)
    # Mass matrix
    M = mass_matrix(n)

    # Create tensor product basis
    x, y = symbols('x, y')
    sym_basis = [vx*(vy.subs(x, y))
                 for vx, vy in product(shenp_basis(n), shenp_basis(n))]

    # Assemble right hand side b
    f = lambdify([x, y], f)
    basis = [lambdify([x, y], v) for v in sym_basis]
    b = np.zeros((n, n))
    for k, v in enumerate(basis):
        i, j = k // n, k % n
        b[i, j] = quad(lambda x, y: v(x, y)*f(x, y), [-1, 1], [-1, 1])

    # Solve by tensor product method
    lmbda, Q = eigh(A, M)
    # Map the rhs to eigen space
    bb = (Q.T).dot(b.dot(Q))
    # Solve the system in eigen space
    UU = np.array([[bb[i, j]/(lmbda[i] + lmbda[j]) for j in range(n)]
                    for i in range(n)])
    # Map the solution back to real space
    U = Q.dot(UU.dot(Q.T))
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
    # Rest the poisson solvers
    # Rate in L2 norm should spectral

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
