from __future__ import division
from sympy import Symbol, lambdify, exp, pi, symbols
from sympy.mpmath import quad
import numpy as np
import numpy.linalg as la
from math import log as ln, sqrt
from eigen_basis import eigen_basis
from itertools import product
from scipy.linalg import eigh

def poisson_solver_1d(f, n):
    '''
    Simple Poisson solver for -u`` = f in (-1, 1) with Dirichlet bcs. Uses
    eigenbasis.
    '''
    # Assemble stiffness matrix A, we know about A-orthogonality
    eigenvalues = [float((pi/2 + k*pi/2)**2) for k in range(n)]
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


def poisson_solver_2d(f, n):
    '''
    Simple Poisson solver -laplace(u) = f in (-1, 1)^2 with Dirichlet bcs.
    Uses tensor product of eigenbasis.
    '''
    # Use the tensor product method so only 1d matrices are needed
    # Stiffness matrix
    eigenvalues = [float((pi/2 + k*pi/2)**2) for k in range(n)]
    A = np.diag(eigenvalues)
    # Mass matrix
    M = np.eye(n)

    # Create tensor product basis
    x, y = symbols('x, y')
    basis = [vx*(vy.subs(x, y))
             for vx, vy in product(eigen_basis(n), eigen_basis(n))]

    # Assemble right hand side b
    f = lambdify([x, y], f)
    basis = [lambdify([x, y], v) for v in basis]
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
    uh = lambda x, y: sum(Uk*v(x, y) for Uk, v in zip(U, basis))
    return uh

# ----------------------------------------------------------------------------

if __name__ == '__main__':
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
