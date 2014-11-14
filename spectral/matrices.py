from sympy import symbols, lambdify
from quadrature import GLQuadrature
import numpy as np


def assemble_matrix(basis_range, basis_domain=None, is_symmetric=True,
                    quadrature=None, quadrature_degree=None):
    '''
    Assemble matrix A with entries
        a[i, j]\int_-1^1 bi(x)*bj(x)*dx for bi in basis_range,
        for bj in basis_domain

    A is then a map from basis_domain to basis_range.
    Functions in basis_* are lambdas/python functions not symbolic.
    '''
    if basis_domain is None:
        basis_domain = basis_range

    if quadrature is None:
        m = len(basis_range)
        n = len(basis_domain)
        # We assuem that we are deal with integral where the ingrand has degree
        # (m-1)*(n-1) which is integrated exactly by GL quadrature of order
        # max(m, n)
        quadrature = GLQuadrature(max(m, n))
    else:
        assert quadrature_degree is not None and quadrature_degree > 0
        quadrature = quadrature(quadrature_degree)

    A = np.zeros((m, n))

    if is_symmetric:
        for i, v in enumerate(basis_range):
            A[i, i] = quadrature.eval(lambda x: v(x)*v(x), [[-1, 1]])
            for j, u in enumerate(basis_domain[i+1:], i+1):
                A[i, j] = quadrature.eval(lambda x: v(x)*u(x), [[-1, 1]])
                A[j, i] = A[i, j]
    else:
        for i, v in enumerate(basis_range):
            for j, u in enumerate(basis_domain):
                A[i, j] = quadrature.eval(lambda x: v(x)*u(x), [[-1, 1]])

    return A


def assemble_mass_matrix(basis):
    '''
    Assemble the mass matrix of space spanned by basis(symbolic).
    Matrix of the identity operator.
    '''
    # Transform the basis
    x = symbols('x')
    basis_lambda = map(lambda f: lambdify(x, f), basis)
    return assemble_matrix(basis_lambda, is_symmetric=True)


def assemble_stiffness_matrix(basis):
    '''
    Assemble the stiffness matrix of space spanned by basis.
    Matrix of the Laplace operator.
    '''
    # Transform the basis
    x = symbols('x')
    basis_lambda = map(lambda f: lambdify(x, f.diff(x, 1)) for f in basis)
    return assemble_matrix(basis_lambda, is_symmetric=True)


def assemble_biharmonic_matrix(basis):
    '''
    Assemble matrix of the biharmonic operator.
    '''
    x = symbols('x')
    basis_lambda = map(lambda f: lambdify(x, f.diff(x, 2)) for f in basis)
    return assemble_matrix(basis_lambda, is_symmetric=True)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from functions import lagrange_basis
    from points import gauss_legendre_points as gl_points
    from mpi4py import MPI
    import time

    Ns = range(2, 45)
    times = []
    for N in Ns:
        start = time.time()
        assemble_mass_matrix(lagrange_basis([gl_points([N])]))
        stop = time.time() - start
        times.append(stop)

    comm = MPI.COMM_WORLD
    proc = comm.Get_rank()

    if proc == 0:
        plt.figure()
        plt.plot(Ns, times, '*-')
        plt.xlabel('$N$')
        plt.ylabel('$s$')
        plt.show()
