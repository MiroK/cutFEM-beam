from __future__ import division
from sympy import symbols, lambdify, sin, pi, sqrt
from sympy.mpmath import quad
from math import pi as mpi
import numpy as np
import time
import numpy.linalg as la
from itertools import product


def sine_basis(n):
    '''
    Return n functions sin(k*pi*x) for k = 1, ..., n
    '''
    x = symbols('x')
    return [sqrt(2)*sin(k*pi*x) for k in range(1, n+1)]


def solve(params, eigs_only=False):
    '''
    Solve the plate-beam problem.
    The plate occupies region [0, 1]^2 and has stiffness E_plate.
    The beam is a line from A--B and has stiffness E_beam.
    The system is loaded with f.

    The deformation of both plate and beam is governed by laplace equation.
    We use boundary conditions u=0 on the boundary.
    '''
    # Problem properties
    E_plate = params.get('E_plate', 1)
    E_beam = params.get('E_beam', 1)
    A = params.get('A', np.array([0, 0]))
    B = params.get('B', np.array([1, 1]))
    f = params.get('f', None)
    # Solver properties, i.e. number of sines in each direction for plate
    #                         number of sines in each direction for beam
    #                         number of sines for lagrange multipliers
    n_plate = params.get('n_plate')
    n_beam = params.get('n_beam', n_plate)
    n_lambda = params.get('n_lambda', n_beam)

    assert f is not None

    # Beam length
    L = np.hypot(*(B-A))
    assert L > 0

    print A, B

    # The system is [[A0,    0,   -B0],  [[u0]  [[F],
    #                [0,     A1,   B1], * [u1] = [0],
    #                [-B0.T, B1.T, 0]]    [p]]   [0]]
    # Block A0 has size n_plate**2 x n_plate**2
    A0 = np.zeros((n_plate**2, n_plate**2))
    A0[np.diag_indices_from(A0)] = np.array([(mpi*i)**2 +
                                            (mpi*j)**2
                                            for i in range(1, n_plate+1)
                                            for j in range(1, n_plate+1)])
    A0 *= E_plate

    # Block A1 has size n_beam x n_beam
    A1 = np.zeros((n_beam, n_beam))
    A1[np.diag_indices_from(A1)] = np.array([(mpi*i)**2
                                             for i in range(1, n_beam+1)])
    A1 *= E_beam
    A1 /= L

    # Basis functions of plate flattened
    x, y = symbols('x, y')
    # For each direction
    basis_i = map(lambda f: lambdify(x, f), sine_basis(n_plate))

    basis_plate = [lambda x, y, bi=bi, bj=bj: bi(x)*bj(y)
                   for (bi, bj) in product(basis_i, basis_i)]

    # Basis functions of beam
    basis_beam = map(lambda f: lambdify(x, f), sine_basis(n_beam))

    # Basis functions for lagrange multiplier space
    basis_p = map(lambda f: lambdify(x, f), sine_basis(n_lambda))

    # To describe the beam we need a mapping from 0, 1 to (x, y) coordinates
    # of beam, i,e (x, y) = A + (B-A)*s
    def x_hat(s):
        return A[0] + (B[0] - A[0])*s

    def y_hat(s):
        return A[1] + (B[1] - A[1])*s

    # Block B0
    start = time.time()
    B0 = np.zeros((n_plate**2, n_lambda))
    for k, phi_k in enumerate(basis_plate):
        for j, chi_j in enumerate(basis_p):
            B0[k, j] = quad(lambda s: phi_k(x_hat(s), y_hat(s))*chi_j(s),
                            [0, 1])

    B0 *= -L
    assemble_B0 = time.time() - start
    print '\tassembled B0 in %g s' % assemble_B0

    # Block B1
    B1 = np.zeros((n_beam, n_lambda))
    for k in range(n_lambda):
        B1[k, k] = 1
    B1 *= L

    # Group A0, A1 into A and B0, B1 into B
    A = np.zeros((n_plate**2 + n_beam, n_plate**2 + n_beam))
    A[:n_plate**2, :n_plate**2] = A0
    A[n_plate**2:, n_plate**2:] = A1

    B = np.zeros((n_plate**2 + n_beam, n_lambda))
    B[:n_plate**2, :] = B0
    B[n_plate**2:, :] = B1

    if eigs_only:
        # Eigenvalues of Schur
        Ainv = np.zeros_like(A)
        Ainv[np.diag_indices_from(Ainv)] = A.diagonal()**-1
        # Put together the C matrices
        # Mass matrix
        C0 = np.eye(n_lambda)*L
        # Laplace
        C1 = np.diag(np.array([(i*mpi)**2 for i in range(1, n_lambda+1)]))/L

        Cs = {'mass': C0, 'laplace': C1}

        eigenvalues = {}
        S = B.T.dot(Ainv.dot(B))
        for key, C in Cs.iteritems():
            Mat = C.dot(S)
            eigs = la.eigvals(Mat)
            lmbda_min = sorted(eigs)[:3]
            eigenvalues[key] = lmbda_min

        return eigenvalues

    # Assemble the system from blocks
    N = n_plate**2 + n_beam + n_lambda
    AA = np.zeros((N, N))
    AA[:n_plate**2 + n_beam, :n_plate**2 + n_beam] = A
    AA[:n_plate**2 + n_beam, n_plate**2 + n_beam:] = B
    AA[n_plate**2 + n_beam:, :n_plate**2 + n_beam] = B.T

    # Right hand side
    # Make f for fast evaluation
    f = lambdify([x, y], f)

    # F vector, n_plate**2 long
    F = np.zeros(n_plate**2)
    start = time.time()
    for i, b in enumerate(basis_plate):
        F[i] = quad(lambda x, y: b(x, y)*f(x, y), [0, 1], [0, 1])
    assemble_F = time.time() - start
    print '\tassembled F in %g s' % assemble_F

    bb = np.zeros(N)
    bb[:n_plate**2] = F

    # Solve the system
    U = la.solve(AA, bb)

    # Extract displacement of beam
    U_plate = U[:n_plate**2]
    U_beam = U[n_plate**2:(n_plate**2 + n_beam)]
    U_lmbda = U[(n_plate**2 + n_beam):]

    # Return expansion coefficients and symbolic basis (plots nicer and faster)
    # combined into function u0 and u1 for plate and beam displacement and
    # lmbda for the lagrange multiplier
    x, y, s = symbols('x, y, s')

    # Symbolic, each direction
    basis_x = sine_basis(n_plate)
    basis_y = [v.subs({x: y}) for v in basis_x]
    # Combine, tensor product
    basis_plate = [bi*bj for (bi, bj) in product(basis_x, basis_y)]
    u0 = sum(coef_i*basis_i for (coef_i, basis_i) in zip(U_plate, basis_plate))

    # Symbolic for beam, note that s extends from 0, 1
    basis_s = sine_basis(n_beam)
    basis_beam = [v.subs({x: s}) for v in basis_s]
    # Combine
    u1 = sum(coef_i*basis_i for (coef_i, basis_i) in zip(U_beam, basis_beam))

    # For penalty we use the same function space as for beam
    lmbda = sum(coef_i*basis_i
                for (coef_i, basis_i) in zip(U_lmbda, basis_beam))

    return u0, u1, lmbda

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.plotting import plot3d, plot

    # Define problem
    x, y, s = symbols('x, y, s')
    f = 1
    A = np.array([0.25, 0])
    B = np.array([0.5, 1])
    params = {'E_plate': 1.,
              'E_beam': 10.,
              'A': A,
              'B': B,
              'f': f,
              'n_plate': 10,
              'n_beam': 10,
              'n_lambda': 10}

    # Solve
    u0, u1, lmbda = solve(params)

    # Plot
    # Plate displacement
    plot3d(u0, (x, 0, 1), (y, 0, 1), xlabel='$x$', ylabel='$y$',
        title='Plate deflection')
    # Displacement of beam and plate on the beam
    x_s = A[0] + (B[0] - A[0])*s
    y_s = A[1] + (B[1] - A[1])*s
    u0_beam = u0.subs({x: x_s, y: y_s})
    plot(u1, u0_beam, (s, 0, 1), xlabel='$s/L$',
        title='Beam deflection vs plate deflection on the beam')
    # Lagrange multiplier
    plot(lmbda, (s, 0, 1), xlabel='$s/L$',
        title='Lagrange multiplier')

