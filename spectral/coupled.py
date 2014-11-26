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


def solve(params):
    '''
    Solve the plate-beam problem.
    The plate occupies region [0, 1]^2 and has stiffness E_plate.
    The beam is a line from A--B and has stiffness E_beam.
    The system is loaded with f.

    We use boundary conditions u=laplace(u)=0 on the boundary
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

    # The system is [[A0,    0,   -B0],  [[u0]  [[F],
    #                [0,     A1,   B1], * [u1] = [0],
    #                [-B0.T, B1.T, 0]]    [p]]   [0]]
    # Block A0 has size n_plate**2 x n_plate**2
    A0 = np.zeros((n_plate**2, n_plate**2))
    A0[np.diag_indices_from(A0)] = np.array([(mpi*i)**4 +
                                            2.*(mpi*i)**2*(mpi*j)**2 +
                                            (mpi*j)**4
                                            for i in range(1, n_plate+1)
                                            for j in range(1, n_plate+1)])
    A0 *= E_plate

    # Block A1 has size n_beam x n_beam
    A1 = np.zeros((n_beam, n_beam))
    A1[np.diag_indices_from(A1)] = np.array([(mpi*i)**4
                                             for i in range(1, n_beam+1)])
    A1 *= E_beam
    A1 /= L**3

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
    B0 = np.zeros((n_plate**2, n_lambda))
    B0 *= -L

    # Block B1
    B1 = np.zeros((n_beam, n_lambda))
    B1 *= L

    # Make f for fast evaluation
    f = lambdify([x, y], f)

    # F vector, n_plate**2 long
    F = np.zeros(n_plate**2)
    start = time.time()
    for i, b in enumerate(basis_plate):
        F[i] = quad(lambda x, y: b(x, y)*f(x, y), [0, 1], [0, 1])
    assemble_F = time.time() - start
    print 'assembled F in %g s' % assemble_F

    # Assemble the system from blocks
    N = n_plate**2 + n_beam + n_lambda
    A = np.zeros((N, N))

    A[:n_plate**2, :n_plate**2] = A0
    A[:n_plate**2, (n_plate**2 + n_beam):] = B0

    A[n_plate**2:(n_plate**2 + n_beam), n_plate**2:(n_plate**2 + n_beam)] = A1
    A[n_plate**2:(n_beam**2 + n_beam), (n_plate**2 + n_beam):] = B1

    A[(n_plate**2 + n_beam):, :n_plate**2] = B0.T
    A[(n_plate**2 + n_beam):, (n_plate**2):(n_plate**2 + n_beam)] = B1.T

    b = np.zeros(N)
    b[:n_plate**2] = F
    # Solve the system

    # Extract displacement of beam

    # Return expansion coefficients and symbolic basis (plots nicer and faster)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.spy(A)
    plt.show()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Define problem
    f = 2
    params = {'E_plate': 1.,
              'E_beam': 1.,
              'A': np.array([0.5, 0]),
              'B': np.array([0.5, 1]),
              'f': f,
              'n_plate': 3,
              'n_beam': 3,
              'n_lambda': 3}
    # Solve
    solve(params)
