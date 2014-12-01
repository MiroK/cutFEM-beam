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


def solve(params, eigs_only):
    '''
    Plate with multiple beams that don't intersect.
    '''
    # Extract plate material and degree of space
    E_plate = params.get('E_plate', 1)
    q_plate = params.get('q_plate')

    beams_params = params.get('beams')
    n_beams = len(beams_params)
    # Extract beam positions, material and degree of spaces for deflection
    # and penalty
    A_beams, B_beams, E_beams, q_beams, q_lambdas = [], [], [], [], []
    for beam in beams_params:
        A_beams.append(beam['A'])
        B_beams.append(beam['B'])
        E_beams.append(beam.get('E_beam', 1))
        q_beams.append(beam['n_beam'])
        q_lambdas.append(beam['n_lambda'])

    # For scaling we will need length of each beam
    L_beams = [np.hypot(*(A-B)) for A, B in zip(A_beams, B_beams)]

    # For integration we will need a mapping F_i(s) = [x_i(s), y_i(s)] that maps
    # s \in [0, 1] to coordinates of each beam
    F_beams = [lambda s, A=A, B=B: (A[0] + (B[0]-A[0])*s, A[1] + (B[1]-A[1])*s)
               for (A, B) in zip(A_beams, B_beams)]

    # To assemble the linear system we need some block info
    # The linear system becomes   AA = [[A, B],]
    #                                   [B.T, 0]
    #
    # with A = diag(A0, A1, A2, ..., A_{n_beams})
    # and B has columns Bcol_i = [B^i_0, 0*i, B_i] with B^i_0 a block for
    # constraint plate-penalty_i, and B_i a block for constrain beam_i-penalty_i
    Asizes = [q_plate**2] + [q_beam for q_beam in q_beams]
    n_cols_A = sum(Asizes)
    n_rows_A = n_cols_A
    Aoffsets = [sum(Asizes[:i]) for i in range(len(Asizes)+1)]

    Bsizes = [q_lambda for q_lambda in q_lambdas]
    n_cols_B = sum(Bsizes)
    n_rows_B = n_rows_A
    Boffsets = [sum(Bsizes[:i]) for i in range(len(Bsizes)+1)]

    # Compute the blocks of A
    Ablocks = []
    # Plate is special
    A0 = np.zeros((Asizes[0], Asizes[0]))
    A0[np.diag_indices_from(A0)] = np.array([(mpi*i)**4 +
                                            2.*(mpi*i)**2*(mpi*j)**2 +
                                            (mpi*j)**4
                                            for i in range(1, q_plate + 1)
                                            for j in range(1, q_plate + 1)])
    A0 *= E_plate
    Ablocks.append(A0)

    # Beams follow template
    for beam in range(n_beams):
        # Avoid plate
        A_beam = np.zeros((Asizes[1+beam], Asizes[1+beam]))
        A_beam[np.diag_indices_from(A_beam)] = \
            np.array([(mpi*i)**4 for i in range(1, Asizes[1+beam] + 1)])
        A_beam *= E_beams[beam]
        A_beam /= L_beams[beam]**3

        Ablocks.append(A_beam)

    # Compute blocks of B that are due beam-penalty
    B1blocks = []
    for beam in range(n_beams):
        # Omit the plate in Asizes
        B1 = np.zeros((Asizes[1+beam],  Bsizes[beam]))
        for k in range(Bsizes[beam]):
            B1[k, k] = 1
        B1 *= L_beams[beam]

        B1blocks.append(B1)

    # Compute blocks of B that are due plate-penalty, need to integrate here
    B0blocks = []
    # Basis functions of plate flattened
    x, y = symbols('x, y')
    # For each direction
    basis_i = map(lambda f: lambdify(x, f), sine_basis(q_plate))
    basis_plate = [lambda x, y, bi=bi, bj=bj: bi(x)*bj(y)
                   for (bi, bj) in product(basis_i, basis_i)]

    start = time.time()
    for beam in range(n_beams):
        # Basis functions for lagrange multiplier space
        basis_p = map(lambda f: lambdify(x, f), sine_basis(Bsizes[beam]))

        B0 = np.zeros((q_plate**2, Bsizes[beam]))
        # for k, phi_k in enumerate(basis_plate):
        #     for j, chi_j in enumerate(basis_p):
        #         B0[k, j] = quad(lambda s: phi_k(*F_beams[beam](s))*chi_j(s),
        #                         [0, 1])

        B0 *= -L_beams[beam]
        B0blocks.append(B0)
        print '.'

    assemble_B0 = time.time() - start
    print '\tassembled B0s in %g s' % assemble_B0

    # Assemble A
    A = np.zeros((n_rows_A, n_cols_A))
    for block in range(n_beams+1):
        m, n = Aoffsets[block], Aoffsets[block+1]
        A[m:n, m:n] = Ablocks[block]

    # Assemble B
    # B0
    B = np.zeros((n_rows_B, n_cols_B))
    for block in range(n_beams):
        col0, coln = Boffsets[block], Boffsets[block+1]
        B[:q_plate**2, col0:coln] = B0blocks[block]
    # B1
    for block in range(n_beams):
        col0, coln = Boffsets[block], Boffsets[block+1]
        row0, rown = q_plate**2 + col0, q_plate**2 + coln
        B[row0:rown, col0:coln] = B1blocks[block]

    # Optionally return at eigenvalues of the Schur complement
    # TODO

    # Assemble AA system matrix
    AA = np.zeros((n_rows_A + n_cols_B, n_rows_A + n_cols_B))
    AA[:n_rows_A, :n_cols_A] = A
    AA[:n_rows_A, n_cols_A:] = B
    AA[n_rows_A:, :n_cols_A] = B.T

    # Compute and assemble the right hand side
    # Make f for fast evaluation
    f = params.get('f')
    f = lambdify([x, y], f)

    # F vector, n_plate**2 long
    F = np.zeros(q_plate**2)
    start = time.time()
    # for i, b in enumerate(basis_plate):
    #    F[i] = quad(lambda x, y: b(x, y)*f(x, y), [0, 1], [0, 1])
    assemble_F = time.time() - start
    print '\tassembled F in %g s' % assemble_F

    bb = np.zeros(AA.shape[0])
    bb[:q_plate**2] = F

    # Solve the system AA*U = bb
    U = np.zeros(A.shape[0])#la.solve(AA, bb)

    # Split the vector to get expansions coeffs of each unknown and return
    # a sympy expression that represents the solution
    U_plate = U[:q_plate**2]
    U_beams = [U[Aoffsets[i]:Aoffsets[i+1]] for i in range(1, n_beams)]
    Boffsets = [offset + n_rows_A for offset in Boffsets]
    U_lambdas = [U[Boffsets[i]:Boffsets[i+1]] for i in range(0, n_beams)]

    # Symbolic basis for plate
    x, y, s = symbols('x, y, s')

    # Symbolic, each direction
    basis_x = sine_basis(q_plate)
    basis_y = [v.subs({x: y}) for v in basis_x]
    # Combine, tensor product
    basis_plate = [bi*bj for (bi, bj) in product(basis_x, basis_y)]
    u_plate = sum(coef_i*basis_i
                  for (coef_i, basis_i) in zip(U_plate, basis_plate))

    # For beam
    u_beams = []
    for q_beam, U_beams in zip(q_beams, U_beams):
        basis_beam = [v.subs({x: s}) for v in sine_basis(q_beam)]
        u_beam = sum(coef_i*basis_i
                     for coef_i, basis_i in zip(U_beams, basis_beam))
        u_beams.append(u_beam)

    # For multipliers
    u_lambdas = []
    for q_lambda, U_lambda in zip(q_lambdas, U_lambdas):
        basis_lambda = [v.subs({x: s}) for v in sine_basis(q_lambda)]
        u_lambda = sum(coef_i*basis_i
                       for coef_i, basis_i in zip(U_lambda, basis_lambda))
        u_lambdas.append(u_lambda)

    return (u_plate, u_beams, u_lambdas)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.plotting import plot3d, plot
    import matplotlib.pyplot as plt

    x, y, s = symbols('x, y, s')
    f = 1
    A1, B1 = np.array([0, 0]), np.array([1/3, 1])
    A2, B2 = np.array([1/3, 0]), np.array([2/3, 1])
    A3, B3 = np.array([2/3, 0]), np.array([1, 1])
    params = {'f': f,
              'E_plate': 1.,
              'q_plate': 10,
              'beams': [{'E_beam': 10.,
                         'A': A1,
                         'B': B1,
                         'n_beam': 10,
                         'n_lambda': 10},
                        #
                        {'E_beam': 5.,
                         'A': A2,
                         'B': B2,
                         'n_beam': 8,
                         'n_lambda': 8},
                        #
                        {'E_beam': 2.,
                         'A': A3,
                         'B': B3,
                         'n_beam': 6,
                         'n_lambda': 6}]
              }
    # Plo the beam positions
    fig = plt.figure()
    ax = fig.gca()
    for beam in params['beams']:
        ax.plot([beam['A'][0], beam['B'][0]], [beam['A'][1], beam['B'][1]])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Solve
    u_plate, u_beams, u_lambdas = solve(params, eigs_only=True)

    # Plot beam displacement
    plot3d(u_plate, (x, 0, 1), (y, 0, 1), xlabel='$x$', ylabel='$y$',
           title='Plate deflection')

    plt.show()
