from __future__ import division
from sympy import symbols, lambdify, sin, pi, sqrt
from sympy.mpmath import quad
from math import pi as mpi
import numpy as np
from numpy import concatenate as cat
import time
import numpy.linalg as la
from itertools import product


class IntersectMap(dict):
    '''
    The intersect order of beams are stored here. The map must reflect the fact
    that intersect of beam_i and beam_j, map[(i, j)], is the same as intersect
    as beam_j and beam_i, that is map[(j, i)]
    '''
    def __init__(self, *args):
        dict.__init__(self, args)

    def __getitem__(self, key):
        '''
        The intersect order is accessed by pair of indices that is sorted if
        needed. If there is no itersect -1
        '''
        assert len(key) == 2 and key[0] != key[1]
        if not (key[0] < key[1]):
            key = (key[1], key[0])
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return -1

    def __setitem__(self, key, value):
        'When the entries are computed in Incident matrix we always have i < j.'
        assert len(key) == 2 and key[0] != key[1]
        assert key[0] < key[1]
        dict.__setitem__(self, key, value)


def intersects(beams):
    '''
    Compute intersects of list of beams.
    Return the incidence matrix which is skew and map that to each pair
    of beams assigns the beam intersect if there is one, else crashes.
    '''
    # Incident matrix
    n = len(beams)
    I = np.zeros((n, n))

    # Map of intersects (i, j) --> order
    I_map = IntersectMap()

    # Intersects coordinates
    I_x = []

    # Matric to compute the intersect
    A = np.zeros((2, 2))
    n_intersects = 0
    for i in range(n):
        for j in range(i+1, n):
            A0, B0 = beams[i]['A'], beams[i]['B']
            A1, B1 = beams[j]['A'], beams[j]['B']

            A[:, 0] = B0 - A0
            A[:, 1] = A1 - B1

            try:
                s, t = la.solve(A, A1-A0)
                # We are only intersted in the interior intersects
                if 1E-13 < s < (1. - 1E-13):
                    # Store to intersect position
                    P = A0 + (B0-A0)*s
                    I_x.append(P)
                    # Record the intersect in matrix
                    I[i, j] = 1
                    I[j, i] = -1
                    # Store the intersect order
                    I_map[(i, j)] = n_intersects
                    n_intersects += 1
            # Skip parallel lines ...etc
            except la.LinAlgError:
                pass

    return I, I_map, np.array(I_x)


def sine_basis(n):
    '''
    Return n functions sin(k*pi*x) for k = 1, ..., n
    '''
    x = symbols('x')
    return [sqrt(2)*sin(k*pi*x) for k in range(1, n+1)]


def solve(params, eigs_only):
    '''
    Plate problem with beams(0 or more) that can intersect.
    '''
    # Extract plate material and degree of space
    E_plate = params.get('E_plate', 1)
    q_plate = params.get('q_plate')

    beams_params = params['beams'] if 'beams' in params else []
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
    # s \in [0, 1] to coordinates of i-th beam
    F_beams = [lambda s, A=A, B=B:
               np.array([A[0] + (B[0]-A[0])*s, A[1] + (B[1]-A[1])*s])
               for (A, B) in zip(A_beams, B_beams)]

    # Learn about intersects in the system
    # Incident matrix, map i, j to intersect order k, intersect is then I_x[k]
    _, Imap, I_x = intersects(beams_params)
    n_intersects = len(I_x)

    # To assemble the linear system we need some block info
    # The linear system becomes   AA = [[A, B],]
    #                                   [B.T, 0]
    #
    # with A = diag(A0, A1, A2, ..., A_{n_beams})
    # and B has columns Bcol_i = [B^i_0, 0*i, B_i] with B^i_0 a block for
    # constraint plate-penalty_i, and B_i a block for constrain beam_i-penalty_i
    # This holds for first n_beam columns. Then there are columns which
    # enforce conditions on interects - there n_intersects of these columns
    Asizes = [q_plate**2] + [q_beam for q_beam in q_beams]
    n_cols_A = sum(Asizes)
    n_rows_A = n_cols_A
    Aoffsets = [sum(Asizes[:i]) for i in range(len(Asizes)+1)]

    Bsizes = [q_lambda for q_lambda in q_lambdas] + [1]*n_intersects
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
        # Avoid plate, Asizes[0] is plate
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
        # This is just a mass matrix
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
    # Tensor product
    basis_plate = [lambda x, y, bi=bi, bj=bj: bi(x)*bj(y)
                   for (bi, bj) in product(basis_i, basis_i)]

    start = time.time()
    for beam in range(n_beams):
        # Basis functions for lagrange multiplier space
        basis_p = map(lambda f: lambdify(x, f), sine_basis(Bsizes[beam]))

        B0 = np.zeros((Asizes[0], Bsizes[beam]))
        for k, phi_k in enumerate(basis_plate):
            for j, chi_j in enumerate(basis_p):
                B0[k, j] = quad(lambda s: phi_k(*F_beams[beam](s))*chi_j(s),
                                [0, 1])

        B0 *= -L_beams[beam]
        B0blocks.append(B0)
        print '.'*(beam+1)

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
        B[:Asizes[0], col0:coln] = B0blocks[block]
    # B1
    for block in range(n_beams):
        col0, coln = Boffsets[block], Boffsets[block+1]
        row0, rown = Aoffsets[block+1], Aoffsets[block+2]
        B[row0:rown, col0:coln] = B1blocks[block]

    # Fill B with constraints on intersects
    for ij in Imap:
        i_sect_order = Imap[ij]
        i_sect = I_x[i_sect_order]
        col = Boffsets[n_beams + i_sect_order]

        for beam, sign in zip(ij, (1, -1)):
            # Represent i_sect as s such that F_i(s) = i_sect
            F_beam = F_beams[beam]
            Q = F_beam(0)
            vec = 0.5*(F_beam(1) - F_beam(-1))
            s_ = np.hypot(*(i_sect - Q))/np.hypot(*vec)
            assert np.allclose(i_sect, F_beam(s_))

            # Compute values that go into colums
            basis = map(lambda f: lambdify(x, f),
                        sine_basis(q_beams[beam]))
            values = np.array([v(s_) for v in basis])
            values *= sign
            # Avoid the plate!
            row0, rown = Aoffsets[beam+1], Aoffsets[beam+2]
            B[row0:rown, col] = values

    # Optionally return at eigenvalues of the Schur complement
    if eigs_only:
        # Eigenvalues of Schur
        Ainv = np.zeros_like(A)
        Ainv[np.diag_indices_from(Ainv)] = A.diagonal()**-1

        # Put together the C matrices over lambda spaces
        C0 = np.zeros((n_cols_B, n_cols_B))
        C1 = np.zeros((n_cols_B, n_cols_B))
        C2 = np.zeros((n_cols_B, n_cols_B))

        # We will identy for the intersect constraints
        C_isects = np.ones(n_intersects)

        # Mass matrix
        C0_diag = cat([L_beam*np.ones(q_lambda)
                       for (L_beam, q_lambda) in zip(L_beams, q_lambdas)])
        C0[np.diag_indices_from(C0)] = cat([C0_diag, C_isects])

        # Laplace
        C1_diag = cat([np.array([(i*mpi)**2
                                 for i in range(1, q_lambda+1)])/L_beam
                       for (L_beam, q_lambda) in zip(L_beams, q_lambdas)])
        C1[np.diag_indices_from(C1)] = cat([C1_diag, C_isects])

        # Biharmonic
        C2_diag = cat([np.array([(i*mpi)**4
                                 for i in range(1, q_lambda+1)])/L_beam**3
                       for (L_beam, q_lambda) in zip(L_beams, q_lambdas)])
        C2[np.diag_indices_from(C2)] = cat([C2_diag, C_isects])

        Cs = {'mass': C0, 'laplace': C1, 'biharmonic': C2}

        eigenvalues = {}
        S = B.T.dot(Ainv.dot(B))
        for key, C in Cs.iteritems():
            Mat = C.dot(S)
            eigs = la.eigvals(Mat)
            lmbda_min = sorted(eigs)[:3]
            eigenvalues[key] = lmbda_min

        return eigenvalues

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
    F = np.zeros(Asizes[0])
    start = time.time()
    for i, b in enumerate(basis_plate):
        F[i] = quad(lambda x, y: b(x, y)*f(x, y), [0, 1], [0, 1])
    assemble_F = time.time() - start
    print '\tassembled F in %g s' % assemble_F

    bb = np.zeros(AA.shape[0])
    bb[:Asizes[0]] = F

    # Solve the system AA*U = bb
    U = la.solve(AA, bb)

    # Split the vector to get expansions coeffs of each unknown and return
    # a sympy expression that represents the solution
    U_plate = U[:Asizes[0]]
    U_beams = [U[Aoffsets[i]:Aoffsets[i+1]] for i in range(1, n_beams+1)]
    # Need global offset of penalty
    Boffsets = [offset + n_rows_A for offset in Boffsets]
    U_lambdas = [U[Boffsets[i]:Boffsets[i+1]] for i in range(n_beams)]

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
    for q_beam, U_beam in zip(q_beams, U_beams):
        assert len(U_beam) == q_beam
        basis_beam = [v.subs({x: s}) for v in sine_basis(q_beam)]
        u_beam = sum(coef_i*basis_i
                     for coef_i, basis_i in zip(U_beam, basis_beam))
        u_beams.append(u_beam)

    # For multipliers
    u_lambdas = []
    for q_lambda, U_lambda in zip(q_lambdas, U_lambdas):
        assert len(U_lambda) == q_lambda
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
    A1, B1 = np.array([0., 0.]), np.array([1., 1.])
    A2, B2 = np.array([0, 0.75]), np.array([0.75, 0])
    A3, B3 = np.array([1, 0.5]), np.array([0.5, 1])
    params = {'f': f,
              'E_plate': 1.,
              'q_plate': 15,
              'beams': [{'E_beam': 5.,
                         'A': A1,
                         'B': B1,
                         'n_beam': 15,
                         'n_lambda': 15},
                        #
                        {'E_beam': 5.,
                         'A': A2,
                         'B': B2,
                         'n_beam': 10,
                         'n_lambda': 10},
                        #
                        {'E_beam': 5.,
                         'A': A3,
                         'B': B3,
                         'n_beam': 10,
                         'n_lambda': 10}
                        ]
              }
    beams = params['beams'] if 'beams' in params else []
    if beams:
        # Plot the beam positions
        fig = plt.figure()
        ax = fig.gca()
        for i, beam in enumerate(params['beams']):
            ax.plot([beam['A'][0], beam['B'][0]], [beam['A'][1], beam['B'][1]],
                    label=str(i))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        # Add intersects
        I, Imap, I_x = intersects(beams)
        for X in I_x:
            ax.plot(X[0], X[1], 'o')
        plt.legend(loc='best')

        # Compute
        u_plate, u_beams, lmbdas = solve(params, eigs_only=False)

        # Plot beam displacement
        plot3d(u_plate, (x, 0, 1), (y, 0, 1), xlabel='$x$', ylabel='$y$',
               title='Plate deflection')

        # Displacement of beams and plate on the beams
        for i, (beam, u_beam) in enumerate(zip(beams, u_beams)):
            A, B = beam['A'], beam['B']
            x_s = A[0] + (B[0] - A[0])*s
            y_s = A[1] + (B[1] - A[1])*s
            # Restrict plate to beam
            u0_beam = u_plate.subs({x: x_s, y: y_s})
            plot(u_beam, u0_beam, (s, 0, 1), xlabel='$s/L$',
                 title='Beam deflection vs plate deflection on beam %d' % i)

        # Lagrange multipliers on beams
        for i, lmbda in enumerate(lmbdas):
            plot(lmbda, (s, 0, 1), xlabel='$s/L$',
                 title='Lagrange multiplier on beam %d' % i)

        # Values of beam deflaction in intersects
        for ij in Imap:
            i_sect_order = Imap[ij]
            i_sect = I_x[i_sect_order]
            values = []
            for k in ij:
                beam = beams[k]
                # Represent i_sect as s such that F_i(s) = i_sect
                A, B = beam['A'], beam['B']
                Q = [A[0], A[1]]
                vec = [B[0]-A[0], B[1]-A[1]]
                s_value = np.hypot(*(i_sect - Q))/np.hypot(*vec)
                # Extract beam deflections
                u_beam = u_beams[k]
                values.append((s_value, u_beam.evalf(subs={s: s_value})))
            print 'At intersect', i_sect, 'beam deflections [s, value]', values
    # No beams
    else:
        # Only compute plate
        u_plate = solve(params, eigs_only=False)

        # Plot beam displacement
        plot3d(u_plate, (x, 0, 1), (y, 0, 1), xlabel='$x$', ylabel='$y$',
               title='Plate deflection')

    plt.show()
