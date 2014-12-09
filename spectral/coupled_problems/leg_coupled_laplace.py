from __future__ import division
from sympy import symbols, lambdify, S, legendre as legendre_sym
from sympy.mpmath import quad, legendre
from math import pi as mpi, sqrt
import numpy as np
import time
import numpy.linalg as la
from scipy.linalg import logm
from itertools import product
from functools import partial


def c(k):
    'Weights of Shen basis'
    return sqrt(4*k + 6)**-1


def shen_basis(m):
    '''
    Shen basis of Legendre polynomials with zeros on [-1, 1].
    M is the highest order polynomial used in all basis functions.
    '''
    return [lambda x, i=i:
            c(i)*(partial(legendre, n=i)(x=x) - partial(legendre, n=i+2)(x=x))
            for i in range(m-1)]


def shen_basis_symbolic(m, x):
    '''
    Shen basis of Legendre polynomials with zeros on [-1, 1].
    M is the highest order polynomial used in all basis functions.
    '''
    return [c(i)*(legendre_sym(i, x) - legendre_sym(i+2, x))
            for i in range(m-1)]


def shen_mass_matrix(n):
    '''
    Return mass matrix corresponding to shen basis of length n.
    '''
    # Mass matrix on [-1, 1]
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = c(i)*c(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, n):
            M[i, j] = -c(i)*c(j)*(2./(2*j+1)) if i+2 == j else 0
            M[j, i] = M[i, j]
    return M


def shen_stiffness_matrix(n):
    '''
    Return mass matrix corresponding to shen basis of length n.
    '''
    return np.eye(n)


def solve(params, eigs_only=False, fractions=None):
    '''
    Solve the plate-beam problem.
    The plate and beam deformation are governed by the laplace equation.
    We use boundary conditions u=0 on the boundary.

    The plate occupies region rectangular defined by point V0, V1.
    By defualt, these are V0 = [0, 0] and V1 = [1, 1].
    The beam is a line from P, Q somewhere on the boundary of place.
    The system is loaded with f.
    '''
    # Problem properties
    E_plate = params.get('E_plate', 1)
    V0 = params['V0'] if 'V0' in params else np.array([0, 0])
    V1 = params['V1'] if 'V1' in params else np.array([1, 1])
    E_beam = params.get('E_beam', 1)
    P = params.get('P', np.array([0, 0]))
    Q = params.get('Q', np.array([1, 1]))
    f = params.get('f', None)
    # Solver properties, i.e. highest Legendre polynomial to be used
    N_plate = params.get('n_plate')
    N_beam = params.get('n_beam', N_plate)
    N_lmbda = params.get('n_lmbda', N_beam)

    assert min([N_plate, N_beam, N_lmbda]) > 1

    assert f is not None

    # Plate len in x dir L0, in y dir L1, and center
    L0 = np.abs(V1[0] - V0[0])
    L1 = np.abs(V1[1] - V0[1])
    assert L0 > 0 and L1 > 0
    center0 = 0.5*(V1[0] + V0[0])
    center1 = 0.5*(V1[0] + V0[0])

    # Beam length, the beam is parametrized [0, L] --> beam
    L = np.hypot(*(Q-P))
    assert L > 0
    center = 0.5

    if eigs_only:
        assert isinstance(fractions, np.ndarray)
        assert len(fractions)

    # The plate problem is solved on domain [-1, 1]^2 which is mapped to plate
    # by mapping F2(x_hat, y_hat)
    def F2(x_hat, y_hat):
        return np.array([L0*x_hat/2 + center0, L1*y_hat/2 + center1])

    # its inverse
    def F2_inv(x, y):
        return np.array([2*(x - center0)/L0, 2*(y - center1)/L1])

    # The beam needs to be mapped to [-1, 1]^2
    P_hat = F2_inv(*P)
    Q_hat = F2_inv(*Q)

    # To describe the beam we need a mapping from -1, 1 to (x_hat, y_hat)
    # coordinates of beam
    def Beam_map(s_hat):
        return np.array([(P_hat[0]+Q_hat[0])/2 + s_hat*(Q_hat[0]-P_hat[0])/2,
                         (P_hat[1]+Q_hat[1])/2 + s_hat*(Q_hat[1]-P_hat[1])/2])

    # This is already lambdified, for 1d
    basis_plate = shen_basis(N_plate)
    basis_beam = shen_basis(N_beam)
    basis_lmbda = shen_basis(N_lmbda)
    # Get sizes related to block sizes
    n_plate, n_beam, n_lmbda = map(len, [basis_plate, basis_beam, basis_lmbda])
    # The system is [[A0,    0,   -B0],  [[u0]  [[F],
    #                [0,     A1,   B1], * [u1] = [0],
    #                [-B0.T, B1.T, 0]]    [p]]   [0]]
    # A0 block for plate is Ap x Mp + Mp x Ap
    Ap = shen_stiffness_matrix(n_plate)
    Mp = shen_mass_matrix(n_plate)
    A0 = np.kron(Ap, Mp)*(L1/L0)
    A0 += np.kron(Mp, Ap)*(L0/L1)
    assert A0.shape == (n_plate**2, n_plate**2)
    A0 *= E_plate
    print '.'

    # Block A1 being the stiffness matrix on beam is identity
    A1 = shen_stiffness_matrix(n_beam)
    A1 *= E_beam
    A1 *= 2./L
    print '.'

    # Block B1 is mass matrix on the beam but chopped
    B1 = shen_mass_matrix(max(n_beam, n_lmbda))
    B1 = B1[:n_beam, :n_lmbda]
    B1 *= L/2
    print '.'

    # We create the tensor product basis of plate, note over [-1, 1]^2
    basis_plate = [lambda x, y, bi=bi, bj=bj: bi(x)*bj(y)
                   for (bi, bj) in product(basis_plate, basis_plate)]
    assert len(basis_plate) == n_plate**2

    # Block B0 must be assembled
    start = time.time()
    B0 = np.zeros((n_plate**2, n_lmbda))
    print '\tAssembling B0 ...'
    for k, phi_k in enumerate(basis_plate):
        for j, chi_j in enumerate(basis_lmbda):
            B0[k, j] = quad(lambda s_hat: phi_k(*Beam_map(s_hat))*chi_j(s_hat),
                            [-1, 1])
            print '*',
        print

    B0 *= -L/2
    assemble_B0 = time.time() - start
    print '\tassembled B0 in %g s' % assemble_B0

    # Group A0, A1 into A and B0, B1 into B
    A = np.zeros((n_plate**2 + n_beam, n_plate**2 + n_beam))
    A[:n_plate**2, :n_plate**2] = A0
    A[n_plate**2:, n_plate**2:] = A1

    B = np.zeros((n_plate**2 + n_beam, n_lmbda))
    B[:n_plate**2, :] = B0
    B[n_plate**2:, :] = B1

    if eigs_only:
        # Eigenvalues of Schur
        # A inversed
        Ainv = la.inv(A)

        # Put together the C matrices
        # The matrices are scaled by L to the L_power = 2*(0.5 - s)
        L_powers = 2*(0.5 - fractions)
        # We know that shen basis yields diagonal stiffness matrix and
        # tridiagonal mass matrix.
        # then norm(u, 0) = U*M*U = U*(V*lmbda^1*V.T)*U
        # and  norm(u, 1) = U*A*U = U*(V*lmbda^0*V.T)*U
        #      norm(u, s) =         U*(V*lmbda^{1-s}*V.T)*U
        # () -- norm matrix + add L scaling
        e_powers = 1 - fractions

        M_penalty = shen_mass_matrix(n_lmbda)
        e_values, V = la.eigh(M_penalty)

        # Now assemble different C and use it to precondition Schur and compute
        # the eigenvalues - stored with key=str(s)
        eigenvalues = {}
        S = logm(B.T.dot(Ainv.dot(B)))
        for i, s in enumerate(fractions):
            L_power = L_powers[i]
            e_power = e_powers[i]

            diagonal = e_values**e_power
            C = V.dot(np.diag(diagonal).dot(V.T))
            C *= (0.5*L)**L_power
            C = logm(C)

            # Preconditioned
            Mat = C.dot(S)
            eigs = la.eigvals(Mat)
            lmbda_min = sorted(eigs)[:1]
            eigenvalues[str(s)] = lmbda_min

        return eigenvalues

    # Assemble the system from blocks
    N = n_plate**2 + n_beam + n_lmbda
    AA = np.zeros((N, N))
    AA[:n_plate**2 + n_beam, :n_plate**2 + n_beam] = A
    AA[:n_plate**2 + n_beam, n_plate**2 + n_beam:] = B
    AA[n_plate**2 + n_beam:, :n_plate**2 + n_beam] = B.T

    # To get the right hand side map f from domain to [-1, 1]^2
    x, y, s = symbols('x, y, s')
    f_lambda = lambdify([x, y], f)

    # Assemble the plate rhs
    start = time.time()
    F = np.zeros(n_plate**2)
    print '\tAssembling F ...'
    for i, v in enumerate(basis_plate):
        F[i] = quad(lambda x_hat, y_hat:
                    f_lambda(*F2(x_hat, y_hat))*v(x_hat, y_hat),
                    [-1., 1.], [-1., 1.])
        print ' '*i, '*'

    # Scale
    F *= L0*L1/4
    print '\tAssembled F in %g s.' % (time.time() - start)

    # Place in global right hand
    bb = np.zeros(N)
    bb[:n_plate**2] = F

    # Solve the system
    U = la.solve(AA, bb)

    # Extract displacement of beam
    U_plate = U[:n_plate**2]
    U_beam = U[n_plate**2:(n_plate**2 + n_beam)]
    U_lmbda = U[(n_plate**2 + n_beam):]

    # Return expansion coefficients and basis combined into functions
    # defined over V0 x V1 and [0, 1], [0, 1] respectively
    # First map the functions
    basis = shen_basis_symbolic(N_plate, x)
    basis_plate = [v0.subs({x: 2*(x-center0)/L0})*v1.subs({x: 2*(y-center1)/L1})
                   for v0, v1 in product(basis, basis)]

    basis_beam = [v.subs({s: 2*(s-center)})
                  for v in shen_basis_symbolic(N_beam, s)]
    basis_lmbda = [v.subs({s: 2*(s-center)})
                   for v in shen_basis_symbolic(N_lmbda, s)]

    assert len(U_plate) == len(basis_plate)
    assert len(U_beam) == len(basis_beam)
    assert len(U_lmbda) == len(basis_lmbda)

    u_plate = sum(Ui*vi for Ui, vi in zip(U_plate, basis_plate))

    u_beam = sum(Ui*vi for Ui, vi in zip(U_beam, basis_beam))

    u_lmbda = sum(Ui*vi for Ui, vi in zip(U_lmbda, basis_lmbda))

    return u_plate, u_beam, u_lmbda

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.plotting import plot3d, plot

    # Define problem
    x, y, s = symbols('x, y, s')
    f = S(1)
    P = np.array([0.25, 0])
    Q = np.array([0.5, 1])
    V0 = np.array([0, 0])
    V1 = np.array([1, 1])
    params = {'E_plate': 1.,
              'E_beam': 10.,
              'P': P,
              'Q': Q,
              'V0': V0,
              'V1': V1,
              'f': f,
              'n_plate': 10,
              'n_beam': 10,
              'n_lmbda': 10}

    # Solve
    u_plate, u_beam, u_lmbda = solve(params)

    # Plot
    # Plate displacement
    plot3d(u_plate, (x, V0[0], V1[0]), (y, V0[1], V1[1]),
           xlabel='$x$', ylabel='$y$', title=r'$u_h$')
    # Displacement of beam and plate on the beam
    x_s = P[0] + (Q[0] - P[0])*s
    y_s = P[1] + (Q[1] - P[1])*s
    u_plate_beam = u_plate.subs({x: x_s, y: y_s})
    plot(u_beam - u_plate_beam, (s, 0, 1), xlabel='$s L$',
         title=r'$u_h|_{B} - w_h$')
    # Lagrange multiplier
    plot(u_lmbda, (s, 0, 1), xlabel='$s L$',
         title=r'$\lambda$')
