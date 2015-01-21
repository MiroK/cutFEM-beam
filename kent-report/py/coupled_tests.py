'''
In all functions below A is [[Ap, 0],  B is [[Bp],
                             [0, Ab]]        [Bb]]
Ap, Ab, Bp, Bb is what the problem can compute. It can also assemble the
system_matrix [[A, B],      and the Schur complement B.T*inv(A)*B.
               [B.T, 0]]
'''

import numpy as np
import numpy.linalg as nla
import scipy.linalg as la


def brezzi_coercivity_arnold(problem, M):
    '''
    Compute the smallest eigenvalue of the problem
    
    [[A, B]   [[U], = [[M, 0], [[U],
     [B.T, 0]] [P]]    [0, 0]]   [P]]
    '''
    # Make sure we have M with correct size
    m, n = problem.m, problem.n
    size = m+n
    assert M.shape == (size, size)        
    # System
    S = problem.system_matrix()
    # Build the rhs
    T = np.zeros_like(S)
    T[:size, :size] = M
    # TODO
    eigenvalues = la.eigvals(S, T)
    eigenvalues = np.sqrt(eigenvalues**2).real
    print eigenvalues
    return np.min(eigenvalues)


def brezzi_infsup_qin(problem, M, N):
    pass


def babuska(problem, M, N):
    '''
    Compute the smallest eigenvalue of the problem
    
    [[A, B]   [[U], = [[M, 0], [[U],
     [B.T, 0]] [P]]    [0, N]]  [P]]
    '''
    # Make sure we have M, N with correct size
    m, n, r = problem.m, problem.n, problem.r
    assert M.shape == (m+n, m+n)
    assert N.shape == (r, r)

    # System
    S = problem.system_matrix()
    # Build the rhs
    T = np.zeros_like(S)
    size = m + n
    T[:size, :size] = M
    T[size:, size:] = N
  
    # Solve
    eigenvalues = la.eigvals(S, T)
    eigenvalues = np.sqrt(eigenvalues**2).real
    return np.min(eigenvalues)


def schur(problem, N):
    'Compute smallest eigenvalues of (B.T*inv(A)*B)*P = \lambda N*P'
    # Make sure dimension are okay
    S = problem.schur_complement_matrix(norms=[None])[0]
    assert S.shape == N.shape

    # Solve
    eigenvalues = la.eigvals(S, N)
    eigenvalues = np.sqrt(eigenvalues**2).real
    return np.min(eigenvalues)


def common_kernel(problem):
    pass

def qin_separately(problem, Mp, Nb, N):
    pass


def schur_separately(problem, N, what):
    '''
    Compute smallest eigenvalues of (Bx.T*inv(Ax)*Bx)*P = \lambda N*P, x is 
    specified by what (plate or beam)
    '''
    if what == 'plate':
        A, B = problem.Ap_matrix(), problem.Bp_matrix()
    elif what == 'beam':
        A, B = problem.Ab_matrix(), problem.Bb_matrix()
    else:
        raise ValueError('plate or beam')

    # Make sure dimension are okay
    S = B.T.dot(nla.inv(A).dot(B))
    assert S.shape == N.shape

    # Solve
    eigenvalues = la.eigvals(S, N)
    eigenvalues = np.sqrt(eigenvalues**2).real
    return np.min(eigenvalues)


def preconditioned_problem(problem, blocks):
    'Check condition number of the preconditioned and not preconditioned system'
    P = problem.preconditioner(blocks)
    S = problem.system_matrix()
    P = P.dot(S)
    return nla.cond(P), nla.cond(S)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from coupled_eigen_laplace import CoupledEigenLaplace
    from plate_beam import LineBeam
    from eigen_poisson import mass_matrix, laplacian_matrix

    # Beam
    A = np.array([-1., -1.])
    B = np.array([0., 1.])
    beam = LineBeam(A, B)
    # Material
    params = {'plate_E': 2.,
              'beam_E': 10.}
    
    # Different discretizations with eigen functions
    # Check schur
    for n in range(2, -11):
        problem = CoupledEigenLaplace(ms=[n, n], n=n, r=n, beam=beam,
                                      params=params)
       
        # This does not decrease so fast only for -1, -0.5
        Nmat = problem.C_matrix(norm=-1)
        lmin = schur_separately(problem, Nmat, what='plate')
        print n, lmin
    
    # Check Babuska
    for size in range(2, -11):
        problem = CoupledEigenLaplace(ms=[size, size], n=size, r=size,
                                      beam=beam, params=params)
        
        # Build the norm matrices
        ms = problem.ms
        m, n = problem.m, problem.n
        size = m + n 
        Mmat = np.zeros((size, size))
        
        # H1 seminorm
        # Top left block is tensor product of laplacians
        # Bottom right ...
        Mmat[:m, :m] = np.kron(laplacian_matrix(ms[0]), mass_matrix(ms[1])) +\
                       np.kron(mass_matrix(ms[0]), laplacian_matrix(ms[1]))
        # Bottom right is over beam
        Mmat[m:,m:] = laplacian_matrix(n)/float(beam.Jac)

        # Have learned before that Nmat is okay as C_matrix for norm -1, -0.5
        Nmat = problem.C_matrix(norm=-1)
        
        lmin = babuska(problem, Mmat, Nmat)
        print n, lmin


    for n in range(2, 11):
        problem = CoupledEigenLaplace(ms=[n, n], n=n, r=n, beam=beam,
                                      params=params)

        # Build the blocks of preconditioner
        ms = problem.ms
        m, n = problem.m, problem.n
        P0 = np.kron(laplacian_matrix(ms[0]), mass_matrix(ms[1])) +\
             np.kron(mass_matrix(ms[0]), laplacian_matrix(ms[1]))
        P0 = nla.inv(P0)

        P1 = laplacian_matrix(n)/float(beam.Jac)
        P1 = nla.inv(P1)
       
        P2 = la.inv(problem.C_matrix(norm=-1))

        blocks = [[P0, 0, 0], [0, P1, 0], [0, 0, P2]]
        cond_p, cond_nop = preconditioned_problem(problem, blocks)
        print n, cond_p, cond_nop
