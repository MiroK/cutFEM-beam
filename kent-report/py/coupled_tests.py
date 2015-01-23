'''
In all functions below A is a block matrix [[Ap, 0], [0, Ab]] and likewise B is
a block matrix [[Bp], [Bb]]. Submatrices Ap, Ab, Bp, Bb is what the problem can
compute. It can also assemble the system_matrix [[A, B], [B.T, 0]] and the Schur
complement B.T*inv(A)*B. The functions below involve eigenvalues of the
matrices.
'''

import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
import scipy.sparse.linalg as sla


def brezzi_infsup_qin(problem, M, N):
    '''
    Compute the smallest eigenvalue of the problem
        
        [[M, B], [B.T, 0]]*[[U], [P]] = -lamnda * [[0, 0], [0, N]][[U], [P]]

    This becomes B.T*inv(M)*B*P = lmbda N*P.
    '''
    B = problem.B_matrix()
    S = B.T.dot(nla.inv(M).dot(B))
    # Make sure shapes are okay
    assert S.shape == N.shape

    # Solve
    eigenvalues = la.eigvals(S, N)
    eigenvalues = np.sqrt(eigenvalues**2).real
    return np.min(eigenvalues)


def brezzi_infsup_qins_components(problem, Mp, Mb, N):
    '''
    Matrix M is block [[Mp, 0], [0, Mb]] and B has also block structure. We 
    break the problem B.T*inv(M)*B*P = lmbda N*P into two problems, one for
    p-block, the other for b-block. Specifically

        Bp.T*inv(Mp)*Bp*P = lmbda N*P

        Bb.T*inv(Mb)*Bb*P = lmbda N*P

    The idea is that perhaps the bad scaling is more due to one guy than the
    other. 
    '''
    Bs = [problem.Bp_matrix(), problem.Bb_matrix()]
    Ms = [Mp, Mb]
    eigs = []
    for B, M in zip(Bs, Ms):
        S = B.T.dot(nla.inv(M).dot(B))
        assert S.shape == N.shape

        # Solve
        eigenvalues = la.eigvals(S, N)
        eigenvalues = np.sqrt(eigenvalues**2).real
        eigs.append = np.min(eigenvalues)

    return tuple(eigs)


def babuska(problem, M, N):
    '''
    Compute the smallest eigenvalue of the problem

        [[A, B], [B.T, 0]]*[[U], [P]] = lambda [[M, 0], [0, N]]*[[U], [P]]
    '''
    # Make sure we have M, N with correct size
    m, n, r = problem.m, problem.n, problem.r
    assert M.shape == (m+n, m+n)

    # System
    S = problem.system_matrix()
    # Build the rhs
    T = np.zeros_like(S)
    size = m + n
    T[:size, :size] = M

    if not isinstance(N, list):
        N = [N]

    eigs = []
    for Nmat in N:
        assert Nmat.shape == (r, r)
        T[size:, size:] = Nmat
  
        # Solve
        eigenvalues = la.eigvals(S, T)
        eigenvalues = np.sqrt(eigenvalues**2).real
        lmin = np.min(eigenvalues)
        eigs.append(lmin)
    return eigs


def schur(problem, N):
    'Compute smallest eigenvalues of (B.T*inv(A)*B)*P = lambda N*P'
    S = problem.schur_complement_matrix(norms=[None])[0]
    # Single matrix
    if not isinstance(N, list):
        N = [N]
    # Now loop
    eigs = []
    for Nmat in N:
        # Make sure dimension are okay
        assert S.shape == Nmat.shape

        # Solve
        eigenvalues = la.eigvals(S, Nmat)
        eigenvalues = np.sqrt(eigenvalues**2).real
        lmin = np.min(eigenvalues)
        eigs.append(lmin)
    return eigs


def schur_components(problem, N):
    '''
    Due to block structure of A and B we brake the Schur problem into problems
    for plate and beam blocks to perhaps isolate the scaling.

        (Bp.T*inv(Ap)*Bp)*P = lambda N*P and

        (Bb.T*inv(Ab)*Bb)*P = lambda N*P and
    '''
    As = [problem.Ap_matrix(), problem.Ab_matrix()]
    Bs = [problem.Bp_matrix(), problem.Bb_matrix()]
    # These are all eigs of Sp and Sb
    eigs = []
    # Always make sure that we can iterate
    if not isinstance(N, list):
        N = [N]
    for A, B in zip(As, Bs):
        # Assemble S only once per compoenent
        S = B.T.dot(nla.inv(A).dot(B))
        # Get the comp eigs
        component_eigs = []
        for Nmat in N:
            assert S.shape == Nmat.shape

            # Solve
            eigenvalues = la.eigvals(S, Nmat)
            eigenvalues = np.sqrt(eigenvalues**2).real
            component_eigs.append(np.min(eigenvalues))
        # Append to all
        eigs.append(component_eigs)

    return eigs


def preconditioned_problem(problem, Pblocks=[]):
    'Check condition number of preconditioned and not preconditioned systems'
    # Make all the preconditioners from blocks
    Ps = [problem.preconditioner(blocks) for blocks in Pblocks]
    # Assemblel system once
    S = problem.system_matrix()
    # Loop the preconditioners
    conds = [nla.cond(P.dot(S)) for P in Ps]
    return conds


def brezzi_coercivity_arnold(problem, M):
    '''
    Get smallest eigenvalues of 
        [[A, B], [B.T, 0]][[U], [P]] = lambda*[[M, 0], [0, 0]][[U], [P]]
    '''
    raise NotImplementedError('!')
    # Make sure we have M, N with correct size
    # m, n = problem.m, problem.n
    # assert M.shape == (m+n, m+n)

    # System
    # S = problem.system_matrix()
    # Build the rhs
    # T = np.zeros_like(S)
    # size = m + n
    # T[:size, :size] = M

    # U, sigma, V = nla.svd(problem.B_matrix().T)
    # print sigma
  
    # Solve
    # eigenvalues = sla.eigs(S, k=3, M=T, which='SM', sigma=0.01)
    # eigenvalues = np.sqrt(eigenvalues**2).real
    # return np.min(eigenvalues)
    # return 'xxx'

def ker_tests(problem):
    m, n, r = problem.ms[0], problem.n, problem.r

    #Bp = problem.Bp_matrix()
    Bb = problem.Bb_matrix()

    U, sigmas, V = nla.svd(Bb)
    s = len(np.where(sigmas > 1E-12)[0])
    print sigmas
    print np.where(sigmas > 1E-12), Bb.shape[1]
    print Bb
    print Bb.shape[1] - s

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
    
    problem = CoupledEigenLaplace([4, 4], 5, 7, beam, params)
    print ker_tests(problem)
