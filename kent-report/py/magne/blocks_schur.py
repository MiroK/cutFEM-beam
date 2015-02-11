from dolfin import *
import numpy as np
import scipy.sparse.linalg as la
from scipy.sparse import csr_matrix, bmat

parameters['linear_algebra_backend'] = 'uBLAS'


def to_sparse(mat):
    'Convert DOLFIN uBLAS matrix to scipy.sparse matrix'
    rows, cols, values = mat.data()
    return csr_matrix((values, cols, rows))


def solve_problem(L, p=1, c1 = 1., c2 = 5.,
                  return_eigs = False,
                  plotting = False,
		  constraint='H1'):
    """
    Solves a 2D-1D diffusion problem on a domain.
    """
    assert L > 2
    assert constraint in ('H1', 'H10', 'L2')

    # 2D source term:
    f = Expression("12*x[0]*x[1]")

    # 1D source term
    g = Constant(0.)

    N = 2 ** L
    h = 1./N
    mesh = UnitSquareMesh(N,N)

    # We will only have one function space V
    # Plate unknown will use all its unknowns
    # Beam unknown and the multiplier will only use the part on the beam
    V = FunctionSpace(mesh, 'CG', p)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Beam - for defining beam measure
    beam = CompiledSubDomain("fabs(x[1]-.5)<delta", delta=DOLFIN_EPS)

    # Plate\Beam - for constraining extra dofs of beam, multiplier spaces
    off_beam = CompiledSubDomain("fabs(x[1]-.5)>delta", delta=1e-6)

    # Measures
    plate_cells = CellFunction('size_t', mesh, 0)
    d_plate = Measure('dx')[plate_cells]

    beam_facets = FacetFunction('size_t', mesh, 0)
    beam.mark(beam_facets, 1)
    d_beam = Measure('dS')[beam_facets]

    # Forms for blocks of system matrix A
    a0 = Constant(c1)*inner(grad(u), grad(v))*d_plate(0)
    a1 = Constant(c2)*inner(avg(u.dx(0)), avg(v.dx(0)))*d_beam(1)
    
    if constraint == 'H1':
        b0 = avg(v.dx(0))*avg(u.dx(0))*d_beam(1) + avg(v)*avg(u)*d_beam(1)
        # Form for b1 only differs by the sign
    elif constraint == 'H10':
        b0 = avg(v.dx(0))*avg(u.dx(0))*d_beam(1)
    else:
        b0 = avg(v)*avg(u)*d_beam(1)

    # Forms for the right hand side b
    L0 = f*v*d_plate(0)
    L1 = avg(g*v)*d_beam(1)

    # Zeros values or constrain not beam
    bc0 = DirichletBC(V, Constant(0), DomainBoundary())   # Fix boundary value
    bc1 = DirichletBC(V, Constant(0), off_beam)           # Fix plate\beam dofs

    A0, vec0 = assemble_system(a0, L0, bcs=[bc0])
    A1, vec1 = assemble_system(a1, L1, bcs=[bc0, bc1], keep_diagonal=True)
    B0, _ = assemble_system(b0, L0, bcs=[bc0], keep_diagonal=True)
    B1, _ = assemble_system(-b0, L1, bcs=[bc0, bc1], keep_diagonal=True)

    # Assemble sparse A from blocks
    A0 = to_sparse(A0)
    A1 = to_sparse(A1)
    B0 = to_sparse(B0)
    B1 = to_sparse(B1)
    # A is [[A0, 0, B0], [0, A1, B1], [B0.T, B1.T, 0]]
    A = bmat([[A0, None, B0], [None, A1, B1], [B0.T, B1.T, None]])

    # Assemble b from blocks
    # b is [b0, b1, 0]
    m = V.dim()
    b = np.zeros(A.shape[0])
    b[:m] = vec0.array()
    b[m:2*m] = vec1.array()

    # Solve the system
    print 'Solving linear system %d x %d' % A.shape
    x = la.spsolve(A, b)

    # Build dolfin Functions
    uh = Function(V)
    uh.vector()[:] = x[:m]

    ph = Function(V)
    ph.vector()[:] = x[m:2*m]

    lh = Function(V)
    lh.vector()[:] = x[2*m:]
    
    if plotting:
        plot(uh)
        plot(ph)
        plot(lh)
        interactive()

    if return_eigs:
        print V.dim()

        # First B is [[B0], [B1]]
        B = bmat([[B0], [B1]])
        # and A is [[A0, 0], [0, A1]]
        A = bmat([[A0, None], [None, A1]])
        # Get the Schur complement S = B.T inv(A) B
        S = B.T.dot(la.inv(A).dot(B))

        # Create the preconditioner for Shur
        # Forms
        p_s = Constant(1./c2)*inner(avg(u.dx(0)), avg(v.dx(0)))*d_beam(1)
        # Block for preconditioning S
        PS, _ = assemble_system(p_s, L0, bcs=[bc0, bc1], keep_diagonal=True)

        # Solve the generalized eigenvalue problem
        from scipy.linalg import eigvalsh
        print "Computing eigenvalues of S size %d" % S.shape[0]
        eigs = eigvalsh(S.toarray(), PS.array())

        return (uh, ph, lh), eigs

    else:
        return (uh, ph, lh), None
    

if __name__ == "__main__":
    p  = 1    # element order
    c1 = 1.   # 2D coefficient
    c2 = .5   # 1D coefficient

    for L in [3, 4, 5]:    
        _, eigs = solve_problem(L, p=p, c1=c1, c2=c2,
                                plotting=False, return_eigs=True,
                                constraint='L2')

        eigs = np.abs(eigs)
        print np.min(eigs), np.max(eigs)
