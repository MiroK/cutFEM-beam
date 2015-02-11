from dolfin import *
import numpy as np

def solve_problem(L, p=1, c1 = 1., c2 = 5.,
                  return_eigvals = False,
                  plotting = False,
		  constraint='H1'):
    """
    Solves a 2D-1D diffusion problem on a domainx.
    """
    assert L > 2
    assert constraint in ('H1', 'H10', 'L2')
    
    # 2D source term:
    f = Expression("12*x[0]*x[1]")

    # 1D source term
    g = Constant(0.)

    # BC on outer boundary (hom. neumann on hole boundary):
    u_bc_val = Constant(0.)
    u_bc_dom = "on_boundary"
    
    p_bc_val = Constant(0.)
    p_bc_dom = "on_boundary"
    
    N = 2 ** L
    h = 1./N
    mesh = UnitSquareMesh(N,N)
    V = FunctionSpace(mesh, "CG", p)
    W = MixedFunctionSpace([V,V,V])

    # Centerline subdomain for dS Measure on the 1D domain:
    beam = CompiledSubDomain("fabs(x[1]-.5)<delta", delta=DOLFIN_EPS)

    # Subdomain off the centerline for setting DirichletBC:
    off_beam = CompiledSubDomain("fabs(x[1]-.5)>delta", delta=1e-6)

    # V has 0 values on the boundary 
    bcsV = [DirichletBC(W.sub(0), u_bc_val, u_bc_dom)]
    # Q has 0 on the boundary and also in dofs outside the beam
    bcsQ = [DirichletBC(W.sub(1), p_bc_val, p_bc_dom),
            DirichletBC(W.sub(1), 0., off_beam)]
    # Multiplier space is like Q
    bcsL = [DirichletBC(W.sub(2), 0., p_bc_dom),
            DirichletBC(W.sub(2), 0., off_beam)]
    
    bcs = bcsV + bcsQ +bcsL

    # Measures 
    cfV = CellFunction("size_t", mesh)
    cfQ = CellFunction("size_t", mesh)
    ffQ = FacetFunction("size_t", mesh)
    beam.mark(ffQ, 1)
    dxV = Measure("dx")[cfV]
    dxQ = Measure("dx")[cfV]
    dsQ = Measure("dS")[ffQ]

    # We obtain linear system from differentiating functionals.
    w = Function(W)

    # J is the functional we seek to minimize
    J = .5 * Constant(c1) * grad(w[0])**2 * dxV \
      + .5 * Constant(c2) * avg(w[1].dx(0))**2 * dsQ(1) \
      - f *w[0]* dxV\
      - avg(g*w[1]) * dsQ(1)

    # Choose how the u = v on beam is enforces
    if constraint == 'H1':
        J += avg((w[0]-w[1]).dx(0))*avg(w[2].dx(0)) * dsQ(1) \
             + avg((w[0]-w[1]))*avg(w[2]) * dsQ(1)

    elif constraint == 'H10':
        J += avg((w[0]-w[1]).dx(0))*avg(w[2].dx(0)) * dsQ(1)

    else:
        raise NotImplementedError('Wait!')
        # J += avg((w[0]-w[1]))*avg(w[2]) * dsQ(1)

    # First line clear
    # Second line dsQ((1, 2)) the FacetFunction is not multivalueed so 1, 2
    # gives entire line and dsQ(1) only the constrained part
    # .dx is the derivative. For anything on facet we need restrinctions
    # hence avg
    # The point is that is enforces useing full H1 inner product

    # Functional K is just used to define the precondioner
    K = .5 * Constant(c1) * grad(w[0])**2 * dxV \
      + .5 * Constant(c2) * avg(w[0].dx(0))**2 * dsQ(1) \
      + .5 * Constant(c2) * avg(w[1].dx(0))**2 * dsQ(1) \
      + .5 / Constant(c2) * avg(w[2].dx(0))**2 * dsQ(1)
    
    L = derivative(J, w, TestFunction(W))
    a = derivative(L, w, TrialFunction(W))
    t = derivative(K, w, TestFunction(W))
    r = derivative(t, w, TrialFunction(W))
    
    wh = Function(W)

    # Need to pass the keep_diagonal option in order to set BC dofs
    # (due to the use of vanishing measures)
    A, b = assemble_system(a, -L, bcs = bcs, keep_diagonal = True)
    R, _ = assemble_system(r, -L, bcs = bcs, keep_diagonal = True)
    solve(A, wh.vector(), b)
    uh, ph, lh = wh.split(True)

    if plotting:
        plot(uh, title = "Solution on the 2D domain")
        plot(ph, title = "Solution on the 1D domain")
        plot(lh, title = "Multiplier on the 1D domain")
        interactive()
    
    if return_eigvals:
        from scipy.linalg import eigvalsh
        print "Computing eigenvalues of A size %d" % A.size(0)
        eigvals = eigvalsh(A.array(), R.array())

        return (uh, ph, lh), eigvals
    else:
        return (uh, ph, lh), None

if __name__ == "__main__":
    p  = 1    # element order
    c1 = 1.   # 2D coefficient
    c2 = .5   # 1D coefficient
    L = 4

    _, eigs = solve_problem(L, p=p, c1=c1, c2=c2,
                            plotting=False, return_eigvals=True,
                            constraint='H1')

    eigs = np.abs(eigs)
    print np.min(eigs), np.max(eigs)
