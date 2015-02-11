from dolfin import *
import numpy

def solve_problem(L, p=1, c1 = 1., c2 = 5.,
                  return_eigvals = False,
                  plotting = False):
    """
    Solves a 2D-1D diffusion problem on a domain with a hole,
    the 1D domain bridges the hole.
    """

    if L < 3:
        raise RuntimeError("Geometry cannot be resolved for  L < 3")
    
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

    # Hole inner subdomain for setting DirichletBC:
    holeI = CompiledSubDomain("(min(x[0],x[1])>.375+delta)"
                             +"&&(max(x[0],x[1])<.625-delta)",
                             delta = DOLFIN_EPS)
    
    # Hole outer subdomain to subtracted from dx measure:
    holeO = CompiledSubDomain("(min(x[0],x[1])>.375-delta)"
                             +"&&(max(x[0],x[1])<.625+delta)",
                             delta = 3*h/4)
    
    # Centerline subdomain for dS Measure on the 1D domain:
    cl = CompiledSubDomain("fabs(x[1]-.5)<delta",delta=DOLFIN_EPS)

    # Constraint subdomain used for dS Measure:
    clH = CompiledSubDomain("(fabs(x[1]-.5)<delta)"
                            +"&&(min(x[0],x[1])>.375-delta)"
                            +"&&(max(x[0],x[1])<.625+delta)",
                            delta=DOLFIN_EPS)
    
    # Subdomain off the centerline for setting DirichletBC:
    off_cl = CompiledSubDomain("fabs(x[1]-.5)>delta",delta=1e-6)

    # Collect all the boundary conditions
    bcsV = [DirichletBC(W.sub(0), u_bc_val, u_bc_dom),
            DirichletBC(W.sub(0), 0., holeI) if L>3 else \
            DirichletBC(W.sub(0), 0.,
                        holeI.cppcode.replace("delta","DOLFIN_EPS"),
                        method="pointwise")]

    bcsQ = [DirichletBC(W.sub(1), p_bc_val, p_bc_dom),
            DirichletBC(W.sub(1), 0., off_cl),]

    bcsL = [DirichletBC(W.sub(2), 0., p_bc_dom),
            DirichletBC(W.sub(2), 0., off_cl),
            DirichletBC(W.sub(2), 0., holeI) if L>3 else \
            DirichletBC(W.sub(2), 0.,
                        holeI.cppcode.replace("delta","DOLFIN_EPS"),
                        method="pointwise")]
    
    bcs = bcsV + bcsQ +bcsL

    # Measures 
    cfV = CellFunction("size_t", mesh)
    cfQ = CellFunction("size_t", mesh)
    ffQ = FacetFunction("size_t", mesh)
    holeO.mark(cfV,1)
    holeI.mark(cfV,2)
    off_cl.mark(cfQ, 1) 
    cl.mark(ffQ, 1)
    clH.mark(ffQ,2)
    dxV = Measure("dx")[cfV]
    dxQ = Measure("dx")[cfV]
    dsQ = Measure("dS")[ffQ]

    # We obtain linear system from differentiating functionals.
    w = Function(W)

    # J is the functional we seek to minimize
    J = .5 * c1 * grad(w[0])**2 * dxV(0) \
      + .5 * c2 * avg(w[1].dx(0))**2 * dsQ((1,2)) \
      + avg((w[0]-w[1]).dx(0))*avg(w[2].dx(0)) * dsQ(1) \
      + avg((w[0]-w[1]))*avg(w[2]) * dsQ(1) \
      - f *w[0]* dxV(0)\
      - avg(g*w[1]) * dsQ((1,2))

    # First line clear
    # Second line dsQ((1, 2)) the FacetFunction is not multivalueed so 1, 2
    # gives entire line and dsQ(1) only the constrained part
    # .dx is the derivative. For anything on facet we need restrinctions
    # hence avg
    # The point is that is enforces useing full H1 inner product

    # Functional K is just used to define the precondioner
    K = .5 * c1 * grad(w[0])**2 * dxV(0) \
      + .5 * c2 * avg(w[0].dx(0))**2 * dsQ(1) \
      + .5 * c2 * avg(w[1].dx(0))**2 * dsQ((1,2)) \
      + .5 / c2 * avg(w[2].dx(0))**2 * dsQ(1)
    
    L = derivative(J, w, TestFunction(W))
    a = derivative(L, w, TrialFunction(W))
    t = derivative(K, w, TestFunction(W))
    r = derivative(t, w, TrialFunction(W))
    
    wh = Function(W)
    uh, ph, lh = split(wh)

    # Need to pass the keep_diagonal option in order to set BC dofs
    # (due to the use of vanishing measures)
    A, b = assemble_system(a, -L, bcs = bcs, keep_diagonal = True)
    R, _ = assemble_system(r, -L, bcs = bcs, keep_diagonal = True)
    solve(A, wh.vector(), b)

    if plotting:
        plot(uh, title = "Solution on the 2D domain")
        plot(ph, title = "Solution on the 1D domain")
        plot(lh, title = "Multiplier on the 1D domain")
        interactive()
    
    if return_eigvals:
        from scipy.linalg import eigvalsh
        print "Computing eigenvalues for A size %d" % A.size(0)
        eigvals = eigvalsh(A.array(), b = R.array())
        return wh, eigvals
    else:
        return wh, None

if __name__ == "__main__":
    p  = 1    # element order
    c1 = 1.   # 2D coefficient
    c2 = .5   # 1D coefficient

    for L in [3, 4, 5, 6]:
      wh, eigvals = solve_problem(L, p=p, c1=c1, c2=c2,
                                  plotting=False, return_eigvals=True)
      h = wh.function_space().mesh().hmin()
      if eigvals != None:
          lambda_min, lambda_max = numpy.sort(abs(eigvals))[[0,-1]]
          print "h=%g cond=%g lmin=%g" % (h, lambda_max/lambda_min, lambda_min)
    
