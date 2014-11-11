from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Make mesh ghosted for evaluation of DG terms
parameters["ghost_mode"] = "shared_facet"

def cg_solver(mesh, problem, p=2, verbose=False):
    '''
    Solve biharmonic problem:

                laplace^2(u) = f in mesh
                           u = 0 on mesh boundary
                  laplace(u) = 0 on mesh boundary

    We use CG elements of degree p. Interior penalty is used to force continuity
    of laplace u.
    '''
    assert p > 1
    if isinstance(mesh, str):
        mesh = Mesh(mesh)

    u_exact = problem['u']
    f = problem['f']

    V = FunctionSpace(mesh, 'CG', p)
    u = TrialFunction(V)
    v = TestFunction(V)

    h = CellSize(mesh)
    h_avg = (h('+') + h('-'))/2.0
    n = FacetNormal(mesh)

    # Penalty parameter
    alpha = Constant(8)

    # Define bilinear form
    # Standard term
    a = inner(div(grad(u)), div(grad(v)))*dx

    # Ip stab of surface term with grad(v).n
    a += - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
         - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
         + alpha/h_avg**(p-1)*inner(jump(grad(u), n), jump(grad(v), n))*dS

    # Define linear form
    L = inner(f, v)*dx

    # DirichletBC
    bc = DirichletBC(V, Constant(0), DomainBoundary())

    # Solve variational problem
    A, M = PETScMatrix(), PETScMatrix()
    b = PETScVector()

    m = inner(u, v)*dx
    assemble_system(m, L, bc, A_tensor=M, b_tensor=b)
    assemble_system(a, L, bc, A_tensor=A, b_tensor=b)

    try:
        esolver = SLEPcEigenSolver(A)
        esolver.parameters['spectrum'] = 'largest magnitude'
        esolver.solve(1)
        max_r, max_c = esolver.get_eigenvalue(0)

        if mesh.num_cells() < 513:
            print '..'
            esolver.parameters['spectrum'] = 'smallest magnitude'
            esolver.solve(1)
            min_r, min_c = esolver.get_eigenvalue(0)

            print '%2E %2E %2E \t' % (max_r, min_r, max_r/min_r)
    except:
        print 'Eigensolver went wrong'

    u = Function(V)
    solve(A, u.vector(), b)

    # Plot solution
    if verbose:
        plot(u, title='numeric')
        plot(u_exact, mesh=mesh, title='exact')
        interactive()

    e_L2 = errornorm(u_exact, u, 'l2')

    return {'h': mesh.hmax(), 'L2': e_L2, 'a_max': max_r}


def mixed_solver(mesh, problem, p, verbose=False):
    '''
    Solve biharmonic problem:

                laplace^2(u) = f in mesh
                           u = 0 on mesh boundary
                  laplace(u) = 0 on mesh boundary

    by braking it into

            -laplace(u) = sigma
            -laplace(sigma) = f  in mesh
                     u = 0
                     sigma = 0  on mesh boundary

    We use CG elements of degree p for u-space and n for s-space
    '''
    if isinstance(mesh, str):
        mesh = Mesh(mesh)

    u_exact = problem['u']
    f = problem['f']

    # --------

    V = FunctionSpace(mesh, 'CG', p)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Stiffness matrix
    a = inner(grad(u), grad(v))*dx
    m = inner(u, v)*dx

    # Define linear form
    L = inner(f, v)*dx

    # DirichletBC
    bc = DirichletBC(V, Constant(0), DomainBoundary())

    A = PETScMatrix()
    b = PETScVector()
    assemble_system(a, L, bc, A_tensor=A, b_tensor=b)
    B, _ = assemble_system(m, L)

    solver = LUSolver(A)
    solver.parameters['reuse_factorization'] = True

    # Solve first for moment
    sigma = Function(V)
    print '.'
    solver.solve(sigma.vector(), b)
    print '..'
    # Make the rhs for discplacement system
    B.mult(sigma.vector(), b)
    # Solve for u
    u = Function(V)
    solver.solve(u.vector(), b)
    print '...'

    # Plot solution
    if verbose:
        plot(u, title='numeric')
        plot(u_exact, mesh=mesh, title='exact')
        interactive()

    e_L2 = errornorm(u_exact, u, 'l2')

    return {'h': mesh.hmax(), 'L2': e_L2, 'a_max': 1}
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from problem2d import joris_problem

    D, Lx, Ly, = 1, 1, 1
    problem = joris_problem(D, Lx, Ly)

    mesh = RectangleMesh(0, 0, Lx, Ly, 40, 40)
    mixed_solver(mesh, problem, 2, True)
