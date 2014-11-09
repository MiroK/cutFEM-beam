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
    u = Function(V)
    A, b = assemble_system(a, L, bc)
    solve(A, u.vector(), b)

    # Plot solution
    if verbose:
        plot(u, title='numeric')
        plot(u_exact, mesh=mesh, title='exact')
        interactive()

    e_L2 = errornorm(u_exact, u, 'l2')

    return {'h': mesh.hmax(), 'L2': e_L2}


def mixed_solver(mesh, problem, p, vertbose=False):
    pass

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from problem2d import biharmonic2d_problem

    D, Lx, Ly, = 1, 1, 1
    problem = biharmonic2d_problem(D, Lx, Ly)

    mesh = RectangleMesh(0, 0, Lx, Ly, 40, 40)
    cg_solver(mesh, problem, 3, True)
