from solvers import VariationalSolver1d, VariationalSolver2d
from scipy.sparse.linalg import LinearOperator
from polynomials import sine_basis
from quadrature import __EPS__
import numpy as np


def solve_biharmonic_1d(f, N, E=1, a=0, b=1, eps=__EPS__, n_refs=10):
    '''
    Solve biharmonic problem
                         E*u^(4) = f in [a, b]
                             u = u(2) = 0 on a, b

    In the variational formulation use N basis function of type sin(i*pi*x)
    for i = 1, ..., N.
    '''
    class BiharmonicSolver(VariationalSolver1d):
        def assemble_A(self, N):
            diag = E*np.array([np.pi**4*l**4/(b-a)**3 for l in range(1, N+1)])
            A = LinearOperator(shape=(N, N),
                               matvec=lambda v: v*diag,
                               dtype='float64')
            return A

        def basis_functions(self, N):
            return sine_basis([N])

    solver = BiharmonicSolver()
    return solver.solve(f, N, E=E, a=a, b=b, eps=eps, n_refs=n_refs)


def solve_biharmonic_2d(f, MN, E=1, domain=[[0, 1], [0, 1]],
                        eps=__EPS__, n_refs=10):
    '''
    Solve biharmonic problem
                         -E*laplace^2(u) = f in domain=[ax, bx] X [ay, by]
                                       u = 0 on domain boundary

    In the variational formulation use sine_basis with last function
    sin(M*pi*x)*sin(N*pi*y)
    '''
    class BiharmonicSolver(VariationalSolver2d):
        def assemble_A(self, MN):
            M, N = MN
            [[ax, bx], [ay, by]] = domain
            Lx = float(bx - ax)
            Ly = float(by - ay)
            pi = np.pi
            diag = E*np.array([Ly/Lx**3*(pi*i)**4 +
                               2/Lx/Ly*(pi*i)**2*(pi*j)**2 +
                               Lx/Ly**3*(pi*j)**4
                               for i in range(1, M+1)
                               for j in range(1, N+1)])
            A = LinearOperator(shape=(M*N, M*N),
                               matvec=lambda v: v*diag,
                               dtype='float64')
            return A

        def basis_functions(self, MN):
            return sine_basis(MN)

    solver = BiharmonicSolver()
    return solver.solve(f=f, MN=MN, E=E, domain=domain, eps=eps, n_refs=n_refs)


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from problems import manufacture_biharmonic_1d
    from quadrature import errornorm
    from sympy import symbols, sin, pi

    x = symbols('x')
    a = -1
    b = 1.5
    E = 2
    u = sin(pi*(x-a)/(b-a))
    problem1d = manufacture_biharmonic_1d(u=u, a=a, b=b, E=E)
    f = problem1d['f']

    U, basis = solve_biharmonic_1d(f, N=20, E=E, a=a, b=b,
                                   eps=__EPS__, n_refs=10)
    e = errornorm(u, (U, basis), domain=[[a, b]], norm_type='L2')

    assert abs(e) < 1E-15

    from problems import manufacture_biharmonic_2d

    # 2d
    x, y = symbols('x, y')
    E = 2
    ax, bx = -1, 1.
    ay, by = 0, 1.
    domain = [[ax, bx], [ay, by]]
    u = sin(2*pi*(x-ax)/(bx-ax))*sin(pi*(y-ay)/(by-ay))
    problem2d = manufacture_biharmonic_2d(u=u, domain=domain, E=E)
    f = problem2d['f']
    U, basis = solve_biharmonic_2d(f, MN=[3, 3], E=E, domain=domain,
                                   eps=__EPS__, n_refs=10)
    e = errornorm(u, (U, basis), domain=domain, norm_type='L2')
    assert abs(e) < 1E-15
