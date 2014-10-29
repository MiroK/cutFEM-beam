from solvers import VariationalSolver1d, VariationalSolver2d
from scipy.sparse.linalg import LinearOperator
from polynomials import sine_basis
from quadrature import __EPS__
import numpy as np


def solve_sine_1d(f, N, E=1, a=0, b=1, eps=__EPS__, n_refs=10):
    '''
    Solve Poisson problem
                         -E*u^(2) = f in [a, b]
                             u = 0 on a, b

    In the variational formulation use sine_basis with last function
    sin(N*pi*x)
    '''
    class PoissonSolver(VariationalSolver1d):
        def assemble_A(self, N):
            diag = E*np.array([np.pi**2*l**2/(b-a) for l in range(1, N+1)])
            A = LinearOperator(shape=(N, N),
                               matvec=lambda v: v*diag,
                               dtype='float64')
            return A

        def basis_functions(self, N):
            return sine_basis([N])

    solver = PoissonSolver()
    return solver.solve(f, N, E=E, a=a, b=b, eps=eps, n_refs=n_refs)


def solve_sine_2d(f, MN, E=1, domain=[[0, 1], [0, 1]], eps=__EPS__, n_refs=10):
    '''
    Solve Poisson problem
                         -E*laplace(u) = f in domain=[ax, bx] X [ay, by]
                                     u = 0 on domain boundary

    In the variational formulation use sine_basis with last function
    sin(M*pi*x)*sin(N*pi*y)
    '''
    class PoissonSolver(VariationalSolver2d):
        def assemble_A(self, MN):
            M, N = MN
            [[ax, bx], [ay, by]] = domain
            Lx = float(bx - ax)
            Ly = float(by - ay)
            pi = np.pi
            diag = E*np.array([Ly/Lx*pi**2*i**2 + Lx/Ly*pi**2*j**2
                               for i in range(1, M+1)
                               for j in range(1, N+1)])
            A = LinearOperator(shape=(M*N, M*N),
                               matvec=lambda v: v*diag,
                               dtype='float64')
            return A

        def basis_functions(self, MN):
            return sine_basis(MN)

    solver = PoissonSolver()
    return solver.solve(f=f, MN=MN, E=E, domain=domain, eps=eps, n_refs=n_refs)


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from problems import manufacture_poisson_1d
    from quadrature import errornorm
    from sympy import sin, symbols, pi

    # 1d
    x = symbols('x')
    a = -1
    b = 2.
    E = 2
    u = sin(pi*(x-a)/(b-a))
    problem1d = manufacture_poisson_1d(u=u, a=a, b=b, E=E)
    f = problem1d['f']
    U, basis = solve_sine_1d(f, N=20, E=E, a=a, b=b, eps=__EPS__, n_refs=10)
    e = errornorm(u, (U, basis), domain=[[a, b]], norm_type='L2')
    assert abs(e) < 1E-15

    from problems import manufacture_poisson_2d
    # 2d
    x, y = symbols('x, y')
    E = 2
    ax, bx = -1, 2.
    ay, by = 1, 2.
    domain = [[ax, bx], [ay, by]]
    u = sin(3*pi*(x-ax)/(bx-ax))*sin(pi*(y-ay)/(by-ay))
    problem2d = manufacture_poisson_2d(u=u, domain=domain, E=E)
    f = problem2d['f']
    U, basis = solve_sine_2d(f, MN=[3, 3], E=E, domain=domain, eps=__EPS__,
                             n_refs=-1)
    e = errornorm(u, (U, basis), domain=domain, norm_type='L2')
    assert abs(e) < 1E-14
