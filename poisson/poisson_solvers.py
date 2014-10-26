from solvers import VariationalSolver1d
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
            return E*np.diag([np.pi**2*l**2/(b-a) for l in range(1, N+1)])

        def basis_functions(self, N):
            return sine_basis(N)

    solver = PoissonSolver()
    return solver.solve(f, N, E=E, a=a, b=b, eps=eps, n_refs=n_refs)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from problems import manufacture_poisson_1d
    from quadrature import errornorm
    from sympy import sin, symbols, pi

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
