from solvers import VariationalSolver1d
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
            return E*np.diag([np.pi**4*l**4/(b-a)**3 for l in range(1, N+1)])

        def basis_functions(self, N):
            return sine_basis(N)

    solver = BiharmonicSolver()
    return solver.solve(f, N, E=E, a=a, b=b, eps=eps, n_refs=n_refs)


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
