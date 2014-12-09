from __future__ import division
from poisson_solver import PoissonSolver
from sympy import symbols, sin, cos, pi


def fourier_basis():
    'Basis of eigenvectors of laplacian on (-1, 1) with homog dirichlet bcs.'
    x = symbols('x')
    i = 0
    while True:
        phi = (i+1)*pi/2
        if i % 2 == 0:
            yield cos(phi*x)
            i += 1
        else:
            yield sin(phi*x)
            i += 1


class EigenPoissonSolver(PoissonSolver):
    'Poisson Solver using fourier basis.'
    # Just assign the generator. Everything else rely on parrent
    # TODO later redefine assemble_A and assemble_M
    def __init__(self):
        PoissonSolver.__init__(self, fourier_basis)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import S
    from sympy.plotting import plot3d

    solver = EigenPoissonSolver()

    x, y = symbols('x, y')
    f = S(1)
    uh = solver.solve(f, 5)

    plot3d(uh, (x, -1, 1), (y, -1, 1))
