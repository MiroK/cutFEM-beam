from poisson_solver import PoissonSolver
from sympy import legendre, symbols, sqrt


def shen_basis_1():
    '''
    Shen basis that has functions that are zero on (-1, 1). Yields diagonal
    stiffness matrix and tridiag. mass matrix.
    '''
    x = symbols('x')
    i = 0
    while True:
        yield (legendre(i, x) - legendre(i+2, x))/sqrt(4*i + 6)
        i += 1


class Shen1PoissonSolver(PoissonSolver):
    'Poisson Solver using Shen1 basis.'
    # Just assign the generator. Everything else rely on parrent
    # TODO later redefine assemble_A and assemble_M
    def __init__(self):
        PoissonSolver.__init__(self, shen_basis_1)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import S
    from sympy.plotting import plot3d

    solver = Shen1PoissonSolver()

    x, y = symbols('x, y')
    f = S(1)
    uh = solver.solve(f, 5)

    plot3d(uh, (x, -1, 1), (y, -1, 1))
