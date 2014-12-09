from poisson_solver import PoissonSolver
from sympy import legendre, symbols


def shen_basis_2():
    '''
    Shen basis that has functions that are zero on (-1, 1). Leads to dense
    matrices.
    '''
    x = symbols('x')
    i = 0
    while True:
        yield legendre(i+2, x)-(legendre(0, x)
                                if (i % 2) == 0 else legendre(1, x))
        i += 1


class Shen2PoissonSolver(PoissonSolver):
    'Poisson Solver using Shen2 basis.'
    # Just assign the generator. Everything else rely on parrent
    def __init__(self):
        PoissonSolver.__init__(self, shen_basis_2)

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import S
    from sympy.plotting import plot3d

    solver = Shen2PoissonSolver()

    x, y = symbols('x, y')
    f = S(1)
    uh = solver.solve(f, 5)

    plot3d(uh, (x, -1, 1), (y, -1, 1))
