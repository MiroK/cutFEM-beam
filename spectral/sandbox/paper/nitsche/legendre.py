from poisson_solver import PoissonSolver
from sympy import legendre, symbols, lambdify
import numpy as np


class NitschePoissonSolver(PoissonSolver):
    'Nitsche solvers have special stiffness matrices'

    def __init__(self, basis):
        'Set the generator'
        PoissonSolver.__init__(self, basis)

    def assemble_A_matrix(self, n):
        'The stiffness matrix includes boundary integrals.'
        # The volume term
        A = PoissonSolver.assemble_A_matrix(self, n)

        x = symbols('x')
        basis_ = self.list_basis(n)
        # Need lambdified function and the derivatives
        basis = map(lambda f: lambdify(x, f), basis_)
        dbasis = map(lambda f: lambdify(x, f.diff(x, 1)), basis_)

        # The gradient boundary term
        C = np.zeros_like(A)
        for i, dbi in enumerate(dbasis):
            for j, bj in enumerate(basis):
                C[i, j] = dbi(1)*bj(1) - dbi(-1)*bj(-1)

        # Penalty term
        D = np.zeros_like(A)
        for i, bi in enumerate(basis):
            D[i, i] = bi(1)**2 + bi(-1)**2
            for j, bj in enumerate(basis[i+1:], i+1):
                D[i, j] = bi(1)*bj(1) + bi(-1)*bj(-1)
                D[j, i] = D[i, j]
        D *= 10*n

        # The mass matrix is
        A = A - C - C.T + D
        return A


def legendre_basis():
    'Basis of Legendre polynomials'
    x = symbols('x')
    i = 0
    while True:
        yield legendre(i, x)
        i += 1


class LegendrePoissonSolver(NitschePoissonSolver):
    'Poisson Solver using Legendre basis.'
    # Just assign the generator. Everything else rely on parrent
    # TODO later redefine assemble_A and assemble_M
    def __init__(self):
        NitschePoissonSolver.__init__(self, legendre_basis)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import S
    from sympy.plotting import plot3d

    solver = LegendrePoissonSolver()

    x, y = symbols('x, y')
    f = S(1)
    uh = solver.solve(f, 7)

    plot3d(uh, (x, -1, 1), (y, -1, 1))
