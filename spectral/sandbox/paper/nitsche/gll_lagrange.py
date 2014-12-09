import sys
sys.path.insert(0, '../../../')

from poisson_solver import PoissonSolver
from points import gauss_legendre_lobatto_points as gll_points
from functions import lagrange_basis
from sympy import symbols
import random


class GLLLagrangeBasis(object):
    '''
    Hierarchical, that is each new polynomial has higher degree. Take gll
    points and and some lagrange polynomial that is nodal in the interior
    points.
    '''
    def __init__(self):
        self.cached_k = {}

    def __call__(self):
        '''This generator yields random Lagrange polynomials that are 0 at
        the boundary. Generator for object with always yield same polynomials.
        '''
        i = 3
        while True:
            points = gll_points([i])
            if i in self.cached_k:
                k = self.cached_k[i]
            else:
                k = random.choice(range(1, i-1))
                self.cached_k[i] = k

            yield lagrange_basis([points])[k]

            i += 1


class GLLLagrangePoissonSolver(PoissonSolver):
    'Poisson Solver using GLL based Lagrange basis.'
    # Just assign the generator. Everything else rely on parrent
    # TODO later redefine assemble_A and assemble_M
    def __init__(self):
        gll_lagrange_basis = GLLLagrangeBasis()
        PoissonSolver.__init__(self, gll_lagrange_basis)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import S
    from sympy.plotting import plot3d

    solver = GLLLagrangePoissonSolver()

    x, y = symbols('x, y')
    f = S(1)
    uh = solver.solve(f, 5)

    plot3d(uh, (x, -1, 1), (y, -1, 1))
