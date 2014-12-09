import sys
sys.path.insert(0, '../../../')

from legendre import NitschePoissonSolver
from points import gauss_legendre_points as gl_points
from functions import lagrange_basis
from sympy import symbols, S
import random


class GLLagrangeBasis(object):
    '''
    Hierarchical, that is each new polynomial has higher degree
    '''
    def __init__(self):
        self.cached_k = {}

    def __call__(self):
        '''
        This generator yields random Lagrange polynomials that are nodal at
        GL points
        '''
        i = 1
        while True:
            if i == 0:
                yield S(1)
                i += 1
            elif i == 1:
                yield symbols('x')
                i += 1
            else:
                points = gl_points([i])
                if i in self.cached_k:
                    k = self.cached_k[i]
                else:
                    k = random.choice(range(i))
                    self.cached_k[i] = k

                yield lagrange_basis([points])[k]

                i += 1


class GLLagrangePoissonSolver(NitschePoissonSolver):
    'Poisson Solver using GL based Lagrange basis.'
    # Just assign the generator. Everything else rely on parrent
    def __init__(self):
        gl_lagrange_basis = GLLagrangeBasis()
        NitschePoissonSolver.__init__(self, gl_lagrange_basis)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import S
    from sympy.plotting import plot3d

    solver = GLLagrangePoissonSolver()

    x, y = symbols('x, y')
    f = S(1)
    uh = solver.solve(f, 7)

    plot3d(uh, (x, -1, 1), (y, -1, 1))
