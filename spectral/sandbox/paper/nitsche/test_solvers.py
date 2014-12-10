from __future__ import division
import sys
sys.path.insert(0, '../../../')
# Problem
from problems import manufacture_poisson_2d
# Solvers
from shen1 import Shen1PoissonSolver
from shen2 import Shen2PoissonSolver
from eigen import EigenPoissonSolver
from gll_lagrange import GLLLagrangePoissonSolver
from legendre import LegendrePoissonSolver
from gl_lagrange import GLLagrangePoissonSolver

# Postprocessing
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, pi, sin, exp
from sympy.plotting import plot3d
from sympy.mpmath import quad
from math import sqrt, log as ln

# Specs for problem
x, y = symbols('x, y')
u = exp(x-y)*(x**2-1)*sin(pi*(x+y))*(y**4-1)
domain = [[-1, 1], [-1, 1]]
E = 1

# Generate problem from specs
problem = manufacture_poisson_2d(u=u, E=E, domain=domain)
u = problem['u']
f = problem['f']

# plot3d(u, (x, -1, 1), (y, -1, 1))

solvers = {'shen1': Shen1PoissonSolver(),
           'shen2': Shen2PoissonSolver(),
           'eigen': EigenPoissonSolver(),
           'leg_nitsche': LegendrePoissonSolver(),
           # 'gl_nitsche': GLLagrangePoissonSolver(),
           'gll': GLLLagrangePoissonSolver()
           }


ns = range(2, 16)
#plt.figure()

for key in solvers:
    print key
    solver = solvers[key]
    es = []
    for n in ns:
        uh, data = solver.solve(f, n, monitor_cond=True)
        # plot3d(uh, (x, -1, 1), (y, -1, 1))

        error = (uh - u)**2
        e = quad(lambdify([x, y], error), [-1, 1], [-1, 1])
        if e > 0:
            e = sqrt(e)

        if n > 2:
            cond = data['monitor_cond']
            rate = ln(e/e_)/ln(n_/n)
            print '\tn=%d error=%.2E rate=%.2f cond=%.2E' % (n, e, rate, cond)
        #    es.append(e)

        e_ = e
        n_ = n
#    plt.loglog(n, es, label=key)

#plt.legend(loc='best')
#plt.show()
