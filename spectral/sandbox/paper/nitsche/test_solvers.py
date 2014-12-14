from __future__ import division
import sys
sys.path.insert(0, '../../../')
# Problem
from problems import manufacture_poisson_2d
# Solvers
from poisson_solver import PoissonSolver
from basis_generators import __basis_d__
# Postprocessing
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
from sympy import symbols, lambdify, pi, sin, exp
from sympy.mpmath import quad
from math import sqrt, log as ln
from collections import defaultdict
import pickle
# mpi
from mpi4py import MPI
comm = MPI.COMM_WORLD
proc_number = comm.Get_rank()
n_processes = comm.Get_size()


# -----------------------------------------------------------------------------

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

basis = __basis_d__
keys = sorted(basis.keys())
n_keys = len(keys)
loc_size = n_keys//n_processes

my_b = proc_number*loc_size
my_end = my_b + loc_size if proc_number != (n_processes - 1) else n_keys
my_keys = keys[my_b:my_end]
print proc_number, my_keys

ns = range(1, 21)
all_results = {}
for key in my_keys:
    print proc_number, key
    all_results[key] = defaultdict(list)
    solver_results = all_results[key]

    solver = PoissonSolver(basis[key]())
    for n in ns:
        uh, data = solver.solve(f, n, monitor_cond=True)
        # plot3d(uh, (x, -1, 1), (y, -1, 1))

        error = (uh - u)**2
        e = quad(lambdify([x, y], error), [-1, 1], [-1, 1])
        if e > 0:
            e = sqrt(e)

        if n > ns[0]:
            cond_op = data['monitor_cond']['op']
            cond_A = data['monitor_cond']['A']
            cond_M = data['monitor_cond']['M']
            rate = ln(e/e_)/ln(n_/n)

            print '\tproc=%s, n=%d, conds(%.2E, %.2E, %.2E) e=%.2E rate=%.2f'\
                % (proc_number, n, cond_op, cond_A, cond_M, e, rate)

            solver_results['n'].append(n)
            solver_results['cond_A'].append(cond_A)
            solver_results['cond_M'].append(cond_M)
            solver_results['cond_op'].append(cond_op)
            solver_results['e'].append(e)
            solver_results['rate'].append(rate)

            # No point to run this forever
            if e < 1E-13:
                break

        e_ = e
        n_ = n

pickle.dump(all_results,
            open('results/results_H_%d.pickle' % proc_number, 'wb'))
