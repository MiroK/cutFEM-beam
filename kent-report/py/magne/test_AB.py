from blocks import solve_problem as block_solve
from no_hole import solve_problem as solve
import numpy as np

p = 1
c1 = 1.
c2 = 2.
constraint = 'H1'

# Test if block solver and Magne's solver give the same solutions and
# same eigenvalues
print p, c1, c2, constraint
for L in range(3, 6):
    print L

    u, eigs = solve(L, p, c1, c2, True, False, constraint)
    u_block, eigs_block = block_solve(L, p, c1, c2, True, False, constraint)

    for ui, vi in zip(u_block, u):
       ui.vector().axpy(-1, vi.vector())
       print '\tsolution diff', ui.vector().norm('l2')

    eigs = np.abs(eigs)
    lmin, lmax = np.min(eigs), np.max(eigs)

    eigs_block = np.abs(eigs_block)
    blmin, blmax = np.min(eigs_block), np.max(eigs_block)

    print '\t lambda_min diff', abs(lmin - blmin)
    print '\t lambda_max diff', abs(lmax - blmax), lmax, blmax
