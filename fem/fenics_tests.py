import problem2d
from fenics_solvers import cg_solver
from math import log as ln
import matplotlib.pyplot as plt


def test_convergence(meshes, solver):
    'Run solver on series of meshes and compute the convergence rate.'
    hs = []
    es = []
    lmbdas = []
    for mesh in meshes:
        result = solver(mesh)
        hs.append(result['h'])
        es.append(result['L2'])
        lmbdas.append(result['a_max'])

    for i in range(1, len(hs)):
        h, e = hs[i], es[i]
        h_, e_ = hs[i-1], es[i-1]

        rate = ln(e/e_)/ln(h/h_)

        print 'h=%.4E error=%.4E rate=%.2f' % (h, e, rate)

    plt.figure()
    plt.loglog(hs, lmbdas)
    print ' '.join(map(lambda a: '%.2E' % a, lmbdas))

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import RectangleMesh
    from functools import partial
    # Specify the problem
    D, Lx, Ly = 1, 1, 0.5
    problem0 = problem2d.joris_problem(D, Lx, Ly)
    problem1 = problem2d.miro_problem(D, Lx, Ly)

    # Get the meshes
    meshes = [RectangleMesh(0, 0, Lx, Ly, 2**i, 2**i)
              for i in range(3, 8)]

    # Specialize the solver for the problem and polynomial order
    solver0 = partial(cg_solver, problem=problem0, p=2, verbose=False)
    solver1 = partial(cg_solver, problem=problem1, p=3, verbose=False)

    # Now run the test
    test_convergence(meshes, solver0)
    print '*'*79
    test_convergence(meshes, solver1)

    plt.show()
