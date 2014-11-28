from __future__ import division
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
from coupled_biharmonic import solve
import numpy as np
import pickle


class nRule(object):
    def __init__(self, name, plate_n, beam_n, lmbda_n):
        self.name = name
        self.plate = plate_n
        self.beam = beam_n
        self.lmbda = lmbda_n

    def __call__(self, n):
        return (self.plate(n), self.beam(n), self.lmbda(n))

# -----------------------------------------------------------------------------

# The rule that will define number of sines in the test
rule = nRule(name='all_equal',
             plate_n=lambda n: n,
             beam_n=lambda n: n,
             lmbda_n=lambda n: n)

f = 1
params = {'E_plate': 1.,
          'E_beam': 10.,
          'A': None,
          'B': None,
          'f': f,
          'n_plate': 10,
          'n_beam': 10,
          'n_lambda': 10}
# Checkout LBB
# Space for plate is n**2, beam and penalty have n
As = np.array([[0, 0],
               [1/6, 0],
               [2/6, 0],
               [0.5, 0]])

Bs = np.array([[0.5, 1],
               [4/6, 1],
               [5/6, 1],
               [1., 1]])

n_points = As.shape[0]

# Explored beam positions
# Row rotates faster here, store angles in radians
angles = []
fig, axarr = plt.subplots(n_points, n_points)
for i, (A, B) in enumerate(product(As, Bs)):
    row = i % n_points
    col = i // n_points
    ax = axarr[row, col]
    ax.plot([A[0], B[0]], [A[1], B[1]], linewidth=3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    sin_phi = 1./np.hypot(*(A-B))
    phi = np.arcsin(sin_phi)
    angle = np.degrees(phi)

    angles.append(angle)


fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig('biharmonic_positions.pdf')

# Plot the eigenvalues for each row problem as 2, 2 plot
m_points = n_points // 2
counter = 0
figs = []
eigen_data = {}
betas = []
for i, (A, B) in enumerate(product(As, Bs)):
    # Switch to new figure per column
    if i % n_points == 0:
        counter = 0
        fig, axarr = plt.subplots(m_points, m_points,
                                  sharex=True, sharey=True)
        fig.suptitle('Column %d' % (i//n_points))
        figs.append(fig)

    # Decife position in 2x2 figure
    row = counter // m_points
    col = counter % m_points
    ax = axarr[row, col]
    ax.set_title('Row %d' % counter)

    # For each subwindow compute the convergence curve
    eigen_data[i] = defaultdict(list)
    eigs = eigen_data[i]

    ns = []
    params['A'] = A
    params['B'] = B
    for n in range(3, 5):
        n_plate, n_beam, n_lambda = rule(n)
        params['n_plate'] = n_plate
        params['n_beam'] = n_beam
        params['n_lambda'] = n_lambda
        eigenvalues = solve(params, eigs_only=True)

        for key in eigenvalues:
            eigs[key].append(eigenvalues[key][-1])

        ns.append(n)

        print '\t\t', n, ' '.join(map(str, (eigenvalues[key][-1]
                                            for key in eigenvalues)))

        for key in eigs:
            ax.loglog(ns, eigs[key])

    # In biharmonic norm, the eigenvalues should be abount constant
    beta = np.mean(eigs['biharmonic'])
    betas.append(beta)

    counter += 1

# Remeber to save figures and data
for i, fig in enumerate(figs):
    fig.savefig('biharmonic_col%d_%s.pdf' % (i, rule.name))

pickle.dump(eigen_data, open('biharmonic_data_%s.pickle' % rule.name, 'wb'))

angles = np.array(angles)
betas = np.array(betas)
idx = np.argsort(betas)

plt.figure()
plt.plot(angles[idx], betas[idx], '*-')
plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'$\beta$')
plt.show()
