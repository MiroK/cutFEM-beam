from __future__ import division
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
from coupled_biharmonic import solve as solve_biharmonic
from coupled_laplace import solve as solve_laplace
from leg_coupled_laplace import solve as solve_laplace_leg
import numpy as np
import pickle

results_dir = './results'
solvers = {'laplace': solve_laplace,
           'biharmonic': solve_biharmonic,
           'laplace_leg': solve_laplace_leg}

operator = 'laplace_leg'
# Norm to consider with Schur
fractions = np.array([0., 1.])
# Norm in which C*Schur has constant eigenvalues
__NORM__ = str(0)


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
# Checkout LBB for different positions of the beam
# The beam position is given by A = A(s) = [0.5*s, 0],
# B = B(t) = [0.5 + 0.5*t, 1]. Both s, t are in [0, 1]
s = np.linspace(0, 1, 4)
t = np.linspace(0, 1, 4)

As = np.vstack([0.5*s, 0*np.ones_like(s)]).T
Bs = np.vstack([0.5*t + 0.5, np.ones_like(t)]).T

n_rows = len(s)
n_cols = len(t)
n_positions = n_rows*n_cols

# Explored beam positions
fig, axarr = plt.subplots(n_rows, n_cols)
print n_rows, n_cols, n_positions
for i, (A, B) in enumerate(product(As, Bs)):
    row = i // n_cols
    col = i % n_cols
    print '\t', i, row, col
    ax = axarr[row, col]
    ax.plot([A[0], B[0]], [A[1], B[1]], linewidth=3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig('%s/%s_positions.pdf' % (results_dir, operator))

# Plot the eigenvalues of fixed A, that is each n_cols, into tiled plot
# The number of rows and cols in this plots N_rows, N_cols
N_cols = 2
N_rows = 2
counter = 0
figs = []
eigen_data = {}
# To get dependency of beta on angle
abetas, angles = defaultdict(list), defaultdict(list)
row_abetas, row_angles = [], []
# To get dependency of beta on angle
lbetas, lens = defaultdict(list), defaultdict(list)
row_lbetas, row_lens = [], []

for i, (A, B) in enumerate(product(As, Bs)):
    # Switch to new figure with new row
    if i % n_cols == 0:
        # Reset
        row_abetas, row_angles = [], []
        row_lbetas, row_lens = [], []
        counter = 0
        fig, axarr = plt.subplots(N_rows, N_cols, sharex=True, sharey=True)
        print i, n_positions, i // n_cols
        fig.suptitle('Row %d' % (i//n_cols))
        figs.append(fig)

    # Decide position in 2x2 figure
    row = counter // N_cols
    col = counter % N_cols
    ax = axarr[row, col]
    ax.set_title('Column %d' % counter)

    # For each subwindow compute the convergence curve
    eigen_data[i] = defaultdict(list)
    eigs = eigen_data[i]

    ns = []
    params['A'] = A
    params['B'] = B
    for n in range(4, 6):
        n_plate, n_beam, n_lambda = rule(n)
        params['n_plate'] = n_plate
        params['n_beam'] = n_beam
        params['n_lambda'] = n_lambda
        eigenvalues = solvers[operator](params,
                                        eigs_only=True, fractions=fractions)

        for key in eigenvalues:
            eigs[key].append(eigenvalues[key][-1])

        ns.append(n)

        print '\t\t', [(key, values[-1])
                       for key, values in eigenvalues.iteritems()]

    eigen_lines = []
    for key in eigs:
        print 'plot', key, eigs[key]
        line, = ax.loglog(ns, eigs[key], label=key)
        eigen_lines.append(line)

    # In some __NORM__, the eigenvalues should be abount constant
    # Get the average constant
    beta = np.mean(eigs[__NORM__])
    row_abetas.append(beta)
    row_lbetas.append(beta)
    # This will be plotted against the angle and len
    length = np.hypot(*(A-B))
    row_lens.append(length)

    sin_phi = 1./length
    phi = np.arcsin(sin_phi)
    angle = np.degrees(phi)
    row_angles.append(angle)

    # Store the angle dependence from the row if we are switching to new on
    # in the next iteration
    if ((i+1) % n_cols) == 0:
        # Sort angles
        row_angles = np.array(row_angles)
        row_abetas = np.array(row_abetas)

        idx = np.argsort(row_angles)
        row_angles = row_angles[idx]
        row_abetas = row_abetas[idx]

        # Sort lengths
        row_lens = np.array(row_lens)
        row_lbetas = np.array(row_lbetas)

        idx = np.argsort(row_lens)
        row_lens = row_lens[idx]
        row_lbetas = row_lbetas[idx]

        this_row = i//n_cols

        angles[this_row].append(row_angles)
        abetas[this_row].append(row_abetas)

        lens[this_row].append(row_lens)
        lbetas[this_row].append(row_lbetas)

        # Put legend
        fig.legend(eigen_lines, eigs.keys(), 'lower right')

    counter += 1

# Remeber to save figures and data
# Eigenvalues
for i, fig in enumerate(figs):
    fig.savefig('%s/%s_row%d_%s.pdf' % (results_dir, operator, i, rule.name))

pickle.dump(eigen_data, open('%s/%s_eigv_data_%s.pickle' %
                             (results_dir, operator, rule.name), 'wb'))

# Betas as function of angle
fig, axarr = plt.subplots(len(abetas), 1, sharex=True)
fig.suptitle('Based on %s norm' % __NORM__)
for row in range(n_rows):
    ax = axarr[row]
    ax.plot(angles[row][0], abetas[row][0], '*-')
    ax.set_ylabel(r'$\beta$')
axarr[-1].set_xlabel(r'$\theta$ [deg]')
# Save
fig.savefig('%s/%s_betas_angles_%s.pdf' % (results_dir, operator, rule.name))
# Dump data
angle_data = {'angles': angles, 'betas': abetas}
pickle.dump(angle_data, open('%s/%s_angle_data_%s.pickle' %
                             (results_dir, operator, rule.name), 'wb'))

# Betas as function of length
fig, axarr = plt.subplots(len(lbetas), 1, sharex=True)
fig.suptitle('Based on %s norm' % __NORM__)
for row in range(n_rows):
    ax = axarr[row]
    ax.plot(lens[row][0], lbetas[row][0], '*-')
    ax.set_ylabel(r'$\beta$')
axarr[-1].set_xlabel(r'$L$')
# Save
fig.savefig('%s/%s_betas_lens_%s.pdf' % (results_dir, operator, rule.name))
# Dump data
len_data = {'lens': lens, 'betas': lbetas}
pickle.dump(len_data, open('%s/%s_len_data_%s.pickle' %
                           (results_dir, operator, rule.name), 'wb'))

plt.show()
