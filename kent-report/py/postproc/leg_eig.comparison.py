from __future__ import division
# Put py on path
import sys
sys.path.append('../')
# Now for the important stuff
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import eigen_poisson as eigen
import shen_poisson as shen
import numpy.linalg as la

def eig_1d():
    'Plot rate, cond vs n eigen 1d.'
    # Rate data
    data = open('results/eig_p_1d', 'r').readlines()
    ns, L2, H1 = [], [], []
    for line in data[2:-1]:
        line = line.strip().split('&')
        ns.append(int(line[0]))
        L2.append(float(line[1]))
        H1.append(float(line[4]))

    # Condition number, we only need stiffness matrix
    conds = [la.cond(eigen.laplacian_matrix(n)) for n in ns]

    # Plotting
    # Left has n vs norms
    fig, ax1 = plt.subplots()
    ln1 = ax1.loglog(ns, L2, label=r'$L^2$',
                     linestyle='--', marker='o', color='r')
    ln2 = ax1.loglog(ns, H1, label=r'$H^1$',
                     linestyle='--', marker='x', color='b')
    ax1.set_xlabel('$n$')
    ax1.set_ylabel('$\|e\|$')
    # Right has n vs cond of the operator
    ax2 = ax1.twinx()
    ln3 = ax2.loglog(ns, conds, label=r'cond($\mathcal{A}$)',
                     linestyle='--', marker='d', color='g')
    ax2.set_ylabel(r'cond($A$)', color='g')
    for tl in ax2.get_yticklabels():
        tl.set_color('g')

    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    plt.show()


def eig_2d():
    'Plot rate, cond vs n eigen 2d.'
    # Rate data
    data = open('results/eig_p_2d', 'r').readlines()
    ns, L2, H1 = [], [], []
    for line in data[:-2]:
        line = line.strip().split(' ')
        ns.append(int(line[0]))
        L2.append(float(line[1]))
        H1.append(float(line[4]))

    # Condition number, we need global operator
    def operator(n):
        A = eigen.laplacian_matrix(n)
        M = eigen.mass_matrix(n)
        return np.kron(A, M) + np.kron(M, A)

    conds = [la.cond(operator(n)) for n in ns]

    # Plotting
    # Left has n vs norms
    fig, ax1 = plt.subplots()
    ln1 = ax1.loglog(ns, L2, label=r'$L^2$',
                     linestyle='--', marker='o', color='r')
    ln2 = ax1.loglog(ns, H1, label=r'$H^1$',
                     linestyle='--', marker='x', color='b')
    ax1.set_xlabel('$n$')
    ax1.set_ylabel('$\|e\|$')
    # Right has n vs cond of the operator
    ax2 = ax1.twinx()
    ln3 = ax2.loglog(ns, conds, label=r'cond($\mathcal{A}$)',
                     linestyle='--', marker='d', color='g')
    ax2.set_ylabel(r'cond($A$)', color='g')
    for tl in ax2.get_yticklabels():
        tl.set_color('g')

    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    plt.show()


def shen_1d():
    'Plot rate, cond vs n sehn 1d.'
    data = open('results/shen_p_1d', 'r').readlines()
    ns, L2, H1 = [], [], []
    for line in data[2:-1]:
        line = line.strip().split('&')
        ns.append(int(line[0]))
        L2.append(float(line[1]))
        H1.append(float(line[4]))

    # Condition number, we only need stiffness matrix
    conds = [la.cond(shen.laplacian_matrix(n)) for n in ns]

    # Plotting
    # Left has n vs norms
    fig, ax1 = plt.subplots()
    ln1 = ax1.loglog(ns, L2, label=r'$L^2$',
                     linestyle='--', marker='o', color='r')
    ln2 = ax1.loglog(ns, H1, label=r'$H^1$',
                     linestyle='--', marker='x', color='b')
    ax1.set_xlabel('$n$')
    ax1.set_ylabel('$\|e\|$')
    # Right has n vs cond of the operator
    ax2 = ax1.twinx()
    ln3 = ax2.loglog(ns, conds, label=r'cond($\mathcal{A}$)',
                     linestyle='--', marker='d', color='g')
    ax2.set_ylabel(r'cond($A$)', color='g')
    for tl in ax2.get_yticklabels():
        tl.set_color('g')

    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    plt.show()


def shen_2d():
    'Plot rate, cond vs n shen 1d.'
    # Rate data
    data = open('results/shen_p_2d', 'r').readlines()
    ns, L2, H1 = [], [], []
    for line in data[:-2]:
        line = line.strip().split(' ')
        ns.append(int(line[0]))
        L2.append(float(line[1]))
        H1.append(float(line[4]))

    # Condition number, we need global operator
    def operator(n):
        A = shen.laplacian_matrix(n)
        M = shen.mass_matrix(n)
        return np.kron(A, M) + np.kron(M, A)

    conds = [la.cond(operator(n)) for n in ns]

    # Plotting
    # Left has n vs norms
    fig, ax1 = plt.subplots()
    ln1 = ax1.loglog(ns, L2, label=r'$L^2$',
                     linestyle='--', marker='o', color='r')
    ln2 = ax1.loglog(ns, H1, label=r'$H^1$',
                     linestyle='--', marker='x', color='b')
    ax1.set_xlabel('$n$')
    ax1.set_ylabel('$\|e\|$')
    # Right has n vs cond of the operator
    ax2 = ax1.twinx()
    ln3 = ax2.loglog(ns, conds, label=r'cond($\mathcal{A}$)',
                     linestyle='--', marker='d', color='g')
    ax2.set_ylabel(r'cond($A$)', color='g')
    for tl in ax2.get_yticklabels():
        tl.set_color('g')

    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    plt.show()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # eig_1d()
    # eig_2d()
    # shen_1d()
    shen_2d()
