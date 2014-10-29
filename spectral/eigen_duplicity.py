from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
import matplotlib.pyplot as plt
from itertools import product
import numpy.linalg as la
import numpy as np


def eigenvalue_duplicity(dim, operator, N=20):
    'Count unique eigenvalues.'
    # index for each of d components in space
    index = np.array([np.arange(1, N+1) for i in range(dim)])
    # combine indices to tuples
    unique_eigenvalues = set([])
    for indices in product(*[i for i in index]):
        unique_eigenvalues.add(operator(indices))

    return len(unique_eigenvalues)


def laplacian(indices):
    'Eigenvalues of laplacian from tuple of indices.'
    return sum((k**2 for k in indices))


def biharmonic(indices):
    'Eigenvalues of biharmonic from tuple of indices.'
    return laplacian(indices)**2

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    op = biharmonic
    data = {}
    Ns = np.arange(1, 50, 1)
    for dim in range(1, 4):
        n_unique = []
        for N in Ns:
            unique = eigenvalue_duplicity(dim=dim, operator=op, N=N)
            n_unique.append(unique)

        data[dim] = n_unique

    # N dependency
    plt.subplot(2, 1, 1)
    AN = np.vstack([np.log(Ns), np.ones_like(Ns)]).T
    for dim in data:
        slope, _ = la.lstsq(AN, np.log(data[dim]))[0]

        plt.loglog(Ns, data[dim],
                   label='$\Delta_%d, %.2g$' % (dim, slope))

        plt.legend(loc='best')
        plt.xlabel(r'$N$')
        plt.ylabel(r'$n_{\lambda}$')

    # Space size dependency
    plt.subplot(2, 1, 2)
    for dim in data:
        AV = np.vstack([dim*np.log(Ns), np.ones_like(Ns)]).T
        slope, _ = la.lstsq(AV, np.log(data[dim]))[0]

        plt.loglog(Ns**dim, data[dim],
                   label='$\Delta_%d, %.2g$' % (dim, slope))

        plt.legend(loc='best')
        plt.xlabel(r'dim($V_h$)')
        plt.ylabel(r'$n_{\lambda}$')

    plt.show()
