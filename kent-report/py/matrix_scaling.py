from eigen_poisson import mass_matrix as Meig1d
from eigen_poisson import laplacian_matrix as Aeig1d
from shen_poisson import mass_matrix as Mshen1d
from shen_poisson import laplacian_matrix as Ashen1d
import numpy as np
from numpy.linalg import cond

# Build 2d matrices
def Meig2d(n):
    '2d mass matrix w.r.t Eigen basis.'
    M = Meig1d(n)
    return np.kron(M, M)


def Aeig2d(n):
    '2d stiffness matrix w.r.t Eigen basis.'
    A = Aeig1d(n)
    M = Meig1d(n)
    return np.kron(A, M) + np.kron(M, A)


def Mshen2d(n):
    '2d mass matrix w.r.t Shen basis.'
    M = Mshen1d(n)
    return np.kron(M, M)


def Ashen2d(n):
    '2d stiffness matrix w.r.t Shen basis.'
    A = Ashen1d(n)
    M = Mshen1d(n)
    return np.kron(A, M) + np.kron(M, A)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt


    ns = np.arange(2, 20)
    
    # Plot 1d 
    if False:
        condMe = np.array([cond(Meig1d(n)) for n in ns])
        condAe = np.array([cond(Aeig1d(n)) for n in ns])
        condMs = np.array([cond(Mshen1d(n)) for n in ns])
        condAs = np.array([cond(Ashen1d(n)) for n in ns])
    else:
        condMe = np.array([cond(Meig2d(n)) for n in ns])
        condAe = np.array([cond(Aeig2d(n)) for n in ns])
        condMs = np.array([cond(Mshen2d(n)) for n in ns])
        condAs = np.array([cond(Ashen2d(n)) for n in ns])

    # Common marker == commmon basis, Common color == common matrix
    plt.figure()
    plt.loglog(ns, condMe, label='$M_E$', color='b', marker='s', linestyle='--')
    plt.loglog(ns, condAe, label='$A_E$', color='g', marker='s', linestyle='--')
    plt.loglog(ns, condMs, label='$A_S$', color='b', marker='o', linestyle='--')
    plt.loglog(ns, condAs, label='$A_S$', color='g', marker='o', linestyle='--')
    plt.legend(loc='best')
    plt.xlabel('$n$')
    plt.ylabel('$\kappa$')
    plt.show()
