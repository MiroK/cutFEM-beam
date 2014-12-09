import numpy as np
from sympy.mpmath import legendre, quad
from functools import partial
import scipy.linalg as la
from math import sqrt
import matplotlib.pyplot as plt
from collections import defaultdict
from numpy.linalg import cond

def c(k):
    'Shen weights'
    return sqrt(4*k + 6)**-1


def shen_basis():
    'Basis in from Legendre polynomials that have 0 at [-1, 1]'
    i = 0
    while True:
        yield lambda x: c(i)*(partial(legendre, n=i)(x=x) -
                              partial(legendre, n=i+2)(x=x))
        i += 1


def mass_matrix(n):
    'Mass matrix assembled from Shen basis'
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = c(i)*c(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, n):
            M[i, j] = -c(i)*c(j)*(2./(2*j+1)) if i+2 == j else 0
            M[j, i] = M[i, j]
    return M


def stiffness_matrix(n):
    'Stiffness matrix of the Shen basis'
    return np.eye(n)


def schur_complement(n, beam):
    '''
    Plate [-1, -1]**2 with beam = [P, Q] of same material. Both governed by
    laplace equation.
    '''
    P, Q = beam
    L = np.hypot(*(P-Q))

    # Map s_hat \in [-1, 1] to coordinates of beam
    def beam_map(s_hat):
        return 0.5*(Q-P)*s_hat + 0.5*(Q+P)

    # Identity in Shen
    A_ = mass_matrix(n)
    # 3-diagonal
    M_ = stiffness_matrix(n)
    # Poisson for plate becomes AxM + MxA
    A0 = np.kron(A_, M_)
    A0 += np.kron(M_, A_)
    # Beam Poisson is stiffness
    A1 = A_
    # Constraint for plate deflection must be assembled
    B0 = np.zeros((n**2, n))
    row = 0
    for i, plate_i in zip(range(n), shen_basis()):
        for j, plate_j in zip(range(n), shen_basis()):
            # Basis function over plate
            plate = lambda x_hat, y_hat: plate_i(x_hat)*plate_j(y_hat)

            for col, beam_k in zip(range(n), shen_basis()):
                value = quad(lambda s_hat:
                             plate(*beam_map(s_hat))*beam_k(s_hat),
                             [-1, 1])
                B0[row, col] = value

            row += 1
    B0 *= -0.5*L
    # Constraint on beam deflections is mass
    B1 = 0.5*L*M_

    A = np.zeros((n**2+n, n**2+n))
    A[:n**2, :n**2] = A0
    A[n**2:, n**2:] = A1

    B = np.zeros((n**2+n, n))
    B[:n**2, :] = B0
    B[n**2:, :] = B1

    S = (B.T).dot(la.inv(A).dot(B))

    return S

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    n_max = 10
    ss = np.array([0, 0.5, 1])
    ns = np.arange(2, n_max, 1)
    beam = np.array([[-0.5, -1],
                     [0.5, 1]])

    cond_numbers = defaultdict(list)
    for s in ss:
        exponent = 1 - s

        for i, n in enumerate(ns):
            M = mass_matrix(n)

            lmbda, V = la.eigh(M)
            H = (V.dot(np.diag(lmbda**exponent))).dot(V.T)

            cond_number = cond(H)
            cond_numbers[s].append(cond_number)

    for n in ns:
        cond_numbers['schur'].append(cond(schur_complement(n, beam)))

    plt.figure()
    for label, values in cond_numbers.items():
        plt.loglog(ns, values, label=label)
    plt.legend(loc='best')
    plt.show()
