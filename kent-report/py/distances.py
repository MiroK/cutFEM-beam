from __future__ import division
from math import pi, sqrt
import numpy as np

def distance_matrix(n, entry):
    'Distance matrix with entries |phi_i - phi_j| for i, j in [0, n).'
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            Z[i, j] = entry(i, j)
            Z[j, i] = Z[i, j]
    return Z

def eigen_L2(n):
    'L2 distance of Laplacian eigenfunctions.'
    return distance_matrix(n, entry=lambda i, j: sqrt(2))

def eigen_H1(n):
    'H1 distance of Laplacian eigenfunctions.'
    eig = lambda k: (pi/2 + k*pi/2)**2
    return distance_matrix(n, entry=lambda i, j: sqrt(eig(i) + eig(j)))

def shen_L2(n):
    'L2 distance of Shen functions.'
    return distance_matrix(n, entry=lambda i, j: sqrt(
        2/(2*i + 1) + 2/(2*j + 1) + 2/(2*(i+2)+1) + 2/(2*(j+2)+1) +\
                (4/(2*(i+2)+1) if j == i+2 else 0)))

def shen_H1(n):
    'H1 distance of Shen functions.'
    return distance_matrix(n, entry=lambda i, j: sqrt(4*(i+j)+12))

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Z = shen_H1(100)
    Z = np.ma.masked_where(abs(Z) < 1E-10, Z)
    # U, V = np.gradient(Z)

    plt.figure()
    plt.pcolor(Z, snap=True)
    plt.colorbar()
    
    # plt.figure()
    # plt.quiver(U, V)

    plt.figure()
    plt.contour(Z, levels=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    #plt.contour(Z, levels=[100, 101, 102, 103])

    ns = np.arange(2, 100)
    distances = np.array([np.min(Z[n, :n]) for n in ns])
    
    plt.figure()
    plt.plot(ns, distances, label='distance')

    plt.show()
