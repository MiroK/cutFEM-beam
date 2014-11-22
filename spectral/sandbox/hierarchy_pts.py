import sys
sys.path.insert(0, '../')

from points import gauss_legendre_points as gl_points
from points import chebyshev_points as cheb_points
import matplotlib.pyplot as plt
import numpy as np

def plot_hierarchy(N, points):
    plt.figure()
    for n in range(1, N):
        x = points([n])
        y = n*np.ones_like(x)
        plt.plot(x, y, 'o')
    plt.xlim([-1, 1])
    plt.ylim([0.9, N-0.9])

# ------------------------------------------------------------------------

if __name__ == '__main__':
    plot_hierarchy(20, gl_points)

    plot_hierarchy(20, cheb_points)
    plt.show()
