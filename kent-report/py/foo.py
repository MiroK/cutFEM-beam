from __future__ import division
import numpy as np
import numpy.linalg as la
from scipy.linalg import toeplitz

def Mfem_matrix(n):
    'Mass matrix of H10 FEM.'
    h = 2/(n+1)
    row = np.zeros(n)
    row[0] = 4
    row[1] = 1
    M = toeplitz(row)
    M *= h/6.
    return M


for n in [2**i for i in range(1, 8)]:
    M = Mfem_matrix(n)

    print M.shape, la.cond(M),
    eigvals = la.eigvals(M)
    eigvals = np.sort(eigvals)
    print eigvals[0], eigvals[-1], eigvals[-1]/eigvals[0]

