from __future__ import division
import numpy as np
import numpy.linalg as la
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

def Mfem_matrix(n):
    'Mass matrix of H10 FEM.'
    h = 2/(n+1)
    row = np.zeros(n)
    row[0] = 4
    row[1] = 1
    M = toeplitz(row)
    M *= h/6.
    return M

def plot_vec(vec, i):
    values = np.zeros(len(vec)+2)
    values[1:len(vec)+1] = vec
    plt.plot(np.linspace(-1, 1, len(values)), values, label=str(i))

n = 20  # Number of functions
vec = np.random.rand(n)
for i in range(10):
    vec += np.random.rand(n)
    vec -= np.random.rand(n)
vec /= la.norm(vec)

M = Mfem_matrix(n)

print M

plt.figure()
for i in range(200):
    plot_vec(vec, i)
    vec = M.dot(vec)
    vec /= la.norm(vec)
    print vec
# plt.legend(loc='best')
plt.show()

