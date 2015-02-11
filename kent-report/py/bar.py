from __future__ import division
from shen_poisson import mass_matrix
import numpy as np

# We have  m > n, specifically n = m-1, and consider the n x m mass matrix w.r.t
# to the shen basis. The claim is that this matrix has a kernel of dimension 1.
# I want to find it here.

m = 6
n = m - 1
M = mass_matrix(m)

# Chop the n x m matrix
M = M[:n, :m]

# For m even a vector x in the kernel has even entries 0 and unless the vector 
# lies is S_m the only kernel vector is 0. In other words the vector in the 
# kernel of the rectangular matrix must have some nonzero part in S_m. For
# m odd, the odd entries of x will be 0 and the even ones are due to solution
# of the system
# The nenzero entries of the vector are given as a solution to the following
# system.

is_even = m % 2 == 0
# Extract the indices to build up the system
size = m//2 - 1 if is_even else m//2
mat_indices = {}
for i in range(size):
    row = 2*i + 1 if is_even else 2*i
    cols = filter(lambda value: 0 <= value < m, [row - 2, row, row + 2])
    mat_indices[row] = cols

assert size == len(mat_indices)

# Build the system
mat = np.zeros((size, size))
vec = np.zeros(size)
for mat_row, M_row in enumerate(sorted(mat_indices.keys())):
    offset = 0 if mat_row < 2 else mat_row - 1
    for mat_col, M_col in enumerate(mat_indices[M_row]):
        if mat_col + offset == size:
            vec[mat_row] = -M[M_row, M_col]
        else:
            mat[mat_row, mat_col+offset] = M[M_row, M_col]
# Solve for odd
vec = np.linalg.solve(mat, vec)

# Assign to x
x = np.zeros(m)
for i, index in enumerate(sorted(mat_indices.keys())):
    x[index] = vec[i]

# Dont forget we made a choice in the system to have the last guy 1
x[-1] = 1

# See if the M.x = 0 as it should
assert np.allclose(M.dot(x), np.zeros(n))

# Just out of curiosity, what does such a spurious mode look like
from shenp_basis import shenp_basis as shen_basis
from sympy.mpmath import quad
from sympy import Symbol, lambdify
from sympy.plotting import plot

basis = list(shen_basis(m))
spurious_mode = sum(coef*f for coef, f in zip(x, basis))

# We are in the seetting where rows are V space. Spurios mode must be OG
x = Symbol('x')
V = basis[:n]
assert all(abs(quad(lambdify(x, v*spurious_mode), [-1, 1])) < 1E-14
           for v in V)

# Plot!
plot(spurious_mode, (x, -1, 1))
