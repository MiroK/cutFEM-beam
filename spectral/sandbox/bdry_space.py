import sys
sys.path.insert(0, '../')

from points import gauss_legendre_points as gl_points
from functions import lagrange_basis
from sympy import symbols, lambdify, integrate
import numpy as np
import plots

m = 3
n = 2

x_points = gl_points([m])
y_points = gl_points([n])

x, y = symbols('x, y')

basis = [0.5*(1-y)*b for b in lagrange_basis([x_points])]
basis.extend([0.5*(1-x)*b for b in lagrange_basis([y_points], xi=1)])
basis.extend([0.5*(1+y)*b for b in lagrange_basis([x_points])])
basis.extend([0.5*(1+x)*b for b in lagrange_basis([y_points], xi=1)])


def bdry_integral(f):
    'Ingrate over the boundary of [-1, 1]^2'
    # x, fix y = -1
    i = f.subs({y: -1})
    i0 = integrate(i, (x, -1, 1))
    # x, fix y = 1
    i = f.subs({y: 1})
    i1 = integrate(i, (x, -1, 1))
    # fix x = -1, y
    i = f.subs({x: -1})
    i2 = integrate(i, (y, -1, 1))
    # fix x = 1, y
    i = f.subs({x: 1})
    i3 = integrate(i, (y, -1, 1))
    return i0 + i1 + i2 + i3

M = np.zeros((len(basis), len(basis)))
for i, bi in enumerate(basis):
    M[i, i] = bdry_integral(bi*bi)
    for j, bj in enumerate(basis[i+1:], i):
        M[i, j] = bdry_integral(bi*bj)
        M[j, i] = M[i, j]

np.set_printoptions(precision=2)
print np.linalg.cond(M)
print M
Minv = np.linalg.inv(M)

# for bi in basis:
#     plots.plot(bi, [[-1, 1], [-1, 1]])
#     print [bi.evalf(subs={x: c0, y: c1})
#            for c0, c1 in ((-1, -1), (-1, 1), (1, 1), (1, -1))]


points = [(-1, p) for p in y_points]
points.extend([(1, p) for p in y_points])
points.extend([(p, -1) for p in x_points])
points.extend([(p, 1) for p in x_points])

for i, b in enumerate(basis):
    for j, (X, Y) in enumerate(points):
        value = b.evalf(subs={x: X, y: Y})
        value = value if value > 1E-15 else 0
        M[i, j] = value

np.set_printoptions(precision=2)
print np.linalg.cond(M)
print M
Minv = np.linalg.inv(M)
