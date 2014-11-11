# Let's have a look at function values of 0-th, first and second derivative
# of Lagrange polynomials in nodes

from points import gauss_legendre_lobatto_points as gll_points
from points import gauss_legendre_points as gl_points
from points import chebyshev_points as cheb_points
from functions import lagrange_basis
from sympy import lambdify, symbols
import matplotlib.pyplot as plt
import numpy as np

points_d = {'Gauss-Legendre-Lobbatto': gll_points,
            'Gauss-Legendre': gl_points,
            'Chebyshev': cheb_points}

# Decide the points
points_s = 'Gauss-Legendre-Lobbatto'
N = 4
points = points_d[points_s]([N])

# Generate lambdas for 0., 1., 2. derivative
x = symbols('x')
basis_symbolic = lagrange_basis([points])
basis = map(lambda f: lambdify(x, f), basis_symbolic)
d_basis = map(lambda f: lambdify(x, f.diff(x, 1)), basis_symbolic)
dd_basis = map(lambda f: lambdify(x, f.diff(x, 2)), basis_symbolic)

t = np.arange(-1, 1+1E-2, 1E-2)
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
ax1.set_title('$\phi, \phi_x, \phi_{xx}$ for %s polynomials of order %d' %
              (points_s, N-1))
print 'f values at -1, 1'
for b in basis:
    ax1.plot(t, np.array([b(ti) for ti in t]))
    print b(-1), b(1)
print '-'*79

print 'f` values at -1, 1'
for db in d_basis:
    ax2.plot(t, np.array([db(ti) for ti in t]))
    print db(-1), db(1)
print '-'*79

print 'f`` values at -1, 1'
for ddb in dd_basis:
    ax3.plot(t, np.array([ddb(ti) for ti in t]))
    print ddb(-1), ddb(1)
f.subplots_adjust(hspace=0.1)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.xlim([-1, 1])
plt.show()
