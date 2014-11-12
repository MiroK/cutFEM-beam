import sys
sys.path.insert(0, '../')

from points import gauss_legendre_lobatto_points as gll_points
from functions import lagrange_basis as l_basis
from quadrature import GLLQuadrature
from sympy import lambdify, symbols, integrate
import numpy.linalg as la
import numpy as np

x = symbols('x')
# If I N points have points
N = 5
points = gll_points([N])
assert len(points) == N
# This allows me to create N Lagrage polynomials whose degree is N-1
basis = l_basis([points])
basis_l = map(lambda f: lambdify(x, f), basis)
assert len(basis) == N
# If I make the mass matrix which combines these polynomials then the degree
# of the integrand is 2*(N-1) = 2*N - 2 <= 2*N - 1 which means that the inner
# product over [-1, 1] is computed exactly by N+1 point GLL quadrature
quad_n = GLLQuadrature(N)
quad_N = GLLQuadrature(N+1)
ip_n = quad_n.eval(lambda X: basis_l[0](X)*basis_l[1](X), domain=[[-1, 1]])
ip_N = quad_N.eval(lambda X: basis_l[0](X)*basis_l[1](X), domain=[[-1, 1]])
ip = integrate(basis[0]*basis[1], (x, -1, 1))
assert abs(ip - ip_N) < 1E-15
assert not abs(ip - ip_n) < 1E-15

# If I then make the mass matrix using same quadrature as was used to create
# the polynials then the mass matrix will be diagonal if N point rule is used
M = np.zeros((N, N))
for i, bi in enumerate(basis_l):
    for j, bj in enumerate(basis_l[i:], i):
        M[i, j] = quad_n.eval(lambda X: bi(X)*bj(X), domain=[[-1, 1]])
M -= np.diag(M.diagonal())
assert la.norm(M)/N**2 < 1E-15
# But I am underintegrating!
for i, bi in enumerate(basis):
    for j, bj in enumerate(basis[i:], i):
        M[i, j] = integrate(bi*bj, (x, -1, 1))
        M[i, j] -= quad_n.eval(lambda X: lambdify(x, bi)(X)*lambdify(x, bj)(X),
                               domain=[[-1, 1]])
        assert not(abs(M[i, j]) < 1E-11)
