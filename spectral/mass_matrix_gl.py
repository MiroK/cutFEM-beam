from polynomials import gauss_legendre_points as gl_points,\
    lagrange_basis as l_basis
from quadrature import GLQuadrature
from sympy import lambdify, symbols
import numpy.linalg as la
import numpy as np

x = symbols('x')
# If I N points have points
N = 20
points = gl_points([N])
assert len(points) == N
# This allows me to create N Lagrage polynomials whose degree is N-1
basis = l_basis([points])
basis = map(lambda f: lambdify(x, f), basis)
assert len(basis) == N
# If I make the mass matrix which combines these polynomials then the degree
# of the integrand is 2*(N-1) = 2*N - 2 <= 2*N - 1 which means that the inner
# product over [-1, 1] is computed exactly by N point quadrature
quad_N = GLQuadrature(N)
quad_2N = GLQuadrature(2*N)
ip_N = quad_N.eval(lambda X: basis[0](X)*basis[1](X), domain=[[-1, 1]])
ip_2N = quad_2N.eval(lambda X: basis[0](X)*basis[1](X), domain=[[-1, 1]])
assert abs(ip_N - ip_2N) < 1E-15
# If I then make the mass matrix using same quadrature as was used to create
# the polynials then the mass matrix will be diagonal
M = np.zeros((N, N))
for i, bi in enumerate(basis):
    for j, bj in enumerate(basis[i:], i):
        M[i, j] = quad_N.eval(lambda X: bi(X)*bj(X), domain=[[-1, 1]])
M -= np.diag(M.diagonal())
assert la.norm(M)/N**2 < 1E-15
