from sympy.mpmath import chebyt, quad, sqrt, pi, diff
from functools import partial
import numpy.linalg as la
import numpy as np

m = 5
basis = [lambda x, i=i: partial(chebyt, n=i)(x=x)
         for i in range(m)]

# The basis of cheb polynomials yields diagonal mass matrix in (, )_w inner
# product with w(x) = 1/sqrt(1-x**2)
def w(x):
    return sqrt(1-x**2)**-1

# M = np.zeros((m, m))
# for i, bi in enumerate(basis):
#     M[i, i] = quad(lambda x: bi(x)*bi(x)*w(x), [-1, 1])
#     for j, bj in enumerate(basis[i+1:], i+1):
#         M[i, j] = quad(lambda x: bi(x)*bj(x)*w(x), [-1, 1])
#         M[j, i] = M[i, j]
# print M

# The boundary values at -1, 1 are not 0
# print 'T_n(-1)', [f(-1) for f in basis]
# print 'T_n(1)', [f(1) for f in basis]

# Shen basis has bdry values 0
shen_basis = [lambda x, i=i: partial(chebyt, n=i)(x=x) - partial(chebyt, n=i+2)(x=x)
              for i in range(m)]

# print 'S_n(-1)', [f(-1) for f in shen_basis]
# print 'S_n(1)', [f(1) for f in shen_basis]

Mshen = np.zeros((m, m))
for i, bi in enumerate(shen_basis):
    Mshen[i, i] = quad(lambda x: bi(x)*bi(x)*w(x), [-1, 1])
    for j, bj in enumerate(shen_basis[i+1:], i+1):
        Mshen[i, j] = quad(lambda x: bi(x)*bj(x)*w(x), [-1, 1])
        Mshen[j, i] = Mshen[i, j]

# The exact should be
def c(i):
    return 2 if i == 0 else 1

Mshen_ = np.zeros((m, m))
for i in range(m):
    Mshen_[i, i] = pi*(c(i) + 1)/2.
    if i-2 > -1:
        Mshen_[i, i-2] = -pi/2
    if i+2 < m:
        Mshen_[i, i+2] = -pi/2
print la.norm(Mshen - Mshen_)

# For the stifness matrix of the poisson problem use -(u``, v). Par partes
# is `formal`
dd_shen_basis = [lambda x, f=f: diff(f, x, 2) for f in shen_basis]

np.set_printoptions(precision=2)

Ashen = np.zeros((m, m))
for i, bi in enumerate(shen_basis):
    for j, dd_bj in enumerate(dd_shen_basis):
        Ashen[i, j] = -quad(lambda x: bi(x)*dd_bj(x)*w(x), [-1, 1])

Ashen_ = np.zeros((m, m))
for i in range(m):
    Ashen_[i, i] = 2*pi*(i+1)*(i+2)
    for j in range(i+2, m, 2):
        Ashen_[i, j] = 4*pi*(i+1)

print la.norm(Ashen - Ashen_)

