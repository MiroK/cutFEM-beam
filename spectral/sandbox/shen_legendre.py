# Here we investigate properties of polynomials basis of Legendre polynomials

from sympy.mpmath import legendre, sqrt, quad, diff
import matplotlib.pyplot as plt
from functools import partial
import numpy.linalg as la
import numpy as np

# Take n legendre polynomials, these are polynials of degree 0, ..., n
n = 15

if False:
    basis = [lambda x, i=i: partial(legendre, n=i)(x=x) for i in range(n)]

    x_values = np.arange(-1, 1.1, 0.01)
    plt.figure()
    for i, f in enumerate(basis):
        y_values = map(f, x_values)
        plt.plot(x_values, y_values, label='$L^%d$' % i)
    plt.legend(loc='best')
    plt.xlim(-1, 1)

    # Check the endpoint values
    # print 'L_i(-1)', map(lambda f: f(-1), basis)
    # print 'L_i(1)', map(lambda f: f(1), basis)

if False:

    def c(k):
        return sqrt(4*k + 6)**-1

    # We construct a new basis that has always zero endpoint values
    basis = [lambda x, i=i: c(i)*(partial(legendre, n=i)(x=x) -
                                partial(legendre, n=i+2)(x=x))
            for i in range(n)]

    # The end point values should be zero
    assert all(map(lambda f: f(-1) == 0, basis))
    assert all(map(lambda f: f(1) == 0, basis))

    # This basis should yield 3-banded mass matrix
    np.set_printoptions(precision=2)
    n = len(basis)
    M = np.zeros((n, n))
    for i, bi in enumerate(basis):
        M[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
        for j, bj in enumerate(basis[i+1:], i+1):
            M[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
            M[j, i] = M[i, j]
    # With values
    M_ = np.zeros_like(M)
    for i in range(n):
        M_[i, i] = c(i)*c(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, n):
            M_[i, j] = -c(i)*c(j)*(2./(2*j+1)) if i+2 == j else 0
            M_[j, i] = M_[i, j]

    assert la.norm(M-M_)/n < 1E-14

    # Tha stiffness matrix should be identity
    A = np.zeros((n, n))
    d_basis = [lambda x, f=f: diff(f, x) for f in basis]
    for i, bi in enumerate(d_basis):
        A[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
        for j, bj in enumerate(d_basis[i+1:], i+1):
            A[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
            A[j, i] = A[i, j]

    assert la.norm(A-np.eye(n))/n < 1E-14

    C = np.zeros((n, n))
    for i, bi in enumerate(d_basis):
        for j, bj in enumerate(basis):
            C[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])

    C_ = np.zeros((n, n))
    for i, bi in enumerate(d_basis):
        for j, bj in enumerate(basis[i+1:], i+1):
            C_[i, j] = -2*c(i)*c(j) if j == i+1 else 0
            C_[j, i] = -C[i, j]

    assert la.norm(C-C_)/n < 1E-14

    # Not sparse
    D = np.zeros((n, n))
    dd_basis = [lambda x, f=f: diff(f, x, 2) for f in basis]
    for i, bi in enumerate(dd_basis):
        D[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
        for j, bj in enumerate(dd_basis[i+1:], i+1):
            D[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
            D[j, i] = D[i, j]


# New basis that has first and second derivative 0 and D is diagonal
def d(k):
    return sqrt(2*(2*k + 3)**2*(2*k + 5))**-1

# We construct a new basis that has always zero endpoint values
basis = [lambda x, i=i: d(i)*(partial(legendre, n=i)(x=x) -
                              (2.*(2*i+5)/(2*i+7))*partial(legendre, n=i+2)(x=x) +
                              ((2.*i+3)/(2*i+7))*partial(legendre, n=i+4)(x=x))
         for i in range(n)]

d_basis = [lambda x, f=f: diff(f, x) for f in basis]
dd_basis = [lambda x, f=f: diff(f, x, 2) for f in basis]

print 'L_i(-1)', map(lambda f: f(-1), basis)
print 'L_i(1)', map(lambda f: f(1), basis)
print 'L_i`(-1)', map(lambda f: f(-1), d_basis)
print 'L_i`(1)', map(lambda f: f(1), d_basis)
print 'L_i``(-1)', map(lambda f: f(-1), dd_basis)
print 'L_i``(1)', map(lambda f: f(1), dd_basis)


D = np.zeros((n, n))
for i, bi in enumerate(dd_basis):
    D[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
    for j, bj in enumerate(dd_basis[i+1:], i+1):
        D[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
        D[j, i] = D[i, j]
print D
