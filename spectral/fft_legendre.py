from __future__ import division
from sympy.mpmath import quad, legendre, factorial, gamma, sqrt, diff
from math import pi, sin, cos
from functools import partial
import scipy.linalg as la
import numpy as np

# I want integrals int_{-1, 1} f(x)*sin(k*pi*x) and f(x)*cos(k*pi*x)
# where f is n-th Legendre polynomial

def J(alpha, lmbda):
    ans = 0.
    for k in range(100):
        ans += (-1)**k*(0.5*lmbda)**(2*k + alpha)/factorial(k)/gamma(k+alpha+1)
    return ans


def series_coef_leg(n, k):
    '''
    Return k-th coefficient in the Fourier series of n-th Legendre polynomial.
    '''
    coef = (1j)**n*sqrt(2./k)*J(n+0.5, k*pi)
    return coef.real if n % 2 == 0 else coef.imag


def c(k):
    'Weights of Shen basis.'
    return sqrt(4*k + 6)**-1


def shen_basis(m):
    # We construct a new basis that has always zero endpoint values
    return [lambda x, i=i: c(i)*(partial(legendre, n=i)(x=x) -
                                 partial(legendre, n=i+2)(x=x))
            for i in range(m-1)]


def series_coef_shen(n, k):
    '''
    Return k-th coefficient in the Fourier series of n-th Shen's basis.
    '''
    coef_n = (1j)**n*sqrt(2./k)*J(n+0.5, k*pi)
    coef_n2 = (1j)**(n+2)*sqrt(2./k)*J((n+2)+0.5, k*pi)
    coef = c(n)*(coef_n - coef_n2)
    return coef.real if n % 2 == 0 else coef.imag

k_range = range(1, 30)
basis = shen_basis(20)
for n in []: #range(10):
    if True:
        # cos series
        if n % 2 == 0:
            exact_coeffs = [quad(lambda x, k=k:
                                 legendre(n, x)*cos(k*pi*x), [-1, 1])
                            for k in k_range]
        # sin series
        else:
            exact_coeffs = [quad(lambda x, k=k:
                                 legendre(n, x)*sin(k*pi*x), [-1, 1])
                            for k in k_range]

        if k_range[-1] < 10:
            numeric_coeffs = [series_coef_leg(n, k) for k in k_range]
    else:
        phi = basis[n]
        # cos series
        if n % 2 == 0:
            exact_coeffs = [quad(lambda x, k=k: phi(x)*cos(k*pi*x), [-1, 1])
                            for k in k_range]
        # sin series
        else:
            exact_coeffs = [quad(lambda x, k=k: phi(x)*sin(k*pi*x), [-1, 1])
                            for k in k_range]

        if k_range[-1] < 10:
            numeric_coeffs = [series_coef_shen(n, k) for k in k_range]

    # print exact_coeffs
    # print numeric_coeffs
    # print la.norm(np.array(exact_coeffs) - np.array(numeric_coeffs)),

    # print '.'
    # plt.figure()
    # plt.semilogy(np.array(exact_coeffs)**2)

# The conclusion here is that formula is okay for small k
# plt.show()

def mass_matrix(n):
    'Mass matrix assembled from Shen basis'
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = c(i)*c(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, n):
            M[i, j] = -c(i)*c(j)*(2./(2*j+1)) if i+2 == j else 0
            M[j, i] = M[i, j]
    return M

# Make up some u = sum U_j*shen_j
basis = shen_basis(6)
n = len(basis)
U = np.random.random(n)
def uh(x):
    return sum(Ui*v(x) for (Ui, v) in zip(U, basis))

# What is it's norm on [-1, 1] in H^0, H^1
norm_H0 = quad(lambda x: uh(x)**2, [-1, 1])
norm_H1 = quad(lambda x: diff(uh, x)**2, [-1, 1])

# These should be computeble by matrices
norm_H1_ = np.sum(U**2)
norm_H0_ = np.sum(U*(mass_matrix(n).dot(U)))

print norm_H0, norm_H0_
print norm_H1, norm_H1_

def f(i, k):
    if i % 2 == 0:
        return lambda x: cos(k*pi*x)
    else:
        return lambda x: sin(k*pi*x)

# Can we relate it to fourier
m = 15    # Number of terms to include in the sine, cosine series
F = np.zeros((n, m))
# for i in range(n):
#     v = basis[i]
#     for j in range(m):
#         F[i, j] = quad(lambda x: v(x)*f(i, j+1)(x), [-1, 1])

def H_s(s, m):
    M = mass_matrix(m)
    lmbda, V = la.eigh(M)
    return V.dot(np.diag(lmbda**(1-s)).dot(V.T))
