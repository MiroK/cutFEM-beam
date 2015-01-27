from __future__ import division
from sympy.plotting import plot
import matplotlib.pyplot as plt
from sympy import legendre, Symbol, lambdify, sqrt, sin, exp
from sympy.mpmath import quad
from math import log as ln
import sympy as sp
import numpy as np

x = Symbol('x')

def shenp_basis(n):
    '''
    Yield first n basis function due to Shen - combinations of Legendre
    polynomials that have zeros at -1, 1 and yield sparse mass and stiffness
    matrices.
    '''
    x = Symbol('x')
    k = 0
    while k < n:
        weight = 1/sqrt(4*k + 6)
        yield weight*(legendre(k+2, x) - legendre(k, x))
        k += 1


def Mshen_matrix(m):
    'Mass matrix w.r.t to Shen basis.'
    weight = lambda k: 1./sqrt(4*k + 6)
    M = np.zeros((m, m))
    for i in range(m):
        M[i, i] = weight(i)*weight(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, m):
            M[i, j] = -weight(i)*weight(j)*(2./(2*j+1)) if i+2 == j else 0
            M[j, i] = M[i, j]

    return M

# We check properties of the generalied Fourier series for Shen
f = exp(2*x)*(x**2 - 1)
# Number of shen functions to use
for n in range(2, 11):
    basis = list(shenp_basis(n))
    # Function you want to expand, we shall conder only such function taht have
    # zeros on the boundary. 
    # Compute the expansion coefficients
    vec = np.zeros(n)
    for i, v in enumerate(basis):
        integrand = lambdify(x, v*f)
        vec[i] = quad(integrand, [-1, 1])
    mat = Mshen_matrix(n)
    # Row has the expansion coefficients
    row = np.linalg.solve(mat, vec)

    # Assemble the approximation
    f_ = sum(coef*v for (coef, v) in zip(row, basis))

    # Compute the L2 error of aproximation
    e = lambdify(x, (f-f_)**2)
    error = sqrt(quad(e, (-1, 1)))

    if n != 2:
        print n, error, ln(error/error_)/ln(n_/n)
    
    n_ = n
    error_ = error

# Let's plot the power spectrum
plt.figure()
power = np.sqrt(row**2)
power = np.ma.masked_less(power, 1E-15)
plt.loglog(power, 'd-')
plt.xlabel('$n$')
plt.title('Power spectrum of $f=%s$' % sp.latex(f)) 

# What happens with a hat function
n = 30
basis = list(shenp_basis(n))
mat = Mshen_matrix(n)
# Compute the rhs vector of hat
hat_left = 2*(x+0.5)
hat_right = -2*(x-0.5)
vec = np.zeros(n)
for i, v in enumerate(basis):
    left = quad(lambdify(x, v*hat_left), (-0.5, 0))
    right = quad(lambdify(x, v*hat_right), (0, 0.5))
    vec[i] = left+right
# Solve for the coefficients
row = np.linalg.solve(mat, vec)

# Plot the spectrum
plt.figure()
power = np.sqrt(row**2)
# power = np.ma.masked_less(power, 1E-15)
plt.loglog(power, 'd-')
plt.xlabel('$n$')
plt.title('Power spectrum of hat function') 
plt.show()
