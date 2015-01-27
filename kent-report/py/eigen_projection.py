from __future__ import division
from sympy.plotting import plot
import matplotlib.pyplot as plt
from sympy import Symbol, lambdify, sqrt, sin, exp, cos, pi
from sympy.mpmath import quad
from math import log as ln
import math
import sympy as sp
import numpy as np

x = Symbol('x')

def eigen_basis(n):
    'Yield first n eigenfunctions of laplacian over (-1, 1) with Dirichlet bcs'
    x = Symbol('x')
    k = 0
    while k < n:
        alpha = pi/2 + k*pi/2
        if k % 2 == 0:
            yield cos(alpha*x)
        else:
            yield sin(alpha*x)
        k += 1

# We check properties of the generalied Fourier series for Eigen
# Number of eigen functions to use
f = exp(2*x)*(x**2 - 1)
n_list = np.arange(2, 16, 2)
for n in n_list:
    basis = list(eigen_basis(n))
    # Function you want to expand, we shall conder only such function taht have
    # zeros on the boundary. 
    # Compute the expansion coefficients
    row = np.zeros(n)
    for i, v in enumerate(basis):
        integrand = lambdify(x, v*f)
        row[i] = quad(integrand, [-1, 1])
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

# Okay we have n-2 convergence
# Now we are only interested in the expansion coefficients of the hat functions
# First get them with FEniCS
from dolfin import FunctionSpace, Expression, IntervalMesh, assemble, inner,\
    dx, interpolate, Function, pi

# Need basis functions as expression

from sympy.printing.ccode import CCodePrinter

class DolfinCodePrinter(CCodePrinter):
    def __init__(self, settings={}):
        CCodePrinter.__init__(self)

    def _print_Pi(self, expr):
        return 'pi'


def dolfincode(expr, assign_to=None, **settings):
    # Print scalar expression
    dolfin_xs = sp.symbols('x[0] x[1] x[2]')
    xs = sp.symbols('x y z')

    for x, dolfin_x in zip(xs, dolfin_xs):
        expr = expr.subs(x, dolfin_x)
    return DolfinCodePrinter(settings).doprint(expr, assign_to)


def eigen_basis_expr(n):
    '''
    Return first n eigenfunctions of Laplacian over biunit interval with homog.
    Dirichlet bcs. at endpoints -1, 1. Functions of x.
    '''
    k = 0
    functions = []
    while k < n:
        alpha = pi/2 + k*pi/2
        if k % 2 == 0:
            f = Expression('cos(alpha*x[0])', alpha=alpha, degree=8)
        else:
            f = Expression('sin(alpha*x[0])', alpha=alpha, degree=8)
        functions.append(f)
        k += 1
    return functions


# We will consider fixed n now. Only interested in comparing FEniCS matrix with
# my matrix
n = 10

n_elements = 10
mesh = IntervalMesh(n_elements, -1, 1)
V = FunctionSpace(mesh, 'CG', 1)
v = Function(V)

# Derivatives of test function are represented here
mesh_fine = IntervalMesh(10000, -1, 1)
V_fine = FunctionSpace(mesh_fine, 'CG', 1)

# Matrix with expansion coefficients
P = np.zeros((V.dim()-2, n))

# Loop over interior test functions
for i in range(1, V.dim()-1):
    # Get the test function
    v_values = np.zeros(V.dim())
    v_values[i] = 1
    v.vector()[:] = v_values

    v_fine = interpolate(v, V_fine)
 
    # Now fill in the row
    for j, u in enumerate(eigen_basis_expr(n)):
        P[i-1, j] = assemble(inner(v_fine, u)*dx)  # Note the shift!

# The claim is that the matrix can be computed by formula which is really simple
# if homef mesh is used. 
P_ = np.zeros_like(P)
h = 2/(P_.shape[0]+1)
for i in range(P_.shape[0]):
    xi = -1 + (i+1)*h
    x_next = xi + h
    x_prev = xi - h
    for j in range(P_.shape[1]):
        dd_f = lambda x, j=j: math.cos((pi/2 + j*pi/2)*x)/(pi/2 + j*pi/2)**2\
                              if (j % 2) == 0 else \
                              math.sin((pi/2 + j*pi/2)*x)/(pi/2 + j*pi/2)**2

        val = 2*dd_f(xi)/h - (dd_f(x_next) + dd_f(x_prev))/h
        P_[i, j] = val

print np.max(np.abs(P-P_))
# The matrix is okay

# Finally I am interested in power spectrum of a hat function
#

# Use m eigenfunctions
m = 128
basis = eigen_basis(m)
row = np.zeros(m)

hat_left = 2*(x+0.5)
hat_right = -2*(x-0.5)

for i, b in enumerate(basis):
    left = quad(lambdify(x, b*hat_left), (-0.5, 0))
    right = quad(lambdify(x, b*hat_right), (0, 0.5))
    row[i] = left+right


# Let's plot the power spectrum
plt.figure()
power = np.sqrt(row**2)
power = np.ma.masked_less(power, 1e-15)
plt.loglog(power, '-*')
plt.xlabel('$n$')
plt.title('Power spectrum of hat')

plt.show()
