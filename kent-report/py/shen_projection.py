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

# We check properties of the generalied Fourier series for Shen
# Number of shen functions to use
for n in range(2, 11):
    basis = list(shenp_basis(n))
    dbasis = [v.diff(x, 1) for v in basis]
    # Function you want to expand, we shall conder only such function taht have
    # zeros on the boundary. 
    f = exp(2*x)*(x**2 - 1)
    df = f.diff(x, 1)
    # Compute the expansion coefficients
    row = np.zeros(n)
    for i, dv in enumerate(dbasis):
        integrand = lambdify(x, dv*df)
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
# plt.figure()
# power = np.sqrt(row**2)
# power = np.ma.masked_less(power, 1E-15)
# plt.loglog(power, 'd-')
# plt.xlabel('$n$')
# plt.title('Power spectrum of $f=%s$' % sp.latex(f)) 

# Okay we have exponential convergence
# Now we are only interested in the expansion coefficients of the hat functions
# First get them with FEniCS
# from dolfin import *

# To compute with FEniCS, I only need the derivatives of basis functions but as
# expressions

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


def shen_basis_derivatives(n):
    'Return Shen basis of H^1_0((-1, 1)).'
    x = sp.Symbol('x')
    k = 0
    functions = []
    while k < n:
        weight = 1/sp.sqrt(4*k + 6)
        f = weight*(sp.legendre(k+2, x) - sp.legendre(k, x))
        df = f.diff(x, 1)

        # Now we turn the sympy symbolic to expression
        ff = Expression(dolfincode(df), degree=8)

        functions.append(ff)
        k += 1
    return functions

# We will consider fixed n now. Only interested in comparing FEniCS matrix with
# my matrix
if False:
	n = 16
	derivatives = shen_basis_derivatives(n)

	n_elements = 10
	mesh = IntervalMesh(n_elements, -1, 1)
	V = FunctionSpace(mesh, 'CG', 1)
	v = Function(V)

	# Derivatives of test function are represented here
	mesh_fine = IntervalMesh(10000, -1, 1)
	V_fine = FunctionSpace(mesh_fine, 'DG', 0)

	# Matrix with expansion coefficients
	P = np.zeros((V.dim()-2, n))

	# Loop over interior test functions
	for i in range(1, V.dim()-1):
	    # Get the test function
	    v_values = np.zeros(V.dim())
	    v_values[i] = 1
	    v.vector()[:] = v_values

	    # Get its derivative in the finer space
	    dv = project(v.dx(0), V_fine)

	    # Now fill in the row
	    for j, du in enumerate(derivatives):
		P[i-1, j] = assemble(inner(dv, du)*dx)  # Note the shift!

	# The claim is that the matrix can be computed by formula which is really simple
	# if homef mesh is used. 
	h = 2./n_elements
	basis = list(shenp_basis(n))
	vertices = [-1 + i*h for i in range(n_elements+1)]
	Pmat = np.zeros_like(P)
	for i in range(1, 1+Pmat.shape[0]):
	    vertex_p = vertices[i-1]
	    vertex = vertices[i]
	    vertex_n = vertices[i+1]

	    for j, shen in enumerate(basis):
		Pmat[i-1, j] = 2*shen.evalf(subs={x:vertex})/h
		Pmat[i-1, j] += -shen.evalf(subs={x:vertex_p})/h
		Pmat[i-1, j] += -shen.evalf(subs={x:vertex_n})/h

	# print P.shape
	print np.max(np.abs(P-Pmat))

# Finally I am interested in power spectrum of a hat function
#
# Use m
m = 80
basis = list(shenp_basis(m))
dbasis = [v.diff(x, 1) for v in basis]
row = np.zeros(m)

hat_left = 2
hat_right = -2

s = Symbol('s')
h = 0.5
error_max = -1
for i, (b, db) in enumerate(zip(basis, dbasis)):
    left, integral_error = quad(lambdify(x, db*hat_left), (-0.5, 0), error=True)
    right, integra_error = quad(lambdify(x, db*hat_right), (0, 0.5), error=True)
    row[i] = left+right
    # Compare to formulat

    b_unit = b.subs(x, 0.5*s)

    # Some properties of the integral

    value = 2*b_unit.evalf(subs={s:0})/h\
           -b_unit.evalf(subs={s:-1})/h\
           -b_unit.evalf(subs={s:1})/h

    error = abs(row[i] - value)
    # print row[i], value, '--', integral_error
    if error > error_max:
        error_max = error

    print '>> i=%d, mid_val=%g' % (i, b_unit.evalf(subs={s:0})/h), value
    if i % 2 == 1:
        assert abs(b_unit.evalf(subs={s:0})/h) < 1E-15

print 'Max formula error', error_max
# Let's plot the power spectrum
plt.figure()
power = np.sqrt(row**2) + 0.01   # Shift the whole spectrum
plt.loglog(power, marker='x')

print 'power', power
# sorted_indices = np.argsort(power)[-10:]
# power_max = power[sorted_indices]
# print 'power_max', power_max
# plt.loglog(sorted_indices, power_max, 'o-', markersize=12)

plt.title('Power spectrum of hat')
plt.show()
