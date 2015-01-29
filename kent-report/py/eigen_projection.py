from __future__ import division
from sympy.plotting import plot

# Ploting
from matplotlib import rc 
rc('text', usetex=True) 
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
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

if True:
    # We check properties of the generalied Fourier series for Eigen
    # Number of eigen functions to use
    f = exp(2*x)*(x**4 - 1)
    n_list = np.arange(2, 33)
    errors0 = []
    errors1 = []
    for n in n_list:
        basis = list(eigen_basis(n))
        # Function you want to expand, we shall conder only such function taht have
        # zeros on the boundary. 
        # Compute the expansion coefficients
        row = np.zeros(n)
        for i, v in enumerate(basis):
            integrand = lambdify(x, v*f)
            row[i] = float(quad(integrand, [-1, 1]))
        # Assemble the approximation
        f_ = sum(coef*v for (coef, v) in zip(row, basis))

        # Compute the L2 error of aproximation
        e0 = lambdify(x, (f-f_)**2)
        error0 = sqrt(quad(e0, (-1, 1)))
        # Compute the H1 error of aproximation
        e1 = lambdify(x, ((f-f_).diff(x, 1))**2)
        error1 = sqrt(quad(e1, (-1, 1)))

        if n != 2:
            print n, error0, ln(error0/error0_)/ln(n_/n), \
                error1, ln(error1/error1_)/ln(n_/n)
        
        n_ = n
        error0_ = error0
        errors0.append(float(error0))
        error1_ = error1
        errors1.append(float(error1))

    data = {'n_list': n_list,
            'errors0': errors0,
            'errors1': errors1,
            'power': row,
            'f': sp.latex(f)}
    import pickle
    pickle.dump(data, open('eigen_smooth_0.pickle', 'wb'))


if False:
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
# (x_prev, xi, x_next) where xi will be zero or something else to make the
# function nor odd nor even. Also the distance is imporant
if False:
    def hat_coefficient(xi, x_prev, x_next, j):
        'Compute \int_{-1}^{1} phi_j hat_i.'
        # Phase
        alpha = lambda k: math.pi/2 + k*math.pi/2
        # j-th basis functions at x
        phi = lambda x, j: math.cos(alpha(j)*x) if j % 2 == 0 else math.sin(alpha(j)*x)
        # Compute the value of integral
        ans = (phi(xi, j) - phi(x_prev, j))/(xi - x_prev)/alpha(j)/alpha(j)
        ans += (phi(x_next, j) - phi(xi, j))/(xi - x_next)/alpha(j)/alpha(j)
        return ans

    def even_even_coefficient(h, j):
        alpha = lambda k: math.pi/2 + k*math.pi/2
        # print j, 0.5*alpha(j)*h 
        return 4/alpha(j)/alpha(j)/h*(math.sin(0.5*alpha(j)*h)**2)

    # Use m eigenfunctions
    m = 1024
    basis = list(eigen_basis(m))
    row = np.zeros(m)

    # Vary the spacing and check sensitivity
    markers = iter(['o', 'd', 'x', 's', 'v'])
    colors = iter(['r', 'b', 'g', 'm', 'k'])
    labels = iter([r'$\frac{1}{4}$',
                   r'$\frac{1}{8}$',
                   r'$\frac{1}{32}$'])

    plt.figure()
    for h in [1/4., 1/8., 1/32.]:
        xi = 0.1
        x_prev = xi - h
        x_next = xi + h

        hat_left = ((x-x_prev)/(xi-x_prev))
        hat_right = ((x-x_next)/(xi-x_next))


        error_max = -1
        for i, b in enumerate(basis):
            # left = quad(lambdify(x, b*hat_left), (x_prev, xi))
            # right = quad(lambdify(x, b*hat_right), (xi, x_next))
            # row[i] = left+right

            # error = abs(row[i]-hat_coefficient(xi, x_prev, x_next, i))
            # if error > error_max:
            #    error_max = error
            row[i] = hat_coefficient(xi, x_prev, x_next, i)

            # Test nulity of odd modes and look even
            #if i % 2 == 1:
            #    assert abs(row[i]) < 1E-14, '%g @ %i' % (row[i], i)
            #else:
            #    diff = abs(row[i]-even_even_coefficient(h, i))
            #    assert diff < 1E-14, '%g' % diff

        # print 'Formula error', error_max

        # Let's plot the power spectrum
        power = np.sqrt(row**2)
        ns = range(len(power))

        nnz_ns = []
        nnz_powers = []

        for n_val, p_val in zip(ns, power):
            if p_val > 1E-13:
                nnz_ns.append(n_val)
                nnz_powers.append(p_val)

        #plt.loglog(ns[::2], power[::2], marker=next(markers), color=next(colors),
        #           linestyle='--', label=next(labels))
        c = next(colors)
        plt.loglog(nnz_ns[::2], nnz_powers[::2], marker=next(markers), color=c,
                   linestyle='--', label=next(labels))

        # where = [2./h, 2./h]
        # plt.loglog(where, [1, 1e-8], color=c)

        # print power

    plt.xlabel('$k$')
    plt.ylabel(r'$|(f, \varphi_k)|$')
    plt.legend(loc='best')
    plt.savefig('eigen_hat_spectrum_off.pdf')
    plt.show()
