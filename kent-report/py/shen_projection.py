from __future__ import division
from sympy.plotting import plot
import matplotlib.pyplot as plt
from sympy import legendre, Symbol, lambdify, sqrt, sin, exp, pi, simplify
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

M = Mshen_matrix(20)

# Approx for smooth
if True:
    f = exp(2*x)*(x**4 - 1)#*sin(4*pi*x)
    plot(f, (x, -1, 1))
    df = f.diff(x, 1)

    n_list = np.arange(2, 31)
    errors0 = []
    errors1 = []
    for n in n_list:
        basis = list(shenp_basis(n))
        dbasis = [v.diff(x, 1) for v in basis]
        # Function you want to expand, we shall conder only such function taht have
        # zeros on the boundary. 
        # Compute the expansion coefficients

        # using deriv orthogonality
        if True:
            row = np.zeros(n)
            for i, dv in enumerate(dbasis):
                integrand = lambdify(x, dv*df)
                row[i] = float(quad(integrand, [-1, 1]))
        else:
            row = np.zeros(n)
            # Assemble lhs
            for i, v in enumerate(basis):
                integrand = lambdify(x, v*f)
                row[i] = float(quad(integrand, [-1, 1]))
            # Get matrix
            M = Mshen_matrix(len(row))
            row = np.linalg.solve(M, row)

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
    pickle.dump(data, open('shen_smooth_1_H10.pickle', 'wb'))

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
if False:
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
    m = 81
    basis = list(shenp_basis(m))
    dbasis = [v.diff(x, 1) for v in basis]
    row = np.zeros(m)

    hat_left = 2
    hat_right = -2

    s = Symbol('s')
    h = 0.5
    error_max = -1
    # Note only even modes, we know odd are 0
    for i, (b, db) in enumerate(zip(basis[:], dbasis[:])):
        left, integral_error = quad(lambdify(x, db*hat_left), (-0.5, 0), error=True)
        right, integra_error = quad(lambdify(x, db*hat_right), (0, 0.5), error=True)
        row[i] = left+right
        # Compare to formulat

        b_unit = b.subs(x, 0.5*s)

        # Some properties of the integral

        #value = 2*b_unit.evalf(subs={s:0})/h\
        #       -b_unit.evalf(subs={s:-1})/h\
        #       -b_unit.evalf(subs={s:1})/h

        # if i % 2 == 0:
        #    print b_unit.evalf(subs={s:-1}) - b_unit.evalf(subs={s: 1})

        # error = abs(row[i] - value)
        # print row[i], value, '--', integral_error
        
        if integral_error > error_max:
            error_max = integral_error

        #print '>> i=%d, mid_val=%g' % (i, b_unit.evalf(subs={s:0})/h), value
        #if i % 2 == 1:
        #    assert abs(b_unit.evalf(subs={s:0})/h) < 1E-15

    print 'Max formula error', error_max
    # Let's plot the power spectrum
    plt.figure()
    power = np.sqrt(row**2)
    plt.loglog(power, marker='x')  # Only plot nonzeros (even) modes
    plt.loglog(range(len(power))[::2], power[::2],
               linestyle='--')  # Only plot nonzeros (even) modes
    maxima = [4, 16, 28, 40, 52, 64, 76]
    plt.loglog(maxima, power[maxima], 'r')

    # Fit for maxima

    b = np.log(maxima)
    col = -np.log(power[maxima])
    A = np.ones((len(b), 2))
    A[:, 0] = col
    ans = np.linalg.lstsq(A, b)[0]
    p = ans[0]
    print '\tLeast square rate %.2f' % p

    print 'power', power
    # sorted_indices = np.argsort(power)[-10:]
    # power_max = power[sorted_indices]
    # print 'power_max', power_max
    # plt.loglog(sorted_indices, power_max, 'o-', markersize=12)

    plt.title('Power spectrum of hat')
    plt.show()
