from sympy.polys.orthopolys import legendre_poly
from sympy.polys.rootoftools import RootOf
from sympy import sin, pi, sqrt, symbols, diff, Dummy, S
from itertools import product
import numpy as np
import operator
import pickle
import os


def sine_basis(N, xi=None):
    '''
    TODO
    '''
    xyz = symbols('x, y, z')
    if xi is None:
        xi = 0

    dim = len(N)
    if dim == 1:
        # Generate for given k
        try:
            return np.array([sin(k*pi*xyz[xi])*sqrt(2) for k in N[0]])
        # Generate for 1, ... N!!!
        except TypeError:
            return sine_basis([range(1, N[0]+1)], xi=xi)
    else:
        shape = tuple(map(lambda item: item if isinstance(item, int)
                          else len(item), N))
        return np.array([reduce(operator.mul, sin_xyz)
                         for sin_xyz in product(*[sine_basis([N[i]], xi=i)
                                                  for i in range(dim)])]
                        ).reshape(shape)


def chebyshev_points(N):
    '''
    TODO
    '''
    dim = len(N)
    if dim == 1:
        k = np.arange(1, N[0]+1, 1)
        return np.cos((2*k-1)*np.pi/2./N[0])
    else:
        return np.array([[point for point in chebyshev_points([N[i]])]
                         for i in range(dim)])


def equidistant_points(N):
    '''
    TODO
    '''
    dim = len(N)
    if dim == 1:
        return np.linspace(-1, 1, N[0])
    else:
        return np.array([[point for point in equidistant_points([N[i]])]
                         for i in range(dim)])


def gauss_legendre_points(N, n_digits=15):
    '''
    TODO
    This code is taken from original SymPy implementation of
    sympy.integral.quadrature.gauss_legendre()
    '''
    dim = len(N)
    if dim == 1:
        # The points are n(all) zeros of n-th Legendre polynomial
        n = N[0]

        gl_pts_name = '.gl_points_%d.pickle' % n

        if os.path.exists(gl_pts_name):
            xi = pickle.load(open(gl_pts_name, 'rb'))
        else:
            x = Dummy('x')
            p = legendre_poly(n, x, polys=True)
            xi = []
            for r in p.real_roots():
                if isinstance(r, RootOf):
                    r = r.eval_rational(S(1)/10**(n_digits+2))
                xi.append(r.n(n_digits))

            pickle.dump(xi, open(gl_pts_name, 'wb'))

        return np.array(map(float, xi))
    else:
        return np.array([[point for point in gauss_legendre_points([N[i]])]
                         for i in range(dim)])


def gauss_legendre_lobatto_points(N, n_digits=15):
    '''
    TODO
    '''
    dim = len(N)
    if dim == 1:
        # The points are n(all) zeros of n-th Legendre polynomial
        n = N[0]-1

        gll_pts_name = '.gll_points_%d.pickle' % n

        if os.path.exists(gll_pts_name):
            xi = pickle.load(open(gll_pts_name, 'rb'))
        else:
            x = Dummy('x')
            p = legendre_poly(n, x, polys=True).diff()
            xi = []
            for r in p.real_roots():
                if isinstance(r, RootOf):
                    r = r.eval_rational(S(1)/10**(n_digits+2))
                xi.append(r.n(n_digits))
            # Pad with -1, 1
            xi.insert(0, S(-1))
            xi.append(S(1))

            pickle.dump(xi, open(gll_pts_name, 'wb'))

        return np.array(map(float, xi))
    else:
        return np.array([[point
                          for point in gauss_legendre_lobatto_points([N[i]])]
                         for i in range(dim)])


def lagrange_basis(points, xi=None):
    '''
    TODO
    '''
    xyz = symbols('x, y, z')
    if xi is None:
        xi = 0

    dim = len(points)
    if dim == 1:
        points = points[0]
        x = xyz[xi]
        basis_xi = []
        for i, xi in enumerate(points):
            nom = reduce(operator.mul, [x-points[j]
                                        for j in range(len(points)) if j != i])
            den = nom.subs(x, xi)
            basis_xi.append(nom/den)
        return np.array(basis_xi)
    else:
        shape = tuple(map(len, points))
        return np.array([reduce(operator.mul, l_xyz)
                         for l_xyz in product(*[lagrange_basis([points[i]],
                                                               xi=i)
                                                for i in range(dim)])]
                        ).reshape(shape)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import integrate
    from scipy.interpolate import BarycentricInterpolator as BI
    import matplotlib.pyplot as plt

    def f(x):
        return 1./(1 + 16*x**2)

    x = np.linspace(-1, 1, 200)
    y = np.array([f(xi) for xi in x])

    N = 13
    x_e = equidistant_points([N])
    y_e = np.array([f(xi) for xi in x_e])
    e_interpolate = BI(x_e, y_e)
    yy = np.array([e_interpolate(xi) for xi in x])

    x_c = chebyshev_points([N])
    y_c = np.array([f(xi) for xi in x_c])
    e_interpolate = BI(x_c, y_c)
    yyy = np.array([e_interpolate(xi) for xi in x])

    x_gl = gauss_legendre_points([N])
    y_gl = np.array([f(xi) for xi in x_gl])
    e_interpolate = BI(x_gl, y_gl)
    yyyy = np.array([e_interpolate(xi) for xi in x])

    x_gll = gauss_legendre_lobatto_points([N])
    y_gll = np.array([f(xi) for xi in x_gll])
    e_interpolate = BI(x_gll, y_gll)
    y5 = np.array([e_interpolate(xi) for xi in x])

    plt.figure()
    plt.plot(x, y, 'b', label='f')

    plt.plot(x, yy, 'g', label='equidistant')
    plt.plot(x_e, y_e, 'go')

    plt.plot(x, yyy, 'r', label='chebyshev')
    plt.plot(x_c, y_c, 'rs')

    plt.plot(x, yyyy, 'm', label='gauss-legendre')
    plt.plot(x_gl, y_gl, 'md')

    plt.plot(x, y5, 'k', label='gauss-legendre-lobatto')
    plt.plot(x_gll, y_gll, 'kx')

    plt.legend(loc='best')
    plt.show()
    exit()

    # Make sure that the 1d basis is orthonormal
    if False:
        x = symbols('x')
        for i, si in enumerate(sine_basis([2])):
            for j, sj in enumerate(sine_basis([2])):
                print si, sj
                if i == j:
                    assert abs(integrate(si*sj, (x, 0, 1)) - 1) < 1E-15
                else:
                    assert abs(integrate(si*sj, (x, 0, 1))) < 1E-15

    # Make sure that the 2d basis is orthonormal
    if False:
        x, y = symbols('x, y')
        basis = sine_basis([2, 2])
        for i, bi in enumerate(basis.flatten()):
            for j, bj in enumerate(basis.flatten()):
                print bi, bj
                l2_ip = integrate(integrate(bi*bj, (x, 0, 1)), (y, 0, 1))
                if i == j:
                    assert abs(l2_ip - 1) < 1E-15
                else:
                    assert abs(l2_ip) < 1E-15

    import sympy.plotting as s_plot
    x, y = symbols('x, y')
    points_x, points_y = chebyshev_points([2, 2])  # how is it with dim 1
    for lp in lagrange_basis([points_x, points_y]).flatten():
        print lp
        s_plot.plot3d(diff(lp, x, 1), (x, -1, 1), (y, -1, 1))
        s_plot.plot3d(diff(lp, y, 1), (x, -1, 1), (y, -1, 1))
