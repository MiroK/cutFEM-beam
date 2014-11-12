from sympy import pi, Dummy, S, Rational, cos
from sympy.polys.orthopolys import legendre_poly
from sympy.polys.rootoftools import RootOf
from quadrature import __CACHE_DIR__
import numpy as np
import pickle
import os


class PointGenerator(object):
    'Parent class for generation of points such as Gauss-Legendre etc.'
    def __init__(self, N, **kwargs):
        'Generate points for [-1, 1]^d.'
        dim = len(N)
        use_cache = kwargs.get('use_cache', True)
        n_digits = kwargs.get('n_digits', 15)

        if dim == 1:
            n = N[0]
            assert isinstance(n, int)

            # With cache, check if they are available. If not compute and dump
            if use_cache:
                # Make sure we have directory
                if not os.path.exists(__CACHE_DIR__):
                    os.mkdir(__CACHE_DIR__)
                else:
                    os.path.isdir(__CACHE_DIR__)

                pts_name = '%s/.%s_points_%d.pickle' % \
                    (__CACHE_DIR__, self.name, n)

                if os.path.exists(pts_name):
                    xi = pickle.load(open(pts_name, 'rb'))
                else:
                    xi = self._get_points(n, n_digits)
                pickle.dump(xi, open(pts_name, 'wb'))
            else:
                xi = self._get_points(n, n_digits)

            assert len(xi) == n

            # Return 1d points
            self.xi = np.array(map(float, xi))
        # Return dim x N[i] matrix with points whose cartesian product can be
        # used for quadrature, tensor product basis etc
        else:
            self.xi = np.array([[point
                                 for point in self._get_points(N[i],
                                                               n_digits)
                                 ]
                                for i in range(dim)])

    def _get_points(N, n_digits):
        'Generate points in interval [-1, 1].'
        raise NotImplementedError('Implement the method in children.')

    def __str__(self):
        'Introduce yourself.'
        return 'Generator for %s points' % self.name

    def __call__(self):
        'Get the generated points.'
        return self.xi


class ChebyshevPointsGenerator(PointGenerator):
    'Generate Chebyshev points for [-1, 1]^d.'
    def __init__(self, N, n_digits):
        self.name = 'Chebyshev'
        PointGenerator.__init__(self, N, use_cache=False, n_digits=n_digits)

    def _get_points(self, n, n_digits):
        'Generate Chebyshev points for [-1, 1].'
        return map(float,
                   [cos(pi*Rational(2*k-1, 2*n)).n(n_digits)
                    for k in range(1, n+1)])


class EquidistantPointsGenerator(PointGenerator):
    'Generate equidistant points for [-1, 1]^d.'
    def __init__(self, N, n_digits):
        self.name = 'equidistant'
        PointGenerator.__init__(self, N, use_cache=False)

    def _get_points(self, n, n_digits):
        'Generate equidistant points for [-1, 1].'
        h = Rational(2, n-1)
        return map(float, [(-1 + i*h).n(n_digits) for i in range(n)])


class GaussLegendrePointsGenerator(PointGenerator):
    'Generate Gauss-Legendre points for [-1, 1]^d.'
    def __init__(self, N, n_digits):
        self.name = 'Gauss-Legendre'
        PointGenerator.__init__(self, N, use_cache=True, n_digits=n_digits)

    def _get_points(self, n, n_digits):
        'Generate Gauss-Legendre points for [-1, 1].'
        # Points are roots of n-th legendre polynomial
        x = Dummy('x')
        p = legendre_poly(n, x, polys=True)
        xi = []
        for r in p.real_roots():
            if isinstance(r, RootOf):
                r = r.eval_rational(S(1)/10**(n_digits+2))
            xi.append(r.n(n_digits))

        return np.array(map(float, xi))


class GaussLegendreLobattoPointsGenerator(PointGenerator):
    'Generate Gauss-Legendre-Lobatto points for [-1, 1]^d.'
    def __init__(self, N, n_digits):
        self.name = 'Gauss-Legendre-Lobatto'
        PointGenerator.__init__(self, N, use_cache=True, n_digits=n_digits)

    def _get_points(self, n, n_digits):
        'Generate Gauss-Legendre-Lobatto points for [-1, 1].'
        # Points are -1, 1 and roots of derivative of (n-1)st legendre
        # polynomial. It has degree n-2 -> n-2 roots, so in total n-2 pts
        n = n-1

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

        return np.array(map(float, xi))

# ------------------ Make functions for quick access --------------------------


def equidistant_points(N, n_digits=15):
    '''
    Generate equidistant points for cube [-1, 1]^d, d=1, 2,..., dim.
    Number of points in each dimension is specified as N[i] where
    dim = len(N).
    '''
    return EquidistantPointsGenerator(N, n_digits)()


def chebyshev_points(N, n_digits=15):
    '''
    Generate chebyshev points for cube [-1, 1]^d, d=1, 2,..., dim.
    Number of points in each dimension is specified as N[i] where
    dim = len(N).
    '''
    return ChebyshevPointsGenerator(N, n_digits)()


def gauss_legendre_points(N, n_digits=15):
    '''
    Generate Gauss-Legendre points for cube [-1, 1]^d, d=1, 2,..., dim.
    Number of points in each dimension is specified as N[i] where
    dim = len(N).
    '''
    return GaussLegendrePointsGenerator(N, n_digits)()


def gauss_legendre_lobatto_points(N, n_digits=15):
    '''
    Generate Gauss-Legendre-Lobatto points for cube [-1, 1]^d, d=1, 2,..., dim.
    Number of points in each dimension is specified as N[i] where
    dim = len(N).
    '''
    return GaussLegendreLobattoPointsGenerator(N, n_digits)()


# Register here for generation of points by script. Only those that use cache
all_points = [gauss_legendre_points, gauss_legendre_lobatto_points]

# -----------------------------------------------------------------------------

if __name__ == '__main__':
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
