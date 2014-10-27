from sympy import sin, pi, sqrt, symbols
from itertools import product
import numpy as np
import operator


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
        # Generate for 1, ... N-1!!!
        except TypeError:
            return sine_basis([range(1, N[0]+1)], xi=xi)
    else:
        shape = map(lambda item: item if isinstance(item, int) else len(item),
                    N)
        return np.array([reduce(operator.mul, basis_comps)
                         for basis_comps in product(*[sine_basis([N[i]], xi=i)
                                                      for i in range(dim)])]).reshape(tuple(shape))

def legendre_polynomial(N):
    pass


def lagrange_basis(N, point_distribution):
    pass


def equidistant_points(domain, N):
    '''
    [-1, -1], [[-1, 1], [-1, 1]]
    '''
    try:
        a = domain[0]
        b = domain[1]
        return np.linspace(a, b, N)
    except TypeError:
        return np.array([point
                         for point in product(*[equidistant_points(sub, n)
                                                for sub, n in zip(domain, N)])])

def chebyshev_points(domain, N):
    '''
    [-1, -1], [[-1, 1], [-1, 1]]
    '''
    try:
        k = np.arange(1, N+1, 1)
        return np.cos((2*k-1)*np.pi/2/N)  # check that these are correct
    except TypeError:
        return np.array([point
                         for point in product(*[equidistant_points(sub, n)
                                                for sub, n in zip(domain, N)])])

def gauss_legendre_points(domain, N):
    pass

def gauss_lobatto_points(domain, N):
    pass

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
    x_e = equidistant_points([-1, 1], N)
    y_e = np.array([f(xi) for xi in x_e])
    e_interpolate = BI(x_e, y_e)
    yy = np.array([e_interpolate(xi) for xi in x])

    x_c = chebyshev_points([-1, 1], N)
    y_c = np.array([f(xi) for xi in x_c])
    e_interpolate = BI(x_c, y_c)
    yyy = np.array([e_interpolate(xi) for xi in x])

    plt.figure()
    plt.plot(x, y, 'b', label='f')
    plt.plot(x, yy, 'g', label='eq')
    plt.plot(x_e, y_e, 'go')
    plt.plot(x, yyy, 'r', label='cheb')
    plt.plot(x_c, y_c, 'rs')
    plt.legend()
    # plt.show()

    # Make sure that the 1d basis is orthonormal
    x = symbols('x')
    for i, si in enumerate(sine_basis([2])):
        for j, sj in enumerate(sine_basis([2])):
            print si, sj
            if i == j:
                assert abs(integrate(si*sj, (x, 0, 1)) - 1) < 1E-15
            else:
                assert abs(integrate(si*sj, (x, 0, 1))) < 1E-15

    # Make sure that the 2d basis is orthonormal
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
