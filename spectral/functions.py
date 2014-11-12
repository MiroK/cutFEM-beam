from sympy import sin, pi, sqrt, symbols, integrate, diff
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
    from points import chebyshev_points
    from quadrature import zero_mean, GLQuadrature
    import plots

    if True:
        from sympy import lambdify
        x = symbols('x')
        N = 5
        quad = GLQuadrature(N/2 + 2)
        bs = lagrange_basis([chebyshev_points([N])])
        b0_s = zero_mean(bs)
        for b, b0 in zip(bs, b0_s):
            plots.plot(b, [[-1, 1]])
            plots.plot(b0, [[-1, 1]])
            print 'mean of b0', quad.eval(lambdify(x, b0), [[-1, 1]])

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

    if False:
        import sympy.plotting as s_plot
        x, y = symbols('x, y')
        points_x, points_y = chebyshev_points([2, 2])
        for lp in lagrange_basis([points_x, points_y]).flatten():
            s_plot.plot3d(diff(lp, x, 1), (x, -1, 1), (y, -1, 1))
            s_plot.plot3d(diff(lp, y, 1), (x, -1, 1), (y, -1, 1))
