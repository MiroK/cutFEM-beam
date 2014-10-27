from sympy import symbols, diff, lambdify
from sympy.integrals import quadrature
from itertools import product
from math import sqrt
import numpy as np
import pickle
import time
import os

__EPS__ = np.finfo(float).eps


class GLQuadrature(object):
    'Generate Gauss-Legendre points for computing integral over [-1, 1]^d.'
    def __init__(self, N):
        'Create quadrature with number of points given in N.'
        time_q = time.time()

        # See if we are making 1d or higher-d quadrature
        try:
            len(N)
        except TypeError:
            N = [N]
        self.dim = len(N)

        zs_dir = []
        ws_dir = []
        for i in range(self.dim):
            quad_name = '.quadrature_%d.pickle' % N[i]

            # Try loading points that were already computed
            if os.path.exists(quad_name):
                z, w = pickle.load(open(quad_name, 'rb'))
            # Compute points, weights and dump for next time
            else:
                z, w = quadrature.gauss_legendre(N[i], n_digits=15)
                pickle.dump((z, w), open(quad_name, 'wb'))

            assert len(z) == N[i] and len(w) == N[i]
            zs_dir.append(z)
            ws_dir.append(w)

        points = np.array([point for point in product(*zs_dir)])
        weights = np.array([np.product(weight)
                           for weight in product(*ws_dir)])

        time_q = time.time() - time_q
        print 'Computing %s-point quadrature :' % 'x'.join(map(str, N)), time_q

        self.N = np.array(N)
        self.z = points
        self.w = weights

    def __str__(self):
        return '%d-point 1d Gauss-Legendre quadrature' % self.N

    def eval(self, f, domain):
        '''
        Integrate[domain(with tensor product structure)\in R^d] f(x) dx.
        '''
        assert len(domain) == self.dim

        def F(q):
            'Build map for going from [-1, 1]^d to domain.'
            return [0.5*sub[0]*(1-qi) + 0.5*sub[1]*(1+qi)
                    for qi, sub in zip(q, domain)]

        # Jacobian of pull back
        J = np.product([0.5*(b - a) for (a, b) in domain])

        return J*sum(wi*f(*F(zi)) for zi, wi in zip(self.z, self.w))

    def eval_adapt(self, f, domain, eps=__EPS__, n_refs=5):
        '''
        Given starting(initialized) quadrature, this function computes
        integral[domain(with tensor product structure)\in R^d] f(x) dx
        using starting and higher order quadratures until the difference is
        smaller then eps.
        '''
        diff = 1
        ref = 0
        result_N = self.eval(f, domain)
        while diff > eps and ref < n_refs:
            ref += 1

            new_N = self.N + 1
            self.__init__(new_N)
            result_new_N = self.eval(f, domain)

            diff = abs(result_new_N - result_N)
            result_N = result_new_N

        print 'eval_adapt final diff', diff
        return result_N

    def plot_points(self, figure):
        'Plot points of the quadrature.'
        if self.dim > 2:
            ax = figure.gca(projection='3d')
        else:
            ax = figure().gca()

        ax = figure.gca()
        ax.plot(*[self.z[:, i] for i in range(self.dim)], color='blue',
                marker='o', linestyle=' ')


def errornorm(u, (U, basis), norm_type, domain):
    '''
    TODO
    '''
    dim = len(domain)
    assert dim == len(U.shape)
    # Heuristic choice for quadrature order
    N = max(U.shape)
    quad = GLQuadrature([2*N]*dim)

    # Assemble uh as linear combination
    uh = sum(Ui*base for (Ui, base) in zip(U.flatten(), basis.flatten()))

    xyz = symbols('x, y, z')
    e = u - uh
    if norm_type == 'L2':
        f = e**2
    elif norm_type == 'H10':
        f = sum([diff(e, xi, 1)**2 for xi in xyz[:dim]])
    elif norm_type == 'H20':
        laplace_comps = [diff(e, xi, 2) for xi in xyz[:dim]]
        f = sum(c0*c1 for c0 in laplace_comps for c1 in laplace_comps)

    # Make for fast evaluation
    f_lambda = lambdify(xyz[:dim], f)
    norm = quad.eval_adapt(f_lambda, domain, n_refs=10)

    if norm < 0:
        return 1E-16
    else:
        return sqrt(norm)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import integrate, sin, exp, cos
    import matplotlib.pyplot as plt

    # 2d tests
    quad = GLQuadrature([2, 2])
    # Plot the quadrature points
    figure = plt.figure()
    quad.plot_points(figure)
    plt.show()

    # Just area
    assert abs(quad.eval(lambda x, y: 1, domain=[[0, 3], [0, 4]])-12) < 1E-15
    # Linear polynomial
    x, y = symbols('x, y')
    f = x + y
    F = integrate(integrate(f, (x, 0, 3)), (y, 0, 1)).evalf()
    f_lambda = lambdify([x, y], f)
    assert abs(quad.eval(f_lambda, domain=[[0, 3], [0, 1]]) - F) < 1E-15
    # Something harded but adapt should manage
    f = sin(x) + cos(y)
    F = integrate(integrate(f, (x, 0, 3)), (y, 0, 1)).evalf()
    f_lambda = lambdify([x, y], f)
    assert abs(quad.eval_adapt(f_lambda, domain=[[0, 3], [0, 1]]) - F) < 1E-12
    # L2
    U = np.array([[1, 1]])
    basis = np.array([x, y])
    F_ = errornorm(0, (U, basis), domain=[[0, 1], [0, 1]], norm_type='L2')
    f = (x+y)**2
    F = integrate(integrate(f, (x, 0, 1)), (y, 0, 1)).evalf()
    F = F**0.5
    assert abs(F-F_) < 1E-12
    # H10
    U = np.array([[1, 1]])
    basis = np.array([x**2, y**2])
    F_ = errornorm(0, (U, basis), domain=[[0, 1], [0, 1]], norm_type='H10')
    f = 4*(x**2 + y**2)
    F = integrate(integrate(f, (x, 0, 1)), (y, 0, 1)).evalf()
    F = F**0.5
    assert abs(F-F_) < 1E-12
    # H20
    U = np.array([[1, 1]])
    basis = np.array([x**3, 2*y**3])
    F_ = errornorm(0, (U, basis), domain=[[0, 1], [0, 1]], norm_type='H20')
    f = (6*x + 12*y)**2
    F = integrate(integrate(f, (x, 0, 1)), (y, 0, 1)).evalf()
    F = F**0.5
    assert abs(F-F_) < 1E-12

    for N in range(1, 60):
        quad = GLQuadrature([N, N])
        quad_is_exact = True
        p = -1
        while quad_is_exact:
            p += 1
            f = x**p + y**p
            F = integrate(integrate(f, (x, 0, 1)), (y, 0, 1)).evalf()
            e = abs(quad.eval(lambdify([x, y], f), [[0, 1], [0, 1]]) - F)

            quad_is_exact = e < 1E-15
            # print '\t (p=%d) -> %.2E' % (p, e)
        p_exact = p-1
        assert p_exact == p-1

    exit()
    # 1d tests
    x = symbols('x')
    f = x
    # Exact integral
    F = integrate(f, (x, 0, 1)).evalf()
    f_lambda = lambdify(x, f)
    quad = GLQuadrature([2])

    # Integrate with given quadrature
    assert abs(quad.eval(f_lambda, [[0, 1]]) - F) < 1E-15
    # Integrate adaptively
    assert abs(quad.eval_adapt(f_lambda, [[0, 1]]) - F) < 1E-15
    # Let's see how much points we need for some harder function
    f = sin(x)*exp(x)
    f_lambda = lambdify(x, f)
    F_ = quad.eval_adapt(f_lambda, [[0, 1]], n_refs=20)
    F = integrate(f, (x, 0, 1)).evalf()
    assert abs(F - F_) < 1E-15

    u = sin(x)
    U = np.array([1])
    basis = np.array([sin(x)])
    error = errornorm(u, (U, basis), domain=[[-2, 3]], norm_type='L2')
    error = errornorm(u, (U, basis), domain=[[-2, 3]], norm_type='H10')
    error = errornorm(u, (U, basis), domain=[[-2, 3]], norm_type='H20')
    assert abs(error) < 1E-15

    # Verify the 2*n-1 accuracy
    for N in range(1, 60):
        quad = GLQuadrature(N)
        quad_is_exact = True
        p = -1
        while quad_is_exact:
            p += 1
            f = x**p
            F = integrate(f, (x, 0, 1)).evalf()
            e = abs(quad.eval(lambdify(x, f), [[0, 1]]) - F)
            quad_is_exact = e < 1E-15
            # print '\t %d -> %.2E' % (p, e)

        p_exact = p-1
        assert p_exact == p-1
