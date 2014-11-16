from sympy import symbols, diff, lambdify, Rational, Dummy, S
from sympy.polys.orthopolys import legendre_poly
from sympy.polys.rootoftools import RootOf
from sympy.integrals import quadrature
from itertools import product
import matrices
from functions import lagrange_basis
from points import gauss_legendre_points as gl_points
from common import __CACHE_DIR__, __EPS__
from math import sqrt
from mpi4py import MPI
import numpy as np
import pickle
import time
import os


class Quadrature(object):
    '''
    Parent class for computing integrals over [-1, 1]^d. Handles everything
    but creation of points and weights for summation. This must be provided
    by children.
    '''
    def get_points_weights(self, n, n_digits):
        'Compute points and weight of 1d quadrature'
        raise NotImplementedError('Implement in child!')

    def __init__(self, N, n_digits=15):
        'Create quadrature with number of points given in N.'
        time_q = time.time()

        # See if we are making 1d or higher-d quadrature
        try:
            len(N)
        except TypeError:
            N = [N]
        self.dim = len(N)

        if not os.path.exists(__CACHE_DIR__):
            os.mkdir(__CACHE_DIR__)
        else:
            os.path.isdir(__CACHE_DIR__)

        # Split for parallel
        comm = MPI.COMM_WORLD
        proc = comm.Get_rank()
        n_procs = comm.Get_size()

        # Lists for points and weights
        zs_dir = []
        ws_dir = []
        # Only root touches the pickle
        if proc == 0:
            for i in range(self.dim):
                quad_name = '%s/.quadrature_%s_%d.pickle' % \
                    (__CACHE_DIR__, self.name, N[i])

                # Try loading points that were already computed
                if os.path.exists(quad_name):
                    z, w = pickle.load(open(quad_name, 'rb'))
                # Compute points, weights and dump for next time
                else:
                    z, w = self.get_points_weights(N[i], n_digits)
                    pickle.dump((z, w), open(quad_name, 'wb'))

                assert len(z) == N[i] and len(w) == N[i]
                zs_dir.append(z)
                ws_dir.append(w)

        zs_dir = comm.bcast(zs_dir, root=0)
        ws_dir = comm.bcast(ws_dir, root=0)

        points = np.array([point for point in product(*zs_dir)])
        weights = np.array([np.product(weight)
                           for weight in product(*ws_dir)])

        glob_len = len(points)
        loc_len = glob_len/n_procs
        begin = proc*loc_len
        end = begin + loc_len if proc != n_procs-1 else glob_len

        points = points[begin:end]
        weights = weights[begin:end]

        time_q = time.time() - time_q
        print 'Computing %s-point %s quadrature :' % \
            ('x'.join(map(str, N)), self.name), time_q

        self.comm = comm
        self.N = np.array(N)
        self.z = points
        self.w = weights

    def __str__(self):
        return '%s-point %s quadrature' % \
            ('x'.join(map(str, self.N)), self.name)

    def eval(self, f, domain):
        '''
        Integrate[domain(with tensor product structure)\in R^d f(x) dx.
        The quadrature loop is implemented manually.
        '''
        assert len(domain) == self.dim

        def F(q):
            'Build map for going from [-1, 1]^d to domain.'
            return [0.5*sub[0]*(1-qi) + 0.5*sub[1]*(1+qi)
                    for qi, sub in zip(q, domain)]

        # Jacobian of pull back
        J = np.product([0.5*(b - a) for (a, b) in domain])
        local_sum = np.array(J*sum(wi*f(*F(zi))
                                   for zi, wi in zip(self.z, self.w)))

        return self.comm.allreduce(local_sum, MPI.DOUBLE)

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
            ax = figure.gca()

        ax = figure.gca()
        ax.plot(*[self.z[:, i] for i in range(self.dim)], color='blue',
                marker='o', linestyle=' ')


class GLQuadrature(Quadrature):
    'Gauss-Legendre quadrature.'
    def __init__(self, N, n_digits=15):
        self.name = 'Gauss-Legendre'
        Quadrature.__init__(self, N, n_digits)

    def get_points_weights(self, n, n_digits):
        'Points and weights for 1d Gauss-Legendre quadrature.'
        assert isinstance(n, int) and n > 0
        return quadrature.gauss_legendre(n, n_digits=15)


class GLLQuadrature(Quadrature):
    'Guass-Legendre-Lobatto quadrature.'
    def __init__(self, N, n_digits=15):
        self.name = 'Gauss-Legendre-Lobatto'
        Quadrature.__init__(self, N, n_digits)

    def get_points_weights(self, n, n_digits):
        'Points and weights for 1d Gauss-Legendre-Lobatto quadrature.'
        assert isinstance(n, int) and n > 1
        x = Dummy('x')
        p = legendre_poly(n-1, x, polys=True)
        dp = p.diff()
        xi = []
        wi = []
        _w_ = Rational(2, n*(n-1))
        for r in dp.real_roots():
            if isinstance(r, RootOf):
                r = r.eval_rational(S(1)/10**(n_digits+2))
            xi.append(r.n(n_digits))
            wi.append((_w_/p.subs(x, r)**2).n(n_digits))
        # Pad with -1, 1
        xi.insert(0, S(-1))
        xi.append(S(1))

        wi.insert(0, _w_.n(n_digits))
        wi.append(_w_.n(n_digits))

        return (xi, wi)


# This registers quadrature for generation by script
all_quads = [GLQuadrature, GLLQuadrature]


def errornorm(u, (U, basis), norm_type, domain):
    '''
    TODO, Uses GL quadrature
    '''
    dim = len(domain)
    assert dim == len(U.shape)
    # Heuristic choice for quadrature order
    N = max(U.shape)
    # Then for the L2 the max is 2*N for this N+1 is sufficient but u does
    # not have to be polynomials so increase to N+2
    quad = GLQuadrature([N+2]*dim)

    # Assemble uh as linear combination
    uh = sum(Ui*base for (Ui, base) in zip(U.flatten(), basis.flatten()))

    xyz = symbols('x, y, z')
    e = u - uh
    if norm_type.lower() == 'l2':
        f = e**2
    elif norm_type.lower() == 'h10':
        f = sum([diff(e, xi, 1)**2 for xi in xyz[:dim]])
    elif norm_type.lower() == 'h20':
        laplace_comps = [diff(e, xi, 2) for xi in xyz[:dim]]
        f = sum(c0*c1 for c0 in laplace_comps for c1 in laplace_comps)
    # Make for fast evaluation
    f_lambda = lambdify(xyz[:dim], f)
    norm = quad.eval_adapt(f_lambda, domain, n_refs=10)

    if norm < 0:
        return 1E-16
    else:
        return sqrt(norm)


def tensor_errornorm(u, (U, basis), norm_type, domain,
                     n_iters=10, prec=100*__EPS__):
    '''
    The integral in the error definition is computed by matrix-matrix or
    matrix-vector products.
    '''
    dim = len(domain)
    assert 1 < dim < 3
    assert dim == len(U.shape)
    # Assemble uh as linear combination
    uh = sum(Ui*base for (Ui, base) in zip(U.flatten(), basis.flatten()))
    # Now make a symbolic error
    e = u - uh
    # Map the error to [-1, 1]^dim
    xy = symbols('x, y')
    for x, (a, b) in zip(xy, domain):
        e = e.subs(x, Rational(a, 2)*(1-x) + (1+x)*Rational(b, 2))
    # Matrices for higher orther space on [-1, 1]^dim must be scaled
    J0, J1 = [(b-a)/2. for a, b in domain]
    # Finally in the computation we need only lambda of error
    e = lambdify(xy[:dim], e)

    N = max(U.shape)
    # Variables for adaptive loop
    n_iter = 0
    diff = 1
    error_ = None
    while n_iter < n_iters and diff > prec:
        n_iter += 1
        # Represent error in higher order space over [-1, 1]^dim
        # First define points of the higher order space
        N += 1
        points = gl_points([N])
        # Interpolate
        E = np.array([e(*p) for p in product(points, points)]).reshape((N, N))
        # Compute the error using matrices
        # Always need mass matrix
        basis = lagrange_basis([points])
        M = matrices.assemble_mass_matrix(basis)
        if norm_type.lower() == 'l2':
            error = ((J0*M.dot(E))*(E.dot(J1*M))).sum()

        elif norm_type.lower() == 'h10':
            A = matrices.assemble_stiffness_matrix(basis)

            error = (((A/J0).dot(E))*(E.dot(J1*M))).sum()
            error += ((J0*M.dot(E))*(E.dot(A/J1))).sum()
        else:
            raise NotImplementedError

        if n_iter > 1:
            diff = abs(error - error_)/error

        error_ = error

    # Prevent taking square root of negative numbers (likely due to round-off)
    print error
    if error > 0:
        return sqrt(error)
    else:
        return 0


def zero_mean(basis):
    '''
    Return functions with zero mean over [-1, 1] that can be used as basis
    of L^2_0(-1, 1).
    '''
    N = len(basis)
    # Degree of N polynomials is N-1 which is integrated exactly by GL
    # quadrature with N/2 points. Add 1 just to be on the safe side
    quad = GLQuadrature(N/2 + 1)
    x = symbols('x')
    return np.array([f - S(quad.eval(lambdify(x, f), [[-1, 1]])/2)
                     for f in basis])

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import integrate, sin, exp, cos
    import matplotlib.pyplot as plt

    x, y = symbols('x, y')
    u = 1 + x + y + x*y
    basis = np.array([[1, 1*y, y**2],
                      [x*1, x*y, x*y**2],
                      [x**2*1, x**2*y, x**2*y**2]])
    U = np.array([[1.3, 2, 0.1, 1],
                  [-1, 1.2, 0.01, 1],
                  [1, 1, 0, 2]])

    exit()
    two = errornorm(u, (U, basis), 'H10', [[-1, 0], [-1, 2]])
    print 'errornorm', two
    one = tensor_errornorm(u, (U, basis), 'H10', [[-1, 0], [-1, 2]])
    print 'tensor', one
    print abs(one - two)

    if False:
        x, y = symbols('x, y')
        quad = GLQuadrature([45, 45])
        f = lambdify([x, y], sin(exp(x))*cos(y))
        print quad.eval(f, [[-1, 1], [2, 3]])

    if False:
        # GLL 1d
        x = symbols('x')
        f = x
        # Exact integral
        F = integrate(f, (x, 0, 1)).evalf()
        f_lambda = lambdify(x, f)
        quad = GLLQuadrature(2)

        # Integrate with given quadrature
        assert abs(quad.eval(f_lambda, [[0, 1]]) - F) < 1E-15
        # Integrate adaptively
        assert abs(quad.eval_adapt(f_lambda, [[0, 1]]) - F) < 1E-15
        # Let's see how much points we need for some harder function
        f = (1+x**2)**-1
        f_lambda = lambdify(x, f)
        F_ = quad.eval_adapt(f_lambda, [[0, 1]], n_refs=20)
        F = integrate(f, (x, 0, 1)).evalf()
        assert abs(F - F_) < 1E-15

        # GL 1d tests
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

        # GL accuracy, N point integrates exactly polynomials of order 2N-1 and less
        # There is no point in runnning this test for large N because then the
        # numeric errors dominate results
        x = symbols('x')
        for N in range(1, 6):
            quad = GLQuadrature(N)
            for p_degree in range(2*N+3):
                f = legendre_poly(p_degree, x)
                exact = integrate(f, (x, -1, 1))
                numeric = quad.eval(lambdify(x, f), [[-1, 1]])
                e = abs(exact - numeric)
                # Exit with first wrong degree
                if e > 1E-14:
                    break
            p_exact = p_degree - 1
            assert p_exact == 2*N - 1

        # GLL accuracy
        x = symbols('x')
        for N in range(2, 6):
            quad = GLLQuadrature(N)
            for p_degree in range(2*N+3):
                f = legendre_poly(p_degree, x)
                exact = integrate(f, (x, -1, 1))
                numeric = quad.eval(lambdify(x, f), [[-1, 1]])
                e = abs(exact - numeric)
                # Exit with first wrong degree
                if e > 1E-14:
                    break
            p_exact = p_degree - 1
            assert p_exact == 2*N - 3
