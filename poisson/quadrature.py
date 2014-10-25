from sympy import symbols, diff, lambdify
from sympy.integrals import quadrature
from math import sqrt
import numpy as np
import pickle
import time
import os

__EPS__ = np.finfo(float).eps


class GLQuadrature_1d(object):
    'Generate Gauss-Legendre points for computing integral over [-1, 1].'
    def __init__(self, N):
        'Create quadrature with N points.'
        time_q = time.time()

        quad_name = '.quadrature_%d.pickle' % N

        # Try loading points that were already computed
        if os.path.exists(quad_name):
            z, w = pickle.load(open(quad_name, 'rb'))
        # Compute points, weights and dump for next time
        else:
            z, w = quadrature.gauss_legendre(N, n_digits=15)
            pickle.dump((z, w), open(quad_name, 'wb'))

        assert len(z) == N

        time_q = time.time() - time_q
        print 'Computing %d-point quadrature :' % N, time_q

        self.N = N
        self.z = z
        self.w = w

    def __str__(self):
        return '%d-point 1d Gauss-Legendre quadrature' % self.N

    def eval(self, f, a, b):
        'Compute integral[a, b] f(x) dx.'
        assert b > a

        def F(z):
            'Build map for going from [-1, 1] to [a, b].'
            return 0.5*a*(1-z) + 0.5*b*(1+z)

        J = 0.5*(b - a)   # Jacobian of pull back

        return J*sum(wi*f(F(zi)) for zi, wi in zip(self.z, self.w))

    def eval_adapt(self, f, a, b, eps=__EPS__, n_refs=5):
        '''
        Given starting(initialized) N-point quadrature, this function computes
        integral [a, b] f(x) dx using N and higher order quadratures until
        the difference is smaller then eps.
        '''
        diff = 1
        ref = 0
        result_N = self.eval(f, a, b)
        while diff > eps and ref < n_refs:
            ref += 1

            new_N = self.N + 1
            self.__init__(new_N)
            result_new_N = self.eval(f, a, b)

            diff = abs(result_new_N - result_N)
            result_N = result_new_N

        print 'eval_adapt final diff', diff
        return result_N


def errornorm_1d(u, (U, basis), norm_type, a, b):
    '''
    Compute error over [a, b] of u and uh. Numerical solution uh is defined
    as sum(Ui*basefor Ui, base in zip(U, basis)). Both functions should be
    defined over [a, b] (i.e.) no pullbacks will be done here. Further they
    should be symbolic.
    '''
    # Heuristic choice for quadrature order
    N = len(U)
    quad = GLQuadrature_1d(2*N)

    # Assemble uh as linear combination
    uh = sum(Ui*base for (Ui, base) in zip(U, basis))

    x = symbols('x')
    if norm_type == 'L2':
        f = (u - uh)**2
    elif norm_type == 'H10':
        f = (diff(u, x, 1) - diff(uh, x, 1))**2
    elif norm_type == 'H20':
        f = (diff(u, x, 2) - diff(uh, x, 2))**2

    # Make for fast evaluation
    f_lambda = lambdify(x, f)
    norm = quad.eval_adapt(f_lambda, a, b, n_refs=10)

    if norm < 0:
        return 1E-16
    else:
        return sqrt(norm)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import integrate, sin, exp

    x = symbols('x')
    f = x
    # Exact integral
    F = integrate(f, (x, 0, 1)).evalf()
    f_lambda = lambdify(x, f)
    quad = GLQuadrature_1d(2)

    # Integrate with given quadrature
    assert abs(quad.eval(f_lambda, 0, 1) - F) < 1E-15
    # Integrate adaptively
    assert abs(quad.eval_adapt(f_lambda, 0, 1) - F) < 1E-15
    # Let's see how much points we need for some harder function
    f = sin(x)*exp(x)
    f_lambda = lambdify(x, f)
    F_ = quad.eval_adapt(f_lambda, 0, 1, n_refs=20)
    F = integrate(f, (x, 0, 1)).evalf()
    assert abs(F - F_) < 1E-15

    u = sin(x)
    U = [1]
    basis = [sin(x)]
    error = errornorm_1d(u, (U, basis), a=-2, b=3, norm_type='L2')
    assert abs(error) < 1E-15

    # Verify the 2*n-1 accuracy
    for N in range(1, 60):
        quad = GLQuadrature_1d(N)
        quad_is_exact = True
        p = -1
        while quad_is_exact:
            p += 1
            f = x**p
            F = integrate(f, (x, 0, 1)).evalf()
            e = abs(quad.eval(lambdify(x, f), 0, 1) - F)
            quad_is_exact = e < 1E-15
            # print '\t %d -> %.2E' % (p, e)

        p_exact = p-1
        assert p_exact == p-1


