from math import sin, cos, pi, sqrt, log as ln
import numpy.linalg as la
import numpy as np
import sympy as S
from sympy.integrals import quadrature
import pickle
import time
import os

__EPS__ = np.finfo(float).eps


class GLQuadrature(object):
    'Generate Gauss-Legendre points for computing integral [-1, 1].'
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
        return '%d-point Gauss-Legendre quadrature' % self.N

    def eval(self, f, a, b):
        'Compute integral[a, b] f(x) dx.'
        assert b > a

        # Build map for going from [a, b] to [-1, 1]
        def pull_back(z):
            return 0.5*a*(1-z) + 0.5*b*(1+z)
        J = 0.5*(b - a)   # Jacobian of pull back

        return J*sum(wi*f(pull_back(zi)) for zi, wi in zip(self.z, self.w))

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


def errornorm(u, (U, basis), norm_type, a=0, b=1):
    N = len(U)
    quad = GLQuadrature(2*N)

    x = symbols('x')
    uh = sum(Ui*base for (Ui, base) in zip(U, basis))

    if norm_type == 'L2':
        f = (u - uh)**2
    elif norm_type == 'H10':
        f = (S.diff(u, x, 1) - S.diff(uh, x, 1))**2
    elif norm_type == 'H20':
        f = (S.diff(u, x, 2) - S.diff(uh, x, 2))**2

    f_lambda = S.lambdify(x, f)
    norm = quad.eval_adapt(f_lambda, a, b, n_refs=10)

    if norm < 0:
        return 1E-16
    else:
        return sqrt(norm)


def sine_basis(N, a, b):
    x = symbols('x')
    try:
        return [S.sin(k*S.pi*(x-a)/(b - a))*S.sqrt(2)/(pi*k) for k in N]
    except TypeError:
        return sine_basis(range(1, N), a, b)


def solve(f, N, a=0, b=1, eps=__EPS__, n_refs=10):
    '''
    Solve Poisson problem
                         -u^(2) = f in [a, b]
                             u = 0 on a, b

    In the variational formulation use N basis function of type sin(i*pi*x)
    for i = 1, ..., N.
    '''
    assert b > a

    AA = np.zeros((N, N))

    # Assemble matrix
    time_AA = time.time()

    AA = np.diag([1./(b-a) for l in range(1, N+1)])

    time_AA = time.time() - time_AA

    x = S.Symbol('x')

    # Make f for fast evaluation
    f_lambda = S.lambdify(x, f)

    # Make symbolic basis
    basis = sine_basis(N+1, a, b)

    # Make basis functions on fast evaluation
    basis_lambda = map(lambda f: S.lambdify(x, f), basis)

    # Get the quadrature for computing integrals
    quad = GLQuadrature(2*N)

    bb = np.zeros(N)

    # Assemble vector, this is done multiple times until either eps is reached
    # in bb difference or n_refs is exceeded
    time_bb = time.time()

    for j, base_lambda in enumerate(basis_lambda):
        bb[j] = quad.eval(lambda x: base_lambda(x)*f_lambda(x), a=a, b=b)

    bb_norm_ = la.norm(bb)
    diff = 1
    ref = 0
    while diff > eps and ref < n_refs:
        ref += 1

        new_N = quad.N + 1
        quad.__init__(new_N)

        for j, base_lambda in enumerate(basis_lambda):
            bb[j] = quad.eval(lambda x: base_lambda(x)*f_lambda(x), a=a, b=b)

        bb_norm = la.norm(bb)
        diff = abs(bb_norm - bb_norm_)
        bb_norm_ = bb_norm

    print 'Assemble vector, final diff', diff
    time_bb = time.time() - time_bb

    # Vector of exp. coeffs
    time_solve = time.time()
    U = la.solve(AA, bb)
    time_solve = time.time() - time_solve

    print 'Assembling matrix:', time_AA
    print 'Assembling vector:', time_bb
    print 'Solve linear system:', time_solve

    cond_AA = la.cond(AA)
    return U, basis, cond_AA

if __name__ == '__main__':
    from sympy import symbols, sin, exp, lambdify, pi, diff
    import matplotlib.pyplot as plt
    from qux import manufacture_poisson

    x = symbols('x')

    a = 0
    b = 1
    u = exp(x)*x*(1-x)                     # smooth power spectrum
    # u = exp(-x**2)*x*(1-x)*exp(sin(pi*x))  # rough power spectrum
    problem = manufacture_poisson(u=u, a=a, b=b)
    f = problem['f']
    u_lambda = lambdify(x, u)
    f_lambda = lambdify(x, f)

    quad = GLQuadrature(62)

    ks = np.arange(1, 50, 1)
    slope_1 = 1./ks
    slope_2 = 1./ks**2
    slope_3 = 1./ks**3
    basis = sine_basis(ks, a, b)
    basis = map(lambda f: lambdify(x, f), basis)

    # u power spectrum
    uk = np.array([sqrt(quad.eval(lambda x: u_lambda(x)*base(x), a, b)**2)
                   for base in basis])
    mask = np.where(uk < __EPS__, True, False)

    ks_ = np.ma.masked_array(ks, mask=mask)
    uk_ = np.ma.masked_array(uk, mask=mask)

    plt.figure()
    plt.loglog(ks_, uk_, '*-')
    plt.loglog(ks, slope_1, '--', label='1')
    plt.loglog(ks, slope_2, '--', label='2')
    plt.loglog(ks, slope_3, '--', label='3')
    plt.legend(loc='best')
    plt.xlabel('$k$')
    plt.ylabel('$F_k(u)$')

    # f power spectrum
    fk = np.array([sqrt(quad.eval(lambda x: f_lambda(x)*base(x), a, b)**2)
                   for base in basis])
    mask = np.where(fk < __EPS__, True, False)

    ks_ = np.ma.masked_array(ks, mask=mask)
    fk_ = np.ma.masked_array(fk, mask=mask)

    plt.figure()
    plt.loglog(ks_, fk_, '*-')
    plt.xlabel('$k$')
    plt.ylabel('$F_k(f)$')
    plt.loglog(ks, slope_1, '--', label='1')
    plt.loglog(ks, slope_2, '--', label='2')
    plt.loglog(ks, slope_3, '--', label='3')
    plt.legend(loc='best')

    # F = integrate(f, (x, 0, 1)).evalf()

    # f_lambda = lambdify(x, f)

    # quad = GLQuadrature(2)

    # print quad.eval(f_lambda, 0, 1)

    # F_ = quad.eval_adapt(f_lambda, 0, 1)
    # print quad
    # print abs(F - F_)

    # F_ = quad.eval_adapt(f_lambda, 0, 1, n_refs=20)
    # print quad
    # print abs(F - F_)

    Ns = np.arange(1, 25, 1)
    eL2s = []
    eH10s = []
    conds_A = []
    for N in Ns:
        U, basis, cond_A = solve(f, N=N, a=a, b=b)
        uh = sum(Ui*base for (Ui, base) in zip(U, basis))
        uh_lambda = S.lambdify(x, uh)

        eL2 = errornorm(u, (U, basis), 'L2', a=a, b=b)
        eH10 = errornorm(u, (U, basis), 'H10', a=a, b=b)
        eL2s.append(eL2)
        eH10s.append(eH10)

        conds_A.append(cond_A)

    # Plot convergence
    plt.figure()
    plt.loglog(Ns, eL2s, label='$L^2$')
    plt.loglog(Ns, eH10s, label='$H^1_0$')
    plt.ylabel('$e$')
    plt.xlabel('$N$')
    plt.loglog(ks, 1./ks**1.5, '--', label='1.5')
    plt.loglog(ks, 1./ks**2.5, '--', label='2.5')
    plt.legend(loc='best')

    # Plot condition numbers
    plt.figure()
    plt.loglog(Ns, conds_A)
    plt.xlabel('$N$')
    plt.ylabel('$\kappa(A)$')

    # Plot final solution
    t = np.linspace(a, b, 100)
    u_values = np.array([u_lambda(ti) for ti in t])
    uh_values = np.array([uh_lambda(ti) for ti in t])
    plt.figure()
    plt.plot(t, u_values, label='$u$')
    plt.plot(t, uh_values, label='$u_h$')
    plt.xlabel('$x$')
    plt.legend(loc='best')

    plt.show()

    # Lets print the rates
    for i in range(10, len(Ns), 2):
        N, N_ = Ns[i], Ns[i-1]
        eL2, eL2_ = eL2s[i], eL2s[i-1]
        eH10, eH10_ = eH10s[i], eH10s[i-1]

        rate_L2 = ln(eL2/eL2_)/ln(float(N_)/N)
        rate_H10 = ln(eH10/eH10_)/ln(float(N_)/N)

        print 'N=%s, L2->%.2f H10->%.2f' % (N, rate_L2, rate_H10)

