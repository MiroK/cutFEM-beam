from problems import manufacture_poisson_2d
from quadrature import errornorm
from bar import solve_lagrange_2d
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
import matplotlib.pyplot as plt
from sympy import exp, symbols, lambdify, pi, sin
from math import log as ln
from points import gauss_legendre_lobatto_points as gll_points
from quadrature import GLLQuadrature
import plots
import time

result_dir = './results'
# Specs for problem
x, y = symbols('x, y')

u, test_spec = (exp(x)*x*(x-1)*sin(pi*y), 'test2d')
domain = [[0, 1], [0, 1]]
E = 2

# Generate problem from specs
problem = manufacture_poisson_2d(u=u, E=2, domain=domain)
u = problem['u']
f = problem['f']
u_lambda = lambdify(x, u)
f_lambda = lambdify(x, f)

# Frequencies for solver in convergence test
Ns = np.arange(2, 10, 1)
eL2s = []
eH10s = []
for i, N in enumerate(Ns):
    start = time.time()
    U, basis = solve_lagrange_2d(f, E=E, domain=domain, MN=[N, N],
                                 points=gll_points,
                                 quadrature=GLLQuadrature,
                                 method='operator')
    stop_solve = time.time() - start

    start = time.time()
    eL2 = errornorm(u, (U, basis), norm_type='L2', domain=domain)
    eH10 = errornorm(u, (U, basis), norm_type='H10', domain=domain)
    eL2s.append(eL2)
    eH10s.append(eH10)
    stop_error = time.time() - start

    if i > 0:
        rate_L2 = ln(eL2/_errorL2)/ln(float(_N)/N)
        rate_H10 = ln(eH10/_errorH10)/ln(float(_N)/N)
        print '-'*79
        print '\t Solved with %dx%d | %g%% completed' % (N, N,
                                                         100.*(i+1)/len(Ns))
        print '\t Error L2=%.4E, Error H10=%.4E' % (eL2, eH10)
        print '\t Rate L2=%.2f, Rate H10=%.2f' % (rate_L2, rate_H10)
        print '\t Assembly + inverse', stop_solve
        print '\t Error computation', stop_error
        print '-'*79

    # Always remeber
    _N, _errorL2, _errorH10 = N, eL2, eH10

# Plot convergence
plt.figure()
plt.loglog(Ns, eL2s, label='$L^2$')
plt.loglog(Ns, eH10s, label='$H^1_0$')
plt.ylabel('$e$')
plt.xlabel(r'$N$')
plt.legend(loc='best')
plt.savefig('%s/lagrange_poisson_convergence_%s.pdf' % (result_dir, test_spec))

# Plot final solution
plot_u = plots.plot(u, domain, title='$u$', xlabel='$x$', ylabel='$y$',
                    show=False)
plot_uh = plots.plot((U, basis), domain, title='$u$', xlabel='$x$',
                     ylabel='$y$', show=False)

plot_u.save('%s/lagrange_poisson_u_%s.pdf' % (result_dir, test_spec))
plot_uh.save('%s/lagrange_poisson_u_%s.pdf' % (result_dir, test_spec))

# Lets print the rates
for i in range(1, len(Ns)):
    N, N_ = Ns[i], Ns[i-1]
    eL2, eL2_ = eL2s[i], eL2s[i-1]
    eH10, eH10_ = eH10s[i], eH10s[i-1]

    rate_L2 = ln(eL2/eL2_)/ln(float(N_)/N)
    rate_H10 = ln(eH10/eH10_)/ln(float(N_)/N)

    print 'N=%s, L2->%.2f H10->%.2f' % (N, rate_L2, rate_H10)
