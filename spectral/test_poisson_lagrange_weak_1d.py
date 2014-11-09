from problems import manufacture_poisson_1d
from quadrature import errornorm
from cux import solve_lagrange_weak_1d
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
import matplotlib.pyplot as plt
from sympy import exp, symbols, lambdify
from math import log as ln
from points import gauss_legendre_points as gl_points
from quadrature import GLQuadrature
import time

result_dir = './results'
# Specs for problem
x = symbols('x')

# Dicide Nitsche
formulation = 'babuska'

u, test_spec = (exp(x)*x*(x-1), 'test_%s' % formulation)

# Generate problem from specs
problem = manufacture_poisson_1d(u=u, a=0, b=1)
u = problem['u']
f = problem['f']
u_lambda = lambdify(x, u)
f_lambda = lambdify(x, f)

Ns = np.arange(3, 20, 1)
eL2s = []
eH10s = []
for i, N in enumerate(Ns):
    start = time.time()
    U, basis = solve_lagrange_weak_1d(f, N=N, a=0, b=1, E=1,
                                      points=gl_points,
                                      quadrature=GLQuadrature,
                                      formulation=formulation)
    stop = time.time() - start
    eL2 = errornorm(u, (U, basis), norm_type='L2', domain=[[0, 1]])
    eH10 = errornorm(u, (U, basis), norm_type='H10', domain=[[0, 1]])
    eL2s.append(eL2)
    eH10s.append(eH10)

    if i > 0:
        rate_L2 = ln(eL2/_errorL2)/ln(float(_N)/N)
        rate_H10 = ln(eH10/_errorH10)/ln(float(_N)/N)
        print '-'*79
        print '\t Solved with %dx%d | %g%% completed' % (N, N,
                                                         100.*(i+1)/len(Ns))
        print '\t Error L2=%.4E, Error H10=%.4E' % (eL2, eH10)
        print '\t Rate L2=%.2f, Rate H10=%.2f' % (rate_L2, rate_H10)
        print '\t Assembly + inverse', stop
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
plt.savefig('%s/lagrange_poisson_weak_convergence_%s.pdf' % (result_dir,
                                                             test_spec))

# Plot final solution
uh = sum(Ui*base for (Ui, base) in zip(U.flatten(), basis.flatten()))
uh_lambda = lambdify(x, uh)

t = np.linspace(0, 1, 100)
u_values = np.array([u_lambda(ti) for ti in t])
uh_values = np.array([uh_lambda(ti) for ti in t])
plt.figure()
plt.plot(t, u_values, label='$u$')
plt.plot(t, uh_values, label='$u_h$')
plt.xlabel('$x$')
plt.legend(loc='best')
plt.savefig('%s/lagrange_poisson_weak_solution_%s.pdf' % (result_dir,
                                                          test_spec))

# Lets print the rates
for i in range(1, len(Ns)):
    N, N_ = Ns[i], Ns[i-1]
    eL2, eL2_ = eL2s[i], eL2s[i-1]
    eH10, eH10_ = eH10s[i], eH10s[i-1]

    rate_L2 = ln(eL2/eL2_)/ln(float(N_)/N)
    rate_H10 = ln(eH10/eH10_)/ln(float(N_)/N)

    print 'N=%s, L2->%.2f H10->%.2f' % (N, rate_L2, rate_H10)
