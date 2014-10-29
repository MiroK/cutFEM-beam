from problems import manufacture_poisson_2d
from quadrature import errornorm, GLQuadrature, __EPS__
from polynomials import sine_basis
from poisson_solvers import solve_sine_2d
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sympy import exp, cos, pi, symbols, lambdify, latex
from math import sqrt, log as ln
import pickle
import plots

result_dir = './results'
# Specs for problem
x, y = symbols('x, y')

u, test_spec = (y*(1-y)*x*(1-x)*cos(pi*(x**2 + y**2))*exp(-x-y), 'test_2d')

# Generate problem from specs
problem = manufacture_poisson_2d(u=u, domain=[[0, 1], [0, 1]])
u = problem['u']
f = problem['f']
u_lambda = lambdify([x, y], u)
f_lambda = lambdify([x, y], f)

# -------
solver_N_max = 3               # max frequency in solver
power_N_max = 10               # max frequency for power spectrum
plot_power = False              # plot the power spectrum
eps = 10*__EPS__               # tolerance for computing accuracy of rhs
# ------

# Frequencies for solver in convergence test
domain = [[0, 1], [0, 1]]
ns = np.arange(1, solver_N_max+1, 1)  # Freqencies for one direction
Ns = []                               # really freq. consider dim(V_h)
eL2s = []
eH10s = []
for i, N in enumerate(ns):
    U, basis = solve_sine_2d(f, MN=[N, N], domain=domain, eps=eps, n_refs=-1)
    eL2 = errornorm(u, (U, basis), norm_type='L2', domain=domain)
    eH10 = errornorm(u, (U, basis), norm_type='H10', domain=domain)
    eL2s.append(eL2)
    eH10s.append(eH10)
    Ns.append(N)

    if i > 0:
        rate_L2 = ln(eL2/_errorL2)/ln(float(_N)/N)
        rate_H10 = ln(eH10/_errorH10)/ln(float(_N)/N)
        print '-'*79
        print '\t Solved with %dx%d | %g%% completed' % (N, N,
                                                         100.*(i+1)/len(ns))
        print '\t Rate L2=%.2f, Rate H10=%.2f' % (rate_L2, rate_H10)
        print '-'*79

    # Always remeber
    _N, _errorL2, _errorH10 = N, eL2, eH10

# Save the data from convergence
pickle.dump(Ns, open('%s/poisson_%s_Ns.pickle' % (result_dir, test_spec), 'wb'))
pickle.dump(eL2s, open('%s/poisson_%s_eL2s.pickle' % (result_dir, test_spec), 'wb'))
pickle.dump(eH10s, open('%s/poisson_%s_eH10s.pickle' % (result_dir, test_spec), 'wb'))

# Plot convergence
Ns = np.array(Ns)
plt.figure()
plt.loglog(Ns, eL2s, label='$L^2$')
plt.loglog(Ns, eH10s, label='$H^1_0$')
plt.ylabel('$e$')
plt.xlabel(r'$N$')
plt.loglog(Ns, 1./Ns**1.5, '--', label='1.5')
plt.loglog(Ns, 1./Ns**2.5, '--', label='2.5')
plt.legend(loc='best')
plt.savefig('%s/poisson_convergence_%s.pdf' % (result_dir, test_spec))

# Plot final solution
p0 = plots.plot(u, domain=[[0, 1], [0, 1]],
                title='$u$', xlabel='$x$', ylabel='$y$', show=False)
p0.save('%s/poisson_u_%s.pdf' % (result_dir, test_spec))

p1 = plots.plot((U, basis), domain=[[0, 1], [0, 1]],
                title='$u_h$', xlabel='$x$', ylabel='$y$', show=False)
p1.save('%s/poisson_uh_%s.pdf' % (result_dir, test_spec))

# Lets print the rates
for i in range(1, len(Ns)):
    N, N_ = Ns[i], Ns[i-1]
    eL2, eL2_ = eL2s[i], eL2s[i-1]
    eH10, eH10_ = eH10s[i], eH10s[i-1]

    rate_L2 = ln(eL2/eL2_)/ln(float(N_)/N)
    rate_H10 = ln(eH10/eH10_)/ln(float(N_)/N)

    print 'N=%s, L2->%.2f H10->%.2f' % (N, rate_L2, rate_H10)

print 'u', latex(u)
print 'f', latex(f)

if plot_power:
    # Get quadrature for computing power spectrum
    # 2*N plus some margin, heuristics
    quad = GLQuadrature([2*power_N_max+5, 2*power_N_max+5])
    # Frequancies
    ks = np.arange(1, power_N_max, 1)
    ls = np.arange(1, power_N_max, 1)
    basis = sine_basis([ks, ls]).flatten()
    basis = map(lambda f: lambdify([x, y], f), basis)

    # u power spectrum
    ukl = np.array([sqrt(quad.eval(lambda x, y: u_lambda(x, y)*base(x, y),
                                   [[0, 1], [0, 1]])**2)
                    for base in basis])
    # Mask for zeros
    ukl = ukl.reshape((len(ks), len(ls)))
    mask = np.where(ukl < __EPS__, True, False)
    Ukl = np.ma.masked_array(ukl, mask=mask)

    K, L = np.meshgrid(ks, ls)
    K = np.ma.masked_array(K, mask=mask)
    L = np.ma.masked_array(L, mask=mask)

    # plot for power spectrum
    plt.figure()
    pc_u = plt.pcolor(K, L, Ukl, norm=LogNorm())
    plt.xscale('log')
    plt.yscale('log')
    co_u = plt.colorbar(pc_u)
    co_u.set_label(r'$F_{kl}(u)$')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$l$')
    plt.savefig('%s/poisson_power_u_%s.pdf' % (result_dir, test_spec))

    # u power spectrum
    fkl = np.array([sqrt(quad.eval(lambda x, y: f_lambda(x, y)*base(x, y),
                                   [[0, 1], [0, 1]])**2)
                    for base in basis])
    # Mask for zeros
    fkl = fkl.reshape((len(ks), len(ls)))
    mask = np.where(fkl < __EPS__, True, False)
    Fkl = np.ma.masked_array(fkl, mask=mask)

    K, L = np.meshgrid(ks, ls)
    K = np.ma.masked_array(K, mask=mask)
    L = np.ma.masked_array(L, mask=mask)

    # plot for power spectrum
    plt.figure()
    pc_f = plt.pcolor(K, L, Fkl, norm=LogNorm())
    plt.xscale('log')
    plt.yscale('log')
    co_f = plt.colorbar(pc_f)
    co_f.set_label(r'$F_{kl}(f)$')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$l$')
    plt.savefig('%s/poisson_power_f_%s.pdf' % (result_dir, test_spec))
