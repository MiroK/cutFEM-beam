from problems import manufacture_poisson_1d
from quadrature import errornorm_1d, GLQuadrature_1d, __EPS__
from polynomials import sine_basis
from poisson_solvers import solve_sine_1d
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
import matplotlib.pyplot as plt
from sympy import exp, symbols, lambdify, latex
from math import sqrt, log as ln

result_dir = './results'
# Specs for problem
x = symbols('x')

u, test_spec = (exp(x)*x*(x-1), 'test')
# u, test_spec = (exp(-x**2)*sin(pi*x)*exp(sin(pi*x**2)), 'rough')
# u, test_spec = (exp(-x**2)*x*(1-x)*exp(sin(pi*x)), 'rougher')
# u, test_spec = (x*(1-x), 'polyn')
# f, test_spec = (Piecewise((1, x < 0.5), (2*x, True)), 'f_kink')

# Generate problem from specs
problem = manufacture_poisson_1d(u=u, a=0, b=1)
u = problem['u']
f = problem['f']
u_lambda = lambdify(x, u)
f_lambda = lambdify(x, f)

# Get quadrature for computing power spectrum
quad = GLQuadrature_1d(62)
# Frequancies
ks = np.arange(1, 50, 1)
basis = sine_basis(ks)
basis = map(lambda f: lambdify(x, f), basis)

# u power spectrum
uk = np.array([sqrt(quad.eval(lambda x: u_lambda(x)*base(x), 0, 1)**2)
               for base in basis])
# Mask for zeros
mask = np.where(uk < __EPS__, True, False)

ks_ = np.ma.masked_array(ks, mask=mask)
uk_ = np.ma.masked_array(uk, mask=mask)

# plot for power spectrum
plt.figure()
plt.loglog(ks_, uk_, '*-')
plt.loglog(ks, 1./ks, '--', label='1')
plt.loglog(ks, 1./ks**2, '--', label='2')
plt.loglog(ks, 1./ks**3, '--', label='3')
plt.legend(loc='best')
plt.xlabel('$k$')
plt.ylabel('$F_k(u)$')
plt.savefig('%s/poisson_power_u_%s.pdf' % (result_dir, test_spec))

# f power spectrum
fk = np.array([sqrt(quad.eval(lambda x: f_lambda(x)*base(x), 0, 1)**2)
               for base in basis])
# Mask zeros
mask = np.where(fk < __EPS__, True, False)
ks_ = np.ma.masked_array(ks, mask=mask)
fk_ = np.ma.masked_array(fk, mask=mask)

# Plot f power spectrum
plt.figure()
plt.loglog(ks_, fk_, '*-')
plt.xlabel('$k$')
plt.ylabel('$F_k(f)$')
plt.loglog(ks, 1./ks, '--', label='1')
plt.loglog(ks, 1./ks**2, '--', label='2')
plt.loglog(ks, 1./ks**3, '--', label='3')
plt.legend(loc='best')
plt.savefig('%s/poisson_power_f_%s.pdf' % (result_dir, test_spec))

# Frequencies for solver in convergence test
Ns = np.arange(1, 25, 1)
eL2s = []
eH10s = []
for N in Ns:
    U, basis = solve_sine_1d(f, N=N, a=0, b=1)
    uh = sum(Ui*base for (Ui, base) in zip(U, basis))
    uh_lambda = lambdify(x, uh)

    eL2 = errornorm_1d(u, (U, basis), 'L2', a=0, b=1)
    eH10 = errornorm_1d(u, (U, basis), 'H10', a=0, b=1)
    eL2s.append(eL2)
    eH10s.append(eH10)

# Plot convergence
plt.figure()
plt.loglog(Ns, eL2s, label='$L^2$')
plt.loglog(Ns, eH10s, label='$H^1_0$')
plt.ylabel('$e$')
plt.xlabel(r'dim $V_h$')
plt.loglog(ks, 1./ks**1.5, '--', label='1.5')
plt.loglog(ks, 1./ks**2.5, '--', label='2.5')
plt.legend(loc='best')
plt.savefig('%s/poisson_convergence_%s.pdf' % (result_dir, test_spec))

# Plot final solution
t = np.linspace(0, 1, 100)
u_values = np.array([u_lambda(ti) for ti in t])
uh_values = np.array([uh_lambda(ti) for ti in t])
plt.figure()
plt.plot(t, u_values, label='$u$')
plt.plot(t, uh_values, label='$u_h$')
plt.xlabel('$x$')
plt.legend(loc='best')
plt.savefig('%s/poisson_solution_%s.pdf' % (result_dir, test_spec))

# Lets print the rates
for i in range(10, len(Ns), 2):
    N, N_ = Ns[i], Ns[i-1]
    eL2, eL2_ = eL2s[i], eL2s[i-1]
    eH10, eH10_ = eH10s[i], eH10s[i-1]

    rate_L2 = ln(eL2/eL2_)/ln(float(N_)/N)
    rate_H10 = ln(eH10/eH10_)/ln(float(N_)/N)

    print 'N=%s, L2->%.2f H10->%.2f' % (N, rate_L2, rate_H10)

print 'u', latex(u)
print 'f', latex(f)
