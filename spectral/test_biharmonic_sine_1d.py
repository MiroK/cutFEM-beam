from problems import manufacture_biharmonic_1d
from quadrature import errornorm, GLQuadrature, __EPS__
from functions import sine_basis
from biharmonic_solvers import solve_biharmonic_1d
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, latex, sin, pi, exp, cos
from math import sqrt, log as ln
import plots

result_dir = './results'
# Specs for problem
x = symbols('x')

f, test_spec = ((x-0.1)*(x-0.2)*exp(-x**2), 'test')

# Generate problem from specs
problem = manufacture_biharmonic_1d(f=f, a=0, b=1)
u = problem['u']
f = problem['f']
u_lambda = lambdify(x, u)
f_lambda = lambdify(x, f)

# Get quadrature for computing power spectrum
quad = GLQuadrature(62)
# Frequancies
ks = np.arange(1, 50, 1)
basis = sine_basis([ks])
basis = map(lambda f: lambdify(x, f), basis)

# u power spectrum
uk = np.array([sqrt(quad.eval(lambda x: u_lambda(x)*base(x), [[0, 1]])**2)
               for base in basis])
# Mask for zeros
mask = np.where(uk < __EPS__, True, False)

ks_ = np.ma.masked_array(ks, mask=mask)
uk_ = np.ma.masked_array(uk, mask=mask)

# plot for power spectrum
plt.figure()
plt.loglog(ks_, uk_, '*-')
plt.loglog(ks, 1./ks**5, '--', label='5')
plt.legend(loc='best')
plt.xlabel('$k$')
plt.ylabel('$F_k(u)$')
plt.savefig('%s/biharmonic_power_u_%s.pdf' % (result_dir, test_spec))

# f power spectrum
fk = np.array([sqrt(quad.eval(lambda x: f_lambda(x)*base(x), [[0, 1]])**2)
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
plt.legend(loc='best')
plt.savefig('%s/biharmonic_power_f_%s.pdf' % (result_dir, test_spec))

# Frequencies for solver in convergence test
Ns = np.arange(1, 25, 1)
eL2s = []
eH20s = []
for N in Ns:
    U, basis = solve_biharmonic_1d(f, N=N, a=0, b=1)
    eL2 = errornorm(u, (U, basis), norm_type='L2', domain=[[0, 1]])
    eH20 = errornorm(u, (U, basis), norm_type='H20', domain=[[0, 1]])
    eL2s.append(eL2)
    eH20s.append(eH20)

# Plot convergence
plt.figure()
plt.loglog(Ns, eL2s, label='$L^2$')
plt.loglog(Ns, eH20s, label='$H^2_0$')
plt.ylabel('$e$')
plt.xlabel('$N$')
plt.loglog(ks, 1./ks**2.5, '--', label='2.5')
plt.loglog(ks, 1./ks**4.5, '--', label='4.5')
plt.legend(loc='best')
plt.savefig('%s/biharmonic_convergence_%s.pdf' % (result_dir, test_spec))

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
plt.savefig('%s/biharmonic_solution_%s.pdf' % (result_dir, test_spec))

# Lets print the rates
for i in range(1, len(Ns)):
    N, N_ = Ns[i], Ns[i-1]
    eL2, eL2_ = eL2s[i], eL2s[i-1]
    eH20, eH20_ = eH20s[i], eH20s[i-1]

    rate_L2 = ln(eL2/eL2_)/ln(float(N_)/N)
    rate_H20 = ln(eH20/eH20_)/ln(float(N_)/N)

    print 'N=%s, L2->%.2f H20->%.2f' % (N, rate_L2, rate_H20)

print 'u', latex(u)
print 'f', latex(f)
