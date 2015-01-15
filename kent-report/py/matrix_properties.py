from __future__ import division
import eigen_poisson as eigen
import shen_poisson as shen
from eigen_basis import eigen_basis
from shenp_basis import shenp_basis
from sympy import Symbol, lambdify, exp
from sympy.mpmath import quad
import numpy as np
import numpy.linalg as la
from math import log as ln

def get_rate(ns, conds):
    'Rates of the condition number.'
    rates = []
    n0, cond0 = ns[0], conds[0]
    for n, cond in zip(ns[1:], conds[1:]):
        rate = ln(cond/cond0)/ln(n0/n)
        rates.append(rate)

        n0, cond0 = n, cond
    return rates

def F_matrix(m, n):
    '''
    Matric that takes eigen basis of lenght n to shen_basis of lenght m.
    It is an (m, n) matrix.
    '''
    x = Symbol('x')
    s_basis = [lambdify(x, f) for f in shenp_basis(m)]
    e_basis = [lambdify(x, f) for f in eigen_basis(m)]
    F = np.zeros((m, n))
    for i, v in enumerate(s_basis):
        for j, u in enumerate(e_basis):
            F[i, j], error = quad(lambda x: u(x)*v(x), [-1, 1], error=True)

    return F

ns = np.arange(2, 33)

# Condition number of M eigen
cond_eM = np.array([la.cond(eigen.mass_matrix(n)) for n in ns])
rate_eM = get_rate(ns, cond_eM)

# Condition number of A eigen
cond_eA = np.array([la.cond(eigen.laplacian_matrix(n)) for n in ns])
rate_eA = get_rate(ns, cond_eA)

# Condition number of M shen
cond_sM = np.array([la.cond(shen.mass_matrix(n)) for n in ns])
rate_sM = get_rate(ns, cond_sM)

# Condition number of A shen
cond_sA = np.array([la.cond(shen.laplacian_matrix(n)) for n in ns])
rate_sA = get_rate(ns, cond_sA)

m = ns[-1]
F = F_matrix(m, m)
cond_F = np.array([la.cond(F[:n, :n]) for n in ns])
rate_F = get_rate(ns, cond_F)

# Now see for all l_i how well they get approximated by more eigen for growing n
x = Symbol('x')
s_basis = [v for v in shenp_basis(m)]
e_basis = [v for v in eigen_basis(m)]
error_matrix = np.zeros((len(ns), len(s_basis)))
for row, n in enumerate(ns):
    for i, si in enumerate(s_basis):
        coefs = F[i, :]
        # Assemble approximation
        si_ = sum(c_j*u_j for c_j, u_j in zip(coefs[:], e_basis[:n]))
        # Symbolic error
        e = si - si_
        # Error in L2
        e = lambdify(x, e)
        error = quad(lambda x: e(x)**2, [-1, 1])
        error_matrix[row, i] = float(error)

print rate_F
for col in error_matrix.T:
    print col
    print get_rate(ns, col)
    print 

