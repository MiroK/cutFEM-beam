from __future__ import division
from sympy import lambdify, symbols, sqrt
from sympy.mpmath import quad
from itertools import product
from plate_beam import Beam
import numpy as np
import numpy.linalg as la

'''
Given a mapping \chi, \chi : [-1, 1] -> [-1, 1]^d we call \mathcal{B},
(generalized) beam, a set of points x in [-1, 1]^d, (generalized) plate, that
are such that x = \chi(s) for s \in [-1, 1]. Further let V_p and \tilde{V_b} be
function spaces defined over the plate and the beam respectively. That is
u \in V_p def_if u: \mathcal{P} -> R , (We only consider scalar function spaces)
and \tilde{v} in \tilde{V_b} def_if 
\tilde{v}: \mathcal{V} -> R. Now each \tilde{v} can be thought of as a composi-
tion of v, v: [-1, 1] -> R and inv(chi): P --> [-1, 1], i.e 

    \tilde{v}(x) = (v o inv(chi))(x).

Finally, consider function space V_b with v in V_b def_iff v, v: [-1, 1] -> R.
We write \tilde{V_b} = V_b o inv(chi).

We consider a mapping T, T: V_p -> V_b which should behave as a 'restriction',
(the trace operator). For numerical computations, we approximate V_b with n
basis functions. The space V_p is approximated m functions, where m = \Pi m_i
m_i being respectively the number of basis functions in i-th direction. We have
ms d-tuple of mi. The space is V_p = (tensor product)V_pi.
...
We get a linear system R * U = Mb * V. Here Mb is the mass matrix of V_b and
R is m x n matrix, R_{k, j} = (phi_k|b, psi_j)_b with phi_k|b the basis foos
of V_p restricted to beam and psi_j basis foos of V_b and (., .)_b the inner
product over beam.
...
Here we want to get R, M_b, M_p, T for any basis and d = 1, 2, 3.
'''

# Plate variables
xyz = symbols('x, y, z')
x, y, z = xyz
# Beam variables
s = symbols('s')


class TraceOperator(object):
    '''
    All that is needed to compute with trace operator between V_p, V_b'
    '''
    def __init__(self, V_p, V_b, beam):
        # Get the geometry from the beam
        assert isinstance(beam, Beam)
        d = beam.d
        assert d in (1, 2, 3)
        self.d = d
        self.beam = beam
        
        # Plate function space must be list with comps of basis for each dir
        assert isinstance(V_p, list) and len(V_p) == d
        self.V_p = V_p
        ms = map(len, V_p)
        self.m = np.prod(ms)

        self.V_b = V_b
        self.n = len(self.V_b)

    def Mp_matrix(self):
        'Compute the mass matrix of V_p'
        # Matrices for directions
        Ms = []

        for V_pi in self.V_p:
            basis = [lambdify(x, v) for v in V_pi]
            mi = len(V_pi)
            M = np.zeros((mi, mi), dtype='float')
            for i, u in enumerate(basis):
                M[i, i] = quad(lambda x: u(x)**2, [-1, 1])
                for j, v in enumerate(basis[(i+1):], i+1):
                    M[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
                    M[j, i] = M[i, j]

            Ms.append(M)

        if self.d == 1:
            return Ms[0]
        else:
            Mb = np.kron(Ms[0], Ms[1])
            for i in range(2, self.d):
                Mb = np.kron(Mb, Ms[i])

            return Mb

    def Mb_matrix(self):
        'Compute the mass matrix of V_b'
        basis = self.V_b
        beam = self.beam
        
        Mb = np.zeros((self.n, self.n), dtype='float')
        for i, u in enumerate(basis):
            Mb[i, i] = beam.inner_product(u, u)
            for j, v in enumerate(basis[(i+1):], i+1):
                Mb[i, j] = beam.inner_product(u, v)
                Mb[j, i] = Mb[i, j]
        
        return Mb

    def R_matrix(self):
        'Restriction matrix from V_p to V_b'
        beam = self.beam
        Rt = np.zeros((self.m, self.n), dtype='float')
        for k, us in enumerate(product(*self.V_p)):
            # Perform the substitution
            us_sub = (us[i].subs(x, var) for i, var in enumerate(xyz[:self.d]))
            # Combine intro product
            u = reduce(lambda ui, uj: ui*uj, us_sub)
            # Restrict
            u_b = beam.restrict(u)
            for j, v in enumerate(self.V_b):
                Rt[k, j] = beam.inner_product(u_b, v)

        assert Rt.shape == (self.m, self.n)
        return Rt.T

    def T_matrix(self):
        'Matrix of the trace operator'
        Mb = self.Mb_matrix()
        R = self.R_matrix()
        T = la.inv(Mb).dot(R)

        return T

# ----------------------------------------------------------------------------- 

if __name__ == '__main__':
    # Test here for some functions the generic R and T matrices against
    # the hand computation
    from sympy import sin, cos, legendre, symbols, pi, lambdify
    from sympy.mpmath import quad
    from plate_beam import LineBeam

    # Some basis, not orthogonal or anything
    x, y, s = symbols('x, y, s')
    basis = [sin(pi*x), cos(pi*x), legendre(2, x), legendre(3, x)]

    # Basis for trace operator
    # Plate
    V_p0 = basis[:3]
    V_p1 = basis[:3]
    V_p = [V_p0, V_p1]
    # Beam
    V_b = [v.subs(x, s) for v in basis]

    # Own basis
    # Plate
    p_basis = []
    for u0 in V_p0:
        for u1 in V_p1:
            u = u0*(u1.subs(x, y))
            p_basis.append(u)

    b_basis = [v for v in V_b]

    # Now we need a beam
    A = np.array([-1, 0])
    B = np.array([1, 1])
    beam = LineBeam(A, B)

    # Trace opeartor
    to = TraceOperator(V_p, V_b, beam)

    # Hand beam mass matrix
    m = len(b_basis)
    MM = np.zeros((m, m))
    for i, u in enumerate(b_basis):
        for j, v in enumerate(b_basis):
            MM[i, j] = beam.inner_product(u, v)
    # Mb from operator
    Mb = to.Mb_matrix()
    assert np.allclose(Mb-MM, np.zeros_like(Mb))

    # Hand plate mass matrix
    pp_basis = [lambdify([x, y], v) for v in p_basis]
    n = len(p_basis)
    MM = np.zeros((n, n))
    for i, u in enumerate(pp_basis):
        for j, v in enumerate(pp_basis):
            MM[i, j] = quad(lambda x, y: u(x, y)*v(x, y), [-1, 1], [-1, 1])
    # Mb from operator
    M = to.Mp_matrix()
    assert np.allclose(M-MM, np.zeros_like(M))

    # Restriction and T by hand
    chi0 = A[0]*(1-s)/2 + B[0]*(1+s)/2
    chi1 = A[1]*(1-s)/2 + B[1]*(1+s)/2

    RR = np.zeros((m, n))
    for i, v in enumerate(b_basis):
        for j, u in enumerate(p_basis):
            ub = u.subs({x: chi0, y: chi1})
            RR[i, j] = beam.inner_product(v, ub)
    
    # R from operator
    R = to.R_matrix()
    assert np.allclose(R - RR, np.zeros_like(R))

    # T hand and oper.
    TT = la.inv(Mb).dot(RR)
    T = to.T_matrix()
    assert np.allclose(TT - T, np.zeros_like(T))
