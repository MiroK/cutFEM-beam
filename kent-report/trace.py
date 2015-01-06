from __future__ import division
from sympy import lambdify, symbols
from sympy.mpmath import quad
from itertools import product
import numpy as np
import numpy.linalg as la

'''
Let \mathcal{P} = [-1, 1]^2 be a plate. Given points A, B on the boundary of
\mathcal{P} let chi, chi : [-1, 1] -> \mathcal{P} and
\mathcal{B} = {(x, y) \in \mathcal{P}, (x, y) = chi(s), x in [-1, 1]}. This set
is the beam. Further let V_P and V_B be function spaces defined over the plate
and the beam respectively.

We consider a mapping T, T: V_p -> V_b which should behave as a 'restriction',
(the trace operator). For numerical computations, we approximate V_b with n
basis functions. The space V_p is approximated m functions, where m = m1 * m2,
m_i being respectively the number of basis functions in x and y direction. We
write V_p = V_p1 x V_p2 (tensor product).

...

We get a linear system R * U = Mb * V. Here Mb is the mass matrix of V_b and
R is m x n matrix, R_{k, j} = (phi_k|b, psi_j)_b with phi_k|b the basis foos
of V_p restricted to beam and psi_j basis foos of V_b and (., .)_b the inner
product over beam.

...

Here we want to get R, M_b, M_p, T for any basis.
'''

x, y, s = symbols('x, y, s')

def on_boundary(V):
    'Check if point V lies on the plate boundary'
    return (np.allclose(V[0], -1) or np.allclose(V[0], 1)) and 
            (np.allclose(V[1], -1) or np.allclose(V[1], 1))

class Beam(object):
    'Beam is a segment defined by A and B on the plate boundary'
    def __init__(self, A, B):
        if isinstance(A, list):
            A = np.array(A)

        if isinstance(B, list):
            B = np.array(B)

        assert on_boundary(A) and on_boundary(B)
        
        # Beam length
        self.L = np.hypot(*(A - B))
        # Beam parametrization [-1, 1] -> beam coordinates
        self.chi = (A[0]/2*(1 - s) + B[0]/2*(1 + s),
                    A[1]/2*(1 - s) + B[1]/2*(1 + s))

    def inner_product(self, u, v):
        '''
        Inner product over beam. Functions must be symbolic functions of
        parameter s
        '''
        assert s in u.atoms() and s in v.atoms()
        u = lambdify(s, u)
        v = lambdify(s, v)
        return quad(lambda s: u(s)*v(s), [-1, 1])/2/self.L

    def restrict(self, u):
        'Restrict function of x, y to beam, i.e. make it only function of s'
        assert x in u.atoms() and y in u.atoms()
        return u.subs({x: self.chi[0], y: self.chi[1]})


class TraceOperator(object):
    '''
    All that is needed to compute with trace operator between V_p, V_b'
    '''
    def __init__(self, V_p, V_b, beam):
        # Need list with basis of x and y dirs
        assert isinstance(V_p, list) and len(V_p) == 2
        self.V_p = V_p
        m1, m2 = map(len, V_p)
        self.m = self.m1*self.m2

        self.V_b = V_b
        self.n = len(self.V_b)

        assert isinstance(beam, Beam)
        self.beam = beam

    def Mp_matrix(self):
        'Compute the mass matrix of V_p'
        # Matrices for directions
        Ms = []

        for basis in V_p:
            mi = len(basis)
            M = np.zeros((self.mi, self.mi))
            for i, u in enumerate(basis):
                M[i, i] = quad(lambda x, y: u(x, y)**2, [-1, 1], [-1, 1])
                for j, v in enumerate(basis[(i+1):], i+1):
                    M[i, j] = quad(lambda x, y: u(x, y)*v(x, y),
                                   [-1, 1], [-1, 1])
                    M[j, i] = M[i, j]

            Ms.append(M)

        Mp = np.outer(*Ms)
        return Mb

    def Mb_matrix(self):
        'Compute the mass matrix of V_b'
        basis = self.V_b
        beam = self.beam
        
        Mb = np.zeros((self.n, self.n))
        for i, u in enumerate(basis):
            Mb[i, i] = beam.inner_product(u, u)
            for j, v in enumerate(basis[(i+1):], i+1):
                Mb[i, j] = beam.inner_product(u, v)
                Mb[j, i] = Mb[i, j]
        
        return Mb

    def R_matrix(self):
        'Restriction matrix from V_p to V_b'
        beam = self.beam
        R = np.zeros((self.m, self.n))
        for k, (u0, u1) in enumerate(product(*self.V_p)):
            u = u0*(u1.subs(x, y))
            u_b = beam.restriction(u)
            for j, v in enumerate(self.V_b):
                R[k, j] = beam.inner_product(u_b, v)

        assert R.shape == (self.m, self.n)
        return R

    def T_matrix(self):
        'Matrix of the trace operator'
        Mb = self.Mb_matrix()
        R = self.R_matrix()
        T = la.inv(Mb).dot(R)

        return T

# ----------------------------------------------------------------------------- 
