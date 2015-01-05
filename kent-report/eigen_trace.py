from __future__ import division
from sympy import lambdify, symbols
from sympy.mpmath import quad
from eigen_basis import eigen_basis
from itertools import product
import numpy as np
import numpy.linalg as la

x, y, s = symbols('x, y, s')

class Beam(object):
    'Beam is a segment defined by A and B'
    def __init__(self, A, B):
        # Beam length
        if isinstance(A, list):
            A = np.array(A)

        if isinstance(B, list):
            B = np.array(B)

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
        return 2*quad(lambda s: u(s)*v(s), [-1, 1])/self.L

    def restrict(self, u):
        'Restrict function of x, y to beam, i.e. make it only function of s.'
        assert x in u.atoms() and y in u.atoms()
        return u.subs({x: self.chi[0], y: self.chi[1]})

    def mass_matrix(self, basis):
        'Beam mass matrix from basis.'
        # Note that this is not optimized for eigen but targets generality
        n = len(basis)
        M = np.zeros((n, n))
        for i, v in enumerate(basis):
            M[i, i] = self.inner_product(v, v)
            for j, u in enumerate(basis[(i+1):], i+1):
                M[i, j] = self.inner_product(u, v)
        return M

    def restriction_matrix(self, pbasis, bbasis):
        '''
        Restriction matrix of pbasis to beam with bbasis

        The idea is that for every function u on plate expressed in pbasis
        there exists a function v on beam expressed in bbeasis such that
            (Tu, w) = (u_restricted, w) = (v, w) for all w on beam.
        Here, the (o, o) denotes the beam inner product. This becomes a
        system R*U = M_b*V, with U, V the expansion coefficients of u, v.
        inv(M_b)*R = T is then maps U to V and is related to the trace ope-
        rator.
        '''
        n = len(bbasis)
        m = len(pbasis)
        R = np.zeros((n, m))
        for i, v in enumerate(bbasis):
            for j, u in enumerate(pbasis):
                R[i, j] = self.inner_product(self.restrict(u), v)

        return R

    def trace_matrix(self, pbasis, bbasis):
        'Matrix of the trace operator'
        R = self.restriction_matrix(pbasis, bbasis)
        M = self.mass_matrix(bbasis)
        return la.inv(M).dot(R)

def beam_basis(n):
    'Symbolic basis of beam using eigen basis. Functions of parameter s.'
    return [v.subs(x, s) for v in list(eigen_basis(n))]

def plate_basis(m):
    '''
    Symbolic basis of plate using tensor product of (m x m) eigen basis.
    Functions of parameter x, y
    '''
    return [vx*vy.subs(x, y) 
            for vx, vy in product(eigen_basis(m), eigen_basis(m))]

def plate_mass_matrix(basis):
    'Plate mass matrix from basis.'
    # As list and lambdify
    m = len(basis)
    M = np.eye(m)
    return M

if __name__ == '__main__':
    from sympy import sin, pi

    A = [-1, -1]
    B = [0, 1]

    beam = Beam(A, B)

    u = sin(pi*s)
    v = sin(pi*s)

    print beam.inner_product(u, v)

    bbasis = beam_basis(2)
    for v in bbasis:
        print v

    pbasis = plate_basis(2)
    for v in pbasis:
        print v, beam.restrict(v)

    print beam.mass_matrix(bbasis)
    print 2/beam.L

    print beam.restriction_matrix(pbasis, bbasis)
    print beam.trace_matrix(pbasis, bbasis)

