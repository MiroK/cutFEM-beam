from coupled_problem import CoupledProblem
from plate_beam import LineBeam
from eigen_basis import eigen_basis
import eigen_poisson
import eigen_biharmonic
from sympy import symbols
from math import pi, sqrt
import numpy as np
import numpy.linalg as la

x, y, s = symbols('x, y, s')

class CoupledEigen(CoupledProblem):
    '''
    Parent for coupled problems with spaces Vp, Vb, Q are spanned by
    functions from eigenbasis
    '''
    def __init__(self, ms, n, r, beam, params):
        'Solver with ms[i] functions for i-th comp of Vp, n for Vb and r for Q.'
        # For now ms = [m, m]
        assert len(set(ms)) == 1

        Vp = [list(eigen_basis(m)) for m in ms]
        Vb = [q.subs(x, s) for q in eigen_basis(n)]
        Q = [mu.subs(x, s) for mu in eigen_basis(r)]
        CoupledProblem.__init__(self, Vp, Vb, Q, beam)

        self.params = params

    def Bb_matrix(self):
        'Matrix of the constraint on the beam'
        # Straight beam simple form
        if isinstance(self.beam, LineBeam):
            # Need (n, r) mass matrix between Q and Vp
            dim = max(self.n, self.r)
            Bb = eigen_poisson.mass_matrix(dim)
            # Chop to proper size
            Bb = Bb[:self.n, :self.r]
            # Jacobian term, Jac is just a number
            Bb *= float(self.beam.Jac)
        # Otherwise use integration
        else:
            Bb = CoupledProblem.Bb_matrix(self)
        return Bb

    def C_matrix(self, norm):
        'H^norm matrices of Q'
        if isinstance(self.beam, LineBeam):
            # These are eigenvalues of u''
            diag = np.array([(pi/2 + k*pi/2)**2 for k in range(self.r)],
                            dtype='float')
            # Matrix has their power on the diagonal
            C = np.diag(diag**norm)
            # Scale the jacobian according to norm
            J = self.beam.Jac   
            J = J**(1-2*norm)
            C *= float(J)
        else:
            C  = CoupledProblem.C_matrix(self, norm)

        return C

# ----------------------------------------------------------------------------- 
