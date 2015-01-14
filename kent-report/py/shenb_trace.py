from __future__ import division
from trace_operator import TraceOperator
from shenb_basis import shenb_basis, mass_matrix
from plate_beam import LineBeam
from sympy import symbols
import numpy as np
from math import sqrt

x, s = symbols('x, s')

class ShenbTraceOperator(TraceOperator):
    '''
    Trace operator betwee V_p and V_b on LineBeam. Spaces are spanned by
    shenb_basis.
    '''
    def __init__(self, ms, n, beam):
        assert isinstance(beam, LineBeam)
        d = len(ms)
        V_p = map(list, [shenb_basis(mi) for mi in ms])
        V_b = [v.subs(x, s) for v in shenb_basis(n)]
        TraceOperator.__init__(self, V_p, V_b, beam)

    def Mp_matrix(self):
        'Plate mass matrix'
        Ms = []
        for m in map(len, self.V_p):
            M = ShenbTraceOperator.M_matrix(m)
            Ms.append(M)

        Mp = np.kron(Ms[0], Ms[1])
        for i in range(2, len(self.V_p)):
            Mp = np.kron(Mp, Ms[i])
        return Mp

    @staticmethod
    def M_matrix(m):
        'Auxiliary mass matrix for V_pi'
        return mass_matrix(m)

    def Mb_matrix(self):
        'Beam mass matrix'
        Mb = ShenbTraceOperator.M_matrix(self.n)
        return Mb*float(self.beam.Jac)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from shenb_basis import shenb_basis
    import numpy.linalg as la

    # Beam
    A = np.array([-1, -1])
    B = np.array([1, 1])
    beam = LineBeam(A, B)

    # Symbolic basis 
    V_p0 = list(shenb_basis(3))
    V_p1 = list(shenb_basis(2))
    V_p = [V_p0, V_p1]

    V_b = [v.subs(x, s) for v in shenb_basis(4)]

    # Numeric basis
    ms = map(len, V_p)
    n = len(V_b)

    sym = TraceOperator(V_p, V_b, beam)
    num = ShenbTraceOperator(ms, n, beam)

    # Test generic agaist specialized
    assert la.norm(num.Mb_matrix() - sym.Mb_matrix()) < 1E-13
    assert la.norm(num.Mp_matrix() - sym.Mp_matrix()) < 1E-13
