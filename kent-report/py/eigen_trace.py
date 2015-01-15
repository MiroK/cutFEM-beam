from trace_operator import TraceOperator
from eigen_basis import eigen_basis
from plate_beam import LineBeam
from sympy import symbols
import numpy as np

x, s = symbols('x, s')

class EigenTraceOperator(TraceOperator):
    'Trace operator for LineBeam with V_p, V_b spanned by eigen_basis'
    def __init__(self, ms, n, beam):
        assert isinstance(beam, LineBeam)
        d  = len(ms)
        V_p = map(list, [eigen_basis(mi) for mi in ms])
        # Beam functions are functions of s
        V_b = [v.subs(x, s) for v in eigen_basis(n)]
        TraceOperator.__init__(self, V_p, V_b, beam)

    def Mp_matrix(self):
        'Plate mass matrix'
        return np.eye(self.m)

    def Mb_matrix(self):
        'Beam mass matrix'
        return np.eye(self.n)*float(self.beam.Jac)

# -----------------------------------------------------------------------------  

if __name__ == '__main__':
    from eigen_basis import eigen_basis
    import numpy.linalg as la

    # Beam
    A = np.array([-1, -1])
    B = np.array([1, 1])
    beam = LineBeam(A, B)

    # Symbolic basis 
    V_p0 = list(eigen_basis(3))
    V_p1 = list(eigen_basis(2))
    V_p = [V_p0, V_p1]

    V_b = [v.subs(x, s) for v in eigen_basis(4)]

    # Numeric basis
    ms = map(len, V_p)
    n = len(V_b)

    sym = TraceOperator(V_p, V_b, beam)
    num = EigenTraceOperator(ms, n, beam)
    
    # Test generic agaist specialized
    assert la.norm(num.Mb_matrix() - sym.Mb_matrix()) < 1E-13
    assert la.norm(num.Mp_matrix() - sym.Mp_matrix()) < 1E-13
