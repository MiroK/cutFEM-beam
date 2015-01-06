from trace import TraceOperator
from eigen_basis import eigen_basis
import numpy as np

class EigenTraceOperator(TraceOperator):
    def __init__(self, ms, n, beam):
        d  = len(ms)
        V_p = [eigen_basis(ms[i]) for i in range(d)]
        V_b = eigen_basis(n)
        TraceOperator.__init__(self, V_p, V_b, beam)

    def Mp_matrix(self):
        return np.eye(self.m)

    def Mb_matrix(self):
        return np.eye(self.n)/2/self.beam.L

# -----------------------------------------------------------------------------  
