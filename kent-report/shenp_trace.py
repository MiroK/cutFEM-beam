from __future__ import division
from trace import TraceOperator
from shenp_basis import shenp_basis
import numpy as np
from math import sqrt

class ShenpTraceOperator(TraceOperator):
    def __init__(self, ms, n, beam):
        d = len(ms)
        V_p = [shenp_basis(i) for i in range(d)]
        V_b = shenp_basis(n)
        TraceOperator.__init__(self, V_p, V_b, beam)

    def Mp_matrix(self):
        Ms = []
        for m in map(len, self.V_p):
            M = M_matrix(m)
            Ms.append(M)

        M = np.outer(Ms[0], Ms[1])
        for i in range(2, len(self.V_p)):
            M = np.outer(M, Ms[i])
        return M

    def Mb_matrix(self):
        Mb = M_matrix(self.n)
        return Mb/2/self.beam.L

    def M_matrix(m):
        weight = lambda k: float(1/sqrt(4*k + 6))
        M = np.zeros((m, m))
        for i in range(m):
            M[i, i] = weight(i)*weight(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
            for j in range(i+1, m):
                M[i, j] = -weight(i)*weight(j)*(2./(2*j+1)) if i+2 == j else 0
                M[j, i] = M[i, j]
        return M
