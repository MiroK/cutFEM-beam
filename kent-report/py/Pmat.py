from __future__ import division
import numpy as np
import numpy.linalg as la
from math import cos, sin, pi

def P_matrix(n, m):
    'Transformation from FEM(with n functions!) to Eigen.'
    h = 2/(n+1)
    P = np.zeros((n, m))
    for i in range(n):
        xi = -1 + (i+1)*h
        x_next = xi + h
        x_prev = xi - h
        for j in range(m):
            dd_f = lambda x, j=j: cos((pi/2 + j*pi/2)*x)/(pi/2 + j*pi/2)**2\
                                  if (j % 2) == 0 else \
                                  sin((pi/2 + j*pi/2)*x)/(pi/2 + j*pi/2)**2

            val = 2*dd_f(xi)/h - (dd_f(x_next) + dd_f(x_prev))/h
            P[i, j] = val
    return P

with open('Pmm_eigen.data', 'w') as f:
    for n in [2**i for i in range(2, 14)]:
        P = P_matrix(n, n)
        cond = la.cond(P.T.dot(P))
        f.write('%d %g\n' % (n, cond))
