from __future__ import division
import numpy as np
import numpy.linalg as la
from math import cos, sin, pi
from sympy.mpmath import legendre, sqrt

def P_matrix(n, m, what):
    'Transformation from FEM(with n functions!) to Eigen.'
    h = 2/(n+1)
    P = np.zeros((n, m))
    if what == 'eigen':
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

    elif what == 'shen':
        # All mesh vertices
        vertices = [-1 + h*i for i in range(n+2)]
        # Shen functions 
        Sk = lambda k, x: (legendre(k+2, x) - legendre(k, x))/sqrt(4*k + 6)
        
        P = np.zeros((n, m))
        # This loops computet integrals: i-th has * j-th shen
        for i in range(n):
            vertex_p = vertices[i]
            vertex = vertices[i+1]
            vertex_n = vertices[i+2]

            for j in range(m):
                P[i, j] = 2*Sk(j, vertex)/h
                P[i, j] -= Sk(j, vertex_p)/h
                P[i, j] -= Sk(j, vertex_n)/h

    return P

what = 'shen'
# with open('Pmm_%s.data' % what, 'w') as f:
#     for n in [4, 16, 32, 64, 128]:
#         P = P_matrix(n, n, what)
#         condP = la.cond(P)
#         condPP = la.cond(P.T.dot(P))
#         f.write('%d %g %g\n' % (n, condP, condPP))

import matplotlib.pyplot as plt

what = 'eigen'
m = 128
P = P_matrix(m, 2*m, what)
P = np.sqrt(P**2)
plt.figure()
plt.pcolor(P)
plt.colorbar()
plt.axis('tight')
plt.show()

