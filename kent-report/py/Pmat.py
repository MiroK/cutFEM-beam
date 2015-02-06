from __future__ import division
import numpy as np
import numpy.linalg as la
from math import cos, sin, pi, sqrt
from sympy.mpmath import legendre

def P_matrix_eigen(n, m):
    'Transformation from FEM(with n functions!) to Eigen.'
    h = 2/(n+1)
    P = np.zeros((n, m))
    # The row has expansion coefs of i-th hat function
    for i in range(n):
        xi = -1 + (i+1)*h
        x_next = xi + h
        x_prev = xi - h
        for j in range(m):
            if j % 2 == 0:
                dd_f_p = sin((pi/2 + j*pi/2)*x_prev)/(pi/2 + j*pi/2)**2
                dd_f_i = sin((pi/2 + j*pi/2)*xi)/(pi/2 + j*pi/2)**2
                dd_f_n = sin((pi/2 + j*pi/2)*x_next)/(pi/2 + j*pi/2)**2
            else:
                dd_f_p = cos((pi/2 + j*pi/2)*x_prev)/(pi/2 + j*pi/2)**2
                dd_f_i = cos((pi/2 + j*pi/2)*xi)/(pi/2 + j*pi/2)**2
                dd_f_n = cos((pi/2 + j*pi/2)*x_next)/(pi/2 + j*pi/2)**2

            val = 2*dd_f_i/h - (dd_f_n + dd_f_p)/h
            P[i, j] = val

    return P


def P_matrix_shen(n, m):
    'Transformation from FEM(with n functions!) to Shen.'
    h = 2/(n+1)
    P = np.zeros((n, m))
    # All mesh vertices
    vertices = [-1 + h*i for i in range(n+2)]
    
    P = np.zeros((n, m))
    # This loops computet integrals: i-th has * j-th shen
    for i in range(n):
        vertex_p = vertices[i]
        vertex = vertices[i+1]
        vertex_n = vertices[i+2]
        for j in range(m):
            P[i, j] = 2*(legendre(j+2, vertex) - legendre(j, vertex))
            P[i, j] -= (legendre(j+2, vertex_p) - legendre(j, vertex_p))
            P[i, j] -= (legendre(j+2, vertex_n) - legendre(j, vertex_n))
            P[i, j] /= sqrt(4*j + 6)*h

    return P

what = 'shen'
# with open('Pmm_%s.data' % what, 'w') as f:
#     for n in [4, 16, 32, 64, 128]:
#         P = P_matrix(n, n, what)
#         condP = la.cond(P)
#         condPP = la.cond(P.T.dot(P))
#         f.write('%d %g %g\n' % (n, condP, condPP))


# Numbalize
from numba import jit, double, int_
fast_P_matrix_eigen = jit(double[:, :](int_, int_))(P_matrix_eigen)


m = 128
P = P_matrix_eigen(m, 2*m)
P = np.sqrt(P**2)
P /= np.max(P)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.pcolor(P)
plt.colorbar()
plt.axis('tight')
plt.ylabel('$n$')
plt.xlabel('$m$')
plt.show()

