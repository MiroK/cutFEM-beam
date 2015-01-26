from __future__ import division
from sympy import sin, cos, pi, sqrt, symbols, lambdify, Piecewise, S,\
        integrate
from sympy.mpmath import quad
import numpy as np

x = symbols('x')

def eigen_basis(n):
    '''
    Return first n eigenfunctions of Laplacian over biunit interval with homog.
    Dirichlet bcs. at endpoints -1, 1. Functions of x.
    '''
    k = 0
    functions = []
    while k < n:
        alpha = pi/2 + k*pi/2
        if k % 2 == 0:
            functions.append(cos(alpha*x))
        else:
            functions.append(sin(alpha*x))
        k += 1
    return functions

def CG1_basis(num_elements):
    vertices = np.linspace(-1, 1, num_elements+1, endpoint=True)
    functions = []
    for i in range(1, num_elements):
        v_prev = S(vertices[i-1])
        v = S(vertices[i])
        v_next = S(vertices[i+1])
        
        def f(x, v_prev=v_prev, v=v, v_next=v_next):
            print v_prev, v, v_next
            if (x-v)*(x-v_prev) <= 0:
                return (x-v_prev)/(v-v_prev)
            elif (x-v)*(x-v_next) <= 0:
                return (x-v_next)/(v-v_next)
            else:
                return 0

        functions.append(f)
    return functions

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_elements = 6
    basis = CG1_basis(num_elements)

    interval = np.linspace(-1, 1, 6)
    plt.figure()
    for j, v in enumerate(basis):
        values = [v(xi) for xi in interval]
        print values
        plt.plot(interval, values, label='%d' % j)
    plt.show()

    #n = len(basis)
    #mat = np.zeros((n, n))
    #for i, v in enumerate(basis):
    #    for j, u in enumerate(basis):
    #        integrand = lambdify(x, u*v)
   #         mat[i, j] = quad(integrand, (-1, 1))
   # print mat
