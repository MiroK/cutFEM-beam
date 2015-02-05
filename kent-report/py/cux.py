from eigen_basis import eigen_basis
from shenp_basis import shenp_basis as shen_basis
import numpy as np
from sympy import symbols, lambdify
from sympy.mpmath import quad

x, s = symbols('x, s')

def eigen_Bb(m, n):
    'Return m x n matrix of beam constraint in Eigen basis'
    tests = list(eigen_basis(m))
    trials = list(eigen_basis(n))
    
    mat = np.zeros((m, n))
    for i, v in enumerate(tests):
        for j, u in enumerate(trials):
            mat[i, j] = quad(lambdify(x, v*u), [-1, 1])
    return mat


def eigen_Bp(m, n, A=[0, -1], B=[0, 1]):
    'Return m**2 x n matrix of beam constraint in Eigen basis'
    chi0 = 0.5*A[0]*(1-s) + 0.5*B[0]*(1+s)
    chi1 = 0.5*A[1]*(1-s) + 0.5*B[1]*(1+s)
    tests = [u.subs(x, chi0)*v.subs(x, chi1) for u in list(eigen_basis(m))
             for v in list(eigen_basis(m))]

    trials = [p.subs(x, s) for p in list(eigen_basis(n))]
    
    mat = np.zeros((m**2, n))
    for i, v in enumerate(tests):
        for j, u in enumerate(trials):
            mat[i, j] = quad(lambdify(s, v*u), [-1, 1])
    return mat


def shen_Bb(m, n):
    'Return m x n matrix of beam constraint in Shen basis'
    tests = list(shen_basis(m))
    trials = list(shen_basis(n))
    
    mat = np.zeros((m, n))
    for i, v in enumerate(tests):
        for j, u in enumerate(trials):
            mat[i, j] = quad(lambdify(x, v*u), [-1, 1])
    return mat


def shen_Bp(m, n, A=[0, -1], B=[0, 1]):
    'Return m**2 x n matrix of beam constraint in Shen basis'
    chi0 = 0.5*A[0]*(1-s) + 0.5*B[0]*(1+s)
    chi1 = 0.5*A[1]*(1-s) + 0.5*B[1]*(1+s)
    tests = [u.subs(x, chi0)*v.subs(x, chi1) for u in list(shen_basis(m))
             for v in list(shen_basis(m))]

    trials = [p.subs(x, s) for p in list(shen_basis(n))]
    
    mat = np.zeros((m**2, n))
    for i, v in enumerate(tests):
        for j, u in enumerate(trials):
            mat[i, j] = quad(lambdify(s, v*u), [-1, 1])
    return mat

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if False:
        # Bb@Eigen
        # The claim is that if m=n the rank is m
        print np.linalg.matrix_rank(eigen_Bb(m=5, n=5))
        # if m > n the rank is n (the matrix has full column rank)
        print np.linalg.matrix_rank(eigen_Bb(m=7, n=5))
        # if m < n the rank is m. And there is a kernel of size n-m
        print np.linalg.matrix_rank(eigen_Bb(m=5, n=7))

        # Bb@shen
        # The claim is that if m=n the rank is m
        print np.linalg.matrix_rank(shen_Bb(m=5, n=5))
        # if m > n the rank is n (the matrix has full column rank)
        print np.linalg.matrix_rank(shen_Bb(m=7, n=5))
        # if m < n the rank is m. And there is a kernel of size n-m
        print np.linalg.matrix_rank(shen_Bb(m=5, n=7))

    if True:
        A = [0., -1]
        B = [0., 1]
        # Bp@Eigen
        # The claim is that if m=n the rank is m
        print np.linalg.matrix_rank(eigen_Bp(m=3, n=3, A=A, B=B))
        # if m > n the rank is n (the matrix has full column rank)
        print np.linalg.matrix_rank(eigen_Bp(m=5, n=3, A=A, B=B))
        # if m < n the rank is m. Observed n if n < 7 else 7
        print np.linalg.svd(eigen_Bp(m=3, n=11, A=A, B=B))[1]

        # Bp@Shen
        # The claim is that if m=n the rank is m
        print np.linalg.matrix_rank(shen_Bp(m=3, n=3, A=A, B=B))
        # if m > n the rank is n (the matrix has full column rank)
        print np.linalg.matrix_rank(shen_Bp(m=5, n=3, A=A, B=B))
        # if m < n the rank is m. Observed n if n < 5 else 5
        print np.linalg.matrix_rank(shen_Bp(m=3, n=6, A=A, B=B))
