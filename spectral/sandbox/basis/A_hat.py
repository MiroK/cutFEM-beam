import numpy as np
import numpy.linalg as la
from math import sqrt

m = 8

G_ = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        G_[i, j] = 0.5*j*(j+1)*(1 + (-1)**(i+j))

A_ = G_
for i in range(m):
    for j in range(m):
        A_[i, j] -= j*(j+1) - i*(i+1) if (i < j-1) and ((i+j) % 2) == 0 else 0

def eigv0(m, orthogonal):
    E = np.zeros((m, m-2))
    if not orthogonal:
        for j in range(m-2):
            E[j, j] = 1
            E[j+2, j] = -1

    # for i, u in enumerate(E):
    #     for v in E[:i]:
    #         u -= u.dot(v)*v
    #     u /= la.norm(u)
    else:
        def triangular_number(k):
            return 0.5*k*(k+1)

        for j in range(m-2):
            k = (j+2)/2
            i_range = range(j%2, j+1, 2)

            value = 1/sqrt(triangular_number(k))
            for i in i_range:
                E[i, j] = value
            E[j+2, j] = -k*value

        E *= sqrt(2)/2.

    # rows are the eigenvectors
    return E.T

U = eigv0(m, False).T

n = m-2

def c(k):
    return sqrt(4*k + 6)**-1

alpha_s = np.zeros((n, m))
def alpha_psi(n, m):
    alpha = np.zeros((n, m))
    for i in range(n):
        alpha[i, i] = c(i)
        alpha[i, i+2] = -c(i)
    return alpha

n = m - 2
alpha_s = alpha_psi(n, m)

C_psi = np.zeros((n, n))
for i in range(n):
    C_psi[i, i] = 2  # c(i)

alpha_s = (U.dot(C_psi)).T

A = alpha_s.dot(A_.dot(alpha_s.T))

UAU = np.zeros((n, n))
for i in range(n):
    UAU[i, i] = 2*(2*(i+1) + 1)

assert la.norm(UAU - U.T.dot(A_.dot(U)))/n < 1E-15

# ---------------------------------------
np.set_printoptions(precision=2)
M = np.zeros((m, m))
for i in range(m):
    M[i, i] = 2./(2*i + 1)

C = np.zeros((m, m))
for i in range(m):
    C[i, i] =1/sqrt((2./(2*i+1)))

def Rmat(m):
    '''
    eig0(m, True).T*R  = eig0(m, False)
    '''
    R = np.zeros((m-2, m-2))
    def triangular_number(k):
        return 0.5*k*(k+1)

    # Generate diagonal
    for i in range(m-2):
        k = (i+2)/2
        R[i, i] = sqrt(triangular_number(k))/k

    for i in range(m-4):
        k = (i+2)/2
        R[i, i+2] = -sqrt(triangular_number(k))/(k+1)


    R *= 2./sqrt(2)

    return R

# Let's try to make mass matrix diagonal
V = eigv0(m, True).T
foo_s = (C.dot(V)).T
assert la.norm(foo_s.dot(M.dot(foo_s.T)) - np.eye(n))/n < 1E-15

