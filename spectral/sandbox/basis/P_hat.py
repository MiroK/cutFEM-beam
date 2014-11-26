import numpy as np
import numpy.linalg as la
from math import sqrt

def matrix(m):
    M = np.zeros((m, m))
    for i in range(0, m, 2):
        for j in range(0, m, 2):
            M[i, j] = 2

    for i in range(1, m, 2):
        for j in range(1, m, 2):
            M[i, j] = 2

    return M

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

    return E.T

def eigv2(m):
    E = np.zeros((m, 2))
    # Corresponds to eigenvae m+1
    for i in range(0, m, 2):
        E[i, 0] = 1
    # Corresponds to eigenvalue m-1
    for i in range(1, m, 2):
        E[i, 1] = 1
    return E.T

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

m = 6
assert m > 2
M = matrix(m)
vectors0 = eigv0(m, True)
vectors2 = eigv2(m)

for vec0 in vectors0:
    assert la.norm(M.dot(vec0))/len(vec0) < 1E-15

for i, vec2 in enumerate(vectors2):
    if (m % 2) == 0:
        assert la.norm(M.dot(vec2) - m*vec2)/m < 1E-15
    else:
        if i == 0:
            assert la.norm(M.dot(vec2) - (m+1)*vec2)/m < 1E-15
        else:
            assert la.norm(M.dot(vec2) - (m-1)*vec2)/m < 1E-15

# Check the orthogonality of eigenvectors of nonzero eigenvalues
assert abs(vectors2[0].dot(vectors2[1])) < 1E-15
# We also have mutual orthogonality of vectors correspnd to 0 and non-zero
assert la.norm(vectors2.dot(vectors0.T))/m < 1E-15

assert la.norm(vectors0.dot(vectors0.T) - np.eye(m-2))/(m-2) < 1E-15

V = eigv0(m, False)
U = eigv0(m, True)

# Not that this should be for V = U*R, U.T*V but you have everything transposed
# to make iteration easy
R = Rmat(m)
assert la.norm(V.T - U.T.dot(R)) < 1E-15

# We nor claim that transformation matrices alpha_phi, alpha_psi can be com-
# puted from V and special matrices C
def c(k):
    return sqrt(4*k + 6)**-1

# Now build transformation matrix
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
    C_psi[i, i] = c(i)

assert la.norm(alpha_s.T - V.T.dot(C_psi))/m < 1E-15

# -------------------

def alpha_phi(n, m):
    alpha_phi = np.zeros((n, m))
    for i in range(alpha_phi.shape[0]):
        if (i % 2) == 0:
            alpha_phi[i, 0] = -1
        else:
            alpha_phi[i, 1] = -1
        alpha_phi[i, i+2] = 1
    return alpha_phi

alpha_h = alpha_phi(n, m)
C_phi = np.zeros((n, n))
for j in range(n):
    if j % 2 == 0:
        i_range = range(0, j+1, 2)
    else:
        i_range = range(1, j+1, 2)

    for i in i_range:
        C_phi[i, j] = -1

#print alpha_h.T
assert la.norm(alpha_h.T - V.T.dot(C_phi))/m < 1E-15

# C_psi is symmetric, positive definite invertible, diagonal
# C_phi is only invertible
x = np.zeros(n)
# print la.solve(C_psi, x)
# print la.solve(C_phi, x)

G = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        G[i, j] = 0.5*j*(j+1)*(1 + (-1)**(i+j))

for v in V:
    assert la.norm(v.dot(G))/len(v) < 1E-13

D = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        D[i, j] = j*(j+1) - i*(i+1) if (i < j-1) and ((i+j) % 2) == 0 else 0
print D
DD = D[:-2, 2:]

vals, vecs = la.eig(DD)

print vals
print vecs

X = np.zeros((m, m-2))
for i, vec in enumerate(vecs.T):
    X[2:, i] = vec

print
print X
print
for x, val in zip(X.T, vals):
    print x
    print D.dot(x)
