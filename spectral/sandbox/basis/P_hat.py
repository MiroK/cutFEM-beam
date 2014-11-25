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

m = 40
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

