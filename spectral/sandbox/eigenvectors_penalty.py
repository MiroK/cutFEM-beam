import numpy as np
import numpy.linalg as la

def get_P(m):
    P = np.zeros((m, m))
    for i in range(m):
        P[i, i] = 1 + (-1)**(2*i)
        for j in range(i+1, m):
            P[i, j] = 1 + (-1)**(i+j)
            P[j, i] = P[i, j]

    return P

def ker_vectors(m):
    N = np.zeros((m, m-2))
    for i in range(m-2):
        N[i, i] = 1.
        N[i+2, i] = -1.

    N, _ = la.qr(N)
    return N.T

# First we claim is that P has m-2 zero eigenvalues

for m in range(3, 5):
    P = get_P(m)
    lmbdas = la.eigvals(P)
    n_zeros = np.where(np.abs(lmbdas) < 1E-14)[0].shape[0]
    assert n_zeros == m-2

    vecs = ker_vectors(m)
    for vec in vecs:
        norm = la.norm(P.dot(vec))/(m-2)
        print norm
        assert norm < 1E-15
    assert la.norm(vecs.dot(vecs.T) - np.eye(m-2))/(m-2) < 1E-15
    print vecs.T.dot(vecs)


