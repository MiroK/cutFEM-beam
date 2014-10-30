import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np


def solve_lagrange_1d(m, n):
    '''
    Solve biharmonic problem
                         E*u^(4) = f in [a, b]
                             u = 0 on a, b
                          u^(2) = 0 on a, b


    We rewrite the problem  as           sigma = -E*u^(2)  in [a, b]
                                    -sigma^(2) = f
                                             u = 0  on a, b
                                         sigma = 0

    We let m be the dimension of space for sigma(W) and n the of space for u(V).
    The linear system is then [[Bm, -E*Am], [An^T, 0]] = [[0], [Bn*F]].

    Here Bm is the mxm mass matrix of W
         Am is the mxn stiffness matrix between W and V
         An is a transpose of Am
         Bn is the nxn mass matrix of V

         F vector in R^n of expansion coefficients of f interpolant in V
    '''
    # Take a, b as 0, 1 and use sines
    # Build blocks
    Am = np.identity(m)

    Bm = np.zeros((m, n))
    for row in range(m):
        for col in range(n):
            if row == col:
                Bm[row, col] = (np.pi*row+1)**2
    Bn = -Bm.T

    AA = np.zeros((m+n, m+n))
    # Put the block together
    AA[:m, :m] = Am
    AA[:m, m:] = Bm
    AA[m:, :m] = Bn

    # plt.figure()
    # plt.spy(AA)
    # plt.show()

    bb = np.zeros(m+n)
    bb[m:] = np.random.random(n)

    return la.cond(AA)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    Ms = np.arange(10, 80, 5)
    conds_0 = np.array([solve_lagrange_1d(m=M, n=M/2) for M in Ms])
    conds_1 = np.array([solve_lagrange_1d(m=M, n=M) for M in Ms])
    conds_2 = np.array([solve_lagrange_1d(m=M, n=2*M/3) for M in Ms])


    plt.figure()
    plt.loglog(Ms, conds_0, label='n=m/2')
    plt.loglog(Ms, conds_1, label='n=m')
    plt.loglog(Ms, conds_2, label='n=m-3')
    plt.legend(loc='best')
    plt.show()

    # Make sense of abote
    # Try with polynoomials
    # Solve it first and then think more details

    # You take m = n+e and keep increasing n, solvability
    # You take m=n and incrase
    # You take m-e = n and keep increasing n, solvability
