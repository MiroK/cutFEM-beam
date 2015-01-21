import numpy as np
import numpy.linalg as la
from trace_operator import TraceOperator
from sympy import symbols, lambdify, S
from sympy.mpmath import quad
from itertools import product

x, y, s = symbols('x, y, s')

class CoupledProblem(object):
    '''
    Parent class for investigating plate-beam problems coupled by a constraint
    with no differential operator. The entire system is a saddle point problem
    with matrix

            [[Ap, 0, Bp]
             [0, Ab, Bb],
             [Bp.T, Bb.T 0]].

    which we write as

            [[A, B],
             [B.T, 0]]

    The class knows how to construct Bp, Bb from Vp, Vb, Q spaces and if child
    implements Ap/Ab_matrix methods which give Ap, Ab it can assemble A 
    as well as the Schur complement. If there is some nice relat. between Vb
    and Q it is smart to implement Bb_matrix and C_matrix method (Q H^norm
    norm matrices) that are used to precondition the Schur complement.
    '''
    def __init__(self, Vp, Vb, Q, beam):
        'Set the beam and basis for plate, beam and Lagrange multiplier spaces.'
        # Vp(x, y), Vb(s), Q(s) are symbolic basis
        assert isinstance(Vp, list)
        self.Vp = Vp
        ms = map(len, Vp)
        self.ms = ms   # Dimension of components
        self.m = np.prod(ms) # Total dimension of Vp is m

        self.Vb = Vb
        self.n = len(Vb) # Dimension of Vb

        self.Q = Q
        self.r = len(Q) # Dimension of Q

        # We don't really want here the trace matrix but Bp is like R with Vp, Q
        self.beam = beam
        self.trace_operator = TraceOperator(Vp, Q, beam)

    def Ap_matrix(self):
        'Return matrix of the bilinear form Vp x Vp that has plate physics.'
        raise NotImplementedError

    def Ab_matrix(self):
        'Return matrix of the bilinear form Vb x Vb that has plate physics.'
        raise NotImplementedError

    def A_matrix(self):
        'Return blocks Ap, Ab combined into A block.'
        # A = [[Ap, 0]]
        #       [0, Ab]
        # with Ap = m x m, Ab = n x n
        m, n = self.m, self.n

        Ap = self.Ap_matrix()
        assert Ap.shape == (m, m)

        Ab = self.Ab_matrix()
        assert Ab.shape == (n, n)

        A = np.zeros((m + n, m + n))
        A[:m, :m] = Ap
        A[m:, m:] = Ab
        return A

    def Bp_matrix(self):
        'Return matrix enforcing constraint on plate.'
        # (u|b, q)_b for u in Vp, q in Q -- R matrix
        # Note that the matrix uses beam inner product and has the Jacobian, e.g
        # L/2 for the linear beam, included
        Bp = -self.trace_operator.R_matrix().T
        return Bp

    def Bb_matrix(self):
        'Return matrix enforcing constraint on beam. Fallback-uses integration.'
        # (v, q)_b for u in Vb, q in Q 
        n, r = self.n, self.r
        Bp = np.zeros((n, r))
        for i, p in enumerate(self.Vb):
            for j, q in enumerate(self.Q):
                Bp[i, j] = self.beam.inner_product(p, q) # Jacobian is included
        return Bp

    def B_matrix(self):
        'Return blocks Bp, Bb combined into B block.'
        # A = [[Bp]]
        #       [Bb]
        # with Bp = m x r, Ab = n x r
        m, n, r = self.m, self.n, self.r

        Bp = self.Bp_matrix()
        assert Bp.shape == (m, r)

        Bb = self.Bb_matrix()
        assert Bb.shape == (n, r)

        B = np.zeros((m + n, r))
        B[:m, :] = Bp
        B[m:, :] = Bb
        return B

    def C_matrix(self, norm):
        'Return Q x Q matrix that is H^norm matrix in the Q basis.'
        # We can do it for positive ints
        assert isinstance(norm, int) and norm >= 0
        r = self.r
        C = np.zeros((r, r))
        # Differentiate according to norm
        dQ = [mu.diff(s, norm) for mu in self.Q]
        # J = |d\vec{\chi}/ds|
        # Beam inner product has J ds included but from the derivatives here
        # you'll get 1/(J)**2norm which needs to be bundled to integrand
        term = self.beam.Jac**(-2*norm)

        for i, lmbda in enumerate(dQ):
            C[i, i] = self.beam.inner_product(lmbda*term, lmbda)
            for j, mu in enumerate(dQ[i+1], i+1):
                C[i, j] = self.beam.inner_product(lmbda*term, mu)
                C[j, i] = C[i, j]
        return C

    def schur_complement_matrix(self, norms):
        'Compute the Schur complement. Optionally precondition by norms.'
        # Note that we consider symmetric system, so just B.T
        # -B.T would work as well
        # B.T inv(A) B

        A = self.A_matrix()
        B = self.B_matrix()
        S = B.T.dot(la.inv(A).dot(B))

        matrices = []
        # Precondition
        for norm in norms:
            if norm is None:
                matrices.append(S)
            else:
                C = self.C_matrix(norm)
                mat = C.dot(S)
                matrices.append(mat)

        return matrices

    def system_matrix(self):
        '''
        The system matrix is
          
            [[Ap, 0, Bp]
             [0, Ab, Bb],
             [Bp.T, Bb.T 0]]

        or
            [[A, B],
             [B.T, 0]]
        # No -B.T so that we have a symmetric system 
        '''
        # Get blocks
        A = self.A_matrix()
        B = self.B_matrix()
        
        m, n, r = self.m, self.n, self.r

        # Assemble system
        system = np.zeros((m+n+r, m+n+r))
        system[:(m+n), :(m+n)] = A
        system[:(m+n), (m+n):] = B
        system[(m+n):, :(m+n)] = B.T

        return system

    def preconditioner(self, blocks):
        'Assemble preconditioner from 9 blocks in 3x3 structure'
        # Make sure we have all blocks
        assert len(blocks) == 3 and all(len(row) == 3 for row in blocks)
        # Assemble the preconditioner
        P = np.zeros((self.m + self.n + self.r, self.m + self.n + self.r))
        # Block are expected to have sizes[i, j] - if they fail the assignment
        # will raise. Zero block are auto expanded
        # Loop up table for sizes
        sizes = (self.m, self.n, self.r)

        for i, row in enumerate(blocks):
            # Where the block goes row-wise
            block_row_start = sum(sizes[:i])
            block_row_stop = block_row_start + sizes[i]
            for j, block in enumerate(row):
                # Where the block goes column-wise
                block_col_start = sum(sizes[:j])
                block_col_stop = block_col_start + sizes[j]

                P[block_row_start:block_row_stop,
                  block_col_start:block_col_stop] = block

        return P

    def solve(self, f, as_sym=True):
        '''
        Solve the problem system_matrix.[[U, P, lmbda]].T = [[F, 0, 0]].T with
        F_k = ((v_k, f)) for v_k in Vp
        # Works only for 2d so far
        '''
        assert self.beam.d == 2
        # lhs
        A = self.system_matrix()

        # rhs, needs to be integrated
        f = lambdify([x, y], f)
        b = np.zeros(self.m+self.n+self.r)
        # For Vp need to combine components. Comps are only funcions of x so
        # for the second component we need to substitute
        Vp = [v0*(v1.subs(x, y)) for v0, v1 in product(*self.Vp)]
        for k, v in enumerate(Vp):
            v = lambdify([x, y], v)
            b[k] = quad(lambda x, y: v(x, y)*f(x, y), [-1, 1], [-1, 1])

        # Get expansion coefficients
        UPLmbda = la.solve(A, b)
        # Split for plate, beam and multiplier
        m, n, r = self.m, self.n, self.r
        U = UPLmbda[:m]
        P = UPLmbda[m:(m+n)]
        Lmbda = UPLmbda[(m+n):]

        # Return the assambled symbolic or lambdas solutions
        # First need basis, we have symbolic Vp, other symbolic:
        Vb = self.Vb
        Q = self.Q
        # Assembly
        # Plate deformation
        u = sum(U_k*v_k for U_k, v_k in zip(U, Vp))
        # Beam deformation 
        # Note that we compute v_hat = v o beam, v_hat: [-1, 1] -> R
        p = sum(P_k*q_k for P_k, q_k in zip(P, Vb))
        # Lagrange multiplier
        # Note that we compute lmbda_hat = lmbda o beam, v_hat: [-1, 1] -> R
        lmbda = sum(Lmbda_k*mu_k for Lmbda_k, mu_k in zip(Lmbda, Q))

        # Lambdify optionally
        if not as_sym:
            u = lambdify([x, y], u)
            p = lambdify(s, p)
            lmbda = lambdify(s, lmbda)

        return (u, p, lmbda)
