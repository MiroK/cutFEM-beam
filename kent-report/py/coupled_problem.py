import numpy as np
import numpy.linalg as la
from trace import TraceOperator
from sympy import symbols, lambdify
from sympy.mpmath import quad

class CoupledBroblem(object):
    '''
    This class defines an interface for investigating plate-beam problems
    coupled by a constraint with no differential operator. The entire system
    is a saddle point problem with matrix

            [[Ap, 0, Bp]
             [0, Ab, Bb],
             [Bp.T, Bb.T 0]].

    which we write as

            [[A, B],
             [B.T, 0]]

    The class knows how to construct Bp, Bb from Vp, Vb, Q spaces and if child
    implements Ap/Ab_matrix method which give Ap, Ab it can assemble A 
    as well as the Schur complement. If there is some nice relat. between Vb
    and Q it is smart to implement Bb_matrix. The child can also implement the
    norm_ matrix method (H^s norm matrices) that are used to precondition the
    Schur complement.
    '''
    def __init__(self, Vp, Vb, Q, beam):
        'Set the beam and basis for plate, beam and Lagrange multiplier spaces.'
        # Vp(x, y), Vb(s), Q(s) are symbolic basis
        assert isinstance(V_p, list)
        self.Vp = Vp
        ms = map(len, Vp)
        self.m = np.prod(ms) # Total dimension of Vp is m

        self.Vb
        self.n = len(Vb) # Dimension of Vb

        self.Q
        self.r = r # Dimension of Q

        # We don't really want here the trace matrix but Bp is like R with Vp, Q
        self.beam = beam   # Keep beam
        self.trace_operator = TraceOperator(Vp, Q, beam)

    def Ap_matrix(self):
        'Return matrix of the bilinear form Vp x Vp that has plate physics.'
        raise NotImplementedError

    def Ab_matrix(self):
        'Return matrix of the bilinear form Vb x Vb that has plate physics.'
        raise NotImplementedError

    def assemble_A(self):
        'Return blocks Ap, Ab combined into A block.'
        # A = [[Ap, 0]]
        #       [0, Ab]
        # with Ap = m x m, Ab = n x n
        m, n = self.m, self.n

        Ap = self.Ap_matrix()
        assert Ap.shape == (m, m)

        Ab = Ab.beam_matrix()
        assert Ab.shape == (n, n)

        A = np.zeros((m + n, m + n))
        A[:m, :m] = Ap
        A[m:, m:] = Ab
        return A

    def Bp_matrix(self):
        'Return matrix enforcing constraint on plate.'
        # (u|b, q)_b for u in Vp, q in Q -- R matrix
        return self.trace_operator.R_matrix()

    def Bp_matrix(self):
        'Return matrix enforcing constraint on beam.'
        # (v, q)_b for u in Vb, q in Q
        n, r = self.n, self.r
        Bp = np.zeros((n, r))
        for i, p in enumerate(Vb):
            for j, q in enumerate(Q):
                Bp[i, j] = self.beam.inner_product(p, q)
        return Bp

    def assemble_B(self):
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

    def norm_matrix(self, norm):
        'Return Vb x Vb matrix that is H^s norm in the Vb basis.'
        # We can do it for positive ints
        assert isinstance(norm, int) and norm >= 0
        n = self.n
        C = np.zeros((n, n))
        # Differentiate according to norm
        dVb = [q.diff(s, norm) for q in self.Vb]
        for i, p in enumerate(dVb):
            C[i, i] = self.beam.inner_product(p, p)
            for j, q in enumerate(dVb[i+1], i+1):
                C[i, j] = self.beam.inner_product(p, q)
                C[j, i] = C[i, j]
        return C

    def schur_complement(self, norm=None):
        'Compute the Schur complement. For norm not None precondition.'
        # B.T inv(A) B
        A = self.assemble_A()
        B = self.assemble_B()
        S = B.T.dot(la.inv(A).dot(B))

        # Precondition
        if norm is not None:
            C = self.norm_matrix(norm)
            return C.dot(S)

# TODO: handle here assembly and schur
# TODO: PoissonEigen class
# TODO: BiharmonicEigen class
# TODO: iff okay, 1) Check that we have match with theory
#                 2) Sensitivity study with the beam position - data
#                 3)                                          \ postprooc
