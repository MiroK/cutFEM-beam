class CoupledBroblem(object):
    '''
    This class defines an interface for investigating plate-beam problems
    coupled by a constraint with no differential operator. The entire system
    is a saddle point problem with matrix

            [[Ap, 0, Bp]
             [0, Ab, Bb],
             [-Bp.T, -Bb.T 0]].

    which we write as

            [[A, B],
             [-B.T, 0]]

    The class knows how to construct Bp, B from Vp and Vb spaces and if child
    implements plate/beam_matrix method which give Ap, Ab it can assemble A 
    as well as the Schur complement. The child can also implement the norm_
    matrix method (H^s norm matrices) that are used to precondition the Schur
    complement.
    '''
    def __init__(self, Vp, Vb, beam):
        'Set the beam and basis for plate and beam spaces.'
        self.Vp
        self.Vb
        self.beam = beam

    def plate_matrix(self):
        'Return matrix of the bilinear form Vp x Vp that has plate physics.'
        raise NotImplementedError

    def beam_matrix(self):
        'Return matrix of the bilinear form Vb x Vb that has plate physics.'
        raise NotImplementedError

    def norm_matrix(self, norm):
        'Return Vb x Vb matrix that is H^s norm in the Vb basis.'
        raise NotImplementedError

    def assemble_A(self):
        'Return blocks A0, A1 combined into A block.'
        pass
    
    def assemble_B(self):
        'Return combined trace operator matrix and beam matrix into B block.'
        pass

    def schur_complement(self, norm):
        'Compute the Schur complement. For norm > 0 precondition.'
        pass

# TODO: handle here assembly and schur
# TODO: PoissonEigen class
# TODO: BiharmonicEigen class
# TODO: iff okay, 1) Check that we have match with theory
#                 2) Sensitivity study with the beam position - data
#                 3)                                          \ postprooc
