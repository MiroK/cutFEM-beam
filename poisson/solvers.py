from quadrature import GLQuadrature
from sympy import symbols, lambdify
import numpy.linalg as la
import numpy as np
import time


class VariationalSolver1d(object):
    'Template for assembling AU=b system in variational problems.'
    def __init__(self):
        pass

    def assemble_A(self, N):
        raise NotImplementedError('Implement in child')

    def basis_functions(self, N):
        raise NotImplementedError('Implement in child')

    def solve(self, f, N, E, a, b, eps, n_refs):
        'Assemble linear system and solve.'
        assert b > a

        # Assemble matrix the matrix on reference domain [0, 1]
        time_AA = time.time()

        AA = self.assemble_A(N)

        time_AA = time.time() - time_AA

        x = symbols('x')

        # Fast evaluate f with x in [a, b]. In integral we need f pulled back to
        # reference domain [0, 1]
        f_lambda = lambdify(x, f)

        def F(x_hat):
            'Pullback map [0, 1] to [a, b]'
            return (b-a)*x_hat + a

        def f_F(x_hat):
            'Function f pulled back to reference domain'
            return f_lambda(F(x_hat))

        # Make symbolic basis on reference domain [0, 1]
        basis = self.basis_functions(N+1)

        # Make basis functions on fast evaluation
        basis_lambda = map(lambda f: lambdify(x, f), basis)

        # Get the quadrature for computing integrals
        quad = GLQuadrature(2*N)

        bb = np.zeros(N)

        # Assemble vector, this is done multiple times until either eps
        # is reached in bb difference or n_refs is exceeded
        time_bb = time.time()

        for j, base_lambda in enumerate(basis_lambda):
            bb[j] = quad.eval(lambda x: base_lambda(x)*f_F(x), [[0, 1]])
        bb *= (b-a)
        bb_norm_ = la.norm(bb)

        diff = 1
        ref = 0
        while diff > eps and ref < n_refs:
            ref += 1

            new_N = quad.N + 1
            quad.__init__(new_N)

            for j, base_lambda in enumerate(basis_lambda):
                bb[j] = quad.eval(lambda x: base_lambda(x)*f_F(x), [[0, 1]])

            bb *= (b-a)
            bb_norm = la.norm(bb)
            diff = abs(bb_norm - bb_norm_)
            bb_norm_ = bb_norm

        print 'Assemble vector, final diff', diff
        time_bb = time.time() - time_bb

        # Vector of exp. coeffs
        time_solve = time.time()
        U = la.solve(AA, bb)
        time_solve = time.time() - time_solve

        print 'Assembling matrix:', time_AA
        print 'Assembling vector:', time_bb
        print 'Solve linear system:', time_solve

        # Map the basis back to [a, b]
        basis = np.array(map(lambda base: base.subs(x, (x-a)/(b-a)), basis))

        # Return expension coefficients and basis functions
        return U, basis
