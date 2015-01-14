from coupled_problem import CoupledProblem
from plate_beam import LineBeam
from shenp_basis import shenp_basis
import shen_poisson
from sympy import symbols
from math import pi
import numpy as np
import numpy.linalg as la

x, y, s = symbols('x, y, s')

class CoupledLaplace(CoupledProblem):
    '''
    Coupled problem with spaces Vp, Vb, Q are spanned by functions from
    shenp basis and physics on plate and beam given by Laplacians.
    '''
    def __init__(self, ms, n, r, beam, params):
        'Solver with ms[i] functions for i-th comp of Vp, n for Vb and r for Q.'
        # For now ms = [m, m]
        assert len(set(ms)) == 1

        Vp = [list(shenp_basis(m)) for m in ms]
        Vb = [q.subs(x, s) for q in shenp_basis(n)]
        Q = [mu.subs(x, s) for mu in shenp_basis(r)]
        CoupledProblem.__init__(self, Vp, Vb, Q, beam)

        self.params = params

    def Ap_matrix(self):
        'Stretching energy of the plate'
        # Ap = E(kron(A, M) + kron(M, A)), A = 1d laplace, M = 1d mass
        m_1d = self.ms[0]
        M = shen_poisson.mass_matrix(m_1d)
        A = shen_poisson.laplacian_matrix(m_1d)
        Ap = np.kron(A, M) + np.kron(M, A)

        # Don't forget mat. props
        E = self.params['plate_E']
        Ap *= E
        return Ap

    def Ab_matrix(self):
        'Stretching energy of the beam'
        # If the beam is straing things are simple
        if isinstance(self.beam, LineBeam):
            Ab = shen_poisson.laplacian_matrix(self.n)
            # Jacobian term
            J = float(self.beam.Jac)
            Ab /= J
        # For the curved need integration
        else:
            # (p, q/J**2)_b, The is J in beam inner so the end is 1/J
            # which is 2/L for the line
            Ab = np.zeros((self.n, self.n))
            J = self.beam.Jac
            for i, p in enumerate(self.Vb):
                Ab[i, i] = self.beam.inner_product(p/J**2, p)
                for j, q in enumerate(self.Vb[i+1], i+1):
                    Ab[i, j] = self.beam.inner_product(p/J**2, q)
                    Ab[j, i] = Ab[i, j]

        # Don't forget material properties
        E = self.params['beam_E']
        Ab *= E
        return Ab

    def Bb_matrix(self):
        'Matrix of the constraint on the beam'
        # Straight beam simple form
        if isinstance(self.beam, LineBeam):
            # Need (n, r) mass matrix between Q and Vp
            dim = max(self.n, self.r)
            Bb = shen_poisson.mass_matrix(dim)
            # Chop to proper size
            Bb = Bb[:self.n, :self.r]
            # Jacobian term, Jac is just a number
            Bb *= float(self.beam.Jac)
        # Otherwise use integration
        else:
            Bp = CoupledProblem.Bb_matrix(self)

        return Bb

    def C_matrix(self, norm):
        'H^norm matrices of Q'
        # Straight beam and 0, 1 reuse mass and stiffness
        if isinstance(self.beam, LineBeam) and norm in (0, 1):
            if norm == 0:
                C = shen_poisson.mass_matrix(self.r)
            else:
                C = shen_poisson.laplacian_matrix(self.r)
            # Remember the Jacobian
            J = self.beam.Jac   
            J = J**(1-2*norm)
            C *= J
        # Really only know how to do this for ints
        else:
            C  = CoupledProblem.C_matrix(self, norm)

        return C

def shen_laplace_Pblocks0(problem):
    'Something that might work as a preconditioner for CoupledLaplace problem'
    # Put inverses of Ap, Ab, H^0 on the diagonal
    Ap = la.inv(problem.Ap_matrix())
    Ab = la.inv(problem.Ab_matrix())
    D = problem.C_matrix(0)

    blocks = [[Ap, 0, 0],
              [0, Ab, 0],
              [0, 0, D]]

    return blocks


def shen_laplace_Pblocks1(problem):
    'Something that might work as a preconditioner for CoupledLaplace problem'
    # Put inverses of Ap, Ab, H^1 on the diagonal
    Ap = la.inv(problem.Ap_matrix())
    Ab = la.inv(problem.Ab_matrix())
    D = problem.C_matrix(1)

    blocks = [[Ap, 0, 0],
              [0, Ab, 0],
              [0, 0, D]]

    return blocks

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.plotting import plot3d, plot
    from itertools import product
    from sympy.mpmath import quad
    import numpy.linalg as la
    from sympy import S, lambdify, simplify

    # Beam
    A = np.array([-1., -1.])
    B = np.array([0., 1.])
    beam = LineBeam(A, B)
    # Material
    params = {'plate_E': 1.,
              'beam_E': 10.}

    # Check the matrices, [OK]
    if False:
        solver = CoupledLaplace(ms=[2, 2], n=2, r=2, beam=beam, params=params)

        Vp = [v0*(v1.subs(x, y))
              for v0, v1 in product(shenp_basis(2), shenp_basis(2))]
        Vb = [q.subs(x, s) for q in shenp_basis(2)]
        Q = [mu.subs(x, s) for mu in shenp_basis(2)]

        # Beam physics
        Ab = np.zeros((2, 2))
        for i, p in enumerate(Vb):
            for j, q in enumerate(Vb):
                pp = lambdify(s, p.diff(s, 1))
                qq = lambdify(s, q.diff(s, 1))
                Ab[i, j] = quad(lambda s: pp(s)*qq(s), [-1, 1])
        L = np.hypot(*(B-A))
        E = params['beam_E']
        Ab *= 2.*E/L
        assert np.allclose(Ab, solver.Ab_matrix(), 1E-13)
        print 'OK'

        # Beam constraint
        Bb = np.zeros((2, 2))
        for i, q in enumerate(Vb):
            for j, mu in enumerate(Q):
                u = lambdify(s, q)
                v = lambdify(s, mu)
                Bb[i, j] = quad(lambda s: u(s)*v(s), [-1, 1])
        Bb *= L/2.
        assert np.allclose(Bb, solver.Bb_matrix(), 1E-13)
        print 'OK'

        # Plate constraint
        Bp = np.zeros((4, 2))
        for i, v in enumerate(Vp):
            for j, q in enumerate(Q):
                u = beam.restrict(v)
                u = lambdify(s, u)
                p = lambdify(s, q)
                Bp[i, j] = quad(lambda s: u(s)*p(s), [-1, 1])
        Bp *= L/2.
        assert np.allclose(-Bp, solver.Bp_matrix(), 1E-13)
        print 'OK'

        # Norms
        for norm in [0, 1]:
            C = np.zeros((2, 2))
            for i, lmbda in enumerate(Q):
                for j, mu in enumerate(Q):
                    u = lambdify(s, lmbda.diff(s, norm))
                    v = lambdify(s, mu.diff(s, norm))
                    C[i, j] = quad(lambda s: v(s)*u(s), [-1, 1])

            C *= (L/2.)**(1-2*norm)
            assert np.allclose(C, solver.C_matrix(norm), 1E-13)
            print 'OK'

        assert np.allclose(solver.C_matrix(1),
                           shen_poisson.laplacian_matrix(len(Q))*2./L, 1E-13)
        assert np.allclose(solver.C_matrix(0),
                           shen_poisson.mass_matrix(len(Q))*L/2., 1E-13)
        print 'OK'

        # Plate physics
        Ap = np.zeros((4, 4))
        for i, v in enumerate(Vp):
            for j, u in enumerate(Vp):
                du = u.diff(x, 1), u.diff(y, 1)
                dv = v.diff(x, 1), v.diff(y, 1)
                term = du[0]*dv[0] + du[1]*dv[1]
                Ap[i, j] = quad(lambdify([x, y], term), [-1, 1], [-1, 1])
        assert np.allclose(Ap, solver.Ap_matrix(), 1E-13)
        print 'OK'

    # Solve the system with some f, seems [OK]
    if False: 
        # Spaces and solver
        solver = CoupledLaplace(ms=[8, 8], n=8, r=8, beam=beam, params=params)

        x, y, s = symbols('x, y, s')
        # Define problem
        # Force
        f = S(1)

        # Solve
        u, p, lmbda = solver.solve(f, as_sym=True)

        # Plot
        # Plate displacement
        plot3d(u, (x, -1, 1), (y, -1, 1), xlabel='$x$', ylabel='$y$',
            title='Plate deflection')
        # Displacement of beam and plate on the beam
        u_beam = beam.restrict(u)
        plot(p - u_beam, (s, -1, 1), xlabel='$s$',
             title='Beam deflection - plate deflection on the beam')
        # Lagrange multiplier
        plot(lmbda, (s, -1, 1), xlabel='$s$',
             title='Lagrange multiplier')

    # For different n, let's see about the eigenvalues
    if True:
        for n in range(2, 7):
            solver = CoupledLaplace(ms=[n, n], n=n, r=n,
                                    beam=beam, params=params)
        
            matrices = solver.schur_complement_matrix(norms=[None, 0, 1])
            print n,
            for mat in matrices:
                eigenvalues = la.eigvals(mat)
                eigenv_min = np.min(eigenvalues[-1])
                print eigenv_min,
            print
