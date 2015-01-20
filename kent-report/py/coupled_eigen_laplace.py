from coupled_eigen import CoupledEigen
from plate_beam import LineBeam
from eigen_basis import eigen_basis
import eigen_poisson
import eigen_biharmonic
from sympy import symbols
from math import pi, sqrt
import numpy as np
import numpy.linalg as la
import scipy.linalg as spla

x, y, s = symbols('x, y, s')

class CoupledEigenLaplace(CoupledEigen):
    'Physics of both beam and plate are due to Laplacian.'
    def __init__(self, ms, n, r, beam, params):
        CoupledEigen.__init__(self, ms, n, r, beam, params)

    def Ap_matrix(self):
        'Stretching energy of the plate'
        # Ap = E(kron(A, M) + kron(M, A)), A = 1d laplace, M = 1d mass
        m_1d = self.ms[0]
        M = eigen_poisson.mass_matrix(m_1d)
        A = eigen_poisson.laplacian_matrix(m_1d)
        Ap = np.kron(A, M) + np.kron(M, A)

        # Don't forget mat. props
        E = self.params['plate_E']
        Ap *= E
        return Ap

    def Ab_matrix(self):
        'Stretching energy of the beam'
        # If the beam is straing things are simple
        if isinstance(self.beam, LineBeam):
            Ab = eigen_poisson.laplacian_matrix(self.n)
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
                for j, q in enumerate(self.Vb[(i+1):], i+1):
                    Ab[i, j] = self.beam.inner_product(p/J**2, q)
                    Ab[j, i] = Ab[i, j]

        # Don't forget material properties
        E = self.params['beam_E']
        Ab *= E
        return Ab


def eigen_laplace_Pblocks01(problem):
    '''
    Something that might work as a preconditioner for CoupledEigenLaplace 
    problem
    '''
    # Put inverses of Ap, Ab, H^0.5 on the diagonal
    Ap = la.inv(problem.Ap_matrix())
    Ab = la.inv(problem.Ab_matrix())
    D = problem.C_matrix(0.5)

    blocks = [[Ap, 0, 0],
              [0, Ab, 0],
              [0, 0, D]]

    return blocks


def eigen_laplace_Pblocks1(problem):
    '''
    Something that might work as a preconditioner for CoupledEigenLaplace
    problem
    '''
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
    from sympy import S, lambdify, simplify, cos, pi, sin, integrate
    from plate_beam import Beam

    # This is just for fun - a curved beam
    if False:
        # Something more exotic
        chi = (-1 + 2*cos(pi*(s+1)/4), -1 + 2*sin(pi*(s+1)/4))
        beam = Beam(chi)
        params = {'plate_E': 1.,
                  'beam_E': 40.}
        # Spaces and solver
        solver = CoupledEigenLaplace(ms=[8, 8], n=8, r=8, beam=beam, params=params)

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

    # More serious tests with linear beam
    A = np.array([-1., -1.])
    B = np.array([0., 1.])
    beam = LineBeam(A, B)
    # Material
    params = {'plate_E': 2.,
              'beam_E': 10.}

    # Check matrix assembly
    if False:
        solver = CoupledEigenLaplace(ms=[2, 2], n=2, r=2, beam=beam,
                                     params=params)

        Vp = [v0*(v1.subs(x, y))
              for v0, v1 in product(eigen_basis(2), eigen_basis(2))]
        Vb = [q.subs(x, s) for q in eigen_basis(2)]
        Q = [mu.subs(x, s) for mu in eigen_basis(2)]

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
                           eigen_poisson.laplacian_matrix(len(Q))*2./L, 1E-13)
        assert np.allclose(solver.C_matrix(0),
                           eigen_poisson.mass_matrix(len(Q))*L/2., 1E-13)
        print 'OK'

        # Plate physics
        Ap = np.zeros((4, 4))
        for i, v in enumerate(Vp):
            for j, u in enumerate(Vp):
                du = u.diff(x, 1), u.diff(y, 1)
                dv = v.diff(x, 1), v.diff(y, 1)
                term = du[0]*dv[0] + du[1]*dv[1]
                Ap[i, j] = E*quad(lambdify([x, y], term), [-1, 1], [-1, 1])
        assert np.allclose(Ap, solver.Ap_matrix(), 1E-13)
        print 'OK'

    # Solve the system with some f
    if False: 
        # Spaces and solver
        solver = CoupledEigenLaplace(ms=[8, 8], n=8, r=8, beam=beam,
                                     params=params)

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

    if True:
        # For different n, let's see about the eigenvalues of Schur
        for n in range(2, 16):
            solver = CoupledEigenLaplace(ms=[n, n], n=n, r=n,
                                         beam=beam, params=params)
       
            norms = [None, 0, 0.5, 1]
            matrices = solver.schur_complement_matrix(norms=norms)
            for norm, mat in zip(norms[1:], matrices[1:]):
                eigenvalues = la.eigvals(mat)
                eigenv_min = np.min(eigenvalues[-1])
                print eigenv_min,

                eigenvalues = spla.eigvals(matrices[0],
                        la.inv(solver.C_matrix(norm)))
                eigenv_min = np.min(eigenvalues[-1])
                print '[%g]' % eigenv_min,
            
            
            print

        print

        # For different n, let's see about condition number of the system
        # and preconditioned system
        for n in range(2, 1):
            solver = CoupledEigenLaplace(ms=[n, n], n=n, r=n,
                                         beam=beam, params=params)
            # No preconditioning
            S = solver.system_matrix()
            print 'S', la.cond(S),

            # Make preconditioner, just an example
            blocks = eigen_laplace_Pblocks(solver)
            # These block go into the diagonal
            P = solver.preconditioner(blocks)
            SP = P.dot(S)
            print 'SP', la.cond(SP)
