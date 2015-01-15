from coupled_eigen import CoupledEigen
from plate_beam import LineBeam
from eigen_basis import eigen_basis
import eigen_poisson
import eigen_biharmonic
from sympy import symbols
from math import pi, sqrt
import numpy as np
import numpy.linalg as la

x, y, s = symbols('x, y, s')

class CoupledEigenBiharmonic(CoupledEigen):
    'Physics of both beam and plate are due to biharmonic operator.'
    def __init__(self, ms, n, r, beam, params):
        CoupledEigen.__init__(self, ms, n, r, beam, params)

    def Ap_matrix(self):
        'Bending energy of the plate'
        # Ap = E(kron(B, M) + 2kron(A, A) + kron(M, B))
        # A = 1d laplace, M = 1d mass, B = 1d biharmonic
        m_1d = self.ms[0]
        M = eigen_poisson.mass_matrix(m_1d)
        A = eigen_poisson.laplacian_matrix(m_1d)
        B = eigen_biharmonic.biharmonic_matrix(m_1d)
        Ap = np.kron(B, M) + 2*np.kron(A, A) + np.kron(M, B)

        # Don't forget mat. props
        E = self.params['plate_E']
        Ap *= E
        return Ap

    def Ab_matrix(self):
        'Bending energy of the beam'
        # If the beam is straing things are simple
        if isinstance(self.beam, LineBeam):
            Ab = eigen_biharmonic.biharmonic_matrix(self.n)
            # Jacobian term
            J = float(self.beam.Jac)
            Ab /= J**3
        # For the curved need integration
        else:
            # (p, q/J**4)_b, The is J in beam inner so the end is 1/J**2
            # which is (2/L)**3 for the line
            Ab = np.zeros((self.n, self.n))
            J = self.beam.Jac
            for i, p in enumerate(self.Vb):
                Ab[i, i] = self.beam.inner_product(p/J**4, p)
                for j, q in enumerate(self.Vb[(i+1):], i+1):
                    Ab[i, j] = self.beam.inner_product(p/J**4, q)
                    Ab[j, i] = Ab[i, j]

        # Don't forget material properties
        E = self.params['beam_E']
        Ab *= E
        return Ab

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.plotting import plot3d, plot
    from itertools import product
    from sympy.mpmath import quad
    import numpy.linalg as la
    from sympy import S, lambdify, simplify, cos, pi, sin, integrate
    from plate_beam import Beam

    # Beam
    A = np.array([-1., -1.])
    B = np.array([0., 1.])
    beam = LineBeam(A, B)
    # Material
    params = {'plate_E': 2.,
              'beam_E': 10.}

    # Chech the matrix assembly
    if False:
        solver = CoupledEigenBiharmonic(ms=[2, 2], n=2, r=2, beam=beam,
                                        params=params)

        Vp = [v0*(v1.subs(x, y))
              for v0, v1 in product(eigen_basis(2), eigen_basis(2))]
        Vb = [q.subs(x, s) for q in eigen_basis(2)]
        Q = [mu.subs(x, s) for mu in eigen_basis(2)]

        # Beam physics
        Ab = np.zeros((2, 2))
        for i, p in enumerate(Vb):
            for j, q in enumerate(Vb):
                pp = lambdify(s, p.diff(s, 2))
                qq = lambdify(s, q.diff(s, 2))
                Ab[i, j] = quad(lambda s: pp(s)*qq(s), [-1, 1])
        L = np.hypot(*(B-A))
        E = params['beam_E']
        Ab *= E*((2./L)**3)
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
        for norm in [0, 1, 2]:
            C = np.zeros((2, 2))
            for i, lmbda in enumerate(Q):
                for j, mu in enumerate(Q):
                    u = lambdify(s, lmbda.diff(s, norm))
                    v = lambdify(s, mu.diff(s, norm))
                    C[i, j] = float(quad(lambda s: v(s)*u(s), [-1, 1]))

            C *= (L/2.)**(1-2*norm)
            assert np.allclose(C, solver.C_matrix(norm), 1E-13)
            print 'OK'

        assert np.allclose(solver.C_matrix(2),
                           eigen_biharmonic.biharmonic_matrix(len(Q))*(2./L)**3,
                                                              1E-13)
        assert np.allclose(solver.C_matrix(1),
                           eigen_poisson.laplacian_matrix(len(Q))*2./L, 1E-13)
        assert np.allclose(solver.C_matrix(0),
                           eigen_poisson.mass_matrix(len(Q))*L/2., 1E-13)
        print 'OK'

        # Check if biharmonic Ap_matrix is correct
        m = int(sqrt(solver.m))
        C_eigenvalues = np.array([(pi/2 + k*pi/2)**2
                                  for k in range(m)], dtype='float')
        # Eigenvalues of biharmonic operator
        A_eigenvalues = C_eigenvalues**2
        # Diagonal
        Ap = np.array([(A_eigenvalues[i] + \
                         2*C_eigenvalues[i]*C_eigenvalues[j] + \
                         A_eigenvalues[j])
                         for j in range(m)
                        for i in range(m)])
        Ap = np.diag(Ap)*params['plate_E']
        assert np.allclose(Ap, solver.Ap_matrix(), 1E-13)
        print 'OK'

        # Plate physics
        Ap = np.zeros((4, 4))
        for i, u in enumerate(Vp):
            for j, v in enumerate(Vp):
                ddu = u.diff(x, 2) + u.diff(y, 2)
                ddv = v.diff(x, 2) + v.diff(y, 2)
                term = ddu * ddv
                # print simplify(term),
                # print float(integrate(integrate(term, (x, -1, 1)), (y, -1, 1))),
                val = quad(lambdify([x, y], term), [-1, 1], [-1, 1])
                # print val
                Ap[i, j] = val
        Ap *= params['plate_E']
        assert np.allclose(Ap, solver.Ap_matrix(), 1E-12)
        print 'OK'

    # Solve the system with some f, seems [OK]
    if False: 
        A = np.array([0.5, -1.])
        B = np.array([0., 1.])
        beam = LineBeam(A, B)

        params = {'plate_E': 1, 'beam_E': 100}

        # Spaces and solver
        solver = CoupledEigenBiharmonic(ms=[8, 8], n=8, r=8,
                                        beam=beam, params=params)

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
            solver = CoupledEigenBiharmonic(ms=[n, n], n=n, r=n,
                                            beam=beam, params=params)
        
            norms = [None, 0, 0.5, 1, 1.5, 2]
            matrices = solver.schur_complement_matrix(norms=norms)
            print n,
            for mat in matrices:
                eigenvalues = la.eigvals(mat)
                eigenv_min = np.min(eigenvalues[-1])
                print eigenv_min,
            print
