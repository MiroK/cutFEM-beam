from points import gauss_legendre_lobatto_points as gl_points
from scipy.sparse.linalg import LinearOperator, cg
from sympy import symbols, lambdify, Rational, diff
from functions import lagrange_basis
from quadrature import GLQuadrature
from itertools import product
import numpy as np


def solve_lagrange_2d(f, E, domain, m, method):
    '''
    For now only 4-th order, as operator and equal order in x, y
    '''
    [[ax, bx], [ay, by]] = domain
    x, y = symbols('x, y')

    nodes_m = gl_points([m])
    quad_m = GLQuadrature(m)
    basis_functions_m = lagrange_basis([nodes_m])
    # Make lambads of basis
    basis_lambda_m = map(lambda f: lambdify(x, f), basis_functions_m)
    # Derivetives are only needed as lambdas
    d1_basis_functions_m = map(lambda f: lambdify(x, diff(f, x, 1)),
                               basis_functions_m)
    d2_basis_functions_m = map(lambda f: lambdify(x, diff(f, x, 2)),
                               basis_functions_m)

    # Everything is assembled in [[-1, 1], [-1, 1]]. We need to map the load
    # vector and  scale the matrices
    # Scaling factors
    Lx, Ly = bx - ax, by - ay

    # The mapping (x, y) in [[ax, bx], [ay, by]] <-
    # P((s, t)) = (0.5*(1-z)*ax + 0.5*(1+s)*bx,0.5*(1-t)*ay + 0.5*(1+t)*by
    # for (s, t) in [[-1, 1], [-1, 1]]
    # We only work with components
    P_s = Rational(ax, 2)*(1-x) + (1+x)*Rational(bx, 2)
    P_t = Rational(ay, 2)*(1-y) + (1+y)*Rational(by, 2)
    # For mapping the basis functions back to we need inverses
    Pi_x = Rational(2, Lx)*x - Rational(bx+ax, Lx)
    Pi_y = Rational(2, Ly)*y - Rational(by+ay, Ly)

    f_ref = f.subs({x: P_s, y: P_t})
    # Make f for fast evaluation
    f_lambda = lambdify([x, y], f_ref)

    # Represent f_ref in the space H^1([[-1, 1], [-1, 1]]) spanned by
    # basis_functions F is a mxn matrix
    F = np.array([[f_lambda(xi, yj) for yj in nodes_m] for xi in nodes_m])
    assert F.shape == (m, m)

    # Assembling the matrices
    # Matrix of biharmonic problem ddf_i * ddf_j
    B = np.zeros((m, m))
    for i, bi in enumerate(d2_basis_functions_m):
        B[i, i] = quad_m.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(d2_basis_functions_m[i+1:], i+1):
            B[i, j] = quad_m.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            B[j, i] = B[i, j]
    Bx = B*E*(2./Lx)**3
    By = B*E*(2./Ly)**3

    # Matrix of the laplacian df_i * df_j
    A = np.zeros_like(B)
    for i, bi in enumerate(d1_basis_functions_m):
        A[i, i] = quad_m.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
        for j, bj in enumerate(d1_basis_functions_m[i+1:], i+1):
            A[i, j] = quad_m.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            A[j, i] = A[i, j]
    Ax = A*E*(2./Lx)
    Ay = A*E*(2./Ly)

    # Mass matrix f_i* f_J
    M = np.zeros_like(A)
    for i, bi in enumerate(basis_lambda_m):
        M[i, i] = quad_m.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
        for j, bj in enumerate(basis_lambda_m[i+1:], i+1):
            M[i, j] = quad_m.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            M[j, i] = M[i, j]
    Mx = M*(Lx/2.)
    My = M*(Ly/2.)

    # Apply the boundary conditions
    for mat in [Bx, By, Ax, Ay]:
        # Zeros
        mat[0, :], mat[-1, :], mat[:, 0], mat[:, -1] = 0, 0, 0, 0
        # Ones
        mat[0, 0], mat[-1, -1] = 1, 1

    # Assemble the rhs
    # Right hand side is a projection
    b = (Mx.dot(F)).dot(My)
    # Apply boundary conditions
    b[0, :], b[-1, :], b[:, 0], b[:, -1] = 0, 0, 0, 0

    # Let's solve the discrete problem now
    if method not in ['operator', 'tensor']:
        raise ValueError('method %s not supported' % method)

    elif method == 'operator':
        # Flatten everything
        b_flat = b.flatten()  # m,n vector collapsed by row

        # Define A as an m.n x m.n opeerator acting on m.n vector
        def matvec(vec):
            # Applying Bx.T diad My
            y = Bx.dot(np.vstack([vec[i*m:(i+1)*m].dot(My) for i in range(m)]))
            # Applying Ax.T diad A.x
            #y += 2*Ax.dot(np.vstack([vec[i*m:(i+1)*m].dot(Ay) for i in range(m)]))
            # Applying Mx.T diad My
            y += Mx.dot(np.vstack([vec[i*m:(i+1)*m].dot(By) for i in range(m)]))

            return y.flatten()

        A = LinearOperator((m*m, m*m), matvec=matvec, dtype='float64')

        # Counting iterations of cg
        class Counter(object):
            def __init__(self, n=0):
                self.n = n
            def __call__(self, x):
                self.n += 1
            def __str__(self):
                return str(self.n)

        # Solve the system to get expansion coeffs for basis
        # of [[-1, 1], [-1, 1]]
        iter_counter = Counter()
        U_flat, info = cg(A, b_flat, tol=1E-12, callback=iter_counter)
        assert info == 0

        print 'CG finished in', iter_counter, 'iterations'
        U = U_flat.reshape((m, m))

    elif method == 'tensor':
        raise NotImplementedError

    # Map the basis back to [a, b]
    basis_functions_x = np.array(map(lambda f: f.subs({x: Pi_x}),
                                     basis_functions_m))
    # Note that everything is defined in terms of x, its's the substitution
    # that makes the basis fuctions of y
    basis_functions_y = np.array(map(lambda f: f.subs({x: Pi_y}),
                                     basis_functions_m))

    # Create the basis phi(x, y)
    basis_functions = np.array([phi_x*psi_y
                                for phi_x, psi_y in product(basis_functions_x,
                                                            basis_functions_y)
                                ]).reshape((m, m))

    return (U, basis_functions)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from problems import manufacture_biharmonic_2d
    from quadrature import errornorm
    from sympy import sin, pi
    import plots

    x, y = symbols('x, y')
    E = 1
    ax, bx = -1, 1.
    ay, by = -1, 1.
    domain = [[ax, bx], [ay, by]]
    u = sin(pi*(x-ax)/(bx-ax))*sin(pi*(y-ay)/(by-ay))

    problem = manufacture_biharmonic_2d(u=u, domain=domain, E=E)
    u = problem['u']
    f = problem['f']

    U, basis = solve_lagrange_2d(f, m=10, E=E, domain=domain, method='operator')

    plots.plot(u, domain)
    plots.plot((U, basis), domain)

    # e = errornorm(u, (U, basis), domain=domain, norm_type='L2')
