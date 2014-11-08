from scipy.sparse.linalg import LinearOperator, cg
from sympy import symbols, lambdify, diff, Rational
from functions import lagrange_basis
from itertools import product
import scipy.linalg as la
import numpy as np


def solve_lagrange_2d(f, E, domain, MN, points, quadrature, method):
    '''
    Solve Poisson problem
                         -E*laplace(u) = f in [[ax, bx], [ay, by]]
                                     u = 0 on the boundary

    In the variational formulation use m=MN[0] and n=MN[1] lagrange basis
    functions with nodes in points in x-direction and y-direction. Assemble
    matrices with quadrature.

    With
        method == 'operator' we solve the problem Au=b where A is an operator
        acting on vector of lenght m*n

        method == 'tensor' the solution is sought using eigenvalue problems
        of size mxm and nxn
    '''
    [[ax, bx], [ay, by]] = domain
    [M, N] = MN

    # Get the nodal points and create lagrange basis functions such that
    # basis_f_i(xj) ~ \delta_{ij} for x-direction and y_direction
    x, y = symbols('x, y')
    nodes_x = points([M])
    nodes_y = points([N])

    # Functions and their derivatives to be used in stiffness matrix and
    # mass matrix for x direction
    basis_functions_x = lagrange_basis([nodes_x])
    dbasis_functions_x = map(lambda f: diff(f, x, 1), basis_functions_x)

    # Functions and their derivatives to be used in stiffness matrix and
    # mass matrix for y direction
    basis_functions_y = lagrange_basis([nodes_y], xi=1)
    dbasis_functions_y = map(lambda f: diff(f, y, 1), basis_functions_y)

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
    F = np.array([[f_lambda(xi, yj) for yj in nodes_y] for xi in nodes_x])
    assert F.shape == (M, N)

    quadM = quadrature(M)
    # Assemble the stiffness matrix in x
    B_m = np.zeros((M, M))
    basis_lambda_x = map(lambda f: lambdify(x, f), basis_functions_x)
    for i, bi in enumerate(basis_lambda_x):
        B_m[i, i] = quadM.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(basis_lambda_x[i+1:], i+1):
            B_m[i, j] = quadM.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            B_m[j, i] = B_m[i, j]
    # Scale
    B_m *= 0.5*Lx

    quadN = quadrature(N)
    # Assemble the stiffness matrix in y
    B_n = np.zeros((N, N))
    basis_lambda_y = map(lambda f: lambdify(y, f), basis_functions_y)
    for i, bi in enumerate(basis_lambda_y):
        B_n[i, i] = quadN.eval(lambda y: bi(y)*bi(y), [[-1, 1]])
        for j, bj in enumerate(basis_lambda_y[i+1:], i+1):
            B_n[i, j] = quadN.eval(lambda y: bi(y)*bj(y), [[-1, 1]])
            B_n[j, i] = B_n[i, j]
    # Scale
    B_n *= 0.5*Ly

    # Assemble the stiffness matrix for x direction
    A_m = np.zeros_like(B_m)
    dbasis_lambda_x = map(lambda f: lambdify(x, f), dbasis_functions_x)
    for i, bi in enumerate(dbasis_lambda_x):
        A_m[i, i] = quadM.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(dbasis_lambda_x[i+1:], i+1):
            A_m[i, j] = quadM.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            A_m[j, i] = A_m[i, j]
    # Material props
    A_m *= E
    # Scale
    A_m *= 2./Lx

    # Assemble the stiffness matrix for x direction
    A_n = np.zeros_like(B_n)
    dbasis_lambda_y = map(lambda f: lambdify(y, f), dbasis_functions_y)
    for i, bi in enumerate(dbasis_lambda_y):
        A_n[i, i] = quadN.eval(lambda y: bi(y)*bi(y), [[-1, 1]])
        for j, bj in enumerate(dbasis_lambda_y[i+1:], i+1):
            A_n[i, j] = quadN.eval(lambda y: bi(y)*bj(y), [[-1, 1]])
            A_n[j, i] = A_n[i, j]
    # Material props
    A_n *= E
    # Scale
    A_n *= 2./Ly

    # Apply boundary conditions to A
    # Zero rows and cols
    A_m[0, :], A_m[-1, :], A_m[:, 0], A_m[:, -1] = 0, 0, 0, 0
    A_n[0, :], A_n[-1, :], A_n[:, 0], A_n[:, -1] = 0, 0, 0, 0
    B_m[0, :], B_m[-1, :], B_m[:, 0], B_m[:, -1] = 0, 0, 0, 0
    B_n[0, :], B_n[-1, :], B_n[:, 0], B_n[:, -1] = 0, 0, 0, 0
    # Puts ones
    A_m[0, 0], A_m[-1, -1] = 1, 1
    A_n[0, 0], A_n[-1, -1] = 1, 1
    B_m[0, 0], B_m[-1, -1] = 1, 1
    B_n[0, 0], B_n[-1, -1] = 1, 1

    # Right hand side is a projection
    b = (B_m.dot(F)).dot(B_n)          # mxn matrix
    # Apply bcs to b
    b[0, :], b[-1, :], b[:, 0], b[:, -1] = 0, 0, 0, 0

    # Let's solve the discrete problem now
    if method not in ['operator', 'tensor']:
        raise ValueError('method %s not supported' % method)

    elif method == 'operator':
        # Flatten everything
        b_flat = b.flatten()  # m,n vector collapsed by row

        # Define A as an m.n x m.n opeerator acting on m.n vector
        def matvec(vec):
            # Applying first A_m diad B_n
            dir_x = A_m.dot(np.vstack([vec[i*N:(i+1)*N].dot(B_n)
                                    for i in range(M)]))
            # Applying B_m diad A_n
            dir_x += B_m.dot(np.vstack([vec[i*N:(i+1)*N].dot(A_n)
                                        for i in range(M)]))

            return dir_x.flatten()

        A = LinearOperator((M*N, M*N), matvec=matvec, dtype='float64')

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
        U = U_flat.reshape((M, N))

    elif method == 'tensor':
        pass



    # Map the basis back to [a, b]
    basis_functions_x = np.array(map(lambda f: f.subs({x: Pi_x}),
                                     basis_functions_x))
    basis_functions_y = np.array(map(lambda f: f.subs({y: Pi_y}),
                                     basis_functions_y))

    # Create the basis phi(x, y)
    basis_functions = np.array([phi_x*psi_y
                                for phi_x, psi_y in product(basis_functions_x,
                                                            basis_functions_y)
                                ]).reshape((M, N))
    return (U, basis_functions)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from points import gauss_legendre_lobatto_points as gll_points
    from quadrature import GLLQuadrature
    from problems import manufacture_poisson_2d
    from quadrature import errornorm
    import plots

    # 2d
    x, y = symbols('x, y')
    ax, bx = -1, 2
    ay, by = 0, 2
    E = 2.5
    domain = [[ax, bx], [ay, by]]

    u = (x-ax)*(x-bx)*(y-ay)*(y-by)
    problem = manufacture_poisson_2d(u=u, domain=domain, E=E)
    f = problem['f']

    (U, basis) = solve_lagrange_2d(f=f, E=E, domain=domain, MN=[3, 5],
                                   points=gll_points, quadrature=GLLQuadrature,
                                   method='operator')

    plots.plot(u, domain)
    plots.plot((U, basis), domain)

    e = errornorm(u, (U, basis), domain=domain, norm_type='L2')
    print e
