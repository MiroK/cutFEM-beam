'''
For 1d biharmonic problems we can

(i)
    use Lagrange polynomials at gll points and
    (a) the mixed formulation to control both u and u``
    (b) u`` are enforced weakly
(ii)
    points = gl_points or cheb_points
    (a) use mixed formulation and enforce bdry values weakly by nitsche
    (b) use Lagrange polynomials at points with sym nitsche
        to enforce weakly u. u`` is enforced weakly

* I could also do (ii) with enforcing bdry values by babuska but that will
have to wait
'''

from points import gauss_legendre_lobatto_points as gll_points
from points import gauss_legendre_points as gl_points
from points import chebyshev_points as cheb_points
from quadrature import GLQuadrature, GLLQuadrature
from sympy import symbols, lambdify, diff, Rational
from functions import lagrange_basis
from functools import partial
import numpy.linalg as la
import numpy as np


def solve_biharmonic_1d_BAA(f, N, E, a, b, points):
    '''
    The problem is recast into

                               -sqrt(E)*u^(2) = sigma
                           -sqrt(E)*sigma^(2) = f        in (a, b)
                                            u = 0
                                        sigma = 0        in {a, b}
    It consists of two laplacians A_m, A_n
    [[B, -A_m], [[S],     [[0],
     [A_n, 0]]  [U]]     [b]]

     Solve A_n*S = b
     Solve A_m*U = S

    The space for u and sigma have N[0] and N[1] nodes respectively.
    '''
    assert points in ['gll', 'gl', 'cheb']
    x = symbols('x')

    # Now decide M, N
    try:
        [M, N] = N
    except TypeError:
        # If not list make everything square
        return solve_biharmonic_1d_BAA(f, [N, N], E, a, b, points)

    # Make a decision for points and quadrature based on their type and M, N
    if points == 'gll':
        nodes = gll_points
        quad = GLLQuadrature
    elif points == 'gl':
        nodes = gl_points
        quad = GLQuadrature
    else:
        nodes = cheb_points
        quad = GLQuadrature

    x = symbols('x')

    nodes_u = nodes([M])
    basis_functions_u = lagrange_basis([nodes_u])
    d_basis_functions_u = map(lambda f: diff(f, x, 1), basis_functions_u)
    quadM = quad(M)

    if M == N:
        basis_functions_s = basis_functions_u
        d_basis_functions_s = d_basis_functions_u
        quadN = quadM
    else:
        basis_functions_s = lagrange_basis([nodes([N])])
        d_basis_functions_s = map(lambda f: diff(f, x, 1), basis_functions_s)
        quadN = quad(N)

    # Everything is assembled in [-1, 1]. We need to map the load vector and
    # scale the matrices
    # Scaling factor
    L = b - a
    # The mapping x in [a, b] <- P(z) = 0.5*(1-z)*a + 0.5*(1+z)*b, z in [-1, 1]
    P_z = Rational(a, 2)*(1-x) + (1+x)*Rational(b, 2)
    # For mapping the basis functions back to [a, b] we will need inverse of P_
    Pi_x = Rational(2, L)*x - Rational(b+a, L)
    f_ref = f.subs({x: P_z})
    # Make f for fast evaluation
    f_lambda = lambdify(x, f_ref)

    # Assemble the m x m matrix for laplacian on u-space
    Am = np.zeros((M, M))
    d_basis_lambda_u = map(lambda f: lambdify(x, f), d_basis_functions_u)
    for i, bi in enumerate(d_basis_lambda_u):
        Am[i, i] = quadM.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(d_basis_lambda_u[i+1:], i+1):
            Am[i, j] = quadM.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            Am[j, i] = Am[i, j]
    # Material props and scale
    Am *= np.sqrt(E)*(2./L)

    # Assemble the n x n matrix for laplacian on s-space
    if M == N:
        An = Am
    else:
        An = np.zeros((N, N))
        d_basis_lambda_s = map(lambda f: lambdify(x, f), d_basis_functions_s)
        for i, bi in enumerate(d_basis_lambda_s):
            An[i, i] = quadN.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
            for j, bj in enumerate(d_basis_lambda_s[i+1:], i+1):
                An[i, j] = quadN.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
                An[j, i] = An[i, j]
        # Material props and scale
        An *= np.sqrt(E)*(2./L)

    # Assemble the m x n mass matrix of the two spaces
    quad = quadM if M >= N else quadN
    B = np.zeros((M, N))
    basis_lambda_u = map(lambda f: lambdify(x, f), basis_functions_u)
    basis_lambda_s = map(lambda f: lambdify(x, f), basis_functions_s)
    for i, bi in enumerate(basis_lambda_u):
        for j, bj in enumerate(basis_lambda_s):
            B[i, j] = quad.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
    # Scale
    B *= 0.5*L
    # Compute the load vector
    # Represent f_ref in the space H^1(-1, 1) spanned by basis_functions of u
    # F is vector of expansion coefficients that has length M
    F = np.array([f_lambda(xj) for xj in nodes_u])
    b = B.T.dot(F)
    # Apply bcs strong with gll
    if points == 'gll':
        b[0] = 0
        b[-1] = 0
        An[0, :], An[:, 0], An[0, 0] = 0, 0, 1
        An[-1, :], An[:, -1], An[-1, -1] = 0, 0, 1

        Am[0, :], Am[:, 0], Am[0, 0] = 0, 0, 1
        Am[-1, :], Am[:, -1], Am[-1, -1] = 0, 0, 1

    S = la.solve(An, b)
    b = B.dot(S)
    U = la.solve(Am, b)

    # Map the basis back to [a, b]
    basis_functions_u = np.array(map(lambda f: f.subs({x: Pi_x}),
                                     basis_functions_u))
    return (U, basis_functions_u)


def solve_biharmonic_1d_A(f, N, E, a, b, points):
    '''
    We only solve the displacement. Boundary conditions on laplace are
    imposed weakly. The boundary values are enforced by the choice of
    space or weakly - this is decided by points
    '''
    assert points in ['gll', 'gl', 'cheb']
    x = symbols('x')

    # Decide points and quadrature
    if points == 'gll':
        nodes = gll_points([N])
        quad = GLLQuadrature(N)
    elif points == 'gl':
        nodes = gl_points([N])
        quad = GLQuadrature(N)
    else:
        nodes = cheb_points([N])
        quad = GLQuadrature(N)

    # Get the nodal points and create lagrange basis functions such that
    # basis_f_i(xj) ~ \delta_{ij}
    # Functions
    basis_functions = lagrange_basis([nodes])
    # Second derivatives
    dd_basis_functions = map(lambda f: diff(f, x, 2), basis_functions)

    # Everything is assembled in [-1, 1]. We need to map the load vector and
    # scale the matrices
    # Scaling factor
    L = b - a
    # The mapping x in [a, b] <- P(z) = 0.5*(1-z)*a + 0.5*(1+z)*b, z in [-1, 1]
    P_z = Rational(a, 2)*(1-x) + (1+x)*Rational(b, 2)
    # For mapping the basis functions back to [a, b] we will need inverse of P_
    Pi_x = Rational(2, L)*x - Rational(b+a, L)
    f_ref = f.subs({x: P_z})
    # Make f for fast evaluation
    f_lambda = lambdify(x, f_ref)

    # Represent f_ref in the space H^1(-1, 1) spanned by basis_functions
    # F is vector of expansion coefficients
    F = np.array([f_lambda(xj) for xj in nodes])

    # Assemble the mass matrix, using symmetry
    B = np.zeros((N, N))
    basis_lambda = map(lambda f: lambdify(x, f), basis_functions)
    for i, bi in enumerate(basis_lambda):
        B[i, i] = quad.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(basis_lambda[i+1:], i+1):
            B[i, j] = quad.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            B[j, i] = B[i, j]
    # Scale
    B *= 0.5*L
    # Mass matrix is only used to compute the rhs
    b = B.dot(F)

    # Now matrices for the lhs
    # Assemble the stiffness matrix  u``*v``
    A = np.zeros_like(B)
    dd_basis_lambda = map(lambda f: lambdify(x, f), dd_basis_functions)
    for i, bi in enumerate(dd_basis_lambda):
        A[i, i] = quad.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(dd_basis_lambda[i+1:], i+1):
            A[i, j] = quad.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            A[j, i] = A[i, j]
    # Material props and scale
    A *= E*(2./L)**3

    # With gll apply bcs strongly
    if points == 'gll':
        # Apply the boundary conditions to have u = 0
        b[0] = 0
        b[-1] = 0

        A[0, :] = 0
        A[:, 0] = 0
        A[0, 0] = 1

        A[-1, :] = 0
        A[:, -1] = 0
        A[-1, -1] = 1

    # Solve the system to get expansion coeffs for basis of [-1, 1]
    U = la.solve(A, b)
    # Map the basis back to [a, b]
    basis_functions = np.array(map(lambda f: f.subs({x: Pi_x}),
                                   basis_functions))
    return (U, basis_functions)


def solve_biharmonic_1d(f, N, E, a, b, method):
    '''
    Solve biharmonic problem
                         E*u^(4) = f in [a, b]
                               u = 0 on a, b
                             u`` = 0 on a, b

    In the variational formulation use N lagrange basis functions.

    method : 'gll_mixed' gives (i)(a)
             'gll_weak`  gives (i)(b)
             'gl_mixed'        gives (ii)(a) with gll pts
             'gl_weak'      gives (ii)(b)
             'cheb_mixed'        gives (ii)(a) with cheb pts
             'cheb_weak'      gives (ii)(b)

             the mixed method can provide N as list of len 2 of points for
             u and sigma.
    '''
    methods = {'gll_mixed': partial(solve_biharmonic_1d_BAA, points='gll'),
               'gll_weak': partial(solve_biharmonic_1d_A, points='gll'),
               'gl_mixed': partial(solve_biharmonic_1d_BAA, points='gl'),
               'gl_weak': partial(solve_biharmonic_1d_A, points='gl'),
               'cheb_mixed': partial(solve_biharmonic_1d_BAA, points='cheb'),
               'cheb_weak': partial(solve_biharmonic_1d_A, points='cheb')}
    assert method in methods

    return methods[method](f, N, E, a, b)

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from problems import manufacture_biharmonic_1d
    from quadrature import errornorm
    from sympy import sin, pi
    import plots

    x = symbols('x')
    a = -1
    b = 1.5
    E = 2
    u = sin(pi*(x-a)/(b-a))
    problem1d = manufacture_biharmonic_1d(u=u, a=a, b=b, E=E)
    f = problem1d['f']

    U, basis = solve_biharmonic_1d(f, N=[9, 8], E=E, a=a, b=b,
                                   method='gll_mixed')
    plots.plot(u, [[a, b]])
    plots.plot((U, basis), [[a, b]])
    e = errornorm(u, (U, basis), domain=[[a, b]], norm_type='L2')
    print e
