from sympy import symbols, lambdify, diff, Rational
from functions import lagrange_basis
import numpy.linalg as la
import numpy as np


def solve_lagrange_nitsche_1d(f, N, E, a, b, points, quadrature, formulation):
    '''
    Solve Poisson problem
                         -E*u^(2) = f in [a, b]
                             u = 0 on a, b

    In the variational formulation use N lagrange basis functions with nodes
    in points. Assemble matrices with quadratures. Unlike foo here bcs are
    set weakly with nitsche method.

    formulation : 'sym' -- symmetric Nitsche
                  'skew' -- skew symmetric Nitsche
    '''
    # Get the nodal points and create lagrange basis functions such that
    # basis_f_i(xj) ~ \delta_{ij}
    x = symbols('x')
    nodes = points([N])
    # Functions
    basis_functions = lagrange_basis([nodes])
    # Derivatives for stiffness matrix
    dbasis_functions = map(lambda f: diff(f, x, 1), basis_functions)

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

    # For GL points N point quadrature is exact for all the terms
    quad = quadrature(N)

    # Assemble the stiffness matrix, using symmetry
    A = np.zeros((N, N))
    dbasis_lambda = map(lambda f: lambdify(x, f), dbasis_functions)
    # dx contribution
    for i, bi in enumerate(dbasis_lambda):
        A[i, i] = quad.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(dbasis_lambda[i+1:], i+1):
            A[i, j] = quad.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            A[j, i] = A[i, j]

    # Assemble the bdry term consistency
    C = np.zeros_like(A)
    basis_lambda = map(lambda f: lambdify(x, f), basis_functions)
    for i, bi in enumerate(basis_lambda):
        for j, dbj in enumerate(dbasis_lambda):
            C[i, j] = bi(1)*dbj(1) - bi(-1)*dbj(-1)

    # Put together the system matrix
    if formulation == 'skew':
        # Stiffness, consistency (skew) symmetry
        A += -C + C.T
    elif formulation == 'sym':
        # Need to assemble the penalty term
        P = np.zeros_like(A)
        for i, bi in enumerate(basis_lambda):
            P[i, i] = bi(1)*bi(1) - bi(-1)*bi(-1)
            for j, bj in enumerate(basis_lambda[i+1:], i+1):
                P[i, j] = bi(1)*bj(1) - bi(-1)*bj(-1)
                P[j, i] = P[i, j]

        # Multiply the penalty
        P *= 10*N      # ad-hoc

        # Stiffness, consistency, symetry, penalty
        A += -C - C.T + P

    # Material props
    A *= E
    # Scale
    A *= 2./L

    # Assemble the mass matrix, using symmetry
    B = np.zeros_like(A)
    for i, bi in enumerate(basis_lambda):
        B[i, i] = quad.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(basis_lambda[i+1:], i+1):
            B[i, j] = quad.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            B[j, i] = B[i, j]
    # Scale
    B *= 0.5*L

    # Right hand side is a projection
    b = B.dot(F)

    # Solve the system to get expansion coeffs for basis of [-1, 1]
    U = la.solve(A, b)
    # Map the basis back to [a, b]
    basis_functions = np.array(map(lambda f: f.subs({x: Pi_x}),
                                   basis_functions))
    return (U, basis_functions)


def solve_lagrange_babuska_1d(f, N, E, a, b, points, quadrature):
    '''
    Solve Poisson problem
                         -E*u^(2) = f in [a, b]
                             u = 0 on a, b

    In the variational formulation use N lagrange basis functions with nodes
    in points. Assemble matrices with quadratures. Unlike foo here bcs are
    set weakly with babuska's lagrange multiplier method.
    '''
    # Get the nodal points and create lagrange basis functions such that
    # basis_f_i(xj) ~ \delta_{ij}
    x = symbols('x')
    nodes = points([N])
    # Functions
    basis_functions = lagrange_basis([nodes])
    # Derivatives for stiffness matrix
    dbasis_functions = map(lambda f: diff(f, x, 1), basis_functions)

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

    # For GL points N point quadrature is exact for all the terms
    quad = quadrature(N)

    # Assemble the stiffness matrix, using symmetry
    A = np.zeros((N, N))
    dbasis_lambda = map(lambda f: lambdify(x, f), dbasis_functions)
    # dx contribution
    for i, bi in enumerate(dbasis_lambda):
        A[i, i] = quad.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(dbasis_lambda[i+1:], i+1):
            A[i, j] = quad.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            A[j, i] = A[i, j]

    # Material props
    A *= E
    # Scale
    A *= 2./L

    # Assemble the multiplier term
    # Basis of function space over boundary
    lbasis = [0.5*(x+1), 0.5*(1-x)]
    lbasis_lambda = map(lambda f: lambdify(x, f), lbasis)
    # Lambdify basis of space over domain
    basis_lambda = map(lambda f: lambdify(x, f), basis_functions)
    C = np.zeros((N, 2))
    for i, phi_i in enumerate(basis_lambda):
        for j, psi_j in enumerate(lbasis_lambda):
            C[i, j] = phi_i(1)*psi_j(1) - phi_i(-1)*psi_j(-1)

    # Assemble the mass matrix, using symmetry
    B = np.zeros_like(A)
    for i, bi in enumerate(basis_lambda):
        B[i, i] = quad.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(basis_lambda[i+1:], i+1):
            B[i, j] = quad.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            B[j, i] = B[i, j]
    # Scale
    B *= 0.5*L

    # Right hand side is a projection
    b = B.dot(F)

    # Put together the block system
    AA = np.zeros((N+2, N+2))
    AA[:N, :N] = A
    AA[:N, N:] = C
    AA[N:, :N] = C.T

    bb = np.zeros(N+2)
    bb[:N] = b

    # Solve the system to get expansion coeffs for basis of [-1, 1]
    U_lambda = la.solve(AA, bb)
    # Don't need the multiplier
    U = U_lambda[:N]
    # Map the basis back to [a, b]
    basis_functions = np.array(map(lambda f: f.subs({x: Pi_x}),
                                   basis_functions))
    return (U, basis_functions)


def solve_lagrange_weak_1d(f, N, E, a, b, points, quadrature, formulation):
    'Dispatch Poisson solver'
    formulations = ['sym', 'skew', 'babuska']
    if formulation not in formulations:
        raise ValueError

    if formulation in ['sym', 'skew']:
        return solve_lagrange_nitsche_1d(f, N, E, a, b, points, quadrature,
                                         formulation)
    else:
        return solve_lagrange_babuska_1d(f, N, E, a, b, points, quadrature)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from points import gauss_legendre_points as gl_points
    from points import chebyshev_points as ch_points
    from quadrature import GLQuadrature
    from problems import manufacture_poisson_1d
    from quadrature import errornorm
    import plots

    # 1d
    x = symbols('x')
    a = -1.
    b = 3.
    E = 2.3
    u = (x-a)*(x-b)**2
    problem1d = manufacture_poisson_1d(u=u, a=a, b=b, E=E)
    f = problem1d['f']

    (U, basis) = solve_lagrange_weak_1d(f=f, N=4, E=E, a=a, b=b,
                                        points=ch_points,  # gl_points
                                        quadrature=GLQuadrature,
                                        formulation='babuska')

    plots.plot(u, [[a, b]])
    plots.plot((U, basis), [[a, b]])

    e = errornorm(u, (U, basis), domain=[[a, b]], norm_type='L2')
    assert abs(e) < 1E-13
