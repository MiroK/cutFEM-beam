from sympy import symbols, lambdify, diff, Rational
from functions import lagrange_basis
import numpy.linalg as la
import numpy as np


def solve_lagrange_1d(f, N, E, a, b, points, quadrature):
    '''
    Solve Poisson problem
                         -E*u^(2) = f in [a, b]
                             u = 0 on a, b

    In the variational formulation use N lagrange basis functions with nodes
    in points. Assemble matrices with quadratures.
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

    quad = quadrature(N)

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

    # Assemble the stiffness matrix
    A = np.zeros_like(B)
    dbasis_lambda = map(lambda f: lambdify(x, f), dbasis_functions)
    for i, bi in enumerate(dbasis_lambda):
        A[i, i] = quad.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(dbasis_lambda[i+1:], i+1):
            A[i, j] = quad.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            A[j, i] = A[i, j]
    # Material props
    A *= E
    # Scale
    A *= 2./L

    # Right hand side is a projection
    b = B.dot(F)

    # Apply the boundary conditions
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

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from points import gauss_legendre_lobatto_points as gll_points
    from quadrature import GLLQuadrature
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

    (U, basis) = solve_lagrange_1d(f=f, N=7, E=E, a=a, b=b,
                                   points=gll_points, quadrature=GLLQuadrature)

    plots.plot(u, [[a, b]])
    plots.plot((U, basis), [[a, b]])

    e = errornorm(u, (U, basis), domain=[[a, b]], norm_type='L2')
    assert abs(e) < 1E-13
