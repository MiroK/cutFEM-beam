from points import gauss_legendre_lobatto_points as gl_points
from sympy import symbols, lambdify, Rational, diff
from functions import lagrange_basis
from quadrature import GLLQuadrature
from itertools import product
import scipy.linalg as la
import numpy as np


def solve_lagrange_2d(f, E, domain, m):
    [[ax, bx], [ay, by]] = domain

    x, y = symbols('x, y')
    nodes = gl_points([m])

    basis_functions = lagrange_basis([nodes])
    dbasis_functions = map(lambda f: diff(f, x, 1), basis_functions)

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
    F = np.array([[f_lambda(xi, yj) for yj in nodes] for xi in nodes])
    assert F.shape == (m, m)

    quad = GLLQuadrature(m)

    # Mass
    B = np.zeros((m, m))
    basis_lambda = map(lambda f: lambdify(x, f), basis_functions)
    for i, bi in enumerate(basis_lambda):
        B[i, i] = quad.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(basis_lambda[i+1:], i+1):
            B[i, j] = quad.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            B[j, i] = B[i, j]

    # Assemble the stiffness matrix for x direction
    A = np.zeros_like(B)
    dbasis_lambda = map(lambda f: lambdify(x, f), dbasis_functions)
    for i, bi in enumerate(dbasis_lambda):
        A[i, i] = quad.eval(lambda x: bi(x)*bi(x), [[-1, 1]])
        for j, bj in enumerate(dbasis_lambda[i+1:], i+1):
            A[i, j] = quad.eval(lambda x: bi(x)*bj(x), [[-1, 1]])
            A[j, i] = A[i, j]

    # Apply boundary conditions to A
    # Zero rows and cols
    A[0, :], A[-1, :], A[:, 0], A[:, -1] = 0, 0, 0, 0
    B[0, :], B[-1, :], B[:, 0], B[:, -1] = 0, 0, 0, 0
    # Puts ones
    A[0, 0], A[-1, -1] = 1, 1
    B[0, 0], B[-1, -1] = 1, 1

    # Right hand side is a projection
    b = (B.dot(F)).dot(B)
    b *= 0.25*Lx*Ly
    # Apply bcs to b
    b[0, :], b[-1, :], b[:, 0], b[:, -1] = 0, 0, 0, 0

    # Let's solve the discrete problem now
    lmbda, Q = la.eigh(A, B)

    # S
    # Map the right hand side to eigen space
    b_ = (Q.T).dot(b.dot(Q))
    # Apply the inverse in eigen space
    S_ = np.array([[b_[i, j]/(lmbda[i]*E*Ly/Lx + lmbda[j]*E*Lx/Ly)
                    for j in range(m)]
                    for i in range(m)])
    # Map back to physical space
    S = (Q).dot(S_.dot(Q.T))

    # U
    b = (B.dot(S)).dot(B)
    b *= 0.25*Lx*Ly

    b_ = (Q.T).dot(b.dot(Q))

    # Apply the inverse in eigen space
    U_ = np.array([[b_[i, j]/(lmbda[i]*E*Ly/Lx + lmbda[j]*E*Lx/Ly)
                    for j in range(m)]
                    for i in range(m)])
    # Map back to physical space
    U = (Q).dot(U_.dot(Q.T))

    # Map the basis back to [a, b]
    basis_functions_x = np.array(map(lambda f: f.subs({x: Pi_x}),
                                     basis_functions))
    basis_functions_y = np.array(map(lambda f: f.subs({x: Pi_y}),
                                     basis_functions))

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
    plots.plot(u, domain)

    U, basis = solve_lagrange_2d(f, m=8, E=E, domain=domain)
    plots.plot((U, basis), domain)
    e = errornorm(u, (U, basis), domain=domain, norm_type='L2')
    print e

    U, basis = solve_lagrange_2d(f, m=10, E=E, domain=domain)
    plots.plot((U, basis), domain)
    e = errornorm(u, (U, basis), domain=domain, norm_type='L2')
    print e

    U, basis = solve_lagrange_2d(f, m=12, E=E, domain=domain)
    plots.plot((U, basis), domain)
    e = errornorm(u, (U, basis), domain=domain, norm_type='L2')
    print e
