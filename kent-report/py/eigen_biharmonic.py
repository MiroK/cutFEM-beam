from __future__ import division
from sympy import Symbol, lambdify, exp, pi, symbols, S
from sympy.mpmath import quad
import numpy as np
import numpy.linalg as la
from math import log as ln, sqrt
from eigen_basis import eigen_basis
from itertools import product
from scipy.linalg import eigh


def biharmonic_matrix(m):
    'Matrix of 1d biharmonic operator(4th derivative) w.r.t to eigen basis'
    return np.diag([float((pi/2 + k*pi/2)**4) for k in range(m)])


def biharmonic_solver_1d(f, n, as_sym=False):
    '''
    Simple solver for u(4) = f in (-1, 1) with Dirichlet bcs on u and u(2).
    Uses eigenbasis.
    '''
    # Assemble stiffness matrix A, we know about A-orthogonality
    eigenvalues = [float((pi/2 + k*pi/2)**4) for k in range(n)]
    A = np.diag(eigenvalues)

    # Assemble right hand side b
    x = Symbol('x')
    f = lambdify(x, f)
    sym_basis = list(eigen_basis(n))
    basis = [lambdify(x, v) for v in sym_basis]
    b = np.zeros(n)
    for i, v in enumerate(basis):
        b[i] = quad(lambda x: v(x)*f(x), [-1, 1])

    # Solve and return solution - assembled linear combination
    U = la.solve(A, b)
    if as_sym:
        uh = sum(Uk*v for Uk, v in zip(U, sym_basis))
    else:
        uh = lambda x: sum(Uk*v(x) for Uk, v in zip(U, basis))
    return uh

def biharmonic_solver_2d(f, n, as_sym=False):
    '''
    Simple solver for laplace(laplace) = f in (-1, 1)^2 with zero boundary 
    values on u and laplace(u). Uses tensor product of eigenbasis.
    '''
    # Use the tensor product method so only 1d matrices are needed
    # Eigenvalues of laplacian
    C_eigenvalues = np.array([(pi/2 + k*pi/2)**2 for k in range(n)],
                             dtype='float')
    # Eigenvalues of biharmonic operator
    A_eigenvalues = C_eigenvalues**2
    A = np.diag(A_eigenvalues)
    # Mass matrix
    M = np.eye(n)

    # Create tensor product basis
    x, y = symbols('x, y')
    sym_basis = [vx*(vy.subs(x, y))
                 for vx, vy in product(eigen_basis(n), eigen_basis(n))]
    basis = [lambdify([x, y], v) for v in sym_basis]

    # Assemble right hand side b
    f = lambdify([x, y], f)
    b = np.zeros((n, n))
    for k, v in enumerate(basis):
        i, j = k // n, k % n
        b[i, j] = quad(lambda x, y: v(x, y)*f(x, y), [-1, 1], [-1, 1])

    # Solve by tensor product method - very easy - no mappings :)
    U = np.array([[b[i, j]/(A_eigenvalues[i] + \
                            2*C_eigenvalues[i]*C_eigenvalues[j] + \
                            A_eigenvalues[j])
                   for j in range(n)]
                  for i in range(n)])
    # Flatten the 2d representation so match the basis
    U = U.flatten()
    assert len(U) == len(basis)

    # Assemble the solution as a linear combination
    if as_sym:
        uh = sum(Uk*v for Uk, v in zip(U, sym_basis))
    else:
        uh = lambda x, y: sum(Uk*v(x, y) for Uk, v in zip(U, basis))
    return uh

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import exp, integrate, sin, pi, simplify
    from numpy.linalg import solve, lstsq
    # Test the biharmonic solvers
    # Rate in L2 norm should be 4

    # 1D
    # Exact solution computed from f
    x = Symbol('x')
    f = x*exp(x)
    # u4 = f
    u3 = integrate(f, x)
    u2 = integrate(u3, x)
    u1 = integrate(u2, x)
    u = integrate(u1, x) # + ax^3/6 + bx^/2 + cx + d

    mat = np.array([[-1/6, 1/2, -1, 1],
                    [1/6, 1/2, 1, 1],
                    [-1, 1, 0, 0],
                    [1, 1, 0, 0]])

    vec = np.array([-u.subs(x, -1),
                    -u.subs(x, 1),
                    -u2.subs(x, -1),
                    -u2.subs(x, 1)])

    a, b, c, d = la.solve(mat, vec)
    u += a*x**3/6 + b*x**2/2 + c*x + d

    # Check that it is the solution
    assert abs(u.evalf(subs={x: -1})) < 1E-15
    assert abs(u.evalf(subs={x: 1})) < 1E-15
    assert abs(u.diff(x, 2).evalf(subs={x: -1})) < 1E-15
    assert abs(u.diff(x, 2).evalf(subs={x: 1})) < 1E-15
    assert (simplify(u.diff(x, 4)) - f) == 0

    # Right hand side
    u = lambdify(x, u)
    # Numerical solution
    ns = range(2, 16)
    print '1D'

    # Colect data for least square fit of the rate
    b = []
    col0 = []
    for n in ns:
        uh = biharmonic_solver_1d(f, n)
        # L2 Error as lambda
        e = lambda x: (u(x) - uh(x))**2
        error = sqrt(quad(e, [-1, 1]))

        if n > ns[0]:
            rate = ln(error/error_)/ln(n_/n)
            print '\t', n, error, rate

        error_ = error
        n_ = n

        b.append(ln(error))
        col0.append(ln(n))

    # The rate should be e = n**(-p) + shift
    A = np.ones((len(b), 2))
    A[:, 0] = col0
    ans = lstsq(A, b)[0]
    p = -ans[0]
    print '\tLeast square rate %.2f' % p

    # 2d
    # Does this satisfy u, u`` = 0 on the boundary
    y = symbols('y')
    u = (x-1)**2*(x+1)**2*sin(pi*x)*(y-1)**2*(y+1)**2*sin(-pi*y)
    # Boundary values
    assert u.subs(x, 1) == 0
    assert u.subs(x, -1) == 0
    assert u.subs(y, 1) == 0
    assert u.subs(y, -1) == 0
    # Combined these would make up laplacian on the boundary
    assert u.diff(x, 2).subs(x, 1) == 0
    assert u.diff(x, 2).subs(x, -1) == 0
    assert u.diff(y, 2).subs(y, 1) == 0
    assert u.diff(y, 2).subs(y, -1) == 0

    # If here, we can compute f for which u is the solution
    u_xx = u.diff(x, 2)
    u_yy = u.diff(y, 2)

    u_xxxx = u_xx.diff(x, 2)
    u_xxyy = u_xx.diff(y, 2)
    u_yyxx = u_yy.diff(x, 2)
    u_yyyy = u_yy.diff(y, 2)
    # Here it is
    f = u_xxxx + u_xxyy + u_yyxx + u_yyyy

    # We are ready for numerics
    u = lambdify([x, y], u)
    # Numerical solution
    ns = range(2, 11)
    print '2D'
    # Data for least square
    b = []
    col0 = []
    for n in ns:
        uh = biharmonic_solver_2d(f, n)
        # L2 Error as lambda
        e = lambda x, y: (u(x, y) - uh(x, y))**2
        error, integral_error = quad(e, [-1, 1], [-1, 1],
                                     maxdegree=40,
                                     error=True)
        error = sqrt(error)

        if n > ns[0]:
            rate = ln(error/error_)/ln(n_/n)
            print '\t', n, error, rate, '[%4e]' % integral_error

        error_ = error
        n_ = n

        b.append(ln(error))
        col0.append(ln(n))

    # The rate should be e = n**(-p) + shift
    A = np.ones((len(b), 2))
    A[:, 0] = col0
    ans = lstsq(A, b)[0]
    p = -ans[0]
    print '\tLeast square rate %.2f' % p
