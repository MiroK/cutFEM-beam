# Here we investigate properties of polynomials basis of Legendre polynomials
from __future__ import division
from sympy.mpmath import legendre, sqrt, quad, diff
import matplotlib.pyplot as plt
from functools import partial
import numpy.linalg as la
import numpy as np
import sys

# Take n legendre polynomials, these are polynials of degree 0, ..., n
m = int(sys.argv[1])

if  True:
    basis = [lambda x, i=i: partial(legendre, n=i)(x=x)
             for i in range(m)]

    d_basis = [lambda x, f=f: diff(f, x) for f in basis]

    np.set_printoptions(precision=2)
    m = len(basis)

    # Mass
    M = np.zeros((m, m))
    for i, bi in enumerate(basis):
        M[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
        for j, bj in enumerate(basis[i+1:], i+1):
            M[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
            M[j, i] = M[i, j]

    M_ = np.zeros((m, m))
    for i, bi in enumerate(basis):
        M_[i, i] = 2/(2*i + 1)
    assert la.norm(M - M_)/m < 1E-15

    A = np.zeros((m, m))
    for i, bi in enumerate(d_basis):
        A[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
        for j, bj in enumerate(d_basis[i+1:], i+1):
            A[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
            A[j, i] = A[i, j]

    A_ = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            A_[i, j] = 0.5*i*(i+1)*(1+(-1)**(i+j))
            if ((i+j) % 2) == 0 and j < i-1:
                A_[i, j] -= (i*(i+1) - j*(j+1))
    assert la.norm(A - A_)/m < 1E-15

    # Penalty
    P = np.zeros((m, m))
    for i, bi in enumerate(basis):
        P[i, i] = bi(1)*bi(1) + bi(-1)*bi(-1)
        for j, bj in enumerate(basis[i+1:], i+1):
            P[i, j] = bi(1)*bj(1) + bi(-1)*bj(-1)
            P[j, i] = P[i, j]

    P_ = np.zeros((m, m))
    for i, bi in enumerate(basis):
        P_[i, i] = 1 + (-1)**(2*i)
        for j, bj in enumerate(basis[i+1:], i+1):
            P_[i, j] = 1 + (-1)**(i+j)
            P_[j, i] = P_[i, j]
    assert la.norm(P-P_)/m < 1E-15

    print P_
    eigenvalues, eigenvectors = la.eig(P_)
    print eigenvalues
    for v in eigenvectors.T:
        print v

    #  Bdry
    G = np.zeros((m, m))
    for i, bi in enumerate(basis):
        for j, dbj in enumerate(d_basis):
            G[i, j] = bi(1)*dbj(1) - bi(-1)*dbj(-1)

    G_ = np.zeros((m, m))
    for i, bi in enumerate(basis):
        for j, dbj in enumerate(d_basis):
            G_[i, j] = 0.5*j*(j+1)*(1 + (-1)**(i+j))
    assert la.norm(G-G_)/m < 1E-15

    # Shen basis
    def c(k):
        return sqrt(4*k + 6)**-1

    # We construct a new basis that has always zero endpoint values
    shen_basis = [lambda x, i=i: c(i)*(partial(legendre, n=i)(x=x) -
                                       partial(legendre, n=i+2)(x=x))
                  for i in range(m-2)]

    d_shen_basis = [lambda x, f=f: diff(f, x) for f in shen_basis]

    n = len(shen_basis)
    # Transformation matrix from basis to shen_basis
    alpha = np.zeros((n, m))
    for i in range(n):
        alpha[i, i] = c(i)
        alpha[i, i+2] = -alpha[i, i]

    # Get the shen_basis mass matrix
    Mshen = alpha.dot(M_.dot(alpha.T))
    # Compare with the one from paper
    Mshen_ = np.zeros((n, n))
    for i in range(n):
        Mshen_[i, i] = c(i)*c(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, n):
            Mshen_[i, j] = -c(i)*c(j)*(2./(2*j+1)) if i+2 == j else 0
            Mshen_[j, i] = Mshen_[i, j]

    assert la.norm(Mshen - Mshen_)/n < 1E-15

    # Get the shen_basis stiffness matrix
    Ashen = alpha.dot(A_.dot(alpha.T))
    Ashen_ = np.eye(n)
    assert la.norm(Ashen - Ashen_)/n < 1E-15

    # Get the bdry matrix
    Gshen = alpha.dot(G_.dot(alpha.T))
    Gshen_ = np.zeros((n, n))
    for i, bi in enumerate(shen_basis):
        for j, dbj in enumerate(d_shen_basis):
            Gshen_[i, j] = bi(1)*dbj(1) - bi(-1)*dbj(-1)

    assert la.norm(Gshen - Gshen_)/n < 1E-15

    # Get the penalty matrix
    Pshen = alpha.dot(P_.dot(alpha.T))
    Pshen_ = np.zeros((n, n))
    for i, bi in enumerate(shen_basis):
        for j, bj in enumerate(shen_basis):
            Pshen_[i, j] = bi(1)*bj(1) + bi(-1)*bj(-1)

    assert la.norm(Pshen - Pshen_)/n < 1E-15

    # Shen basis has functions that are zero on the boundary
    print 'penalty'
    print Pshen_
    print 'boundary'
    print Gshen_

    # The claim now is that shen's choice is a special case of alpha where
    # alpha.T = [cols in ker of P_]*diag(c), K the matrix is called K
    K = np.zeros((m, m-2))
    for i in range(m-2):
        K[i, i] = 1
        K[i+2, i] = -1

    D = np.diag([c(i) for i in range(m-2)])
    alpha = (K.dot(D)).T

    # Now need the basis [phi, ..., ] where phi_i = alpha_ij*legendre_j
    def assemble_basis(alpha):
        basis = [lambda x, row=row: sum(col*legendre(n=i, x=x)
                                        for i, col in enumerate(row))
                 for row in alpha]
        return basis

    foo_basis = assemble_basis(alpha)

    # compare with plot
    x_values = np.linspace(-1, 1, 100)
    plt.figure()
    for f, s in zip(foo_basis, shen_basis):
        yf_values = map(f, x_values)
        ys_values = map(s, x_values)
        plt.plot(x_values, yf_values, label='foo')
        plt.plot(x_values, ys_values, label='shen')
    # plt.show()

    # But D can be anything and the resulting P will still be zero
    D = np.diag(np.random.rand(m-2))
    # Note that 0 diagonal entry make mass matrix loose definiteness
    # Negative values are allowed ?
    for i in range(m-2, 2):
        D[i, i] *= -1


    alpha = (K.dot(D)).T
    any_basis = assemble_basis(alpha)
    # Assemble Pany
    Pany = np.zeros((n, n))
    for i, bi in enumerate(any_basis):
        for j, bj in enumerate(any_basis):
            Pany[i, j] = bi(1)*bj(1) + bi(-1)*bj(-1)
    print Pany

    plt.figure()
    for i, v in enumerate(any_basis):
        y_values = map(v, x_values)
        plt.plot(x_values, y_values, label='$\phi_{%d}$' % i)
    plt.legend(loc='best')
    # plt.show()

    # Does any basis make the stiffness matrix diagonal
    d_any_basis = [lambda x, f=f: diff(f, x) for f in any_basis]

    Aany = np.zeros((n, n))
    for i, bi in enumerate(d_any_basis):
        Aany[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
        for j, bj in enumerate(d_any_basis[i+1:], i+1):
            Aany[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
            Aany[j, i] = Aany[i, j]
    print Aany

    plt.figure()
    plt.spy(Aany, precision=1E-13)
    plt.show()
# -----------------------------------------------------------------------------

if False:

    def c(k):
        return sqrt(4*k + 6)**-1

    # We construct a new basis that has always zero endpoint values
    basis = [lambda x, i=i: c(i)*(partial(legendre, n=i)(x=x) -
                                partial(legendre, n=i+2)(x=x))
            for i in range(n)]

    # The end point values should be zero
    assert all(map(lambda f: f(-1) == 0, basis))
    assert all(map(lambda f: f(1) == 0, basis))

    # This basis should yield 3-banded mass matrix
    np.set_printoptions(precision=2)
    n = len(basis)
    M = np.zeros((n, n))
    for i, bi in enumerate(basis):
        M[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
        for j, bj in enumerate(basis[i+1:], i+1):
            M[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
            M[j, i] = M[i, j]
    # With values
    M_ = np.zeros_like(M)
    for i in range(n):
        M_[i, i] = c(i)*c(i)*(2./(2*i + 1) + 2./(2*(i+2) + 1))
        for j in range(i+1, n):
            M_[i, j] = -c(i)*c(j)*(2./(2*j+1)) if i+2 == j else 0
            M_[j, i] = M_[i, j]

    assert la.norm(M-M_)/n < 1E-14

    # Tha stiffness matrix should be identity
    A = np.zeros((n, n))
    d_basis = [lambda x, f=f: diff(f, x) for f in basis]
    for i, bi in enumerate(d_basis):
        A[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
        for j, bj in enumerate(d_basis[i+1:], i+1):
            A[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
            A[j, i] = A[i, j]

    assert la.norm(A-np.eye(n))/n < 1E-14

    C = np.zeros((n, n))
    for i, bi in enumerate(d_basis):
        for j, bj in enumerate(basis):
            C[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])

    C_ = np.zeros((n, n))
    for i, bi in enumerate(d_basis):
        for j, bj in enumerate(basis[i+1:], i+1):
            C_[i, j] = -2*c(i)*c(j) if j == i+1 else 0
            C_[j, i] = -C[i, j]

    assert la.norm(C-C_)/n < 1E-14

    # Not sparse
    D = np.zeros((n, n))
    dd_basis = [lambda x, f=f: diff(f, x, 2) for f in basis]
    for i, bi in enumerate(dd_basis):
        D[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
        for j, bj in enumerate(dd_basis[i+1:], i+1):
            D[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
            D[j, i] = D[i, j]

if False:
    # New basis that has first and second derivative 0 and D is diagonal
    def d(k):
        return sqrt(2*(2*k + 3)**2*(2*k + 5))**-1

    # We construct a new basis that has always zero endpoint values
    basis = [lambda x, i=i: d(i)*(partial(legendre, n=i)(x=x) -
                                (2.*(2*i+5)/(2*i+7))*partial(legendre, n=i+2)(x=x) +
                                ((2.*i+3)/(2*i+7))*partial(legendre, n=i+4)(x=x))
            for i in range(n)]

    d_basis = [lambda x, f=f: diff(f, x) for f in basis]
    dd_basis = [lambda x, f=f: diff(f, x, 2) for f in basis]

    print 'L_i(-1)', map(lambda f: f(-1), basis)
    print 'L_i(1)', map(lambda f: f(1), basis)
    print 'L_i`(-1)', map(lambda f: f(-1), d_basis)
    print 'L_i`(1)', map(lambda f: f(1), d_basis)
    print 'L_i``(-1)', map(lambda f: f(-1), dd_basis)
    print 'L_i``(1)', map(lambda f: f(1), dd_basis)


    D = np.zeros((n, n))
    for i, bi in enumerate(dd_basis):
        D[i, i] = quad(lambda x: bi(x)*bi(x), [-1, 1])
        for j, bj in enumerate(dd_basis[i+1:], i+1):
            D[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
            D[j, i] = D[i, j]
    print D
