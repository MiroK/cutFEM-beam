from sympy import sin, pi, sqrt, legendre, symbols, lambdify, cos
import matplotlib.pyplot as plt
from sympy.mpmath import quad
import numpy.linalg as la
import numpy as np

x = symbols('x')


def fourier_basis(N):
    'Basis of eigenvectors of laplacian on (-1, 1)'
    cosines = [cos((pi/2 + i*pi)*x) for i in range(N/2)]
    sines = [sin((i+1)*pi*x) for i in range(N/2)]

    basis = []
    for cosine, sine in zip(cosines, sines):
        basis.append(cosine)
        basis.append(sine)
    return basis


def legendre_basis(N):
    'Legendre basis'
    return [legendre(i, x) for i in range(1, N+1)]


def shen_basis_1(N):
    '''
    Shen basis that has functions that are zero on (-1, 1). Yields diagonal
    stiffness matrix and tridiag. mass matrix.
    '''
    return [(legendre(i, x) - legendre(i+2, x))/sqrt(4*i + 6)
            for i in range(N-1)]


def shen_basis_2(N):
    '''
    Shen basis that has functions that are zero on (-1, 1). Leads to dense
    matrices.
    '''
    return [legendre(i+2, x)-(legendre(0, x) if (i%2) == 0 else legendre(1, x))
            for i in range(N-1)]


def assemble_mass_matrix(basis, domain=[-1, 1]):
    'Mass matrix in given basis.'
    n = len(basis)
    _basis = map(lambda f: lambdify(x, f), basis)
    M = np.zeros((n, n))
    for i, bi in enumerate(_basis):
        M[i, i] = quad(lambda x: bi(x)**2, domain)
        for j, bj in enumerate(_basis[i+1:], i+1):
            M[i, j] = quad(lambda x: bi(x)*bj(x), domain)
            M[j, i] = M[i, j]
    return M


def assemble_stiffness_matrix(basis, domain=[-1, 1]):
    'Stiffness matrix in given basis.'
    n = len(basis)
    _basis = map(lambda f: lambdify(x, f.diff(x, 1)), basis)
    A = np.zeros((n, n))
    for i, bi in enumerate(_basis):
        A[i, i] = quad(lambda x: bi(x)**2, domain)
        for j, bj in enumerate(_basis[i+1:], i+1):
            A[i, j] = quad(lambda x: bi(x)*bj(x), domain)
            A[j, i] = A[i, j]
    return A


def laplace2d_matrix(basis, domain=[-1, 1]):
    '''
    Assemble matrix of the 2d laplace problem on [-1, 1] x [-1, 1].
    '''
    M = assemble_mass_matrix(basis, domain)
    A = assemble_stiffness_matrix(basis, domain)

    Mat = np.kron(A, M) + np.kron(M, A)
    return Mat


def cond(space, domain=[-1, 1]):
    '''
    Compute condition numbers of the 2d laplace problem assembled of subspaces
    of given space.
    '''
    numbers = []
    for i in range(2, len(space)):
        subspace = space[:i]
        mat = laplace2d_matrix(subspace, domain)
        numbers.append(la.cond(mat))
    return numbers


def spectrum(basis, fourier_basis):
    '''
    Compute the fourier series matrix for going from fourier basis to basis
    '''
    if isinstance(basis, list):
        return np.array([spectrum(f, fourier_basis) for f in basis])
    else:
        return np.array([quad(lambdify(x, basis*f), [-1, 1])**2
                         for f in fourier_basis])

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    N = 20
    # Arrange N such that the basis have equal length
    fourier = fourier_basis(N)
    leg = legendre_basis(N)
    shen1 = shen_basis_1(N+1)
    shen2 = shen_basis_2(N+1)

    if False:
        plt.figure()
        for label, basis in {'fourier': fourier,
                             'shen1': shen1,
                             'shen2': shen2,
                             'ledendre': leg}.items():
            print label
            conds = cond(basis)
            # Start from 2 so that fourier gives straight line of slope 2
            n = range(2, len(conds)+2)
            plt.loglog(n, conds, label=label)
        plt.legend(loc='best')
        plt.show()
        # For polynomial spaces the condition number grows exponentialy

    # The transformation matrices are square and invertible
    # Its condition number grows exponentially as well
    if True:
        fig, axarr = plt.subplots(3, 1, sharex=True, sharey=True)
        F_shen1 = spectrum(shen1, fourier)
        for row in F_shen1:
            axarr[0].plot(row)

        F_shen2 = spectrum(shen2, fourier)
        for row in F_shen2:
            axarr[1].plot(row)

        F_leg = spectrum(leg, fourier)
        for row in F_leg:
            axarr[2].plot(row)

        plt.show()










