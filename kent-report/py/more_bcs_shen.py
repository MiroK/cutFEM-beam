from __future__ import division
from sympy import legendre, Symbol, lambdify
from sympy.mpmath import quad
import numpy as np

class ShenBasis(object):
    '''
    Polynomials of the form f_k = l_k + alpha_k * l_{k+1} + beta_k * l_{k+2},
    where l_k is the k-th Legendre polynomials. Coefficients alpha_k, beta_k
    are used to meet boundary conditions on -1, 1. Mass matrix and stiffness
    matrix w.r.t to this basis is sparse.
    '''
    def __init__(self):
        pass

    def alpha(self, k):
        raise NotImplementedError('Specify in child!')

    def beta(self, k):
        raise NotImplementedError('Specify in child!')

    def functions(self, n):
        'Yield first n basis functions'
        x = Symbol('x')
        k = 0
        while k < n:
            yield legendre(k, x) + \
                  self.alpha(k)*legendre(k+1, x) + \
                  self.beta(k)*legendre(k+2, x)
            k += 1

    def mass_matrix(self, n):
        'Identity w.r.t between functions(n) and functions(n)'
        M = np.zeros((n, n))
        for i in range(n):
            # Main diagonal
            M[i, i] = 2/(2*i + 1) + \
                      self.alpha(i)**2*2/(2*i + 3) + \
                      self.beta(i)**2*2/(2*i + 5)
            if i+1 < n:
                # First diagonal
                M[i, i+1] = self.alpha(i)*2/(2*i + 3) + \
                            self.beta(i)*self.alpha(i+1)*2/(2*i+5)
                # Symmetry
                M[i+1, i] = M[i, i+1]
                if i+2 < n:
                    # Second diagonal
                    M[i, i+2] = self.beta(i)*2/(2*i + 5)
                    # Symmetry
                    M[i+2, i] = M[i, i+2]
        return M

    def laplacian_matrix(self, n):
        'Matrix of Laplacian between functions(n) and functions(n)'
        A = np.diag([-self.beta(k)*(4*k + 6) for k in range(n)])
        return A


class ShenBasisNeumann(ShenBasis):
    'Polynomials have u`(-1) = u`(1) = 0'
    def __init__(self):
        ShenBasis.__init__(self)

    def alpha(self, k):
        return 0

    def beta(self, k):
        return - k*(k+1)/(k+2)/(k+3)

    def functions(self, n, noconstant=False):
        '''
        Yield functions from basis. If noconstant the first function 1 is
        skipped yielding only functions that are in L^2_0(-1, 1).
        '''
        for i, f in enumerate(ShenBasis.functions(self, n)):
            if noconstant and i == 0:
                pass
            else:
                yield f


class ShenBasisMixed(ShenBasis):
    'Polynomials have u(-1) = u`(1) = 0'
    def __init__(self):
        ShenBasis.__init__(self)

    def alpha(self, k):
        return (2*k+3)/(k+2)**2

    def beta(self, k):
        return -(k+1)**2/(k+2)**2


def test_mass_matrix(cls, n):
    'Verify properties of n x n mass matrix build from class'
    # Get functions
    basis = cls()
    functions = basis.functions(n)
    # Lamdify
    x = Symbol('x')
    functions = [lambdify(x, f) for f in functions]
    # Build numeric
    M = np.zeros((n, n))
    for i, v in enumerate(functions):
        M[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(functions):
            M[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            M[j, i] = M[i, j]
    # Compare
    return np.allclose(M, basis.mass_matrix(n), 1E-13)


def test_laplacian_matrix(cls, n):
    'Verify properties of n x n Laplacian matrix build from class'
    # Get functions
    basis = cls()
    functions = basis.functions(n)
    # Lamdify derivatives
    x = Symbol('x')
    functions = [lambdify(x, f.diff(x, 1)) for f in functions]
    # Build numeric
    A = np.zeros((n, n))
    for i, v in enumerate(functions):
        A[i, i] = quad(lambda x: v(x)**2, [-1, 1])
        for j, u in enumerate(functions):
            A[i, j] = quad(lambda x: u(x)*v(x), [-1, 1])
            A[j, i] = A[i, j]
    # Compare
    return np.allclose(A, basis.laplacian_matrix(n), 1E-13)


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import integrate, sqrt
    # Check boundary values
    x = Symbol('x')
    for f in ShenBasisNeumann().functions(20, noconstant=True):
        assert abs(f.diff(x, 1).subs(x, -1)) < 1E-8
        assert abs(f.diff(x, 1).subs(x, 1)) < 1E-8
        # Just for fun see about the Poincare inequality
        # is_L20 = abs(integrate(f, (x, -1, 1))) < 1E-8
        # if is_L20:
        #     L2_norm = sqrt(integrate(f**2, (x, -1, 1)))
        #     H1_seminorm = sqrt(integrate(f.diff(x, 1)**2, (x, -1, 1)))
        #     K = L2_norm/H1_seminorm
        #     print 'L2=%g H1=%g -> K=%g' % (L2_norm, H1_seminorm, K)

    for f in ShenBasisMixed().functions(15):
        assert abs(f.subs(x, -1)) < 1E-8
        assert abs(f.diff(x, 1).subs(x, 1)) < 1E-8

    exit()

    # Check matrix properties
    assert test_mass_matrix(ShenBasisNeumann, 10)
    assert test_mass_matrix(ShenBasisMixed, 10)
    assert test_laplacian_matrix(ShenBasisNeumann, 10)
    assert test_laplacian_matrix(ShenBasisMixed, 10)
