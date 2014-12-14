import numpy as np
import scipy.linalg as la
from itertools import product
from sympy import symbols, lambdify
from sympy.mpmath import quad
from numpy.linalg import cond
from basis_generators import H1Basis, H10Basis, Hierarchical


class ClassicPoissonSolver(object):
    '''
    Poisson solver that uses tensor product basis constructed from H10
    function. In this way the bcs are built into the space.
    '''
    def __init__(self, basis):
        'Set the basis generator. Symbolic.'
        self.basis = basis

    def list_basis(self, n):
        '''
        Create new approximateion space. n is a counting variable and its
        connection to e.g. polynomial degree of functions dependes on the
        choice basis
        '''
        # For hierchical basis construct the approximation by using first n
        # basis functions
        if isinstance(self.basis, Hierarchical):
            return [bi for _, bi in zip(range(n), self.basis())]
        # Otherwise take the n-th space
        else:
            for i, space in enumerate(self.basis(), 1):
                if i == n:
                    return space

    def solve(self, f, n, **kwargs):
        '''
        Solve the Poisson problem on [-1, 1]**2 with rhs f and homog Dirichlet
        boundary conditions. The basis functions are taken as tensor product of
        functions of space characterized by basis (n parameterizes this
        approximation).

        Return solution and data based on kwargs.
        '''
        data = {}
        # Assembly of 1d matrices that are used in tensor products
        A = self.assemble_A_matrix(n)
        M = self.assemble_M_matrix(n)
        # Assembly of rhs that is 2d
        b = self.assemble_b_vector(f, n)

        assert A.shape == M.shape and M.shape == b.shape
        size = len(A)

        if 'monitor_cond' in kwargs:
            if kwargs['monitor_cond']:
                # Check conditioning number of 2d operator
                # as well as mass and stiffness matrices alone
                op = np.kron(A, M) + np.kron(M, A)
                cond_number_op = cond(op)
                cond_number_A = cond(A)
                cond_number_M = cond(M)
                data['monitor_cond'] = {'op': cond_number_op,
                                        'A': cond_number_A,
                                        'M': cond_number_M}

        data['size'] = size

        # Tensor product solver
        lmbda, Q = la.eigh(A, M)

        # Compute the inverse
        # Map the right hand side to eigen space
        b_ = (Q.T).dot(b.dot(Q))
        # Apply the inverse in eigen space
        U_ = np.array([[b_[i, j]/(lmbda[i] + lmbda[j])
                        for j in range(size)]
                       for i in range(size)])
        # Map back to physical space
        U = Q.dot(U_.dot(Q.T))

        # Assemble the solution as symbolic function
        # Note that basis is 1d(x)
        x, y = symbols('x, y')
        U = U.flatten()
        basis_x = self.list_basis(n)
        basis_y = map(lambda f: f.subs({x: y}), basis_x)
        basis = [bi*bj for bi, bj in product(basis_x, basis_y)]

        assert len(U) == len(basis)

        uh = sum(Uk*basis_k for (Uk, basis_k) in zip(U, basis))

        # System size
        data['size'] = size

        return uh, data

    def assemble_A_matrix(self, n):
        'Default (slow) implementation of A assembly. Sets no boundary conds.'
        basis = self.list_basis(n)
        size = len(basis)
        # Lambdify basis for numeric integration, Derivative
        x = symbols('x')
        basis = map(lambda f: lambdify(x, f.diff(x, 1)), basis)
        A = np.zeros((size, size))
        for i, bi in enumerate(basis):
            A[i, i] = quad(lambda x: bi(x)**2, [-1, 1])
            for j, bj in enumerate(basis[i+1:], i+1):
                A[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
                A[j, i] = A[i, j]
        return A

    def assemble_M_matrix(self, n):
        'Default (slow) implementation of M assembly. Sets no boundary conds.'
        basis = self.list_basis(n)
        size = len(basis)
        # Lambdify basis for numeric integration
        x = symbols('x')
        basis = map(lambda f: lambdify(x, f), basis)
        M = np.zeros((size, size))
        for i, bi in enumerate(basis):
            M[i, i] = quad(lambda x: bi(x)**2, [-1, 1])
            for j, bj in enumerate(basis[i+1:], i+1):
                M[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1])
                M[j, i] = M[i, j]
        return M

    def assemble_b_vector(self, f, n):
        'Default (slow) implementation of b assembly. Sets no boundary conds.'
        basis = self.list_basis(n)
        size = len(basis)
        x, y = symbols('x, y')
        # Lambdify rhs for fast integration
        f = lambdify([x, y], f)
        # Lambdify basis for numeric integration
        basis = map(lambda f: lambdify(x, f), basis)
        b = np.zeros((size, size))
        for i, bi in enumerate(basis):
            for j, bj in enumerate(basis):
                b[i, j] = quad(lambda x, y: bi(x)*bj(y)*f(x, y),
                               [-1, 1], [-1, 1])
        return b


class NitschePoissonSolver(ClassicPoissonSolver):
    '''
    Nitsche solvers have special stiffness matrices that enforce boundary
    conditions. Use H1 fuctions.
    '''

    def __init__(self, basis):
        'Set the generator'
        ClassicPoissonSolver.__init__(self, basis)

    def assemble_A_matrix(self, n):
        'The stiffness matrix includes boundary integrals.'
        # The volume term
        A = ClassicPoissonSolver.assemble_A_matrix(self, n)

        x = symbols('x')
        basis_ = self.list_basis(n)
        # Need lambdified function and the derivatives
        basis = map(lambda f: lambdify(x, f), basis_)
        dbasis = map(lambda f: lambdify(x, f.diff(x, 1)), basis_)

        # The gradient boundary term
        C = np.zeros_like(A)
        for i, dbi in enumerate(dbasis):
            for j, bj in enumerate(basis):
                C[i, j] = dbi(1)*bj(1) - dbi(-1)*bj(-1)

        # Penalty term
        D = np.zeros_like(A)
        for i, bi in enumerate(basis):
            D[i, i] = bi(1)**2 + bi(-1)**2
            for j, bj in enumerate(basis[i+1:], i+1):
                D[i, j] = bi(1)*bj(1) + bi(-1)*bj(-1)
                D[j, i] = D[i, j]
        D *= 10*n

        # The mass matrix is
        A = A - C - C.T + D
        return A


class PoissonSolver(object):
    'Construct appropriate solution method based on basis.'
    def __init__(self, basis):
        assert isinstance(basis, H1Basis)
        # H10 basis are classic
        if isinstance(basis, H10Basis):
            self.solver = ClassicPoissonSolver(basis)
        # H1 basis must be formulated with Nitsche
        elif isinstance(basis, H1Basis):
            self.solver = NitschePoissonSolver(basis)
        else:
            raise ValueError('Not good basis')

    def solve(self, f, n, **kwargs):
        return self.solver.solve(f, n, **kwargs)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import S
    from sympy.plotting import plot3d
    from basis_generators import __basis_d__

    for key in __basis_d__:
        print key
        basis = __basis_d__[key]
        solver = PoissonSolver(basis())

        x, y = symbols('x, y')
        f = S(1)
        for n in range(1, 5):
            uh, _ = solver.solve(f, n)
            print n, _
            print solver.solver.list_basis(n)
            print solver.solver.list_basis(n)

            plot3d(uh, (x, -1, 1), (y, -1, 1))
