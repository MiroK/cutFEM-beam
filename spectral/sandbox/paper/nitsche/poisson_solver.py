import numpy as np
import scipy.linalg as la
from itertools import product
from sympy import symbols, lambdify
from sympy.mpmath import quad
from numpy.linalg import cond


class PoissonSolver(object):
    'Generic Poisson solver that uses tensor product basis'
    def __init__(self, basis):
        'Set the basis generator. Symbolic.'
        self.basis = basis

    def list_basis(self, n):
        'Extract first n basis functions.'
        return [bi for _, bi in zip(range(n), self.basis())]

    def solve(self, f, n, **kwargs):
        '''
        Solve the Poisson problem on [-1, 1]**2 with rhs f and homog Dirichlet
        boundary conditions. The basis functions are taken as tensor product of
        first n basis functions.

        Return solution and data based on kwargs.
        '''
        data = {}
        # Assembly of 1d matrices that are used in tensor products
        A = self.assemble_A_matrix(n)
        M = self.assemble_M_matrix(n)
        # Assembly of rhs that is 2d
        b = self.assemble_b_vector(f, n)

        assert A.shape == (n, n)
        assert M.shape == (n, n)
        assert b.shape == (n, n)

        if 'monitor_cond' in kwargs:
            if kwargs['monitor_cond']:
                # Check conditioning number of 2d operator
                op = np.kron(A, M) + np.kron(M, A)
                cond_number = cond(op)
                data['monitor_cond'] = cond_number

        # Tensor product solver
        lmbda, Q = la.eigh(A, M)

        # Compute the inverse
        # Map the right hand side to eigen space
        b_ = (Q.T).dot(b.dot(Q))
        # Apply the inverse in eigen space
        U_ = np.array([[b_[i, j]/(lmbda[i] + lmbda[j])
                        for j in range(n)]
                       for i in range(n)])
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

        if data:
            return uh, data
        else:
            return uh

    def assemble_A_matrix(self, n):
        'Default (slow) implementation of A assembly. Sets no boundary conds.'
        basis = self.list_basis(n)
        # Lambdify basis for numeric integration, Derivative
        x = symbols('x')
        basis = map(lambda f: lambdify(x, f.diff(x, 1)), basis)
        A = np.zeros((n, n))
        for i, bi in enumerate(basis):
            A[i, i] = quad(lambda x: bi(x)**2, [-1, 1])
            for j, bj in enumerate(basis[i+1:], i+1):
                A[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1], maxdegree=40)
                A[j, i] = A[i, j]
        return A

    def assemble_M_matrix(self, n):
        'Default (slow) implementation of M assembly. Sets no boundary conds.'
        basis = self.list_basis(n)
        # Lambdify basis for numeric integration
        x = symbols('x')
        basis = map(lambda f: lambdify(x, f), basis)
        M = np.zeros((n, n))
        for i, bi in enumerate(basis):
            M[i, i] = quad(lambda x: bi(x)**2, [-1, 1])
            for j, bj in enumerate(basis[i+1:], i+1):
                M[i, j] = quad(lambda x: bi(x)*bj(x), [-1, 1], maxdegree=40)
                M[j, i] = M[i, j]
        return M

    def assemble_b_vector(self, f, n):
        'Default (slow) implementation of b assembly. Sets no boundary conds.'
        basis = self.list_basis(n)
        x, y = symbols('x, y')
        # Lambdify rhs for fast integration
        f = lambdify([x, y], f)
        # Lambdify basis for numeric integration
        basis = map(lambda f: lambdify(x, f), basis)
        b = np.zeros((n, n))
        for i, bi in enumerate(basis):
            for j, bj in enumerate(basis):
                b[i, j] = quad(lambda x, y: bi(x)*bj(y)*f(x, y),
                               [-1, 1], [-1, 1], maxdegree=40)
        return b
