from sympy.mpmath import quad
from itertools import product
import sympy as sp
import numpy as np
from math import pi


def mass_matrix(m):
    '''
    Return m x m matrix corresponding to (u, v)_{-1, 1} for u, v in shen_basis.
    '''
    return np.eye(m)


def stiffness_matrix(m):
    '''
    Return m x m matrix corresponding to (u', v')_{-1, 1} for u, v in shen_basis.
    '''
    return np.diag([(pi/2 + k*pi/2)**2 for k in range(m)])


def hp_matrix(m, norm):
    '''
    Return m^2 x m^2 matrix corresponding to norm(u, v)_plate for u, v in
    shen_basis(m) \otimes shen_basis(m).
    '''
    assert norm in ('L2', 'H10', 'H1')

    if norm == 'L2':
        M = mass_matrix(m)
        return np.kron(M, M)

    elif norm == 'H10':
        M = mass_matrix(m)
        A = stiffness_matrix(m)
        return np.kron(A, M) + np.kron(M, A)

    else:
        return hp_matrix(m, 'L2') + hp_matrix(m, 'H10')


def eigen_basis(m):
    'First m eigenfunctions of 1d laplacian on (-1, 1) with 0 Dirichlet bcs.'
    x = sp. Symbol('x')
    k = 0
    basis = []
    while k < m:
        alpha = pi/2 + k*pi/2
        if k % 2 == 0:
            basis.append(sp.cos(alpha*x))
        else:
            basis.append(sp.sin(alpha*x))
        k += 1
    return basis


def hb_matrix(m, norm, beam):
    '''
    Return m^2 x m^2 matrix corresponding to norm(u|b, v|b)_beam for u, v in
    shen_basis(m) \otimes shen_basis(m).
    '''
    assert norm in ('L2', 'H10', 'H1')

    if norm == 'L2' or norm == 'H10':
        x, y, s = sp.symbols('x, y, s')
        # Basis of the plate, functions of x, y
        basis = [u*v.subs(x, y)
                 for u, v in product(eigen_basis(m), eigen_basis(m))]

        # Beam is a tuple, so that we know how to substitute x, y to a common
        # variable s
        basis = [u.subs({x: beam[0], y: beam[1]}) for u in basis]

        # Decide how to build the integrand
        if norm == 'L2':
            integrand = lambda u, v: u*v
        else:
            integrand = lambda u, v: u.diff(s, 1)*v.diff(s, 1)

        mat = np.zeros((len(basis), len(basis)))
        # Assemble
        for i, u in enumerate(basis):
            for j, v in enumerate(basis):
                # The jacobian term is ignored
                mat[i, j] = quad(sp.lambdify(s, integrand(u, v)), (-1, 1))
        return mat

    else:
        return hb_matrix(m, 'L2', beam) + hb_matrix(m, 'H10', beam)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import pickle
    norm = 'H10'
    ms = np.arange(3, 4)
    s = sp.Symbol('s')

    As = [(-1, -1), (0, -1), (-0.75, -1)]
    Bs = [(1, 1), (0, 1), (0.25, 1)]

    np.set_printoptions(precision=2)
    data = {'beam': [], 'plate': [], 'combined': [], 'm': ms}
    for beam_index, (A, B) in enumerate(zip(As, Bs)):
        beam = (0.5*A[0]*(1-s) + 0.5*B[0]*(1+s),
                0.5*A[1]*(1-s) + 0.5*B[1]*(1+s))
        
        print A, B
        # Remember the beam
        data['beam'].append((A, B))

        # Gather condition numbers
        pconds = []
        cconds = []
        for m in ms:
            # Only the plate norm H(P)
            plate_mat = hp_matrix(m, norm)
            # Only the restricted norm  H(B)
            restricted_mat = hb_matrix(m, norm, beam)

            print restricted_mat

            # Combination H(P) \cap H(B)
            combined_mat = plate_mat + restricted_mat

            pconds.append(np.linalg.cond(plate_mat))
            cconds.append(np.linalg.cond(combined_mat))

            print '\tp=%.2g c=%.2g' % (pconds[-1], cconds[-1])

        # Remember
        data['plate'].append(pconds)
        data['combined'].append(cconds)

    pickle.dump(data, open('data_hp_hb_%s.pickle' % norm, 'wb'))

    if True:
        data = pickle.load(open('data_hp_hb_%s.pickle' % norm, 'rb'))
        ms = data['m']

        import matplotlib.pyplot as plt
        from pylab import getp

        plt.figure()
        for key in range(len(data['beam'])):
            A, B = data['beam'][key]
            AB = tuple(list(A)+list(B))
            plate = data['plate'][key]
            combined = data['combined'][key]

            line, = plt.loglog(ms, plate, label='[%g, %g]-[%g, %g] Hp' % AB,
                               linestyle='--', marker='x')

            color = getp(line, 'color')
            plt.loglog(ms, combined, label='[%g, %g]-[%g, %g] Hp+Hb' % AB,
                       linestyle='--', marker='o', color=color)
        plt.xlabel('m')
        plt.ylabel('$\kappa$')
        plt.legend(loc='best')
        plt.show()
