from plate_beam import LineBeam
from trace_operator import TraceOperator
from eigen_basis import eigen_basis
from shenp_basis import shenp_basis as shen_basis
import numpy as np
from numpy.linalg import cond
from sympy import symbols


def foo(basis, m, beam):
    '''
    Construct space Vp which has dim m**2 and 5-tuple of spaces
    Vb which have dimensions m-2, m-1, m, m+1, m+2. For each pair
    Vp-Vb and the beam return condition number of the restriction
    matrix and Trace matrix.
    '''
    assert basis in ('shen', 'eigen')
    if basis == 'shen':
        basis = shen_basis
    else:
        basis = eigen_basis
    
    # Make sure m makes sense for the 5-tuple
    assert m-2 > 2

    # Results
    Rconds = []
    Tconds = []
    # One Vp as tensor product
    x, y, s = symbols('x, y, s')
    Vp = list(basis(m))
    Vp = [Vp, Vp]

    # Loop Vb
    print 'm=', m
    for n in [m-2, m-1, m, m+1, m+2]:
        # The space is in s!
        Vb = [p.subs(x, s) for p in basis(n)]

        # Trace operator
        op = TraceOperator(Vp, Vb, beam)

        # Get the matrices
        R = op.R_matrix()
        print '\tR@%d' % n
        T = op.T_matrix()
        print '\tT@%d' % n

        # Condition numbers
        Rconds.append(cond(R))
        Tconds.append(cond(T))

    return {'R': Rconds, 'T': Tconds}

# -----------------------------------------------------------------------------

if __name__:
    import pickle
    import matplotlib.pyplot as plt

    # Form 3 beam positions ...    

    A = np.array([-1, -1])
    B = np.array([1, 1])
    beam = LineBeam(A, B)

    ms = range(5, 8)

    # Compute
    if False:
        eigen_data = dict((m, foo(basis='eigen', m=m, beam=beam)) for m in ms)
        shen_data = dict((m, foo(basis='shen', m=m, beam=beam)) for m in ms)

        pickle.dump(eigen_data, open('eigen_trace_scaling0.pickle', 'wb'))
        pickle.dump(shen_data, open('shen_trace_scaling0.pickle', 'wb'))
    # Process
    else:
        eigen_data = pickle.load(open('eigen_trace_scaling0.pickle', 'rb'))
        shen_data = pickle.load(open('shen_trace_scaling0.pickle', 'rb'))
        data = {'shen': shen_data, 'eigen': eigen_data}

        # Postprocessing, _row is data for given beam position
        colors = {'shen': 'g', 'eigen': 'b'}
        markers = ['s', 'x', 'o', 'v', '^']
        labels = ['$m-2$', '$m-1$', '$m$', '$m+1$', '$m+2$']

        key='shen'
        color = colors[key]
        data = data[key]
        plt.figure()
        for (col, label, marker) in zip(range(5), labels, markers):
            col_data = [data[m]['T'][col] for m in ms]
            plt.plot(ms, col_data, label=label, color=color, marker=marker)

        plt.xlabel('$m$')
        plt.ylabel('$\kappa$')
        plt.legend(loc='best')
        plt.show()
