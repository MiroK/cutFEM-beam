from __future__ import division
import sys
sys.path.append('../')
from collections import defaultdict
import pickle
from coupled_problem import CoupledProblem
import coupled_tests
from plate_beam import Beam
import numpy as np
import numpy.linalg as la
from itertools import product

class NRule(object):
    'Spaces sizes as (sizes of Vp, size of Vb, size of Q)'
    def __init__(self, name, rule):
        self.name = name
        self.rule = rule

    def __call__(self, N):
        return self.rule(N)

# Auxiliary functions to buid keys for lmin of Schur complement and
# Schur compement of plate and beam
keyS = lambda norm: 'lmin_S' if norm is None else 'lmin' + str(norm) + '_S'
keySp = lambda norm: 'lmin_Sp' if norm is None else 'lmin' + str(norm) + '_Sp'
keySb = lambda norm: 'lmin_Sb' if norm is None else 'lmin' + str(norm) + '_Sb'
keyBab = lambda norm: 'gamma' if norm is None else 'gamma_' + str(norm)

def test_coupled_problem(params):
    '''
    All kinds of quantities that might be of interest for the coupled problems
    are computed here.

    Params is a dictionary with
        params['problem'] ... class that defines a problems
        params['beam_list'] ... a list of beams to be considered
        params['materials'] ... dict with 'plate_E', 'beam_E' as stiffness of
                                plate and beam
        params['rule'] ... class that has a name and __cal__(N) produces tuple
                           (ms=list, n, r) that determined sizes of spaces in 
                           beam problem
        params['N_list'] ... list of N values passed to rule to produce spaces
        params['norms'] ... norms to be used for Schur complement precondition
        params['Pblocks'] ... this is a list of functions that know how to build
                              blocks for preconditioner to entire system. If 
                              defined we compute (PA) properties
        params['postfix'] ... something to add to name of data file

    Computed quantities
        for each beam:
            lists of 

            B.T*inv(A)*B P = lambda Nmat P and the two partial problems for Nmat
            given by norms.

            babuska constant with Nmat given by norms

            cond(A) condition of system

            for each N

    To decode the datastrucure you need to remember beam_list, N_list, norms and
    order of preconditioners.!
    '''
    # Let's do the input checks
    assert issubclass(params['problem'], CoupledProblem)
    Problem = params['problem']

    assert all(isinstance(beam, Beam) for beam in params['beam_list'])
    beams = params['beam_list']

    assert 'plate_E' in params['materials'] and 'beam_E' in params['materials']
    materials = params['materials']

    assert isinstance(params['N_list'], list)
    Ns = params['N_list']

    assert isinstance(params['rule'], NRule)
    rule = params['rule']

    # If there are no norms indicate by [None] that we want not preconditioned
    norms = params.get('norms', [None])
    # Indicate no preconditioner of system
    Pblocks = params.get('Pblocks', None)
    # Set default postifx to ''
    postfix = params.get('postfix', '')

    # Build the data structure to hold results
    # Always have cond number of entire system
    data = {'cond_A': defaultdict(list)}
    # We have smallest eigenvalue of S, Sp, Sb for each norm
    for norm in norms:
        data[keyS(norm)] = defaultdict(list)
        data[keySp(norm)] = defaultdict(list)
        data[keySb(norm)] = defaultdict(list)
    # We have babuska for each norm:
    for norm in norms:
        data[keyBab(norm)] = defaultdict(list)
    # Finally if there is a preconditioner we will get its cond. number
    #if Pblocks is not None:
    #    for blocks_index in range(len(Pblocks)):
    #        data['cond_PA' + str(blocks_index)] = defaultdict(list)

    # For each beam
    for beam_index, beam in enumerate(beams):
        # Check n-dependency
        for N in Ns:
            # Get the space sizes
            ms, n, r = rule(N)
            # Construct the problem
            problem = Problem(ms, n, r, beam, materials)

            # Schur     
            # Get all the norm matrices
            Nmats = [problem.C_matrix(norm) for norm in norms]
            # Get all S eigs
            lminS = coupled_tests.schur(problem, Nmats)
            # Get all Sp eigs and Sb eigs
            lminSp, lminSb = coupled_tests.schur_components(problem, Nmats)
    
            # Record
            [data[keyS(norm)][beam_index].append(val)
             for norm, val in zip(norms, lminS)]

            [data[keySp(norm)][beam_index].append(val)
             for norm, val in zip(norms, lminSp)]

            [data[keySb(norm)][beam_index].append(val)
             for norm, val in zip(norms, lminSb)]

            # Babuska
            M = la.inv(problem.A_matrix())
            babuska = coupled_tests.babuska(problem, M, Nmats)
            # Record
            [data[keyBab(norm)][beam_index].append(val)
             for norm, val in zip(norms, babuska)]

            # System conditioning
            A = problem.system_matrix()
            cond = la.cond(A)
            data['cond_A'][beam_index].append(cond)

            # Preconditioned system
            # if Pblocks is not None:
            #     # Use given preconditioners
            #     for block_index, blocks in enumerate(Pblocks):
            #         # Make blocks of preconditioner for the problem
            #         blocks_ = blocks(problem) 
            #         # Make the preconditioner
            #         P = problem.preconditioner(blocks_)
            #         PA = P.dot(A)
            #         cond = la.cond(PA)
            #         data['cond_PA' + str(block_index)][beam_index].append(cond)

            # Data for current N as row
            print N,
            print ', '.join([':'.join((key, '%.2g' % data[key][beam_index][-1]))
                            for key in data])
        # New beam
        print

    # We'll add some of the input as well
    data['input'] = {'N_list': params['N_list'],
                     'plate_E': params['materials']['plate_E'],
                     'beam_E': params['materials']['beam_E'],
                     'norms': params['norms']}
        
    # Pickle!
    pickle_name = '_'.join((Problem.__name__, rule.name, postfix))
    pickle_name = '.'.join((pickle_name, 'pickle'))
    pickle.dump(data, open(pickle_name, 'wb'))

    # Make loading easier
    print '\n pickling', pickle_name
    return pickle_name


def as_tex_table(ns, beam_data, row_format, header):
    '''
    Build latex table. Ns is first column, the others are rows of beam_data.
    Format of entire row is in row_format. Table has header.
    '''
    # Build the table without format
    iters = [iter(d) for d in beam_data]
    iters.insert(0, iter(ns))
    # Make sure there same of everything
    assert len(header) == len(row_format) and len(row_format) == len(iters)
    # Make templates
    row_format = ' & '.join(row_format)
    header = ' & '.join(header)
    # Print the table
    print r'\hline'
    print header + r'\\'
    print r'\hline'
    for row in zip(*iters):
        print (row_format % row) + r'\\'
    print r'\hline'


def as_plot(ns, beam_data, line_styles, markers, labels, colors, ylabel):
    '''
    Make loglog plots ns vs. data. Each col in beam data has linestyle, marker,
    label, color. Figure need ylabel.
    '''
    # Build data
    iters = [iter(d) for d in beam_data]
    # Transpose for easier iteration
    cols = np.array([list(row) for row in zip(*iters)]).T
    n_cols = len(cols)
    # Expand line_styles
    if isinstance(line_styles, list):
        assert len(line_styles) == n_cols
    else:
        line_styles = [line_styles]*n_cols
    # Everything must be of exact length
    assert len(markers) == n_cols
    assert len(labels) == n_cols
    assert len(colors) == n_cols
    
    # Now plot
    plt.figure()
    for (ls, mk, lb, c, col) in zip(line_styles, markers, labels, colors, cols):
        plt.loglog(ns, col, linestyle=ls, marker=mk, label=lb, color=c)
    plt.legend(loc='best')
    plt.ylabel(ylabel)
    plt.xlabel('$n$')
    plt.show()


def plot_beams(beams):
    'Plots all the beams into one plot.'
    n = int(np.sqrt(len(beams)))
    assert n > 1
    assert abs(n**2 - len(beams)) < 1E-14

    fig, axarr = plt.subplots(n, n)
    for i, beam in enumerate(beams):
        row = i // n
        col = i % n
        ax = axarr[row, col]
        # Beam knows how to plot itself
        beam.plot(ax)
        # Remove ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.subplots_adjust(hspace=0, wspace=0)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Problem
    from coupled_eigen_laplace import CoupledEigenLaplace
    from plate_beam import LineBeam
    # Postproc
    import matplotlib.pyplot as plt

    # Some beam positions
    A_pos = lambda t: [-1+t, -1]
    B_pos = lambda t: [t, 1]
    ts = [0, 1]
    beams = [LineBeam(A_pos(tA), B_pos(tB)) for tA, tB in product(ts, ts)][:1]
    # plot_beams(beams)
    # plt.show()

    # Common stuff
    params = {'beam_list': beams,
              'materials': {'plate_E': 1, 'beam_E': 20},
              'rule': NRule('all_equal', lambda N: ([N, N], N, N)),
              'N_list': range(2, 8),
              'postfix': 'test'}
    # Unique eigen
    params_eigen = {'problem': CoupledEigenLaplace,
                    'norms': [None, 0, -0.5, -1]}
    # Combine
    params_eigen.update(params)

    pickle_name = test_coupled_problem(params_eigen)
    pickle_name = 'CoupledEigenLaplace_all_equal_test.pickle'
    data = pickle.load(open(pickle_name, 'rb'))

    # All markers, colors
    all_markers = ['x', 'd', 's', 'o', 'v', '^', '+']
    all_colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']

    # Plot 
    key = 'lmin_S'
    assert key in data
    ns = data['input']['N_list']
    norms = data['input']['norms']

    # Suppose a beam (beam position is given)
    beam = 0
    # We definitely want plots showing  
    # (i) ns vs lmin(S), in the norms
    # (ii) ns vs lmin(Sp), in the norms
    # (iii) ns vs lmin(Sb), in the norms
    # (iv) ns vs gamma, in the norms
    # (v) ns vs cond(A) and cond(Pa)
    plot = '(iv)'

    if plot in ('(i)', '(ii)', '(iii)', '(iv)'):
        row_format = ['%d'] + ['%.2E'] * len(norms)
        header = ['$n$']+['$\mathbb{I}$']+['$H^{%g}$' % s for s in norms[1:]]
        line_styles = '--'
        markers = all_markers[:len(norms)]
        colors = all_colors[:len(norms)]
        labels=header[1:]
        
        if plot == '(i)':
            beam_data = [data[keyS(norm)][beam] for norm in norms]
            ylabel='$S\lambda_{min}$'

        if plot == '(ii)':
            beam_data = [data[keySp(norm)][beam] for norm in norms]
            ylabel='$S_p\lambda_{min}$'

        if plot == '(iii)':
            beam_data = [data[keySb(norm)][beam] for norm in norms]
            ylabel='$S_b\lambda_{min}$'

        if plot == '(iv)':
            beam_data = [data[keyBab(norm)][beam] for norm in norms]
            ylabel='$\gamma$'

    elif plot == '(v)':
        beam_data = [data['cond_A'][beam]]
        row_format = ['%d', '%.2E']
        header = ['$n$']+['$A$']
        line_styles = '--'
        markers=['x']
        colors=['r']
        labels=header[1:]
        ylabel='$\kappa$'

        iter_markers = iter(all_markers)
        iter_colors = iter(all_colors)
        for key in data:
            if key.startswith('cond_PA'):
                print '?'
                beam_data.append(data[key][beam])
                row_format.append('%.2E')
                header.append(key)
                markers.append(iter_markers.next())
                colors.append(iter_colors.next())
                labels.append(key)

    as_tex_table(ns, beam_data, row_format, header)
    as_plot(ns, beam_data, line_styles, markers, labels, colors, ylabel)
