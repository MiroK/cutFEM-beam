import pickle
import numpy as np
from matplotlib import rc 
rc('text', usetex=True) 
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt

# Each is a dictionary holdin for n a row of data which is a list to be plotted
# agains ms
ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
colors = ['r', 'b', 'g', 'k', 'm', 'c']
ms = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype='float')
# Dict
Pconds = pickle.load(open('eigen_pcond.pickle'))
Mnorms = pickle.load(open('M_norms.pickle'))
Anorms = pickle.load(open('A_norms.pickle'))

if False:
    # For selected n plot loglog(m vs A_norm)
    plt.figure()
    iter_colors = iter(colors)
    for n in [16, 64, 256, 1024]:
        c = next(iter_colors)
        plt.loglog(ms, Anorms[n], 'o-', label=str(n), color=c)
        plt.loglog([n, n], [1E4, 1E-4], linestyle='--', color=c)
    # Ref. line
    plt.loglog(ms, ms**-1, label='rate 1', color='c', linestyle='-')

    plt.xlabel('$m$')
    plt.ylabel('$||A-A_m||_F$')
    plt.legend(loc='best')
    plt.savefig('eig_A.pdf')

    # For selected n plot loglog(m vs M_norm)
    plt.figure()
    iter_colors = iter(colors)
    for n in [16, 64, 256, 1024]:
        c = next(iter_colors)
        plt.loglog(ms, Mnorms[n], 'o-', label=str(n), color=c)
        plt.loglog([n, n], [1E-1, 1E-11], linestyle='--', color=c)
    # Ref. line
    plt.loglog(ms, ms**-3, label='rate 3', color='c', linestyle='-')

    plt.xlabel('$m$')
    plt.ylabel('$||M-M_m||_F$')
    plt.legend(loc='best')
    plt.savefig('eig_M.pdf')

    # For selected n plot (m vs P)
    plt.figure()
    iter_colors = iter(colors)
    for n in [16, 64, 256, 1024]:
        c = next(iter_colors)
        plt.semilogx(ms, Pconds[n], 'o-', label=str(n), color=c)
        plt.semilogx([n, n], [1, 2.5], linestyle='--', color=c)

    # The convergence line
    plt.semilogx(ms, np.ones_like(ms)*np.sqrt(3), linestyle='-',
                 label='$\sqrt{3}$', color='c')
    plt.xlabel('$m$')
    plt.ylabel('$\kappa(P_m)$')
    plt.legend(loc='best')
    plt.savefig('eig_Pmn.pdf')


# --- Tables --------

# Get the cond number of Pmm as table
Pmm_data = np.loadtxt('Pmm_eigen.data')
n_mag = ['%d' % n for n in Pmm_data[4:, 0]]
P_mag = ['%.2f' % p for p in Pmm_data[4:, 1]]
print '$n$ &', '& '.join(n_mag) 
print '$P$ &', '& '.join(P_mag)

# Get the norm for last m of A, M vs n
A_mag = ['%.2E' % Anorms[n][-1] for n in ns[::2]]
M_mag = ['%.2E' % Mnorms[n][-1] for n in ns[::2]]
n_mag = ['%d' % n for n in ns[::2]]

print '$n$ &', '& '.join(n_mag)
print '$A$ &', '& '.join(A_mag)
print '$M$ &', '& '.join(M_mag)


