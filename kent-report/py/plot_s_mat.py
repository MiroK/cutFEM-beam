import pickle
import numpy as np
from matplotlib import rc 
rc('text', usetex=True) 
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt

# Each is a dictionary holdin for n a row of data which is a list to be plotted
# agains ms
ns = [4, 8, 16, 24, 32, 40, 48, 56, 64]
colors = ['r', 'b', 'g', 'k', 'm', 'c']
ms = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256], dtype='float')
# Dict
Pconds = pickle.load(open('shen_pcond.pickle'))
Mnorms = pickle.load(open('shenM_norms.pickle'))
Anorms = pickle.load(open('shenA_norms.pickle'))

if False:
    # For selected n plot loglog(m vs A_norm)
    plt.figure()
    iter_colors = iter(colors)
    for n in [16, 24, 32, 48, 64]:
        c = next(iter_colors)
        plt.loglog(ms, Anorms[n], 'o-', label=str(n), color=c)
        plt.loglog([n, n], [1E4, 1E-4], linestyle='--', color=c)
    # Ref. line
    plt.loglog(ms, ms**-1, label='rate 1', color='c', linestyle='-')

    plt.xlabel('$m$')
    plt.ylabel('$||A-A_m||_F$')
    plt.legend(loc='best')
    plt.savefig('shen_A.pdf')

    # For selected n plot loglog(m vs M_norm)
    plt.figure()
    iter_colors = iter(colors)
    for n in [16, 24, 32, 48, 64]:
        c = next(iter_colors)
        plt.loglog(ms, Mnorms[n], 'o-', label=str(n), color=c)
        plt.loglog([n, n], [1E-1, 1E-11], linestyle='--', color=c)
    # Ref. line
    plt.loglog(ms, ms**-3, label='rate 3', color='c', linestyle='-')

    plt.xlabel('$m$')
    plt.ylabel('$||M-M_m||_F$')
    plt.legend(loc='best')
    plt.savefig('shen_M.pdf')

    # For selected n plot (m vs P)
    plt.figure()
    iter_colors = iter(colors)
    for n in [16, 24, 32, 48, 64]:
        c = next(iter_colors)
        plt.loglog(ms, Pconds[n], 'o-', label=str(n), color=c)
        plt.loglog([n, n], [1, 2.5], linestyle='--', color=c)

    plt.xlabel('$m$')
    plt.ylabel('$\kappa(P_m)$')
    plt.legend(loc='best')
    plt.savefig('shen_Pmn.pdf')

plt.show()

# --- Tables --------

# Get the cond number of Pmm as table
Pmm_data = np.loadtxt('Pmm_shen.data')
n_mag = ['%d' % n for n in Pmm_data[:, 0]]
P_mag = ['%.2E' % p for p in Pmm_data[:, 1]]
PP_mag = ['%.2E' % p for p in Pmm_data[:, 2]]
print '$n$ &', '& '.join(n_mag) 
print '$P$ &', '& '.join(P_mag)
print '$PP$ &', '& '.join(PP_mag)

data = pickle.load(open('shen_pcond.pickle', 'rb'))
for key in sorted(data.keys()):
    print key, data[key][-1]

# Get the norm for last m of A, M vs n
A_mag = ['%.2E' % Anorms[n][-1] for n in ns[::2]]
M_mag = ['%.2E' % Mnorms[n][-1] for n in ns[::2]]
n_mag = ['%d' % n for n in ns[::2]]

print '$n$ &', '& '.join(n_mag)
print '$A$ &', '& '.join(A_mag)
print '$M$ &', '& '.join(M_mag)


