import pickle
import numpy as np
from matplotlib import rc 
rc('text', usetex=True) 
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt

# Each is a dictionary holdin for n a row of data which is a list to be plotted
# agains ms
ns = [2, 4, 8, 16, 24, 32, 40]
colors = ['r', 'b', 'g', 'k', 'm', 'c']
ms = np.array([2, 4, 8, 16, 24, 32, 40, 48, 64, 72, 80, 96, 128, 256], dtype='float')
# Dict
Pconds = pickle.load(open('se_pcond.pickle'))
Mnorms = pickle.load(open('seM_norms.pickle'))
Anorms = pickle.load(open('seA_norms.pickle'))

if True:
    # For selected n plot loglog(m vs A_norm)
    plt.figure()
    iter_colors = iter(colors)
    for n in [16, 24, 32, 40]:
        c = next(iter_colors)
        
        count = len(Anorms[n])

        plt.plot(ms[:count-1], Anorms[n][:-1], 'o-', label=str(n), color=c)
        # plt.semilogx([n, n], [1E4, 1E-4], linestyle='--', color=c)

    plt.xlabel('$m$')
    plt.ylabel('$||A-A_m||_F$')
    plt.legend(loc='best')
    plt.savefig('seA.pdf')

    # For selected n plot loglog(m vs M_norm)
    plt.figure()
    iter_colors = iter(colors)
    for n in [16, 24, 32, 40]:
        c = next(iter_colors)

        count = len(Mnorms[n])

        plt.plot(ms[:count-1], Mnorms[n][:-1], 'o-', label=str(n), color=c)
        # plt.semilogx([n, n], [1E-1, 1E-11], linestyle='--', color=c)

    plt.xlabel('$m$')
    plt.ylabel('$||M-M_m||_F$')
    plt.legend(loc='best')
    plt.savefig('seM.pdf')

    # For selected n plot (m vs P)
    plt.figure()
    iter_colors = iter(colors)
    for n in [16, 24, 32, 40]:
        c = next(iter_colors)

        count = len(Pconds[n])

        plt.semilogy(ms[:count-1], Pconds[n][:-1], 'o-', label=str(n), color=c)
        # plt.semilogx([n, n], [1, 2.5], linestyle='--', color=c)

    plt.xlabel('$m$')
    plt.ylabel('$\kappa(P_m)$')
    plt.legend(loc='best')
    plt.savefig('sePmn.pdf')

plt.show()

# --- Tables --------

# Need cond of P for all n and largest m
nlist, pmlist = [], []
for n in ns:
    p = Pconds[n][-1] if n < 16 else Pconds[n][-2]
    m = ms[Pconds[n].index(p)]

    nlist.append(n)
    pmlist.append((p, m))

print '& '.join(map(str, nlist))
print '& '.join(['%.2f(%d)' % (p, m) for (p, m) in pmlist])

# Need cond of P for n x n case
nlist, plist = [], []
for n in ns:
    p = Pconds[n][ms.tolist().index(n)]
    nlist.append(n)
    plist.append(p)

print '& '.join(map(str, nlist))
print '& '.join(['%.2E' % p for p in plist])
