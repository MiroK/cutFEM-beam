import pickle
import numpy as np
from matplotlib import rc 
rc('text', usetex=True) 
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt

data0 = pickle.load(open('eigen_smooth_0.pickle', 'rb'))
data1 = pickle.load(open('eigen_smooth_1.pickle', 'rb'))

n_list = data0['n_list']
errors0 = data0['errors']
row0 = data0['power']
row0 = np.sqrt(row0**2)

errors1 = data1['errors']
row1 = data1['power']
row1 = np.sqrt(row1**2)

plt.figure()
plt.loglog(n_list, errors0, '-rs', label='$%s$' % data0['f'])
plt.loglog(n_list, errors1, '-bo', label='$%s$' % data1['f'])
plt.legend(loc='best')
plt.xlabel('$m$')
plt.savefig('eigen_smooth_rate.pdf')

plt.figure()
plt.loglog(row0, '-rs', label='$%s$' % data0['f'])
plt.loglog(row1, '-bo', label='$%s$' % data1['f'])
plt.legend(loc='best')
plt.xlabel('$k$')
plt.ylabel(r'$|(f, \varphi_k)|$')
plt.savefig('eigen_smooth_power.pdf')

plt.show()

