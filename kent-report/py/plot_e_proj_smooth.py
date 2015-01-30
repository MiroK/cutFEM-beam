import pickle
import numpy as np
from matplotlib import rc 
rc('text', usetex=True) 
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt

data_first = pickle.load(open('eigen_smooth_0.pickle', 'rb'))
data_second = pickle.load(open('eigen_smooth_1.pickle', 'rb'))

n_list = data_first['n_list']

errors_first0 = data_first['errors0']  # L2
errors_first1 = data_first['errors1']  # H10
row_first = data_first['power']
row_first = np.sqrt(row_first**2)

errors_second0 = data_second['errors0']  # L2
errors_second1 = data_second['errors1']  # H10
row_second = data_second['power']
row_second = np.sqrt(row_second**2)

# Rates of both functions in L2 and H1
plt.figure()
plt.loglog(n_list, errors_first0, '-rs', label='$%s, L^2$' % data_first['f'])
plt.loglog(n_list, errors_first1, '-ro', label='$%s, H^1_0$' % data_first['f'])

plt.loglog(n_list, errors_second0, '-bs', label='$%s, L^2$' % data_second['f'])
plt.loglog(n_list, errors_second1, '-bo', label='$%s, H^1_0$' % data_second['f'])

plt.legend(loc='best')
plt.xlabel('$m$')
plt.savefig('eigen_smooth_rate.pdf')

# Spectrum
plt.figure()
plt.loglog(row_first, '-rs', label='$%s$' % data_first['f'])
plt.loglog(row_second, '-bo', label='$%s$' % data_second['f'])
plt.legend(loc='best')
plt.xlabel('$k$')
plt.ylabel(r'$|(f, \varphi_k)|$')
plt.savefig('eigen_smooth_power.pdf')

plt.show()

# Least squares fit of rate for both functions

for x, what in zip([errors_first0, errors_first1], ['L2', 'H10']):
    b = np.log(x)
    col = -np.log(n_list)
    A = np.ones((len(b), 2))
    A[:, 0] = col
    ans = np.linalg.lstsq(A, b)[0]
    p = ans[0]
    print '\tLeast square rate in %s is %.2f' % (what, p)

