import pickle
import numpy as np
from matplotlib import rc 
rc('text', usetex=True) 
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt

# We have two projections and each has data for two functions
# Get the data of L2 projection for both functions
data_L2_0 = pickle.load(open('shen_smooth_0_L2.pickle', 'rb'))
data_L2_1 = pickle.load(open('shen_smooth_1_L2.pickle', 'rb'))
# Get the data of H10 projection for both functions
data_H10_0 = pickle.load(open('shen_smooth_0_H10.pickle', 'rb'))
data_H10_1 = pickle.load(open('shen_smooth_1_H10.pickle', 'rb'))
# x axis is common 
n_list = data_L2_0['n_list']

# Compoare projecions in L2 norms for both functions
plt.figure()
plt.loglog(n_list, data_L2_0['errors1'], '-rs', label='$\pi, %s$' % data_L2_0['f'])
plt.loglog(n_list, data_L2_1['errors1'], '-ro', label='$\pi, %s$' % data_L2_1['f'])
plt.loglog(n_list, data_H10_0['errors1'], '-bs', label='$\Pi, %s$' % data_H10_0['f'])
plt.loglog(n_list, data_H10_1['errors1'], '-bo', label='$\Pi, %s$' % data_H10_1['f'])

plt.legend(loc='best')
plt.xlabel('$m$')
plt.savefig('shen_H10proj.pdf')
plt.show()
