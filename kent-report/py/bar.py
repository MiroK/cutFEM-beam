from __future__ import division
from shen_poisson import mass_matrix
from sympy.mpmath import gamma, sqrt, pi
import numpy as np

m = 15
M = mass_matrix(m)

Msub = M[:m-1, :]
vec = np.zeros(m)
vec[0] = 1
vec[2] = -M[0, 0]*vec[0]/M[0, 2]
for i in range(4, m, 2):
    vec[i] = M[i-2, i-2]*vec[i-2] + M[i-2, i-4]*vec[i-4]
    vec[i] /= -M[i-2, i]

print Msub.dot(vec)

def value0(k):
    'Hat centered at zero with wifht 2h vs k. Shen functions.'
    l = k/2
    value = 2*(-1)**(l+1)/sqrt(pi)
    value *= 1./sqrt(4*k + 6)
    value *= (2*l + 1.5)/(l + 1)
    value *= gamma(l+0.5)/gamma(l+1)
    return float(value)

import matplotlib.pyplot as plt
ks = range(0, 101, 2)
values = [value0(k) for k in ks]
plt.figure()
plt.plot(ks, values, '-x')
plt.show()
