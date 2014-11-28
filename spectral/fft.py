from __future__ import division
import numpy as np
from math import pi as mpi, sqrt as msqrt
import numpy.linalg as la
import matplotlib.pyplot as plt
import pyfftw


if False:
    N = 2**16
    x = np.linspace(-1, 1, N)
    # y_exact = x
    # y_exact = np.concatenate([-1*np.ones(N/2), 1*np.ones(N/2)])
    foo = -(x[N/2:]-0.5)**2 + 5/4
    y_exact = np.concatenate([-foo, foo])

    n = 40
    y = np.zeros_like(x)
    bs = []
    for i in range(1, n):
        # bi = 2*(-1)**(i+1)/i/pi    # x  on [-1, 1]
        # bi = 2*(1-(-1)**i)/i/pi      # -1 on [-1, 0], 1 on [0, 1]
        bi = -2*(mpi**2*i**2 + 2)*((-1)**i - 1)/mpi**3/i**3
        y += bi*np.sin(i*mpi*x)

    bs.append(bi)

    plt.figure()
    plt.plot(x, y_exact)
    plt.plot(x, y)
    plt.show()

    print '%d points' % N

    fourier = 2*pyfftw.interfaces.numpy_fft.rfft(y_exact)/N
    fourier[::2] *= -1

    for n_coeffs in range(1, n):
        b_fft = fourier.imag[1:n_coeffs]
        b_exact = bs[:n_coeffs-1]
        print n_coeffs, la.norm(b_fft - b_exact)

    # Convergence rate in linear in N
    # All coefficients are computed with about the same accuracy
else:
    # Let's use this machinery for computing integrals f(x)*sin(k*pi*x) [0, 1]
    from sympy import symbols, sin, integrate, pi, lambdify
    # Function to integrate
    x = symbols('x')
    f = 1 # x**2
    # Analytical integrals
    n = 20
    exact = np.array([integrate(f*sin(i*pi*x), (x, 0, 1)).evalf()
                      for i in range(1, n)])

    # Data for fft
    N = 2**15
    f = lambdify(x, f)
    positive = np.array(map(f, np.linspace(0, 1, N/2)))
    y_values = np.concatenate([-positive[::-1], positive])
    x_values = np.linspace(-1, 1, N)

    numeric = pyfftw.interfaces.numpy_fft.rfft(y_values)/N
    numeric[::2] *= -1
    numeric = numeric[1:n].imag

    e = numeric - exact
    print msqrt(np.sum(e**2))
