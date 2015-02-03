from __future__ import division
from sympy import legendre, Symbol, lambdify, integrate, simplify
from sympy.mpmath import quad

# Verify the integral formulas for legendre polynomials
x = Symbol('x')
a, b = -1, 1  # The interval for integration

def leg_integral(k):
    l_n = legendre(k+1, x)
    l_p = legendre(k-1, x)
    return (l_n - l_p)/(2*k+1)


# Single integral
for i in range(1, 21):
    # Numeric
    num = integrate(legendre(i, x))
    # Symbolic
    sym = leg_integral(i)

    #print num, sym
    print '\t', simplify(num - sym)
