from __future__ import division
from sympy import integrate, lambdify, symbols, legendre
from shenp_basis import shenp_basis as basis
from sympy.plotting import plot
from sympy.mpmath import quad
from sympy.mpmath import legendre as mp_legendre, sqrt as mp_sqrt

x = symbols('x')

def linear_legendre(f, k):
    'Integral of (linear function)` * L_k` from -1, 1'
    df = f.diff(x, 1)
    Lk = legendre(k, x)
    dLk = Lk.diff(x, 1)

    # by symbolic cal
    sym = float(integrate(df*dLk, (x, -1, 1)))

    # by quad
    num = float(quad(lambdify(x, df*dLk), (-1, 1)))

    # by formula
    form = float(df*(Lk.subs(x, 1) - Lk.subs(x, -1)))

    return abs(sym-num)<1E-15 and abs(sym-form)<1E-15 and abs(num-form)<1E-15


def hat_legendre(xp, xi, xn, k):
    '''
    Integral of (hat function(xp, xi, xn))` * L_k` from -1, 1

    Hat function is 0 at xp, xn, 1 at xi, linear between (xp, xi), (xi, xn) and
    zero everywhere else.
    '''
    assert xp < xn
    assert abs(xp)-1 < 1 and abs(xn)-1 < 1
    assert xp < xi and xi < xn
    # Hat function
    left = (x-xp)/(xi-xp)
    right = (x-xn)/(xi-xn)
    d_left = left.diff(x, 1)
    d_right = right.diff(x, 1)

    # L
    Lk = legendre(k, x)
    dLk = Lk.diff(x, 1)

    # by sym
    sym = integrate(d_left*dLk, (x, xp, xi))
    sym += integrate(d_right*dLk, (x, xi, xn))

    # by num
    num = quad(lambdify(x, d_left*dLk), [xp, xi])
    num += quad(lambdify(x, d_right*dLk), [xi, xn])

    # by formula
    form = (Lk.subs(x, xi) - Lk.subs(x, xp))/(xi-xp)
    form += (Lk.subs(x, xn) - Lk.subs(x, xi))/(xi-xn)

    assert abs(sym-num)<1E-13 and abs(sym-form)<1E-13 and abs(num-form)<1E-13


def hat_shen(xp, xi, xn, k):
    '''
    Integral of (hat function(xp, xi, xn))` * S_k` from -1, 1

    Hat function is 0 at xp, xn, 1 at xi, linear between (xp, xi), (xi, xn) and
    zero everywhere else.
    '''
    assert xp < xn
    assert abs(xp)-1 < 1 and abs(xn)-1 < 1
    assert xp < xi and xi < xn
    # Hat function
    left = (x-xp)/(xi-xp)
    right = (x-xn)/(xi-xn)
    d_left = left.diff(x, 1)
    d_right = right.diff(x, 1)

    # S
    Sk = list(basis(k+1))[k]
    dSk = Sk.diff(x, 1)

    # by sym
    sym = integrate(d_left*dSk, (x, xp, xi))
    sym += integrate(d_right*dSk, (x, xi, xn))

    # by num
    num = quad(lambdify(x, d_left*dSk), [xp, xi])
    num += quad(lambdify(x, d_right*dSk), [xi, xn])

    # by formula
    form = (Sk.subs(x, xi) - Sk.subs(x, xp))/(xi-xp)
    form += (Sk.subs(x, xn) - Sk.subs(x, xi))/(xi-xn)

    assert abs(sym-num)<1E-13 and abs(sym-form)<1E-13 and abs(num-form)<1E-13

    return sym, num, form


def hat_shen_h(h, k):
    'Integral of (hat function(-h, 0, h))` * S_k` from -1, 1'
    # Compare against
    # sym, num, form = hat_shen(0-h, 0, 0+h, k)
    # New formula
    Sk = lambda k, x: (mp_legendre(k+2, x) - mp_legendre(k, x))/mp_sqrt(4*k + 6)

    # We have shifted odd and even

    if k%2 == 0:  
        form_h = (2*Sk(k, 0)-2*Sk(k, h))/h
    else:
        form_h = 0

    # assert abs(form_h-sym) < 1E-13, '@%d: %g %g' % (k, form_h, sym)
    return float(form_h)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if False:
        f = 2*x + 1
        for k in range(0, 15):
            assert linear_legendre(f, k=k)
        print 'Okay'

        xp = -0.25
        xi = 0.1
        xn = 0.55
        for k in range(0, 10):
            hat_legendre(xp, xi, xn, k=k)
        print 'Okay'

        for k in range(1, 10):
            hat_shen(xp, xi, xn, k=k)
        print 'Okay'

        for h in [1, 0.5, 0.25, 0.125]:
            print h
            for k in range(2, 15):
                hat_shen_h(h=h, k=k)

    from matplotlib import rc 
    rc('text', usetex=True) 
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    import matplotlib.pyplot as plt
    from pylab import getp
    import numpy as np

    A = 2
    ks = np.arange(2, 257)
    plt.figure()
    labels = iter([r'$\frac{1}{4}$',
                   r'$\frac{1}{8}$',
                   r'$\frac{1}{16}$'])
    for h in [1/4., 1/8, 1/16.]:
        ck = np.array([hat_shen_h(h, k) for k in ks])
        ck = np.sqrt(ck**2)
        line,  = plt.loglog(ks[::2], ck[::2], label=next(labels), marker='x')
        plt.loglog([2./h, 2./h], [1e-4, 1], color=getp(line, 'color'), linestyle='--')

        # print n, ks[np.argmax(ck)], A*n
    plt.loglog(ks[::2], ks[::2]**(-1.), label='rate 1')
    plt.legend(loc='best')
    plt.xlabel('$k$')
    plt.ylabel(r'$|(f^{\prime}, \psi^{\prime}_k)|$')
    plt.savefig('shen_hat_h.pdf')
    plt.show()
