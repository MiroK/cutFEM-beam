import sys
sys.path.insert(0, '../')

from sympy.integrals.quadrature import gauss_legendre
from sympy import lambdify, symbols
from multiprocessing import Pool
from itertools import product
import numpy as np
import time


def foo(points, weights):
    return sum(w*p for p, w in zip(points, weights))


class Foo(object):
    def __init__(self, N, n_threads):
        self.n_threads = n_threads
        start = time.time()
        points, weights = gauss_legendre(N, 15)

        self.points = np.array([point for point in product(points, points)])
        self.weights = np.array([np.product(weight)
                                 for weight in product(weights, weights)])

        stop = time.time() - start
        print 'Got %d x %d points and %d x %d weights in %g s' % \
            (len(points), len(points),
             len(weights), len(weights),
             stop)

    def __call__(self, f, domain):
        [[ax, bx], [ay, by]] = domain
        assert ax < bx and ay < by

        x, y = symbols('x, y')
        Fx = 0.5*ax*(1-x) + 0.5*bx*(1+x)
        Fy = 0.5*ay*(1-y) + 0.5*by*(1+y)

        f = f.subs(x, Fx).subs(y, Fy)
        J = 0.5*(bx-ax)*0.5*(by-ay)

        f_lambda = lambdify([x, y], f)
        values = map(lambda xy: f_lambda(xy[0], xy[1]), self.points)

        if self.n_threads == 0:
            return J*foo(values, self.weights)
        else:
            pool = Pool(self.n_threads)
            results = []
            for ps, ws in zip(np.array_split(values, self.n_threads),
                              np.array_split(self.weights, self.n_threads)):
                results.append(pool.apply_async(foo, (ps, ws)))
            pool.close()
            pool.join()
            return J*sum((r.get() for r in results))

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import cos, integrate
    import matplotlib.pyplot as plt
    x, y = symbols('x, y')
    domain = [[-1, 2], [0, 1]]
    [[ax, bx], [ay, by]] = domain
    f = cos(x)*cos(2*y)
    exact = (integrate(integrate(f, (x, ax, bx)), (y, (ay, by)))).evalf()

    for n_threads in [0, 1, 2, 4]:
        Ns = range(3, 40)
        times = []
        errors = []
        for N in Ns:
            bar = Foo(N=N, n_threads=n_threads)
            start = time.time()
            numeric = bar(f, domain)
            stop = time.time() - start

            print 'Integration took %g s' % stop

            times.append(stop)
            errors.append(abs(exact-numeric))

        fig, ax1 = plt.subplots()
        ax1.set_title('With %d threads' % n_threads)
        ax1.plot(Ns, times, 'b-')
        ax1.set_xlabel('$N$')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('time $s$', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ax2 = ax1.twinx()
        ax2.semilogy(Ns, errors, 'g')
        ax2.set_ylabel('error', color='g')
        for tl in ax2.get_yticklabels():
            tl.set_color('g')

    plt.show()
