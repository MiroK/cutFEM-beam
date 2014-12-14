from __future__ import division
import sys
sys.path.insert(0, '../../../')

from points import gauss_legendre_lobatto_points as gll_points
from points import gauss_legendre_points as gl_points
from points import chebyshev_points as cheb_points
from functions import lagrange_basis, bernstein_basis
from sympy import symbols, sin, cos, pi, legendre, sqrt
import random


class H1Basis(object):
    'Generator of basis function in H1.'
    def __call__(self):
        raise NotImplementedError('I am a template')


class H10Basis(H1Basis):
    'Generator of basis function in H10.'
    pass

# Derive from helper classes to get properties


class Hierarchical(object):
    '''
    Generator for spaces where the new approximation is created by adding
    one new function to the previous ones. This is used for spaces with
    natural hierarchy!
    '''
    pass


class Lagrangian(object):
    '''
    Basis of Lagrange polynomials. For these, the underlying points
    are important.
    '''
    def __init__(self, points):
        self.points = points


class FourierBasis(H10Basis, Hierarchical):
    'Basis of eigenvectors of laplacian on (-1, 1) with homog dirichlet bcs.'
    def __init__(self):
        H10Basis.__init__(self)
        Hierarchical.__init__(self)

    def __call__(self):
        'Yield the next function.'
        x = symbols('x')
        i = 0
        while True:
            phi = (i+1)*pi/2
            if i % 2 == 0:
                yield cos(phi*x)
                i += 1
            else:
                yield sin(phi*x)
                i += 1


class Shen1Basis(H10Basis, Hierarchical):
    '''
    Basis of combinations of Legendre polynomials combined to meet 0 Dirichlet
    bcs in (-1, 1).
    '''
    def __init__(self):
        H10Basis.__init__(self)
        Hierarchical.__init__(self)

    def __call__(self):
        'Yield the next basis function.'
        x = symbols('x')
        i = 0
        while True:
            yield (legendre(i, x) - legendre(i+2, x))/sqrt(4*i + 6)
            i += 1


class Shen2Basis(H10Basis, Hierarchical):
    '''
    Basis of combinations of Legendre polynomials combined to meet 0 Dirichlet
    bcs in (-1, 1).
    '''
    def __init__(self):
        H10Basis.__init__(self)
        Hierarchical.__init__(self)

    def __call__(self):
        'Yield the next basis function.'
        x = symbols('x')
        i = 0
        while True:
            yield legendre(i+2, x)-(legendre(0, x)
                                    if (i % 2) == 0 else legendre(1, x))
            i += 1


class LegendreBasis(H1Basis, Hierarchical):
    'Basis of Legendre polynomials'
    def __init__(self):
        H1Basis.__init__(self)
        Hierarchical.__init__(self)

    def __call__(self):
        'Yield the new basis function.'
        x = symbols('x')
        i = 0
        while True:
            yield legendre(i, x)
            i += 1


class GL_LagrangeBasis(H1Basis, Lagrangian):
    '''
    Space is constructed from Gauss Legendre points. The call to iteratior
    yields n polynomials of degree n-1.
    '''
    def __init__(self):
        H1Basis.__init__(self)
        Lagrangian.__init__(self, gl_points)

    def __call__(self):
        'Yield the new basis functions'
        i = 2
        while True:
            yield lagrange_basis([self.points([i])])
            i += 1


class Cheb_LagrangeBasis(H1Basis, Lagrangian):
    '''
    Space is constructed from Chebyshev points. The call to iteratior
    yields n polynomials of degree n-1.
    '''
    def __init__(self):
        H1Basis.__init__(self)
        Lagrangian.__init__(self, cheb_points)

    def __call__(self):
        'Yield the new basis functions'
        i = 2
        while True:
            yield lagrange_basis([self.points([i])])
            i += 1


class GLL_LagrangeBasis(H1Basis, Lagrangian):
    '''
    Space is constructed from Lagrange polynimials nodal in Gauss Legendre
    Lobatto points.
    '''
    def __init__(self):
        H1Basis.__init__(self)
        Lagrangian.__init__(self, gll_points)

    def __call__(self):
        'Yield the new basis functions'
        i = 2
        while True:
            yield lagrange_basis([self.points([i])])
            i += 1


class GLL_LagrangeBasis0(H10Basis, Lagrangian):
    '''
    Space is constructed from Lagrange polynimials nodal in Gauss Legendre
    Lobatto points and zero on {-1, 1}.
    '''
    def __init__(self):
        H1Basis.__init__(self)
        Lagrangian.__init__(self, gll_points)

    def __call__(self):
        'Yield the new basis functions'
        i = 3
        while True:
            functions = lagrange_basis([self.points([i])])
            yield functions[1:(len(functions)-1)]
            i += 1


class BernsteinBasis(H1Basis):
    'Space is constructed Bernstein polynomials.'
    def __init__(self):
        H1Basis.__init__(self)

    def __call__(self):
        'Yield the new basis function'
        i = 0
        while True:
            yield bernstein_basis(i)
            i += 1


class BernsteinBasis0(H10Basis):
    'Space is constructed Bernstein polynomials that are zero on {-1, 1}'
    def __init__(self):
        H10Basis.__init__(self)

    def __call__(self):
        'Yield the new basis function'
        i = 2
        while True:
            functions = bernstein_basis(i)
            yield functions[1:(len(functions)-1)]
            i += 1


def make_hierarchy(cls):
    'Turn class that is not Hierarchical into one.'

    assert not issubclass(cls, Hierarchical)
    assert issubclass(cls, H1Basis)

    # I want to create class which is like the original but with two exceptions:
    # i) the new class is Hierarchical
    # ii) the call is modified with caching

    parents = [Hierarchical]
    if issubclass(cls, H10Basis):
        parents.append(H10Basis)
    else:
        parents.append(H1Basis)

    class NewClass(parents[0], parents[1]):
        def __init__(self):
            for parent in parents:
                parent.__init__(self)
            self.basis_instance = cls()
            # We have a cache to remember choices across calls to generator
            # The items are chosen randomly in first run
            self.cached_k = {}

        def __call__(self):
            'Yield the new functions from modal basis.'
            i = 0
            for functions in self.basis_instance():
                if i in self.cached_k:
                    k = self.cached_k[i]
                else:
                    if i:
                        k = random.choice(range(len(functions)))
                    else:
                        k = 0
                self.cached_k[i] = k
                yield functions[k]
                i += 1

    return NewClass

GL_LagrangeBasis_H = make_hierarchy(GL_LagrangeBasis)

Cheb_LagrangeBasis_H = make_hierarchy(Cheb_LagrangeBasis)

GLL_LagrangeBasis_H = make_hierarchy(GLL_LagrangeBasis)

GLL_LagrangeBasis0_H = make_hierarchy(GLL_LagrangeBasis0)

BernsteinBasis_H = make_hierarchy(BernsteinBasis)

BernsteinBasis0_H = make_hierarchy(BernsteinBasis0)

# This should be split accross 4 porcesses in such a way that
# GLL, GL all reside on one process. Otherwise the points might
# might be accessed at the same time by different procs - race
# conditions
__basis_d__ = {'0fourier': FourierBasis,
               '0shen1': Shen1Basis,
               '0shen2': Shen2Basis,
               '0legendre': LegendreBasis,
               
               '1gl_lagrange': GL_LagrangeBasis,
               '1gl_lagrange_h': GL_LagrangeBasis_H,
               '1cheb_lagrange': Cheb_LagrangeBasis,
               '1cheb_lagrange_h': Cheb_LagrangeBasis_H,
               
               '2gll_lagrange': GLL_LagrangeBasis,
               '2gll_lagrange0': GLL_LagrangeBasis0,
               '2gll_lagrange_h': GLL_LagrangeBasis_H,
               '2gll_lagrange0_h': GLL_LagrangeBasis0_H,
               
               '3bernstein': BernsteinBasis,
               '3bernstein0': BernsteinBasis0,
               '3bernstein_h': BernsteinBasis_H,
               '3bernstein0_h': BernsteinBasis0_H
               }

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.plotting import plot

    x = symbols('x')
    basis = GLL_LagrangeBasis()

    if isinstance(basis, Hierarchical):
        for f in basis():
            if isinstance(basis, H10Basis):
                assert abs(f.evalf(subs={x: -1})) < 1E-15
                assert abs(f.evalf(subs={x: 1})) < 1E-15
                print 'OK'
            plot(f, (x, -1, 1))
    else:
        for fs in basis():
            print len(fs), fs
            for f in fs:
                if isinstance(basis, H10Basis):
                    assert abs(f.evalf(subs={x: -1})) < 1E-15
                    assert abs(f.evalf(subs={x: 1})) < 1E-15
                    print 'OK'
                plot(f, (x, -1, 1))
