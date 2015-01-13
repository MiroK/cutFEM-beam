from __future__ import division
from sympy import lambdify, symbols, sqrt
from sympy.mpmath import quad
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

'''
Given a mapping \chi, \chi : [-1, 1] -> [-1, 1]^d we all (generalized) beam
a set of points x in [-1, 1]^d, (generalized) plate, that are such that 
x = \chi(s) for s \in [-1, 1].

There is only support for d = 1, 2, 3
'''

# Plate variables
xyz = symbols('x, y, z')
x, y, z = xyz
# Beam variables
s = symbols('s')

def on_boundary(V):
    'Check if point V lies on the plate boundary'
    # Get the geometric dimension of the point
    d = len(V)
    ans = []
    for i in range(d):
        axis = abs(V[i]**2 - 1) < 1E-15
        js = range(d)
        js.pop(i)
        plane = all(((-1 - 1E-15) < V[j] < (1 + 1E-15)) for j in js)
        ans_i = axis and plane
        ans.append(ans_i)
    return any(ans)


class Beam(object):
    'Create the beam from mapping'
    def __init__(self, chi):
        # Geometry
        d = len(chi)
        assert d in (1, 2, 3)
        self.d = d
        # The mapping
        self.chi = chi
        # Compute the jacobian |d_x| = |d_chi/d_s| * |d_s|
        Jac = 0
        for i in range(d):
            Jac += chi[i].diff(s, 1)**2
        Jac = sqrt(Jac)
        # At least some Jac check for degeneracy
        assert all(Jac.subs(s, val) > 0 for val in np.linspace(-1, 1, 20))
        self.Jac = Jac

    def inner_product(self, u, v):
        '''
        Inner product over beam. Functions must be symbolic functions of
        parameter s
        '''
        assert s in u.atoms() and s in v.atoms()
        u = lambdify(s, u)
        v = lambdify(s, v)
        J = lambdify(s, self.Jac)
        return quad(lambda s: u(s)*v(s)*J(s), [-1, 1])

    def restrict(self, u):
        'Restrict function from plate variables to beam variables.'
        assert all(var in u.atoms() for var in xyz[:self.d])
        return u.subs({(var, self.chi[i])
                        for i, var in enumerate(xyz[:self.d])})

    def plot(self, n_points=100, fig=None):
        'Plot beam embedded in 2d plate'
        assert self.d == 2

        chi = lambdify(s, self.chi)
        
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()
        points = np.array([list(chi(val))
                            for val in np.linspace(-1, 1, n_points)])
        ax.plot(points[:, 0], points[:, 1])
        ax.set_axes('equal')
        return fig


class LineBeam(Beam):
    'LineBeam is a segment defined be two points on the boundary.'
    def __init__(self, A, B):
        # Check that A, B are okay points
        if isinstance(A, list):
            A = np.array(A)
        if isinstance(B, list):
            B = np.array(B)

        assert len(A) == len(B)
        d = len(A)
        assert A.shape == B.shape and A.shape == (d, )

        # Check that they are on the bondary
        assert on_boundary(A) and on_boundary(B)

        # Check that they are not identical
        assert not np.allclose(A-B, np.zeros(d), 1E-13)

        # Creata the chi map
        chi = tuple(A[i]/2*(1 - s) + B[i]/2*(1 + s) for i in range(d))
        
        # Call parent
        Beam.__init__(self, chi)

# ----------------------------------------------------------------------------- 

if __name__ == '__main__':
    # 1d
    pts = [[-1], [0.5], [1]]
    vals = True, False, True
    assert all(on_boundary(p) == val for p, val in zip(pts, vals))
    # 2d
    pts = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0.25, 0.25], [2, 2]]
    vals = True, True, True, True, False, False, False, True
    assert all(on_boundary(p) == val for p, val in zip(pts, vals))
    # 3d
    pts = [[1, 0, 0], [0, -1, 1], [0, 0, 0], [2, 2, 2], [1, 1, 1]]
    vals = True, True, False, False, True
    assert all(on_boundary(p) == val for p, val in zip(pts, vals))
    # 4d for fun
    pts = [[1, 0, 0, 1], [-1, 0, -1, 1], [0.5, 0, 0, 0]]
    vals = True, True, False
    assert all(on_boundary(p) == val for p, val in zip(pts, vals))

    A = np.array([-1, -1])
    B = np.array([1, 1])

    beam = LineBeam(A, B)
    # Check jacobian is half of |A-B|
    assert abs(beam.Jac - 0.5*np.hypot(*(A-B))) < 1E-13
    
    # Sines should be orthogonal
    from sympy import sin, pi, cos, simplify
    assert abs(beam.inner_product(sin(pi*s), sin(2*pi*s))) < 1E-13
    assert abs(beam.inner_product(sin(pi*s), sin(pi*s))/beam.Jac - 1) < 1E-13

    # Restriction
    u = sin(x)*cos(2*y)
    assert simplify(beam.restrict(u) - sin(1.0*s)*cos(2.0*s)) == 0

    # Check the points
    chi = lambdify(s, beam.chi)
    all(abs(chi(val)[0] - chi(val)[1]) < 1E-13
            for val in np.linspace(-1, 1, 100))

    fig0 = beam.plot()

    # Something more exotic
    chi = (-1 + 2*cos(pi*(s+1)/4), -1 + 2*sin(pi*(s+1)/4))
    beam = Beam(chi)
    fig1 = beam.plot()

    plt.show()
