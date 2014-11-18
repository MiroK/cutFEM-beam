from sympy import integrate, simplify, diff, symbols, Matrix, lambdify
from itertools import product
import numpy as np


def manufacture_poisson_1d(**kwargs):
    '''
    Create solution for Poisson problem:

            -E*u`` = f in (a, b)
               u = 0 on {a, b}
    '''
    a = kwargs.get('a', 0)
    b = kwargs.get('b', 1)
    E = kwargs.get('E', 1.)
    assert a < b
    u = kwargs.get('u', None)
    f = kwargs.get('f', None)

    if u is None and f is None:
        raise ValueError('Invalid argument. u and f missing')

    x = symbols('x')
    # With both u, f check solution properties
    if u is not None and f is not None:
        # Match bcs on u
        assert abs(u.evalf(subs={x: a})) < 1E-14
        assert abs(u.evalf(subs={x: b})) < 1E-14

        # Match ode
        try:
            assert simplify(f + E*diff(u, x, 2)) == 0
        # If sympy has problems with 1E-16*sin(x), do eval in points
        except AssertionError:
            e = lambdify(x, simplify(f + E*diff(u, x, 2)), 'math')
            assert all(e(xi) < 1E-15 for xi in np.linspace(a, b, 100))

        return kwargs

    # Compute f from u
    if u is not None:
        f = simplify(-E*diff(u, x, 2))
        kwargs['f'] = f
        return manufacture_poisson_1d(**kwargs)

    # Compute u from f
    if f is not None:
        du = integrate(-f/E, x)
        u = integrate(du, x)
        mat = Matrix([[a, 1],
                      [b, 1]])

        vec = Matrix([-u.subs(x, a), -u.subs(x, b)])
        c0, c1 = mat.solve(vec)
        u += c0*x + c1
        kwargs['u'] = simplify(u)
        return manufacture_poisson_1d(**kwargs)


def manufacture_biharmonic_1d(**kwargs):
    '''
    Create solution for biharmonic problem:

            -E*u```` = f in (a, b)
           u = u`` = 0 on {a, b}
    '''
    # Inteval [0, 1] by default
    a = kwargs.get('a', 0)
    b = kwargs.get('b', 1)
    E = kwargs.get('E', 1)
    assert a < b
    u = kwargs.get('u', None)
    f = kwargs.get('f', None)

    # At least one of u, f must be given
    if u is None and f is None:
        raise ValueError('Invalid argument. u and f missing')

    x = symbols('x')
    # If both are given check solution properties
    if u is not None and f is not None:
        # Match bcs on u
        assert abs(u.evalf(subs={x: a})) < 1E-15
        assert abs(u.evalf(subs={x: b})) < 1E-15

        ddu = diff(u, x, 2)
        # Match bcs on u``
        assert abs(ddu.evalf(subs={x: a})) < 1E-15
        assert abs(ddu.evalf(subs={x: b})) < 1E-15

        # Match original ode
        try:
            simplify(f - E*diff(u, x, 4)) == 0
        # If sympy has problems with 1E-16*sin(x), do eval in points
        except AssertionError:
            e = lambdify(x, simplify(f - E*diff(u, x, 4)), 'math')
            assert all(e(xi) < 1E-15 for xi in np.linspace(a, b, 100))
        return kwargs

    # Compute f from u
    if u is not None:
        f = simplify(E*diff(u, x, 4))
        kwargs['f'] = f
        return manufacture_biharmonic_1d(**kwargs)

    # Compute u from f
    if f is not None:
        d3u = integrate(f/E, x)
        d2u = integrate(d3u, x)
        du = integrate(d2u, x)
        u = integrate(du, x)

        mat = Matrix([[a**3, a**2, a, 1],
                      [b**3, b**2, b, 1],
                      [6*a, 2, 0, 0],
                      [6*b, 2, 0, 0]])

        vec = Matrix([-u.subs(x, a),
                      -u.subs(x, b),
                      -d2u.subs(x, a),
                      -d2u.subs(x, b)])
        c0, c1, c2, c3 = mat.solve(vec)

        u += c0*x**3 + c1*x**2 + c2*x + c3

        kwargs['u'] = u
        return manufacture_biharmonic_1d(**kwargs)


def manufacture_poisson_2d(**kwargs):
    '''
    Create solution for Poisson problem:

            -E*u`` = f in [ax, bx] x [ay, by]
                 u = 0 on boundary
    '''
    [[ax, bx], [ay, by]] = kwargs.get('domain', [[0, 1], [0, 1]])
    E = kwargs.get('E', 1.)
    assert ax < bx and ay < by
    u = kwargs.get('u', None)
    f = kwargs.get('f', None)

    if u is None and f is None:
        raise ValueError('Invalid argument. u and f missing')

    x, y = symbols('x, y')
    # With both u, f check solution properties
    if u is not None and f is not None:
        # Match bcs on u
        assert simplify(u.subs(x, ax)) == 0
        assert simplify(u.subs(x, by)) == 0
        assert simplify(u.subs(y, ay)) == 0
        assert simplify(u.subs(y, by)) == 0

        # Match ode
        try:
            assert simplify(f + E*diff(u, x, 2) + E*diff(u, y, 2)) == 0
        # If sympy has problems with 1E-16*sin(x), do eval in points
        except AssertionError:
            e = lambdify([x, y],
                         simplify(f + E*diff(u, x, 2) + E*diff(u, y, 2)),
                         'math')
            assert all(e(xi, yi) < 1E-15
                       for xi, yi in product(np.linspace(ax, bx, 100),
                                             np.linspace(ay, by, 100)))

        return kwargs

    # Compute f from u
    if u is not None:
        f = simplify(-E*diff(u, x, 2) - E*diff(u, y, 2))
        kwargs['f'] = f
        return manufacture_poisson_2d(**kwargs)

    raise NotImplementedError('Getting u for general f is hard!')


def manufacture_biharmonic_2d(**kwargs):
    '''
    Create solution for Poisson problem:

            -E*laplace^2(u) = f in [ax, bx] x [ay, by]
                          u = 0 on boundary
                 laplace(u) = 0 on boundary
    '''
    [[ax, bx], [ay, by]] = kwargs.get('domain', [[0, 1], [0, 1]])
    E = kwargs.get('E', 1.)
    assert ax < bx and ay < by
    u = kwargs.get('u', None)
    f = kwargs.get('f', None)

    if u is None and f is None:
        raise ValueError('Invalid argument. u and f missing')

    x, y = symbols('x, y')
    # With both u, f check solution properties
    if u is not None and f is not None:
        # Match bcs on u
        assert simplify(u.subs(x, ax)) == 0
        assert simplify(u.subs(x, by)) == 0
        assert simplify(u.subs(y, ay)) == 0
        assert simplify(u.subs(y, by)) == 0

        # Match bcs on laplace(u)
        u_xx = diff(u, x, 2)
        u_yy = diff(u, y, 2)
        lap_u = u_xx + u_yy

        assert simplify(lap_u.subs(x, ax)) == 0
        assert simplify(lap_u.subs(x, by)) == 0
        assert simplify(lap_u.subs(y, ay)) == 0
        assert simplify(lap_u.subs(y, by)) == 0

        # Match ode
        try:
            u_xxxx = diff(u_xx, x, 2)
            u_xxyy = diff(u_xx, y, 2)
            u_yyxx = diff(u_yy, x, 2)
            u_yyyy = diff(u_yy, y, 2)

            lhs = simplify(E*(u_xxxx + u_xxyy + u_yyxx + u_yyyy))
            res = f - lhs
            assert simplify(res) == 0
        # If sympy has problems with 1E-16*sin(x), do eval in points
        except AssertionError:
            e = lambdify([x, y], res, 'math')
            assert all(e(xi, yi) < 1E-15
                       for xi, yi in product(np.linspace(ax, bx, 100),
                                             np.linspace(ay, by, 100)))

        return kwargs

    # Compute f from u
    if u is not None:
        u_xx = diff(u, x, 2)
        u_yy = diff(u, y, 2)

        u_xxxx = diff(u_xx, x, 2)
        u_xxyy = diff(u_xx, y, 2)
        u_yyxx = diff(u_yy, x, 2)
        u_yyyy = diff(u_yy, y, 2)

        f = simplify(E*(u_xxxx + u_xxyy + u_yyxx + u_yyyy))
        kwargs['f'] = f
        return manufacture_biharmonic_2d(**kwargs)

    raise NotImplementedError('Getting u for general f is hard!')

# ------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import pi, sin, exp
    x = symbols('x')

    # Create and check 1d poisson solution
    f = 1
    u_dict = manufacture_poisson_1d(f=f, a=-1, b=1, E=4.)
    u_dict['f'] = None
    f_dict = manufacture_poisson_1d(**u_dict)
    assert not (f_dict['f'] - f)

    # Create and check 1d biharmonic solution
    f = exp(x)*sin(x*pi)
    u_dict = manufacture_biharmonic_1d(f=f, a=-1, b=1, E=3.)
    u_dict['f'] = None
    f_dict = manufacture_biharmonic_1d(**u_dict)
    assert not (f_dict['f'] - f)

    # Check 2d poisson solution
    y = symbols('y')
    f = 2.*(x*(1-x) + y*(1-y))
    u = x*(1-x)*y*(1-y)
    f_dict = manufacture_poisson_2d(u=u, domain=[[0, 1], [0, 1]])
    assert not simplify(f_dict['f'] - f)

    # Check 2d biharmonic solution
    y = symbols('y')
    f = 4*pi**4*sin(pi*x)*sin(pi*y)
    u = sin(pi*x)*sin(pi*y)
    ans = manufacture_biharmonic_2d(u=u, domain=[[0, 1], [0, 1]])
    assert not simplify(ans['f'] - f)
