from sympy import integrate, simplify, diff, symbols, Matrix


def manufacture_poisson(**kwargs):
    '''
    Create solution for Poisson problem:

            -u`` = f in (a, b)
               u = 0 on {a, b}
    '''
    a = kwargs.get('a', 0)
    b = kwargs.get('b', 1)
    assert a < b
    u = kwargs.get('u', None)
    f = kwargs.get('f', None)

    if u is None and f is None:
        raise ValueError('Invalid argument. u and f missing')

    x = symbols('x')
    # With both u, f check solution properties
    if u is not None and f is not None:
        # Match bcs on u
        assert abs(u.evalf(subs={x: a})) < 1E-15
        assert abs(u.evalf(subs={x: b})) < 1E-15

        # Match ode
        assert simplify(f + diff(u, x, 2)) == 0
        return kwargs

    # Compute f from u
    if u is not None:
        f = simplify(-diff(u, x, 2))
        kwargs['f'] = f
        return manufacture_poisson(**kwargs)

    # Compute u from f
    if f is not None:
        du = integrate(-f, x)
        u = integrate(du, x)
        mat = Matrix([[a, 1],
                      [b, 1]])

        vec = Matrix([-u.subs(x, a), -u.subs(x, b)])
        c0, c1 = mat.solve(vec)
        u += c0*x + c1
        kwargs['u'] = u
        return manufacture_poisson(**kwargs)


def manufacture_biharmonic(**kwargs):
    '''
    Create solution for biharmonic problem:

            -u```` = f in (a, b)
           u = u`` = 0 on {a, b}
    '''
    # Inteval [0, 1] by default
    a = kwargs.get('a', 0)
    b = kwargs.get('b', 1)
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
        assert simplify(f - diff(u, x, 4)) == 0
        return kwargs

    # Compute f from u
    if u is not None:
        f = simplify(diff(u, x, 4))
        kwargs['f'] = f
        return manufacture_biharmonic(**kwargs)

    # Compute u from f
    if f is not None:
        d3u = integrate(f, x)
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
        return manufacture_biharmonic(**kwargs)

# ------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import pi, sin, exp
    x = symbols('x')

    f = x
    u_dict = manufacture_poisson(f=f, a=-1, b=1)
    u_dict['f'] = None
    f_dict = manufacture_poisson(**u_dict)
    assert not (f_dict['f'] - f)
    print f_dict

    f = exp(x)*sin(x)
    u_dict = manufacture_biharmonic(f=f, a=-1, b=1)
    u_dict['f'] = None
    f_dict = manufacture_biharmonic(**u_dict)
    assert not (f_dict['f'] - f)
    print f_dict

