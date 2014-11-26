from sympy import symbols, diff, simplify, Expr, Number, integrate, Matrix
from sympy.printing.ccode import CCodePrinter
from dolfin import Expression


class DolfinCodePrinter(CCodePrinter):
    'Printer for turning sympy.Expr to code for dolfin expressions.'
    def __init__(self, settings={}):
        CCodePrinter.__init__(self)

    def _print_Pi(self, expr):
        # pi not M_PI as in C, C++
        return 'pi'


def dolfincode(expr, subs=None, assign_to=None, **settings):
    'Turn sympy.Expr into code for doflin expressions.'
    # Print scalar expression
    if isinstance(expr, Expr):
        # Perform user substitutions
        if subs is not None:
            assert isinstance(subs, dict)
            expr = expr.subs(subs)

        # Substitute x, y, z for x[i]
        dolfin_xs = symbols('x[0] x[1] x[2]')
        xs = symbols('x y z')

        for x, dolfin_x in zip(xs, dolfin_xs):
            expr = expr.subs(x, dolfin_x)

        return DolfinCodePrinter(settings).doprint(expr, assign_to, **settings)
    # Number
    elif isinstance(expr, int):
        return DolfinCodePrinter(settings).doprint(Number(expr),
                                                   assign_to, **settings)
    # Recurse if vector or tensor
    elif isinstance(expr, tuple):
        return tuple(dolfincode(e, assign_to, **settings) for e in expr)
    else:
        raise ValueError


def biharm2d(u):
    '''
    Apply biharmonic operator D to u:
        D(u) = u_xxxx + u_xx, yy + u_yy, xx + u_yy
    '''
    x, y = symbols('x, y')
    u_xx = diff(u, x, 2)
    u_yy = diff(u, y, 2)

    u_xxxx = diff(u_xx, x, 2)
    u_xxyy = diff(u_xx, y, 2)
    u_yyxx = diff(u_yy, x, 2)
    u_yyyy = diff(u_yy, y, 2)

    return u_xxxx + u_xxyy + u_yyxx + u_yyyy


def joris_problem(D, Lx, Ly):
    '''
    Compute f such that D*biharm2d(u) == f in [-Lx, Lx] x [-Ly, Ly],
                                   u = 0
                          laplace(u) = 0
    where u is from Joris.
    '''
    from sympy import sin, exp, pi

    x, y = symbols('x, y')
    D_, Lx_, Ly_ = symbols('D_, Lx_, Ly_')

    u = ((2*x/Lx_ - 1)**2 - 1)**2*(x/Lx_ - 1)*(exp(x/Lx_) - 1)
    u *= ((2*y/Ly_ - 1)**2 - 1)**2*sin(2*pi*y/Ly_)**2

    f = D_*biharm2d(u)

    return {'u': Expression(dolfincode(u), Lx_=Lx, Ly_=Ly, degree=8),
            'f': Expression(dolfincode(f), D_=D, Lx_=Lx, Ly_=Ly, degree=8)}


def miro_problem(D, Lx, Ly):
    '''
    Compute f such that D*biharm2d(u) == f in [-Lx, Lx] x [-Ly, Ly],
                                   u = 0
                          laplace(u) = 0.
    '''
    from sympy import sin, pi

    x, y = symbols('x, y')
    D_, Lx_, Ly_ = symbols('D_, Lx_, Ly_')

    u = sin(4*pi/Lx_*x)*sin(3*pi/Ly_*y)

    f = D_*biharm2d(u)

    return {'u': Expression(dolfincode(u), Lx_=Lx, Ly_=Ly, degree=8),
            'f': Expression(dolfincode(f), D_=D, Lx_=Lx, Ly_=Ly, degree=8)}


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

        kwargs['u'] = Expression(dolfincode(u), a=a, b=b, E=E)
        kwargs['f'] = Expression(dolfincode(f), a=a, b=b, E=E)
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

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import plot, RectangleMesh

    Lx = 1
    Ly = 1
    problem = joris_problem(2, Lx=1, Ly=1)

    mesh = RectangleMesh(0, 0, Lx, Ly, 40, 40)
    plot(problem['u'], mesh=mesh)
    plot(problem['f'], interactive=True, mesh=mesh)

