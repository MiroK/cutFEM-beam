from matplotlib.pyplot import colorbar, clabel
from sympy import lambdify, symbols
import sympy.plotting as s_plot
import numpy as np


def plot(f, domain, figure=None, **kwargs):
    '''
    Quick plot of f which is a symbolic expression or tuple of len two,
    where the first item are expansion coefficients and the second one are
    basis functions(symbolic expressions).
    '''
    dim = len(domain)
    if dim > 2:
        raise NotImplementedError('Only 1d, 2d plots work for now')

    xy = list(symbols('x, y'))

    try:
        f_plot = sum(Ui*fi for (Ui, fi) in zip(f[0].flatten(),
                                               f[1].flatten()))
    except TypeError:
        f_plot = f

    if dim == 1:
        bounds = tuple(xy[:dim] + domain[0])
        return s_plot.plot(f_plot, bounds, **kwargs)
    else:
        bounds_x = tuple(xy[:1] + domain[0])
        bounds_y = tuple(xy[1:2] + domain[1])
        # Plot with matplotlib
        if figure is not None:
            f_lambda = lambdify(xy[:dim], f_plot)

            x_array = np.linspace(*bounds_x[1:], num=100)
            y_array = np.linspace(*bounds_y[1:], num=100)
            X, Y = np.meshgrid(x_array, y_array)

            Z = np.array([f_lambda(x_, y_)
                          for (x_, y_) in zip(X.flatten(),
                                              Y.flatten())]).reshape(X.shape)

            ax = figure.gca()
            pc = ax.pcolor(X, Y, Z)
            colorbar(pc)
            co = ax.contour(X, Y, Z, 6, colors='k')
            clabel(co, fontsize=9, inline=1)

            if 'title' in kwargs:
                ax.set_title(kwargs['title'])

            return None

        # Plot with sympy
        else:
            return s_plot.plot3d(f_plot, bounds_x, bounds_y, **kwargs)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x, y = symbols('x, y')

    u = x**2*y*(1-y)
    plot(u, [[0, 1], [2, 3]], title=r'$x^2 y (1-y)$')

    figure = plt.figure()
    U = np.array([1, 2])
    basis = np.array([x**2, y**2])
    plot((U, basis), [[-1, 1], [-1, 1]], figure, title=r'$x^2 + 2y^2$')
    plt.show()

    plot((U, basis), [[-1, 1], [-1, 1]], title=r'$x^2 + 2y^2$')
