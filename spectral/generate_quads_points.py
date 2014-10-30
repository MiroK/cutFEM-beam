from quadrature import __EPS__, __CACHE_DIR__
from quadrature import all_quads
from points import all_points
import os


def generate_all_foo(all_foos, N, n_digits):
    '''
    Generate points and weights for all quadratures in all_quads
    that use up to and including N points. Sympy computation uses
    n_digits but at the end all is converted to float.
    '''
    # FIXME. Cached quadrature does not store precision!
    # FIXME. Cached points do not store precision!
    for n in range(1, N+1):
        for foo in all_foos:
            try:
                # Creating the instance makes the points
                foo([n], n_digits)
            except AssertionError:
                print foo, 'Failed'


def main(argv):
    'Generate or clear cache.'
    # Either clear or generate with 15 digits precision
    assert 1 < len(argv) < 4
    if len(argv) == 2:
        if argv[1].startswith('-') and argv[1][1:] == 'clear':
            print 'Clearing %s' % __CACHE_DIR__
            for f in os.listdir(__CACHE_DIR__):
                os.remove(os.path.join(__CACHE_DIR__, f))
        else:
            argv.append('15')
            return main(argv)
    else:
        try:
            N = int(sys.argv[1])
            n_digits = int(sys.argv[2])
        except ValueError:
            print 'bash> python generate_all_quads 30 15'

        print 'Generating quadratures...'
        generate_all_foo(all_quads, N, n_digits)
        print 'Generating points...'
        generate_all_foo(all_points, N, n_digits)

#------------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
