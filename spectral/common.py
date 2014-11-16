from numpy import finfo

class Counter(object):
    'Couning with __cal__'
    def __init__(self, n=0):
        self.n = n
    def __call__(self, x):
        self.n += 1
    def __str__(self):
        return str(self.n)


# Precision to which things are carried
__EPS__ = finfo(float).eps

# Directory where points, quadratures are cached
__CACHE_DIR__ = '.cache'
