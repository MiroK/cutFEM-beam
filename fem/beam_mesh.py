import numpy as np


def line_mesh(a, b, n_cells, file_name):
    gdim = len(a)
    assert len(a) == len(b)

    n_vertices = n_cells + 1
    coordinates = np.zeros((n_vertices, gdim))

    s = np.linspace(0, 1, n_vertices)
    for i in range(gdim):
        coordinates[:, i] = a[i] + (b[i] - a[i])*s

    top = '<?xml version="1.0"?>\n'
    header_ = '<dolfin xmlns:dolfin="http://fenicsproject.org">\n'
    mesh_ = '\t<mesh celltype="interval" dim="2">\n'

    vertices_ = '\t\t<vertices size="%d">\n' % n_vertices
    vertex = '\t\t\t<vertex index="%d" x="%.16f" y="%.16f"/>\n'
    _vertices = '\t\t</vertices>\n'

    cells_ = '\t\t<cells size="%d">\n' % n_cells
    intervals = '\t\t\t<interval index="%d" v0="%d" v1="%d"/>\n'
    _cells = '\t\t</cells>\n'

    _mesh = '\t</mesh>\n'
    _header = '</dolfin>\n'

    with open(file_name, 'w') as f:
        f.write(top)
        f.write(header_)
        f.write(mesh_)

        f.write(vertices_)
        for i, (x, y) in enumerate(coordinates):
            f.write(vertex % (i, x, y))
        f.write(_vertices)

        f.write(cells_)
        for i in range(n_cells):
            f.write(intervals % (i, i, i+1))
        f.write(_cells)

        f.write(_mesh)
        f.write(_header)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    line_mesh([0, 0], [1, 1], 2, 'foo')
