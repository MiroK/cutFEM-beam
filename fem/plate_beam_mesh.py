import numpy as np
import numpy.linalg as la
import os


def make_mesh(A, B, filename, ts=[], hole=[], msize=1, convert=False):
    '''
    Define GMSH's geo file for the following shape
        
                        B
    [-1, 1] ------------x--- [1, 1]
        |               |        |
        |  ----H1       /        |
        |  |   |       |         |
        |  |   |      /          |
        |  ----      |           |
        | H0        /            |
    [-1, -1] ------x--------- [1, -1]
                   A

    -A, B are end point of the segment and must lie on the bottom and top edges.
    -filename is the .geo file where the mesh is stored
    -if ts is not emty it specified the extra points on |AB|
    -hole is a list with bottom left and top right vertex of rectangular hole or
     all four vertices of the quidrilateral
    -msize controls mesh size of points
    -convert True calls gmsh to make the .msh file and also converts to .xml
    '''
    # Generate the points, lines, etc
    if not hole:
        structure = make_mesh_nohole(A, B, ts)
    else:
        structure = make_mesh_hole(A, B, ts, hole)

    # Write the geo file from structure
    write_geo(structure, filename, msize)


def make_mesh_nohole(A, B, ts):
    'Make structure for mesh with no holes and ts points on the beam.'
    # Make sure points are on proper edges
    assert abs(A[0]) < 1 + 1E-13 and abs(A[1] + 1) < 1E-13, 'Not on bottom edge'
    assert abs(B[0]) < 1 + 1E-13 and abs(B[1] - 1) < 1E-13, 'Not on top edge'
    # Arrays are better for point generation
    if isinstance(A, list):
        A = np.array(A)
    if isinstance(B, list):
        B = np.array(B)

    # Make sure ts specifies points correctly
    if ts:
        # Always sort
        ts = np.sort(ts)
        # prepend -1 and append 1 if already not there
        if abs(ts[0] + 1) > 1E-13:
            ts = np.concatenate((np.array([-1]), ts))
        if abs(ts[-1] -1) > 1E-13:
            ts = np.concatenate((ts, np.array([1])))
    else:
        ts = np.array([-1, 1])

    # Points are vertices and then beams points
    # They are ordered and will be reffered to by python_index + 1!
    points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    beam_points = np.array([0.5*A*(1-t) + 0.5*B*(1+t) for t in ts])
    points = np.concatenate((points, beam_points))

    # Lines, we have the external lines and internal on the beam, point connect
    first = 4 + 1
    last = 4 + len(beam_points)
    lines = [(1, first), (first, 2), (2, 3), (3, last), (last, 4), (4, 1)]
    lines += [(f, f+1) for f in range(first, last)]
    # Remember lines that were on the beam
    beam_lines = range(7, 7+last-first)

    # Data for line loops
    line_loops = [[5, 6, 1] + beam_lines,
                  [2, 3, 4] + [-i for i in reversed(beam_lines)]]

    # Data for plane surfaces, here each surface is single loop but that can
    # change. Note the shift
    plane_surfaces = [[i] for i, _ in enumerate(line_loops, 1)]

    # Data for physical surfaces. Each plane gets a tag
    phys_surfaces = [i for i, _ in enumerate(plane_surfaces, 1)]

    # Data for physical lines, only have 1 tagged as 42, line and tag
    phys_lines = [([1], 42)]

    # Now build the structure
    structure= {'points': points,
                'lines': lines,
                'line_loops':  line_loops,
                'plane_surfaces': plane_surfaces,
                'physical_surfaces': phys_surfaces,
                'physical_lines': phys_lines}

    return structure


def make_mesh_hole(A, B, ts, hole):
    'Make structure for mesh with holes and (at least) ts points on the beam.'
    # Make sure points are on proper edges
    assert abs(A[0]) < 1 + 1E-13 and abs(A[1] + 1) < 1E-13, 'Not on bottom edge'
    assert abs(B[0]) < 1 + 1E-13 and abs(B[1] - 1) < 1E-13, 'Not on top edge'
    # Arrays are better for point generation
    if isinstance(A, list):
        A = np.array(A)
    if isinstance(B, list):
        B = np.array(B)
    
    # Now onto hole
    assert len(hole) in (2, 4)
    # Extend 2 defining points to rectangular shape
    if len(hole) == 2:
        P, Q = hole
        # Always arrays
        if isinstance(P, list):
            P = np.array(P)
        if isinstance(Q, list):
            Q = np.array(Q)

        dx, dy = Q[0] - P[0], Q[1] - P[1]
        assert dx > 0 and dy > 0, 'Points can define a rectangle'
        dx = np.array([dx, 0])
        hole = [P, P + dx, Q, Q-dx]
    else:
        hole = map(lambda P: P if isinstance(P, list) else np.array(P), hole)
    hole = np.array(hole)

    # Make sure all points are inside
    inside = map(lambda P: abs(P[0]) < 1 and abs(P[1]) < 1, hole)
    assert all(inside), 'Not all hole points are inside'

    # Decide when there is definitely no intersect, Remeber side of beam with
    # the hole
    is_intersected = True
    plane_surface = None
    # Hole is on the left side of the beam
    if max(P[0] for P in hole) < min(A[0], B[0]):
        is_intersected = False
        plane_surface = 0
    elif min(P[0] for P in hole) > max(A[0], B[0]):
        is_intersected = False
        plane_surface = 1
    # Check for intersects, each side of rect x beam
    else:
        # Collect interscted edges along with beam coordinate
        intersected_edges = []
        t_intersect = []
        for i in range(4):
            P = hole[i]
            Q = hole[(i+1)%4]

            mat = np.vstack([B-A, P-Q]).T
            vec = -(A+B) + (P+Q)
            try:
                s, t = la.solve(mat, vec)
            except la.LinAlgError:
                continue

            if abs(s)-1 < 1E-13 and abs(t)-1 < 1E-13:
                intersected_edges.append(i)
                t_intersect.append(t)

        # So now we either have two intersects or empty
        assert len(intersected_edges) in (2, 0), 'Wrong number of intesects'

        if len(intersected_edges) == 0:
            is_intersected = False
            # Decide left/right from beam is the hole and then we're done
            if max(P[0] for P in hole) < max(A[0], B[0]):
                plane_surface = 0
            elif min(P[0] for P in hole) < min(A[0], B[0]):
                plane_surface = 1
        else:
            # For now we hard code how to deal with top bottom isect and nothing
            # else is not allowed
            assert set(sorted(intersected_edges)) == {0, 2}, 'Sorry'

    structure = make_mesh_nohole(A, B, ts)
    # With no intersect we only modify the nohole structure a bit
    if not is_intersected and plane_surface is not None:
        # Append points of hole
        points = structure['points']
        # n is the index of last point before new points appended
        n = len(points)
        structure['points'] = np.concatenate((points, hole))
        
        # Append lines
        lines = structure['lines']
        # n is the index of last line before new were added
        n = len(lines)
        new_lines = [(n, n+1), (n+1, n+2), (n+2, n+3), (n+3, n)]
        structure['lines'].extend(new_lines)
        
        # Make loop of new lines
        structure['line_loops'].append(tuple(n+i+1
                                             for i in range(len(new_lines))))

        # Modify the plane surface. Add the new loop to correct side
        n = len(structure['line_loops'])
        structure['plane_surfaces'][plane_surface].append(n)

        # New added line loop should be tagged as some physical pline
        lines_tags = list(structure['line_loops'][-1]), 1
        structure['physical_lines'].append(lines_tags)
    else:
        raise NotImplementedError

    return structure


def write_geo(structure, filename, msize=1):
    'Write data from structure into the geo file.'
    # Check the extension, respect the conventions for .geo
    base, ext = os.path.splitext(filename)
    assert ext == '.geo', 'Must save into .geo file!'

    # Some comments
    comment = lambda s: f.write('\n//%s\n' % s)
    
    with open(filename, 'w') as f:
        # Write points
        comment('Points')
        for i, (x, y) in enumerate(structure['points'], 1):
            f.write('Point(%d) = {%g, %g, 0, %g};\n' % (i, x, y, msize))

        # Write lines
        comment('Lines')
        for i, (a, b) in enumerate(structure['lines'], 1):
            f.write('Line(%d) = {%d, %d};\n' % (i, a, b))

        # Write line loops
        comment('Line Loops')
        for i, loop in enumerate(structure['line_loops'], 1):
            f.write('Line Loop(%d) = {%s};\n' % (i, ', '.join(map(str, loop))))

        # Write plane surfaces
        comment('Plane Surfaces')
        for i, surf in enumerate(structure['plane_surfaces'], 1):
            f.write('Plane Surface(%d) = {%s};\n' % (i,
                                                     ', '.join(map(str, surf))))

        # Write physical surfaces
        comment('Physical Surfaces')
        for i, surf in enumerate(structure['physical_surfaces'], 1):
            f.write('Physical Surface(%d) = {%s};\n' % (i, surf))
        
        # Write physical lines
        comment('Physical Lines')
        for lines_tag in structure['physical_lines']:
            print lines_tag
            lines = ', '.join(map(str, lines_tag[0]))
            tag = lines_tag[1]
            f.write('Physical Line(%d) = {%s};\n' % (tag, lines))

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Beam
    A = [0.75, -1]
    B = [0.75, 1]


    # No hole
    # make_mesh(A, B, 'foo.geo', ts=[-0.25, 0, 0.5, 0.75], hole=[], msize=1)
    
    # Hole no isect
    # hole_r = [np.array([-0.75, 0]), np.array([-0.5, 0.25])]
    # make_mesh(A, B, 'foo.geo', ts=[], hole=hole_r, msize=1)

    # Hole isect
    hole_r = [np.array([-0.75, -0.75]), np.array([0.8, 0.8])]
    make_mesh(A, B, 'foo.geo', ts=[], hole=hole_r, msize=1)
