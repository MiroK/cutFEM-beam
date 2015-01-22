def make_mesh(A, B, filename, lc=1):
    # Make sure A, B are on the boundary
    assert A[0] and A[1]
    assert A[0] and A[1]

    # Write geo file
    with open(filename, 'w') as f:
        # Points:
        f.write('Point(1) = {-1, -1, 0, %g};\n' % lc)
        f.write('Point(2) = {1, -1, 0, %g};\n' % lc)
        f.write('Point(3) = {1, 1, 0, %g};\n' % lc)
        f.write('Point(4) = {-1, 1, 0, %g};\n' % lc)
        f.write('Point(5) = {%g, %g, 0, %g};\n' % (A[0], A[1], lc))
        f.write('Point(6) = {%g, %g, 0, %g};\n' % (B[0], B[1], lc))

        # Lines
        f.write('Line(1) = {1, 5};\n')
        f.write('Line(2) = {5, 2};\n')
        f.write('Line(3) = {2, 3};\n')
        f.write('Line(4) = {3, 6};\n')
        f.write('Line(5) = {6, 4};\n')
        f.write('Line(6) = {4, 1};\n')
        f.write('Line(7) = {5, 6};\n')

        # Loops
        f.write('Line Loop(1) = {1, 7, 5, 6};\n')
        f.write('Line Loop(2) = {2, 3, 4, -7};\n')

        # Plane surfaces
        f.write('Plane Surface(1) = {1};')
        f.write('Plane Surface(2) = {2};')

        # Physical surfaces
        f.write('Physical Surface(1) = {1, 2};\n')

        # Physical lines
        f.write('Physical Line(42) = {7};\n')

    return filename

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import subprocess

    A = [-0.75, -1]
    B = [0.75, 1]

    geo_file = make_mesh(A, B, 'test.geo')
    subprocess.Popen('gmsh -2 %s' % geo_file, shell=True)

