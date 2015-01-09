def merge_tables(files, rows, columns, row_format, header):
    '''
    Create tables with selected row from all files and columns for each file.
    The format of the merged line is in row_format.
    '''
    # As a sanity make sure that each column has a format
    assert sum(map(len, columns)) == len(row_format)
    # and in header there is name for each column
    assert len(row_format) == len(header)

    # Build the row template and header
    row_format = ' & '.join(row_format) + r'\\'
    header = ' & '.join(header) + r'\\'

    # Read all allowed rows from given files
    lines = [open(f, 'r').readlines()[rows[0]:rows[1]] for f in files]
    # Same number of all lines across files
    assert len(set(map(len, lines))) == 1

    print header
    print '\hline'
    # From each line of file extract given columns and build row of merged table
    for line in zip(*lines):
        row_values = []
        for f in range(len(files)):
            values = [v for i, v in enumerate(line[f].split(' '))
                      if i in columns[f]]
            row_values.extend(values)
        # Turn from string to number
        row_values = map(float, row_values)
        # Format the row
        row = row_format % tuple(row_values)
        print row
    print '\hline'
