all: CG_matrices.pdf view

CG_matrices.pdf: CG_matrices.tex CG_matrices.aux graph.mp
	mpost graph.mp
	pdflatex CG_matrices.tex
	bibtex CG_matrices.aux
	pdflatex CG_matrices.tex
	pdflatex CG_matrices.tex

view:
	open CG_matrices.pdf
