all: plate_beam.pdf

plate_beam.pdf: plate_beam.tex plate_beam.bib
	pdflatex plate_beam.tex
	bibtex plate_beam.aux
	pdflatex plate_beam.tex
	pdflatex plate_beam.tex

view:
	open plate_beam.pdf
