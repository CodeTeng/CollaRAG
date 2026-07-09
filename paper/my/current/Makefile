# Makefile for CollaRAG paper
# Usage: make pdf

MAIN = main
BIB = references

.PHONY: pdf clean

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(BIB).bib
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).pdf $(MAIN).nav $(MAIN).snm $(MAIN).fls \
	      $(MAIN).fdb_latexmk $(MAIN).synctex.gz
