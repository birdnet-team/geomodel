# BirdNET Geomodel paper -- build targets
#
#   make            build main.pdf (pdflatex + bibtex, as many passes as needed)
#   make watch      rebuild continuously while editing
#   make arxiv      produce arxiv-submission.tar.gz (includes main.bbl)
#   make clean      remove build artifacts, keep main.pdf
#   make distclean  remove build artifacts and main.pdf

MAIN := main
LATEXMK := latexmk -pdf -interaction=nonstopmode -file-line-error

.PHONY: all watch arxiv check docx docx-from-pdf clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex body.tex appendix.tex references.bib arxiv.sty $(wildcard figures/*)
	$(LATEXMK) $(MAIN).tex

watch:
	$(LATEXMK) -pvc $(MAIN).tex

# arXiv wants a self-contained source bundle: .tex, .sty, .bbl (never .bib),
# and the figures. It runs LaTeX itself but does not run BibTeX.
arxiv: $(MAIN).pdf
	@test -f $(MAIN).bbl || { echo "main.bbl missing -- run 'make' first"; exit 1; }
	tar -czf arxiv-submission.tar.gz \
		$(MAIN).tex body.tex appendix.tex $(MAIN).bbl arxiv.sty figures/
	@echo "arxiv-submission.tar.gz written:"
	@tar -tzf arxiv-submission.tar.gz | head -5
	@echo "  ... $$(tar -tzf arxiv-submission.tar.gz | wc -l) files total"

# Word version for co-authors and submission portals. The LaTeX route gives a
# structured, editable document; docx-from-pdf keeps the printed layout instead.
docx:
	python make_docx.py

docx-from-pdf: $(MAIN).pdf
	python make_docx.py --from-pdf --out main-layout.docx

# Fail loudly on the problems that matter before uploading.
check: $(MAIN).pdf
	@echo "== undefined references/citations =="
	@! grep -E "Citation .* undefined|Reference .* undefined" $(MAIN).log || exit 1
	@echo "none"
	@echo "== overfull boxes =="
	@grep -c "Overfull" $(MAIN).log || true
	@echo "== pages =="
	@pdfinfo $(MAIN).pdf | grep Pages

clean:
	latexmk -c
	rm -f $(MAIN).bbl $(MAIN).run.xml

distclean: clean
	rm -f $(MAIN).pdf arxiv-submission.tar.gz main.docx main-layout.docx
