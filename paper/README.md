# BirdNET Geomodel paper source

> This paper describes V3 (union labels, label propagation, the environment
> auxiliary task). V4 changes the targets, the training and the outputs. Its methods
> are documented in `docs/` until a V4 report replaces this one.

LaTeX source for the arXiv preprint. It was converted from a Markdown draft;
`main.tex` and `body.tex` are the source of truth, and edits belong here.

## Layout

| File | Purpose |
|:-----|:--------|
| `main.tex` | Preamble, title block, abstract, bibliography; the file you compile |
| `body.tex` | Sections 1--9 (`\input` by `main.tex`) |
| `appendix.tex` | Appendices A--E, input after the bibliography |
| `references.bib` | 50 BibTeX entries, one per work cited |
| `arxiv.sty` | arXiv preprint style (vendored, do not fetch at build time) |
| `figures/` | Charts as vector PDF, maps as PNG (see below) |
| `Makefile`, `latexmkrc` | Build targets and latexmk configuration |

Figures 1 and 2 are native TikZ, written directly in `body.tex`.

All charts and the global maps (filter-study panels, observation density map,
richness and propagation maps) are vector PDFs without in-plot titles; the titles
are in the LaTeX captions. Each figure exists as `.png` and `.pdf`, and the paper
uses the `.pdf`. Data-heavy layers are rasterized inside the PDF, so file sizes stay
small while text and axes remain vector. The scripts that generate the figures are
not part of the repository.

The eBird S&T overlap maps are the exception. They are PNGs, cropped to remove their
baked-in titles. They were not regenerated, because that requires the "scale 0.75,
12K species" production model whose metrics Table 3 reports; figures from a
different model would not match the table.

## Building locally

```bash
make            # pdflatex + bibtex until stable -> main.pdf
make watch      # continuous rebuild while editing
make check      # fail on undefined refs/citations, report overfull boxes
make clean      # remove build artifacts, keep the PDF
```

## Word version

```bash
make docx           # main.docx        -- structured and editable (default)
make docx-from-pdf  # main-layout.docx -- looks like the PDF, not editable
```

`make_docx.py` drives both. The default route converts the LaTeX sources with
pandoc. The result has Word headings and tables, embedded figures (the two TikZ
diagrams are compiled to images on the fly), Word equations, and citations resolved
from `references.bib` into author-date form with a reference list. The page layout
is Word's, not the arXiv style's.

`--from-pdf` converts the rendered `main.pdf` through LibreOffice instead. It keeps
the printed appearance, but the text lands in one frame per line, with no headings,
no tables and no practical way to edit it. Use it only to view the paper in Word.

The default route requires pandoc. If pandoc is not on PATH, the script uses the
copy in the project venv (`.venv/bin/pip install pypandoc_binary`). Neither output
is authoritative, so check tables and equations before sending either one.

Requirements: a TeX Live installation with `latexmk`, `bibtex` and the packages
listed in the `main.tex` preamble (all standard). TeX Live 2019 or newer works; the
build is tested on 2019.

### VS Code

With the LaTeX Workshop extension, open `paper/main.tex` and run *LaTeX Workshop:
Build LaTeX project*. The repository does not include VS Code settings.

## arXiv submission

```bash
make arxiv      # -> arxiv-submission.tar.gz
```

arXiv compiles the sources itself but does not run BibTeX, so the bundle includes
`main.bbl` and omits `references.bib`. Everything else it needs
(`main.tex`, `body.tex`, `arxiv.sty`, `figures/`) is in the tarball, and no
package is loaded that is outside the standard TeX Live distribution.

Before uploading, run `make check` and confirm the log shows no undefined
references and no overfull boxes.

## Overleaf

Upload the folder as-is (or the tarball). It compiles with the default
pdfLaTeX + BibTeX setting; do not switch the project to biber.

## Open items

- Appendix E (*Export Validation Tolerances*) was never written in the source
  draft. Its heading is commented out in `appendix.tex` so the PDF does not show
  an empty section; uncomment the two lines once the content exists.
- Figure panel titles inside the eBird S&T maps are small at grid size. The
  species names are repeated as LaTeX subcaptions, so the panels remain
  identifiable. Maps without in-image titles would look cleaner.
