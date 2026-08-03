# BirdNET Geomodel — paper source

LaTeX source for the arXiv preprint. Converted from `report/report_draft.md`;
**`main.tex` + `body.tex` are now the source of truth** — edits belong here, not
in the markdown draft.

## Layout

| File | Purpose |
|:-----|:--------|
| `main.tex` | Preamble, title block, abstract, bibliography — the file you compile |
| `body.tex` | Sections 1--9 (`\input` by `main.tex`) |
| `appendix.tex` | Appendices A--E, input after the bibliography |
| `references.bib` | 50 BibTeX entries, one per work cited |
| `arxiv.sty` | arXiv preprint style (vendored, do not fetch at build time) |
| `figures/` | Charts as vector PDF, maps as PNG (see below) |
| `Makefile`, `latexmkrc` | Build targets and latexmk configuration |

Figures 1 and 2 are native TikZ, written directly in `body.tex`.

All charts and the global maps are vector PDFs, regenerated without in-plot titles
(the titles now live in the LaTeX captions) by:

```bash
python report/scripts/filter_study_geofilter.py     # filter-study panels
python report/scripts/plot_observation_density.py   # observation density map
python report/scripts/plot_label_propagation.py     # richness + propagation maps
```

Each writes a `.png` and a `.pdf`; the paper uses the `.pdf`. Data-heavy layers are
rasterized inside the PDF so file sizes stay reasonable while text and axes remain
vector.

The eBird S&T overlap maps are the exception: they remain PNGs, cropped to remove
their baked-in titles. They were **not** regenerated, because no checkpoint in
`checkpoints/` matches the "scale 0.75, 12K species" production model whose metrics
Table 3 reports, and regenerating with a different checkpoint would put the figures
and the table out of sync. Regenerate them with
`python report/scripts/compare_ebirdst.py --checkpoint <the right one>` once that is
resolved, and drop `set_title` from `plot_overlap_map` at the same time.

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
pandoc: it produces real Word headings, real tables, embedded figures (the two
TikZ diagrams are compiled to images on the fly), Word equations, and citations
resolved from `references.bib` into author-date form with a reference list. The
page layout is Word's, not the arXiv style's.

`--from-pdf` converts the rendered `main.pdf` through LibreOffice instead. It
keeps the printed appearance, but the text lands in one frame per line --- no
headings, no tables, no practical way to edit it. Use it only when someone needs
to look at the paper in Word.

Requires pandoc; if it is not on PATH the script uses the copy in the project
venv (`.venv/bin/pip install pypandoc_binary`). Neither output is authoritative
--- check tables and equations before sending either anywhere.

Requirements: a TeX Live installation with `latexmk`, `bibtex`, and the packages
listed in the `main.tex` preamble (all standard; TeX Live 2019 or newer is
sufficient — verified on 2019).

### VS Code

`.vscode/settings.json` in the repository root configures LaTeX Workshop with a
`latexmk` recipe, build-on-save, SyncTeX, and a tab viewer. Open `paper/main.tex`
and use *LaTeX Workshop: Build LaTeX project*, or just save the file.

Note that `.vscode/` is git-ignored, so those settings stay local to this
machine.

## arXiv submission

```bash
make arxiv      # -> arxiv-submission.tar.gz
```

arXiv compiles the sources itself but **does not run BibTeX**, so the bundle
includes `main.bbl` and omits `references.bib`. Everything else it needs
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
  identifiable; regenerating the maps without in-image titles would look cleaner
  still.
