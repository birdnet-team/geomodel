#!/usr/bin/env python3
"""Convert the paper to main.docx.

    python make_docx.py            # -> main.docx
    python make_docx.py --out x.docx --keep

Word cannot reproduce a LaTeX preprint faithfully, and it does not need to: the
DOCX exists for co-authors and journal submission portals, not for typesetting.
This script therefore rewrites the sources into a plainer LaTeX dialect that
pandoc can read, and lets pandoc build the document structure:

  * the arXiv style, header, and float tuning are dropped -- pandoc stalls while
    parsing arxiv.sty, and none of it survives the conversion anyway;
  * the two TikZ figures are compiled standalone and embedded as images, since
    pandoc cannot draw them;
  * figures given to LaTeX as PDF are swapped for their PNG siblings, because
    Word cannot embed PDF images;
  * \\resizebox around a table is unwrapped and tabularx becomes tabular, both of
    which pandoc otherwise drops along with the table inside them;
  * citations are resolved from references.bib by citeproc and land in the text
    as "(Author, Year)" with a reference list at the end.

Requires pandoc. If it is not on PATH the script looks for the copy bundled with
the project virtualenv (pip install pypandoc_binary), and pdflatex + pdftoppm for
the TikZ figures.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_PANDOC = (HERE.parent / ".venv/lib/python3.11/site-packages/pypandoc/files/pandoc")

# Preamble lines that only matter for print layout; pandoc either ignores them or
# chokes on them.
DROP_LINE_PATTERNS = [
    r"\\usepackage\{arxiv\}",
    r"\\renewcommand\{\\shorttitle\}",
    r"\\renewcommand\{\\headeright\}",
    r"\\fancyhf",
    r"\\lhead", r"\\rhead", r"\\chead", r"\\cfoot",
    r"\\floatstyle", r"\\newfloat", r"\\floatname",
    r"\\captionsetup",
    r"\\setlength\{\\tabcolsep\}",
    r"\\renewcommand\{\\arraystretch\}",
    r"\\renewcommand\{\\tabularxcolumn\}",
    r"\\newcolumntype",
    r"\\raggedbottom",
    r"\\graphicspath",
    r"\\usetikzlibrary",
    r"\\counterwithin",
    r"\\clearpage",
]


def find_pandoc(explicit: str | None) -> str:
    for cand in (explicit, shutil.which("pandoc"), str(VENV_PANDOC)):
        if cand and Path(cand).exists():
            return cand
    sys.exit(
        "pandoc not found. Install it system-wide, or into the project venv with:\n"
        "    .venv/bin/pip install pypandoc_binary"
    )


def strip_preamble(text: str) -> str:
    """Remove layout-only commands and the \\makeatletter block."""
    out, skip = [], False
    for line in text.split("\n"):
        if "\\makeatletter" in line:
            skip = True
            continue
        if "\\makeatother" in line:
            skip = False
            continue
        if skip:
            continue
        if any(re.search(p, line) for p in DROP_LINE_PATTERNS):
            continue
        out.append(line)
    return "\n".join(out)


def render_tikz(body: str, workdir: Path, figdir: Path) -> str:
    """Compile each tikzpicture standalone and swap in an image."""
    pdflatex = shutil.which("pdflatex")
    pdftoppm = shutil.which("pdftoppm")
    count = 0

    while "\\begin{tikzpicture}" in body:
        start = body.index("\\begin{tikzpicture}")
        end = body.index("\\end{tikzpicture}") + len("\\end{tikzpicture}")
        picture = body[start:end]
        count += 1
        name = f"tikz_figure_{count}"

        replacement = ""
        if pdflatex and pdftoppm:
            src = workdir / f"{name}.tex"
            src.write_text(
                "\\documentclass[border=4pt]{standalone}\n"
                "\\usepackage{tikz}\n"
                "\\usetikzlibrary{positioning,fit,calc,arrows.meta}\n"
                "\\usepackage{amsmath,amssymb}\n"
                "\\begin{document}\n" + picture + "\n\\end{document}\n",
                encoding="utf-8",
            )
            ok = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", src.name],
                cwd=workdir, capture_output=True,
            ).returncode == 0
            if ok:
                subprocess.run(
                    [pdftoppm, "-png", "-r", "200", "-singlefile",
                     f"{name}.pdf", f"figures/{name}"],
                    cwd=workdir, capture_output=True,
                )
                if (figdir / f"{name}.png").exists():
                    replacement = f"\\includegraphics[width=\\linewidth]{{figures/{name}.png}}"
                    print(f"  rendered {name}.png")
        if not replacement:
            print(f"  WARNING: could not render {name}; leaving a placeholder")
            replacement = f"\\textit{{[{name}: see the PDF version]}}"
        body = body[:start] + replacement + body[end:]
    return body


def simplify(text: str) -> str:
    """Rewrite constructs pandoc silently drops."""
    # Word cannot embed PDF images; the PNG siblings are already in figures/.
    text = re.sub(r"(\{figures/[^}]+)\.pdf\}", r"\1.png}", text)

    # \resizebox{..}{..}{% <table> } -> plain table
    text = re.sub(r"\\resizebox\{[^{}]*\}\{[^{}]*\}\{%?\s*\n", "", text)
    text = text.replace("\\end{tabular}\n}", "\\end{tabular}")

    # tabularx -> tabular, and the custom wrapping column -> plain left column
    text = re.sub(r"\\begin\{tabularx\}\{[^{}]*\}\{([^{}]*)\}",
                  lambda m: "\\begin{tabular}{" + m.group(1).replace("L", "l") + "}", text)
    text = text.replace("\\end{tabularx}", "\\end{tabular}")

    # float placement hints and spacing that mean nothing in Word
    text = re.sub(r"\\begin\{(figure|table|algorithm)\}\[[^\]]*\]", r"\\begin{\1}", text)
    text = re.sub(r"\\vspace\*?\{[^{}]*\}", "", text)
    text = text.replace("\\centering", "")
    return text


def convert_from_pdf(pdf: Path, out: Path) -> int:
    """Layout-preserving conversion of the rendered PDF via LibreOffice.

    Keeps the printed appearance but produces a document made of one text frame
    per line: no headings, no real tables, effectively not editable. Useful when
    someone needs to *see* the paper in Word, not work on it.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        sys.exit("LibreOffice not found; --from-pdf needs soffice on PATH.")
    if not pdf.exists():
        sys.exit(f"{pdf} not found -- run `make` first.")

    tmp = Path(tempfile.mkdtemp(prefix="pdf2docx_"))
    subprocess.run(
        [soffice, "--headless", "--convert-to", "docx:MS Word 2007 XML",
         "--infilter=writer_pdf_import", str(pdf), "--outdir", str(tmp)],
        capture_output=True, text=True,
    )
    produced = tmp / (pdf.stem + ".docx")
    if not produced.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit("LibreOffice produced no output.")
    shutil.move(str(produced), out)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Wrote {out} ({out.stat().st_size/1024:.0f} kB) from {pdf.name}")
    print("Layout is preserved, but the text sits in per-line frames: readable,\n"
          "not editable. Use the default LaTeX route for anything editorial.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "main.docx"))
    ap.add_argument("--pandoc", default=None, help="path to the pandoc binary")
    ap.add_argument("--keep", action="store_true", help="keep the temporary build directory")
    ap.add_argument("--from-pdf", action="store_true",
                    help="convert the rendered main.pdf via LibreOffice instead "
                         "(keeps the layout, loses editability)")
    args = ap.parse_args()

    if args.from_pdf:
        return convert_from_pdf(HERE / "main.pdf", Path(args.out).resolve())

    pandoc = find_pandoc(args.pandoc)
    print(f"pandoc: {pandoc}")

    tmp = Path(tempfile.mkdtemp(prefix="paper2docx_"))
    figdir = tmp / "figures"
    figdir.mkdir()
    for f in (HERE / "figures").glob("*.png"):
        shutil.copy2(f, figdir / f.name)
    shutil.copy2(HERE / "references.bib", tmp / "references.bib")

    main_tex = simplify(strip_preamble((HERE / "main.tex").read_text(encoding="utf-8")))
    body_tex = simplify((HERE / "body.tex").read_text(encoding="utf-8"))
    appendix_tex = simplify(strip_preamble((HERE / "appendix.tex").read_text(encoding="utf-8")))

    print("Rendering TikZ figures ...")
    body_tex = render_tikz(body_tex, tmp, figdir)

    # Word has no \appendix: without help, appendix sections would continue the
    # numbering and "Appendix~\ref{...}" would render as "Appendix 12". Put the
    # letter in the heading instead and resolve references to it by hand.
    appendix_tex = appendix_tex.replace("\\appendix", "")
    letters = {}
    def letter_heading(m):
        title, label = m.group(1), m.group(2)
        ltr = chr(ord("A") + len(letters))
        letters[label] = ltr
        return f"\\section*{{Appendix {ltr}: {title}}}"

    appendix_tex = re.sub(r"\\section\{([^}]*)\}\s*\n\\label\{([^}]*)\}",
                          letter_heading, appendix_tex)
    appendix_tex = re.sub(r"\\subsection\{([^}]*)\}", r"\\subsection*{\1}", appendix_tex)
    for label, ltr in letters.items():
        pattern = r"\\ref\{" + re.escape(label) + r"\}"
        main_tex = re.sub(pattern, ltr, main_tex)
        body_tex = re.sub(pattern, ltr, body_tex)
        appendix_tex = re.sub(pattern, ltr, appendix_tex)
    print(f"  appendices lettered: {', '.join(sorted(letters.values()))}")

    (tmp / "main.tex").write_text(main_tex, encoding="utf-8")
    (tmp / "body.tex").write_text(body_tex, encoding="utf-8")
    (tmp / "appendix.tex").write_text(appendix_tex, encoding="utf-8")

    out = Path(args.out).resolve()
    print("Running pandoc ...")
    result = subprocess.run(
        [pandoc, "main.tex", "-f", "latex", "-o", str(out),
         "--citeproc", "--bibliography=references.bib",
         "--resource-path=.", "--wrap=preserve"],
        cwd=tmp, capture_output=True, text=True,
    )
    for line in (result.stderr or "").splitlines()[:15]:
        print("  " + line)
    if result.returncode != 0 or not out.exists():
        print("pandoc failed", file=sys.stderr)
        print(f"build directory kept at {tmp}", file=sys.stderr)
        return 1

    print(f"\nWrote {out} ({out.stat().st_size/1024:.0f} kB)")
    if args.keep:
        print(f"build directory: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    print("Check tables and equations before sending it anywhere: Word has no\n"
          "equivalent for some LaTeX constructs, and the PDF remains authoritative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
