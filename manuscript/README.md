# Manuscript build

The default build performs a clean rebuild: verify or install the Python requirements, regenerate numerical figures and JSON, copy generated figures into `figures/`, clean stale LaTeX/REVTeX auxiliary artifacts (including `mainNotes.bib`), compile `main.tex` in `nonstopmode`, and run post-build audit checks on `main.log`, `main.blg`, and `main.pdf`.

```bash
cd manuscript
./build.sh
```

The Python packages are declared in `../requirements.txt` (`numpy`, `scipy`, and `matplotlib`). To install them explicitly:

```bash
python3 -m pip install -r ../requirements.txt
```

Other commands:

```bash
./build.sh rebuild  # default; clean, regenerate figures/JSON, compile, and audit
./build.sh build    # clean stale aux files, compile existing inputs, and audit
./build.sh figures  # regenerate and copy figures and JSON only
./build.sh audit    # verify zero undefined refs/citations, clean BibTeX, layout bounds, and no Type 3 fonts
./build.sh clean    # remove generated and LaTeX build artifacts
```

For LaTeX compilation without figure regeneration, run:

```bash
make build
```

`main.tex` must exist and a TeX distribution (`latexmk`, or `pdflatex` + `bibtex`) must be installed.
