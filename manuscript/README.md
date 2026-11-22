# Manuscript build

The default build performs a clean rebuild: install the Python requirements, regenerate numerical figures and JSON, copy generated figures into `figures/`, clean LaTeX artifacts, and compile `main.tex`.

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
./build.sh rebuild  # default; clean, regenerate, and compile
./build.sh build    # compile existing inputs without regenerating figures
./build.sh figures  # regenerate and copy figures and JSON only
./build.sh clean    # remove generated and LaTeX build artifacts
```

For incremental LaTeX compilation without regeneration, run:

```bash
make build
```

`main.tex` must exist and a TeX distribution with `latexmk` must be installed.
