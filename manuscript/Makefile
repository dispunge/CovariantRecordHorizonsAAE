PYTHON := $(shell if [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; else echo python3; fi)
SCRIPT := ../lf_leak_tests.py
MAIN := main.tex
OUTPUT := main.pdf
FIGURES := figures/lf_leakage_bounds.pdf figures/lf_leakage_bounds.png
JSON := data/lf_verification_results.json
LATEXMK := latexmk

.PHONY: all setup rebuild clean figures build check

all: rebuild

check:
	@if [ ! -f "$(MAIN)" ]; then echo "No $(MAIN) yet; create manuscript/$(MAIN) before building."; exit 1; fi
	@command -v $(LATEXMK) >/dev/null 2>&1 || { echo "latexmk is not installed. Install a TeX distribution and latexmk."; exit 1; }

figures:
	mkdir -p figures data
	$(PYTHON) -m pip install -r ../requirements.txt
	cd .. && $(subst ../,,$(PYTHON)) $(subst ../,,$(SCRIPT)) --plot --save-json lf_verification_results.json
	cp ../lf_leakage_bounds.pdf figures/lf_leakage_bounds.pdf
	cp ../lf_leakage_bounds.png figures/lf_leakage_bounds.png
	cp ../lf_verification_results.json $(JSON)

build: check
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN)

rebuild: clean figures build

clean:
	$(LATEXMK) -C $(MAIN) 2>/dev/null || true
	rm -f $(OUTPUT) $(FIGURES) $(JSON)
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.synctex.gz *.toc
	rm -rf build/

setup:
	@if [ ! -d "../.venv" ]; then echo "No virtual environment found. Creating one..."; python3 -m venv ../.venv; fi
	@echo "Installing required packages..."
	../.venv/bin/python -m pip install -r ../requirements.txt
