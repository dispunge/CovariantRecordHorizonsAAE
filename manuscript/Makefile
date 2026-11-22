PYTHON := $(shell if [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; else echo python3; fi)
SCRIPT := ../lf_leak_tests.py
MAIN := main.tex
BASE := $(basename $(MAIN))
OUTPUT := $(BASE).pdf
FIGURES := figures/lf_leakage_bounds.pdf figures/lf_leakage_bounds.png
JSON := data/lf_verification_results.json
LATEXMK := latexmk
PDFLATEX := pdflatex
BIBTEX := bibtex

.PHONY: all setup rebuild clean clean-aux figures build audit check

all: rebuild

check:
	@if [ ! -f "$(MAIN)" ]; then echo "No $(MAIN) yet; create manuscript/$(MAIN) before building."; exit 1; fi
	@if ! command -v $(LATEXMK) >/dev/null 2>&1; then \
		command -v $(PDFLATEX) >/dev/null 2>&1 || { echo "Neither $(LATEXMK) nor $(PDFLATEX) is installed."; exit 1; }; \
		command -v $(BIBTEX) >/dev/null 2>&1 || { echo "$(BIBTEX) is not installed."; exit 1; }; \
	fi

figures:
	mkdir -p figures data
	$(PYTHON) -c "import numpy, scipy, matplotlib" >/dev/null 2>&1 || $(PYTHON) -m pip install -r ../requirements.txt
	cd .. && $(subst ../,,$(PYTHON)) $(subst ../,,$(SCRIPT)) --plot --save-json lf_verification_results.json
	cp ../lf_leakage_bounds.pdf figures/lf_leakage_bounds.pdf
	cp ../lf_leakage_bounds.png figures/lf_leakage_bounds.png
	cp ../lf_verification_results.json $(JSON)

clean-aux:
	rm -f $(BASE).aux $(BASE).bbl $(BASE).bcf $(BASE).blg $(BASE).fdb_latexmk $(BASE).fls $(BASE).log $(BASE).out $(BASE)Notes.bib $(BASE).run.xml $(BASE).synctex.gz $(BASE).toc

build: check clean-aux
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -pdf -g -interaction=nonstopmode -halt-on-error $(MAIN); \
	else \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN); \
		$(BIBTEX) $(BASE); \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN); \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN); \
	fi
	@$(MAKE) --no-print-directory audit

audit:
	@if grep -E "LaTeX Warning: (Reference|Citation).*undefined|There were undefined" $(BASE).log; then \
		echo "ERROR: Undefined references or citations found in $(BASE).log." >&2; exit 1; \
	else \
		echo "OK: No undefined references or citations."; \
	fi
	@if [ -f "$(BASE).blg" ] && grep -E "Warning--|error message" $(BASE).blg | grep -v "Warning--jnrlst"; then \
		echo "ERROR: BibTeX warnings or errors found in $(BASE).blg." >&2; exit 1; \
	else \
		echo "OK: BibTeX clean."; \
	fi
	@severe=$$(awk '/Overfull \\hbox/ { match($$0, /\([0-9.]+pt too wide\)/); if (RSTART > 0) { pt = substr($$0, RSTART + 1, RLENGTH - 13) + 0; if (pt > 10.0) print $$0 } }' $(BASE).log); \
	if [ -n "$$severe" ]; then \
		echo "$$severe"; \
		echo "ERROR: Severe Overfull \\hbox (> 10pt) detected in $(BASE).log." >&2; exit 1; \
	elif grep -q "Overfull \\\\hbox" $(BASE).log; then \
		grep "Overfull \\\\hbox" $(BASE).log; \
		echo "OK: No severe Overfull hbox (> 10pt) warnings."; \
	else \
		echo "OK: No Overfull hbox warnings."; \
	fi
	@if command -v pdftotext >/dev/null 2>&1; then \
		if pdftotext $(OUTPUT) - | grep -qF "(??)"; then \
			echo "ERROR: (??) still in $(OUTPUT)!" >&2; exit 1; \
		else \
			echo "OK: Zero (??) in $(OUTPUT)."; \
		fi; \
		if pdftotext $(OUTPUT) - | grep -qF "ACKNOWLEDGMENTS"; then \
			echo "WARN: Check ACKNOWLEDGMENTS heading in $(OUTPUT)." >&2; exit 1; \
		else \
			echo "OK: Replaced with Code and Data Availability."; \
		fi; \
	fi
	@if command -v pdffonts >/dev/null 2>&1; then \
		if pdffonts $(OUTPUT) | grep -q "Type 3"; then \
			pdffonts $(OUTPUT) | grep "Type 3"; \
			echo "ERROR: Type 3 fonts in $(OUTPUT) (bitmap TeX fonts or matplotlib's default); their text does not extract or search reliably." >&2; exit 1; \
		else \
			echo "OK: No Type 3 fonts in $(OUTPUT)."; \
		fi; \
	fi

rebuild: clean figures build

clean: clean-aux
	$(LATEXMK) -C $(MAIN) 2>/dev/null || true
	rm -f $(OUTPUT) $(FIGURES) $(JSON)
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *Notes.bib *.run.xml *.synctex.gz *.toc
	rm -rf build/

setup:
	@if [ ! -d "../.venv" ]; then echo "No virtual environment found. Creating one..."; python3 -m venv ../.venv; fi
	@echo "Installing required packages..."
	../.venv/bin/python -m pip install -r ../requirements.txt
