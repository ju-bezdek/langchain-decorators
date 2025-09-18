# Makefile for building documentation

# Prefer the repo's local virtualenv Python if present; fallback to python3
PYTHON := python3
ifneq (,$(wildcard .venv/bin/python))
PYTHON := .venv/bin/python
endif

# Call Sphinx via the Python module so we don't rely on PATH scripts
SPHINXBUILD = $(PYTHON) -m sphinx
SPHINXOPTS = -q
SPHINXHTML = docs/build/html
SPHINXPDF = docs/build/pdf
SPHINXDOXYGEN = docs/build/doxygen

.PHONY: help clean html pdf deploy docu

help:
	@echo "Makefile for building documentation"
	@echo "Usage:"
	@echo "  make clean      - Remove all build artifacts"
	@echo "  make html       - Build HTML documentation"
	@echo "  make pdf        - Build PDF documentation"

clean:
	@echo "Cleaning up build artifacts..."
	rm -rf $(SPHINXHTML) $(SPHINXDOXYGEN)

html:
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) $(SPHINXOPTS) -b html docs $(SPHINXHTML)



deploy:
	@echo "Cleaning dist, building package, and uploading to PyPI..."
	rm -f dist/* || true
	python -m build
	twine upload dist/*