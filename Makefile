# Requires an environment with the dev extra installed:
#   pip install -e ".[dev]"

.PHONY: check test evidence

check:
	ruff check .
	mypy
	pytest

test:
	pytest

# Regenerates evidence/results.json from the committed deterministic corpus.
# Every number in the README must come from this and nowhere else.
evidence:
	python evidence/recompute.py
