.PHONY: tests
tests:
	PYTHONPATH=src pytest

.PHONY: lint
lint: # Run pre-commit on staged/changed files
	pre-commit run

.PHONY: check
check: # Run all pre-commit hooks on all files (useful for CI or full check)
	pre-commit run --all-files

.PHONY: format
format: # Manually run ruff formatter on all files
	ruff format .

.PHONY: pre-commit-install
pre-commit-install: # Install pre-commit hooks changes
	pip install ruff
	pip install pre-commit
	pre-commit install