UV ?= uv

.PHONY: install format format-check lint typecheck test check clean

install:
	$(UV) sync --locked --all-groups

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

format-check:
	$(UV) run ruff format --check .

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy src tests scripts

test:
	$(UV) run pytest

check: format-check lint typecheck test

clean:
	$(UV) run python scripts/clean.py

