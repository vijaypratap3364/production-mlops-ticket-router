UV ?= uv

.PHONY: install format format-check lint typecheck test check download-data normalize-data analyze-data prepare-data train-baselines experiment-candidates evaluate-final recover-final-registration promote-candidate api-dev clean

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

download-data:
	$(UV) run python -m ticket_router.data.download

normalize-data:
	$(UV) run python -m ticket_router.data.normalize

analyze-data:
	$(UV) run python -m ticket_router.data.analyze

prepare-data:
	$(UV) run python -m ticket_router.data.prepare

train-baselines:
	$(UV) run python -m ticket_router.modeling.train_baseline

experiment-candidates:
	$(UV) run python -m ticket_router.modeling.train_candidates

evaluate-final:
	$(UV) run python -m ticket_router.registry.evaluate_final

recover-final-registration:
	$(UV) run python -m ticket_router.registry.recover_final

promote-candidate:
	$(UV) run python -m ticket_router.registry.promote --approve

api-dev:
	$(UV) run uvicorn ticket_router.api.main:app --host 127.0.0.1 --port 8000

clean:
	$(UV) run python scripts/clean.py
