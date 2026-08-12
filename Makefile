UV ?= uv
DOCKER_COMPOSE ?= docker compose

.PHONY: install format format-check lint typecheck test test-unit test-integration test-e2e build-package check download-data normalize-data analyze-data prepare-data train-baselines experiment-candidates evaluate-final recover-final-registration promote-candidate api-dev dashboard-dev db-upgrade db-downgrade db-revision test-db-integration build-monitoring-reference monitor retrain simulate-drift flow-ingest flow-train flow-monitor flow-retraining prefect-deploy fixture-flow docker-build docker-up docker-up-orchestration docker-down docker-logs docker-status docker-smoke docker-monitor docker-retrain migrate bootstrap clean

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

test-unit:
	$(UV) run pytest tests/unit

test-integration:
	$(UV) run pytest tests/integration --no-cov

test-e2e:
	$(UV) run pytest tests/e2e --no-cov

build-package:
	$(UV) build

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

dashboard-dev:
	$(UV) run streamlit run src/ticket_router/dashboard/app.py --server.address 127.0.0.1 --server.port 8501

db-upgrade:
	$(UV) run alembic upgrade head

db-downgrade:
	$(UV) run alembic downgrade -1

db-revision:
	@test -n "$(MESSAGE)" || (echo "Set MESSAGE, for example: make db-revision MESSAGE='add index'" && exit 1)
	$(UV) run alembic revision --autogenerate -m "$(MESSAGE)"

test-db-integration:
	$(UV) run pytest -m integration tests/integration/db

build-monitoring-reference:
	$(UV) run python -m ticket_router.monitoring.build_reference

monitor:
	$(UV) run python -m ticket_router.monitoring.run --lookback-days 7

retrain:
	$(UV) run python -m ticket_router.orchestration retraining

simulate-drift:
	$(UV) run python -m ticket_router.monitoring.simulate_drift

flow-ingest:
	$(UV) run python -m ticket_router.orchestration ingest

flow-train:
	$(UV) run python -m ticket_router.orchestration train-candidate

flow-monitor:
	$(UV) run python -m ticket_router.orchestration monitor

flow-retraining:
	$(UV) run python -m ticket_router.orchestration retraining

prefect-deploy:
	$(UV) run python -m ticket_router.orchestration.deploy

fixture-flow:
	$(UV) run python -m ticket_router.orchestration.fixture_flow

docker-build:
	$(DOCKER_COMPOSE) build api dashboard mlflow prefect-worker

docker-up:
	$(DOCKER_COMPOSE) up -d postgres mlflow migrate api dashboard

docker-up-orchestration:
	$(DOCKER_COMPOSE) --profile orchestration up -d

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs --follow --tail=200

docker-status:
	$(DOCKER_COMPOSE) ps

migrate:
	$(DOCKER_COMPOSE) run --rm migrate

bootstrap:
	$(DOCKER_COMPOSE) --profile bootstrap run --rm bootstrap
	$(DOCKER_COMPOSE) restart api

docker-smoke:
	$(DOCKER_COMPOSE) --profile smoke run --rm smoke-test

docker-monitor:
	$(DOCKER_COMPOSE) --profile orchestration exec prefect-worker python -m ticket_router.orchestration monitor

docker-retrain:
	$(DOCKER_COMPOSE) --profile orchestration exec prefect-worker python -m ticket_router.orchestration retraining

clean:
	$(UV) run python scripts/clean.py
