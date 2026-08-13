# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv
FROM python:3.12-slim-bookworm AS builder

COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group runtime --group monitoring \
    --group orchestration --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group runtime --group monitoring \
    --group orchestration --no-editable

FROM python:3.12-slim-bookworm AS runtime

ARG SOURCE_GIT_COMMIT=""
ARG SOURCE_GIT_DIRTY="false"
ENV PATH=/opt/ticket-router-venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SOURCE_GIT_COMMIT=$SOURCE_GIT_COMMIT \
    SOURCE_GIT_DIRTY=$SOURCE_GIT_DIRTY
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 ticket-router \
    && useradd --uid 10001 --gid ticket-router --create-home ticket-router \
    && mkdir -p /app/data /app/artifacts /app/reports \
    && chown -R ticket-router:ticket-router /app
WORKDIR /app

COPY --from=builder --chown=ticket-router:ticket-router /app/.venv /opt/ticket-router-venv
COPY --chown=ticket-router:ticket-router configs ./configs
COPY --chown=ticket-router:ticket-router migrations ./migrations
COPY --chown=ticket-router:ticket-router scripts ./scripts
COPY --chown=ticket-router:ticket-router alembic.ini ./alembic.ini

USER ticket-router
CMD ["python", "-m", "ticket_router.orchestration", "--help"]
