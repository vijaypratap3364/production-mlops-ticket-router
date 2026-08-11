# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv
FROM python:3.12-slim-bookworm AS builder

COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group mlflow-server --no-install-project

FROM python:3.12-slim-bookworm AS runtime

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN groupadd --gid 10001 ticket-router \
    && useradd --uid 10001 --gid ticket-router --create-home ticket-router \
    && mkdir -p /mlartifacts \
    && chown ticket-router:ticket-router /mlartifacts
WORKDIR /app

COPY --from=builder --chown=ticket-router:ticket-router /app/.venv /app/.venv

USER ticket-router
EXPOSE 5000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=8 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/health', timeout=3).read()"]
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
