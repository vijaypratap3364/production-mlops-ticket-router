"""Regression tests for the public-repository CI safety contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).parents[2]


def _load_workflow(name: str) -> dict[str, Any]:
    workflow_path = REPOSITORY_ROOT / ".github" / "workflows" / name
    loaded = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def _run_commands(workflow: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                commands.append(str(step["run"]))
    return commands


def test_ci_workflow_has_required_quality_and_build_boundaries() -> None:
    workflow = _load_workflow("ci.yml")

    assert set(workflow["on"]) == {"push", "pull_request"}
    assert set(workflow["jobs"]) == {
        "lint",
        "typecheck",
        "unit-tests",
        "integration-tests",
        "package-build",
        "docker-build",
    }
    commands = _run_commands(workflow)
    assert "uv run ruff format --check ." in commands
    assert "uv run ruff check ." in commands
    assert "uv run mypy src tests scripts load_tests" in commands
    assert "uv run pytest tests/unit" in commands
    assert "uv run pytest -m integration tests/integration --no-cov" in commands
    assert "uv build" in commands
    assert not (
        workflow.get("env", {}).get("UV_FROZEN") == "true"
        and any("uv sync --locked" in command for command in commands)
    )
    assert str(workflow["env"]["UV_PROJECT_ENVIRONMENT"]).startswith("${{ runner.temp }}")


def test_ci_uses_disposable_postgres_and_does_not_run_full_training() -> None:
    workflow = _load_workflow("ci.yml")
    integration = workflow["jobs"]["integration-tests"]
    postgres = integration["services"]["postgres"]

    assert postgres["image"] == "postgres:16-alpine"
    assert postgres["env"]["POSTGRES_DB"] == "test_ticket_router"
    assert "TEST_DATABASE_URL" in integration["env"]
    workflow_text = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    forbidden_commands = (
        "ticket_router.data.download",
        "ticket_router.modeling.train_candidates",
        "ticket_router.registry.evaluate_final",
        "ticket_router.registry.promote",
    )
    assert all(command not in workflow_text for command in forbidden_commands)
    assert "push: false" in workflow_text


def test_workflow_actions_are_immutable_and_release_does_not_deploy() -> None:
    for workflow_name in ("ci.yml", "release.yml"):
        workflow_text = (REPOSITORY_ROOT / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        action_lines = [line.strip() for line in workflow_text.splitlines() if "uses:" in line]
        assert action_lines
        for line in action_lines:
            reference = line.split("@", maxsplit=1)[1].split(maxsplit=1)[0]
            assert len(reference) == 40
            assert all(character in "0123456789abcdef" for character in reference)

    release_commands = "\n".join(_run_commands(_load_workflow("release.yml")))
    assert "never publishes or deploys" in release_commands
    assert "docker push" not in release_commands
