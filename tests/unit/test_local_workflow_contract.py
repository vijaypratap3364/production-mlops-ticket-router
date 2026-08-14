"""Regression tests for the lightweight native-development boundary."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]


def _make_recipes() -> dict[str, list[str]]:
    recipes: dict[str, list[str]] = {}
    current_targets: list[str] = []
    for line in (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and ":" in line and "=" not in line:
            current_targets = line.split(":", maxsplit=1)[0].split()
            for target in current_targets:
                recipes.setdefault(target, [])
        elif line.startswith("\t"):
            for target in current_targets:
                recipes[target].append(line.strip())
    return recipes


def test_only_explicit_docker_make_targets_invoke_compose() -> None:
    recipes = _make_recipes()
    docker_targets = {
        target
        for target, commands in recipes.items()
        if any("$(DOCKER_COMPOSE)" in command for command in commands)
    }

    assert docker_targets
    assert all(target.startswith("docker-") for target in docker_targets)


def test_default_and_normal_development_targets_are_native_only() -> None:
    makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
    recipes = _make_recipes()

    assert ".DEFAULT_GOAL := help" in makefile
    for target in ("help", "install", "check", "test", "migrate", "api-dev", "dashboard-dev"):
        assert target in recipes
        assert all("DOCKER_COMPOSE" not in command for command in recipes[target])


def test_agent_instructions_require_explicit_local_docker_authorization() -> None:
    instructions = (REPOSITORY_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "Do not start Docker Desktop" in instructions
    assert "explicitly requests Docker verification" in instructions
