"""Bounded Locust command and realistic workload contract tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ticket_router.benchmarking.config import BenchmarkConfig
from ticket_router.benchmarking.load_test import build_locust_command, main


def test_locust_command_uses_bounded_headless_configuration() -> None:
    config = BenchmarkConfig.load()
    command = build_locust_command(
        config=config,
        host="http://127.0.0.1:8000",
        users=3,
        spawn_rate=1.0,
        duration_seconds=30,
    )

    assert "--headless" in command
    assert command[command.index("--users") + 1] == "3"
    assert command[command.index("--run-time") + 1] == "30s"
    assert command[command.index("--exit-code-on-error") + 1] == "1"


@pytest.mark.parametrize(
    ("host", "users", "duration", "message"),
    [
        ("https://example.com", 1, 10, "remote load-test hosts"),
        ("http://127.0.0.1:8000", 26, 10, "users must be"),
        ("http://127.0.0.1:8000", 1, 301, "duration_seconds must be"),
    ],
)
def test_locust_command_rejects_remote_or_unbounded_traffic(
    host: str,
    users: int,
    duration: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_locust_command(
            config=BenchmarkConfig.load(),
            host=host,
            users=users,
            spawn_rate=1.0,
            duration_seconds=duration,
        )


def test_locust_workload_contains_short_long_batch_and_feedback_tasks() -> None:
    source = Path("load_tests/locustfile.py").read_text(encoding="utf-8")

    assert "SHORT_TICKETS" in source
    assert "LONG_TICKETS" in source
    assert "@task(12)" in source
    assert "def predict_single" in source
    assert "def predict_batch" in source
    assert "def submit_feedback" in source
    assert "wait_time = between(0.25, 1.0)" in source
    assert 'os.environ.get("TICKET_ROUTER_LOAD_TEST_SEED", "42")' in source


def test_load_test_main_passes_reproducible_seed_to_locust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = BenchmarkConfig.load()
    config = original.model_copy(
        update={
            "load_test": original.load_test.model_copy(
                update={
                    "output_prefix": tmp_path / "locust",
                    "html_report_path": tmp_path / "locust.html",
                }
            )
        }
    )
    captured_environment: dict[str, str] = {}

    def fake_run(
        command: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert command[0]
        assert check is False
        captured_environment.update(env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "ticket_router.benchmarking.load_test.BenchmarkConfig.load",
        lambda _path: config,
    )
    monkeypatch.setattr("ticket_router.benchmarking.load_test.subprocess.run", fake_run)

    exit_code = main([])

    assert exit_code == 0
    assert captured_environment["TICKET_ROUTER_LOAD_TEST_SEED"] == "42"
