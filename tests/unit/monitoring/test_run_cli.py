"""Monitoring CLI validation, time-window, and resource-lifecycle tests."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from ticket_router.config import Settings
from ticket_router.monitoring import run

BASE_SETTINGS = Settings.load(env_file=None)


def _settings(database_url: str | None) -> Settings:
    return BASE_SETTINGS.model_copy(
        update={"database_url": SecretStr(database_url) if database_url else None}
    )


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    monkeypatch.setattr(
        run,
        "Settings",
        SimpleNamespace(load=lambda _: settings),
    )


def test_monitoring_cli_requires_database_and_positive_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, _settings(None))
    assert run.main([]) == 1

    _patch_settings(monkeypatch, _settings("sqlite:///fixture.db"))
    assert run.main(["--minimum-events", "0"]) == 1


def test_monitoring_cli_runs_explicit_window_and_closes_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    class FixtureManager:
        session_factory = object()

        def __init__(self, database_url: str) -> None:
            observed["database_url"] = database_url

        def connect(self) -> None:
            observed["connected"] = True

        def close(self) -> None:
            observed["closed"] = True

    def execute(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(
            run_id="monitoring-fixture",
            decision=SimpleNamespace(status="healthy"),
            summary_path=tmp_path / "summary.json",
            html_report_path=tmp_path / "report.html",
            json_report_path=tmp_path / "report.json",
        )

    _patch_settings(monkeypatch, _settings("sqlite:///fixture-monitoring.db"))
    monkeypatch.setattr(run, "DatabaseSessionManager", FixtureManager)
    monkeypatch.setattr(run, "SQLAlchemyMonitoringDataRepository", lambda _: object())
    monkeypatch.setattr(run, "SQLAlchemyMonitoringRunRepository", lambda _: object())
    monkeypatch.setattr(run, "execute_monitoring_run", execute)

    exit_code = run.main(
        [
            "--start",
            "2026-08-01T00:00:00Z",
            "--end",
            "2026-08-02T00:00:00+00:00",
            "--minimum-events",
            "5",
            "--model-version",
            "7",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert observed["connected"] is True
    assert observed["closed"] is True
    assert observed["start"] == datetime(2026, 8, 1, tzinfo=UTC)
    assert observed["end"] == datetime(2026, 8, 2, tzinfo=UTC)
    assert observed["minimum_event_count"] == 5
    assert observed["model_version"] == "7"
    assert '"status": "healthy"' in capsys.readouterr().out


def test_monitoring_cli_closes_database_after_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, bool] = {}

    class FixtureManager:
        session_factory = object()

        def __init__(self, database_url: str) -> None:
            del database_url

        def connect(self) -> None:
            observed["connected"] = True

        def close(self) -> None:
            observed["closed"] = True

    _patch_settings(monkeypatch, _settings("sqlite:///fixture.db"))
    monkeypatch.setattr(run, "DatabaseSessionManager", FixtureManager)
    monkeypatch.setattr(run, "SQLAlchemyMonitoringDataRepository", lambda _: object())
    monkeypatch.setattr(run, "SQLAlchemyMonitoringRunRepository", lambda _: object())
    monkeypatch.setattr(
        run,
        "execute_monitoring_run",
        lambda **_: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )

    assert run.main(["--lookback-hours", "24"]) == 1
    assert observed == {"connected": True, "closed": True}


def test_timestamp_requires_timezone_and_normalizes_to_utc() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="UTC offset"):
        run._timestamp("2026-08-01T12:00:00")

    assert run._timestamp("2026-08-01T07:00:00-05:00") == datetime(
        2026,
        8,
        1,
        12,
        tzinfo=UTC,
    )
