"""Run privacy-safe drift and delayed-label monitoring over PostgreSQL events."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ticket_router.config import Settings
from ticket_router.db.repositories import (
    SQLAlchemyMonitoringDataRepository,
    SQLAlchemyMonitoringRunRepository,
)
from ticket_router.db.session import DatabaseSessionManager
from ticket_router.logging_config import configure_logging, get_logger
from ticket_router.monitoring.config import (
    DEFAULT_MONITORING_CONFIG_PATH,
    MonitoringConfig,
)
from ticket_router.monitoring.service import execute_monitoring_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--monitoring-config", type=Path, default=DEFAULT_MONITORING_CONFIG_PATH)
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--lookback-hours", type=int)
    window.add_argument("--lookback-days", type=int)
    window.add_argument("--start", type=_timestamp)
    parser.add_argument("--end", type=_timestamp)
    parser.add_argument("--minimum-events", type=int)
    parser.add_argument("--model-version")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/monitoring"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    config = MonitoringConfig.load(args.monitoring_config)
    configure_logging(settings.log_level)
    if settings.database_url is None:
        get_logger(__name__).error("monitoring_failed", error="DATABASE_URL is required")
        return 1
    if args.minimum_events is not None and args.minimum_events <= 0:
        get_logger(__name__).error("monitoring_failed", error="minimum events must be positive")
        return 1
    end = args.end or datetime.now(UTC)
    if args.start is not None:
        start = args.start
    elif args.lookback_hours is not None:
        start = end - timedelta(hours=args.lookback_hours)
    else:
        days = args.lookback_days or config.current_window.default_lookback_days
        start = end - timedelta(days=days)
    manager = DatabaseSessionManager(settings.database_url.get_secret_value())
    try:
        manager.connect()
        execution = execute_monitoring_run(
            reference_path=config.reference.output_path,
            reference_manifest_path=config.reference.manifest_path,
            data_repository=SQLAlchemyMonitoringDataRepository(manager.session_factory),
            run_repository=SQLAlchemyMonitoringRunRepository(manager.session_factory),
            config=config,
            start=start,
            end=end,
            output_root=args.output_dir,
            model_version=args.model_version,
            minimum_event_count=args.minimum_events,
        )
    except (ConnectionError, FileNotFoundError, RuntimeError, ValueError) as exc:
        get_logger(__name__).error("monitoring_failed", error=str(exc))
        return 1
    finally:
        manager.close()
    print(
        json.dumps(
            {
                "run_id": execution.run_id,
                "status": execution.decision.status,
                "summary_path": execution.summary_path.as_posix(),
                "html_report_path": (
                    execution.html_report_path.as_posix() if execution.html_report_path else None
                ),
                "json_report_path": (
                    execution.json_report_path.as_posix() if execution.json_report_path else None
                ),
            },
            indent=2,
        )
    )
    return 0


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamps must include a UTC offset")
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    sys.exit(main())
