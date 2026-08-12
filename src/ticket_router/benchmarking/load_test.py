"""Bounded headless Locust runner for the local inference API."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

from ticket_router.benchmarking.config import (
    DEFAULT_BENCHMARK_CONFIG_PATH,
    BenchmarkConfig,
)

LOCAL_LOAD_TEST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "api"})


def build_locust_command(
    *,
    config: BenchmarkConfig,
    host: str,
    users: int,
    spawn_rate: float,
    duration_seconds: int,
    locustfile: Path = Path("load_tests/locustfile.py"),
) -> list[str]:
    """Validate hard traffic caps before constructing a shell-free command."""
    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("load-test host must be an absolute HTTP(S) URL")
    if not config.load_test.allow_remote_host and parsed.hostname not in LOCAL_LOAD_TEST_HOSTS:
        raise ValueError("remote load-test hosts are disabled by configuration")
    if not 1 <= users <= config.load_test.maximum_users:
        raise ValueError(f"users must be between 1 and {config.load_test.maximum_users}")
    if not 0.0 < spawn_rate <= config.load_test.maximum_spawn_rate:
        raise ValueError(
            f"spawn_rate must be positive and at most {config.load_test.maximum_spawn_rate}"
        )
    if not 1 <= duration_seconds <= config.load_test.maximum_duration_seconds:
        raise ValueError(
            f"duration_seconds must be between 1 and {config.load_test.maximum_duration_seconds}"
        )
    return [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(locustfile),
        "--headless",
        "--host",
        host.rstrip("/"),
        "--users",
        str(users),
        "--spawn-rate",
        str(spawn_rate),
        "--run-time",
        f"{duration_seconds}s",
        "--stop-timeout",
        "5",
        "--csv",
        str(config.load_test.output_prefix),
        "--html",
        str(config.load_test.html_report_path),
        "--exit-code-on-error",
        "1",
        "--only-summary",
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=DEFAULT_BENCHMARK_CONFIG_PATH,
    )
    parser.add_argument("--host")
    parser.add_argument("--users", type=int)
    parser.add_argument("--spawn-rate", type=float)
    parser.add_argument("--duration-seconds", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = BenchmarkConfig.load(args.benchmark_config)
    host = args.host or config.api_url
    users = args.users or config.load_test.default_users
    spawn_rate = args.spawn_rate or config.load_test.default_spawn_rate
    duration = args.duration_seconds or config.load_test.default_duration_seconds
    try:
        command = build_locust_command(
            config=config,
            host=host,
            users=users,
            spawn_rate=spawn_rate,
            duration_seconds=duration,
        )
    except ValueError as exc:
        print(f"Load test configuration is invalid: {exc}", file=sys.stderr)
        return 2
    config.load_test.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    config.load_test.html_report_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TICKET_ROUTER_LOAD_TEST_SEED"] = str(config.random_seed)
    completed = subprocess.run(command, check=False, env=environment)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
