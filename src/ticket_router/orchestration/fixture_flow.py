"""Network-free lightweight Prefect flow used to verify the local engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from prefect import flow, task

from ticket_router.hashing import sha256_file


@task(name="inspect fixture tickets", tags=["fixture", "validation"], persist_result=False)
def inspect_fixture_task(path: Path) -> dict[str, object]:
    frame = pl.read_csv(path)
    frame = frame.rename({name: name.strip().casefold() for name in frame.columns})
    required = {"language", "subject", "body", "queue"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("fixture is missing columns: " + ", ".join(missing))
    return {
        "row_count": frame.height,
        "queue_count": frame["queue"].n_unique(),
        "sha256": sha256_file(path),
    }


@flow(name="fixture-smoke-flow", log_prints=False)
def fixture_smoke_flow(path: str = "tests/fixtures/tickets.csv") -> dict[str, object]:
    return inspect_fixture_task(Path(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="tests/fixtures/tickets.csv")
    result = fixture_smoke_flow(parser.parse_args().fixture)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
