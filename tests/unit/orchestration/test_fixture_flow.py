from pathlib import Path
from typing import cast

from ticket_router.orchestration.fixture_flow import inspect_fixture_task


def test_lightweight_fixture_flow() -> None:
    result = inspect_fixture_task.fn(Path("tests/fixtures/tickets.csv"))

    assert cast(int, result["row_count"]) > 0
    assert cast(int, result["queue_count"]) > 0
    assert len(str(result["sha256"])) == 64
