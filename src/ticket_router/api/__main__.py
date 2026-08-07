"""Local API development entry point."""

from __future__ import annotations

import uvicorn

from ticket_router.config import Settings


def main() -> None:
    settings = Settings.load()
    uvicorn.run(
        "ticket_router.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
