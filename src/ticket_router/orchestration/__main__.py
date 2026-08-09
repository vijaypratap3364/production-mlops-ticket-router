"""Run the Prefect flow CLI."""

from ticket_router.orchestration.flows import main

raise SystemExit(main())
