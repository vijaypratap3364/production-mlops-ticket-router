"""Default ASGI application import target."""

from ticket_router.api.app import create_app

app = create_app()
