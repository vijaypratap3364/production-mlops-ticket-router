"""Privacy-safe one-way text fingerprints for monitoring and duplicate analysis."""

from __future__ import annotations

import hashlib
import hmac


def text_fingerprint(value: str, *, hmac_secret: str | None) -> tuple[str, str]:
    """Prefer keyed HMAC; fall back to SHA-256 only for secret-free local development."""
    encoded = value.encode("utf-8")
    if hmac_secret:
        digest = hmac.new(hmac_secret.encode("utf-8"), encoded, hashlib.sha256).hexdigest()
        return digest, "hmac-sha256"
    return hashlib.sha256(encoded).hexdigest(), "sha256"
