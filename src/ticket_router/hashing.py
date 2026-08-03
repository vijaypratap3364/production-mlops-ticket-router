"""Canonical SHA-256 helpers for data and configuration lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = DEFAULT_HASH_CHUNK_SIZE) -> str:
    """Hash a file without loading it fully into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    """Return the SHA-256 digest of canonical JSON."""
    return sha256_bytes(canonical_json_bytes(value))
