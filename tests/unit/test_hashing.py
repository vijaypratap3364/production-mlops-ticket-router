"""Tests for canonical ingestion hashes."""

from __future__ import annotations

from pathlib import Path

from ticket_router.hashing import sha256_bytes, sha256_file, sha256_json


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    source_file = tmp_path / "value.bin"
    source_file.write_bytes(b"abc")

    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_bytes(b"abc") == expected
    assert sha256_file(source_file) == expected


def test_json_hash_is_independent_of_mapping_order() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
