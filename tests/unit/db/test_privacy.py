"""One-way text fingerprint tests."""

from ticket_router.db.privacy import text_fingerprint


def test_text_fingerprint_prefers_keyed_hmac_and_is_deterministic() -> None:
    first, algorithm = text_fingerprint("synthetic ticket", hmac_secret="test-secret")
    second, _ = text_fingerprint("synthetic ticket", hmac_secret="test-secret")
    different, _ = text_fingerprint("synthetic ticket", hmac_secret="other-secret")

    assert algorithm == "hmac-sha256"
    assert len(first) == 64
    assert first == second
    assert first != different
    assert "synthetic ticket" not in first


def test_text_fingerprint_has_documented_local_fallback() -> None:
    digest, algorithm = text_fingerprint("synthetic ticket", hmac_secret=None)

    assert algorithm == "sha256"
    assert len(digest) == 64
