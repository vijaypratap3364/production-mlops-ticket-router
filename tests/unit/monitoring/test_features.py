"""Privacy-safe monitoring feature tests."""

from ticket_router.monitoring.features import derive_text_monitoring_features


def test_monitoring_features_count_structure_without_retaining_text() -> None:
    features = derive_text_monitoring_features(
        subject="URGENT invoice 42!",
        body="Email person@example.com or see https://example.test/a.",
        model_text="[SUBJECT] URGENT invoice 42! [BODY] Email <EMAIL> or see <URL>.",
    )
    values = features.to_dict()

    assert features.subject_length == len("URGENT invoice 42!")
    assert features.combined_length == features.subject_length + features.body_length
    assert features.url_count == 1
    assert features.email_marker_count == 1
    assert 0.0 < features.uppercase_ratio <= 1.0
    assert 0.0 < features.digit_ratio <= 1.0
    assert "subject" not in values
    assert "body" not in values
