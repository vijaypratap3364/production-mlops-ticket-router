"""Generate deterministic aggregate data with planted length, label, and confidence drift."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from ticket_router.data.manifests import atomic_write_json
from ticket_router.monitoring.config import (
    DEFAULT_MONITORING_CONFIG_PATH,
    MonitoringConfig,
)
from ticket_router.monitoring.drift import generate_drift_report
from ticket_router.monitoring.policy import evaluate_alert_policy
from ticket_router.monitoring.quality import calculate_delayed_quality


def simulated_monitoring_frames(
    *, seed: int = 42, rows: int = 500
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return safe aggregate frames; no synthetic ticket text is generated or retained."""
    if rows < 100:
        raise ValueError("simulation requires at least 100 rows")
    generator = np.random.default_rng(seed)
    reference_labels = np.asarray(["Billing", "Returns", "Technical", "General"])
    reference_predictions = generator.choice(
        reference_labels, size=rows, p=[0.35, 0.20, 0.35, 0.10]
    )
    current_predictions = generator.choice(reference_labels, size=rows, p=[0.85, 0.05, 0.08, 0.02])
    reference_subject = generator.integers(12, 70, size=rows)
    reference_body = generator.integers(80, 450, size=rows)
    current_subject = generator.integers(180, 420, size=rows)
    current_body = generator.integers(1500, 4000, size=rows)
    reference_confidence = np.clip(generator.normal(0.82, 0.06, size=rows), 0.0, 1.0)
    current_confidence = np.clip(generator.normal(0.38, 0.08, size=rows), 0.0, 1.0)
    reference = _simulation_frame(
        subject_length=reference_subject,
        body_length=reference_body,
        predicted_queue=reference_predictions,
        confidence=reference_confidence,
        generator=generator,
        drifted=False,
    )
    current = _simulation_frame(
        subject_length=current_subject,
        body_length=current_body,
        predicted_queue=current_predictions,
        confidence=current_confidence,
        generator=generator,
        drifted=True,
    )
    return reference, current


def run_simulation(*, config: MonitoringConfig, output_dir: Path) -> dict[str, object]:
    reference, current = simulated_monitoring_frames()
    drift = generate_drift_report(
        reference=reference,
        current=current,
        settings=config.drift,
        html_path=output_dir / "drift_report.html",
        json_path=output_dir / "drift_report.json",
    )
    quality = calculate_delayed_quality(
        (), minimum_sample_count=config.quality.minimum_feedback_count
    )
    decision = evaluate_alert_policy(
        event_count=current.height,
        minimum_event_count=config.current_window.minimum_event_count,
        drift=drift,
        quality=quality,
        reference_macro_f1=1.0,
        settings=config.alerts,
    )
    summary = {
        "simulation": "deterministic_planted_drift",
        "seed": 42,
        "reference_rows": reference.height,
        "current_rows": current.height,
        "status": decision.status,
        "drift": drift.to_dict(),
        "alert_decision": decision.to_dict(),
        "delayed_label_quality": quality.to_dict(),
        "planted_changes": [
            "longer_subjects_and_bodies",
            "billing_heavy_prediction_distribution",
            "lower_prediction_confidence",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "simulation_summary.json", summary)
    return summary


def _simulation_frame(
    *,
    subject_length: np.ndarray[tuple[int], np.dtype[np.signedinteger]],
    body_length: np.ndarray[tuple[int], np.dtype[np.signedinteger]],
    predicted_queue: np.ndarray[tuple[int], np.dtype[np.str_]],
    confidence: np.ndarray[tuple[int], np.dtype[np.floating]],
    generator: np.random.Generator,
    drifted: bool,
) -> pl.DataFrame:
    rows = len(subject_length)
    multiplier = 1.0 if not drifted else 4.0
    return pl.DataFrame(
        {
            "subject_length": subject_length,
            "body_length": body_length,
            "combined_length": subject_length + body_length,
            "word_count": np.maximum(1, ((subject_length + body_length) / 5).astype(int)),
            "uppercase_ratio": np.clip(
                generator.normal(0.05 * multiplier, 0.01, size=rows), 0.0, 1.0
            ),
            "digit_ratio": np.clip(generator.normal(0.02 * multiplier, 0.005, size=rows), 0.0, 1.0),
            "punctuation_ratio": np.clip(
                generator.normal(0.04 * multiplier, 0.01, size=rows), 0.0, 1.0
            ),
            "url_count": generator.binomial(3 if drifted else 1, 0.5 if drifted else 0.1, rows),
            "email_marker_count": generator.binomial(
                2 if drifted else 1, 0.4 if drifted else 0.08, rows
            ),
            "predicted_queue": predicted_queue,
            "prediction_confidence": confidence,
            "low_confidence": confidence < 0.50,
            "actual_queue": predicted_queue,
            "model_version": ["simulation-champion"] * rows,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitoring-config", type=Path, default=DEFAULT_MONITORING_CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/monitoring/simulation"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_simulation(
        config=MonitoringConfig.load(args.monitoring_config),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
