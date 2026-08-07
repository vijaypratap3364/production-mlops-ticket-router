"""Tests for the irreversible final-test access boundary."""

from pathlib import Path

import pytest

from ticket_router.config import Settings
from ticket_router.registry.config import FinalModelConfig
from ticket_router.registry.evaluate_final import FinalEvaluationError, run_final_evaluation
from ticket_router.registry.promote import PromotionError, promote_candidate
from ticket_router.registry.recover_final import FinalRecoveryError, recover_final_registration


def test_existing_test_access_audit_prevents_repeated_evaluation(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "test_access_audit.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FinalEvaluationError, match="already been recorded"):
        run_final_evaluation(
            settings=Settings.load(env_file=None),
            final_config=FinalModelConfig.load(),
            processed_dir=tmp_path / "processed",
            split_manifest_path=tmp_path / "split_manifest.json",
            model_artifacts_dir=tmp_path / "models",
            reports_dir=reports,
            project_root=tmp_path,
        )


def test_promotion_requires_explicit_human_approval(tmp_path: Path) -> None:
    with pytest.raises(PromotionError, match="--approve"):
        promote_candidate(
            settings=Settings.load(env_file=None),
            final_config=FinalModelConfig.load(),
            split_manifest_path=tmp_path / "split_manifest.json",
            audit_directory=tmp_path / "audits",
            project_root=tmp_path,
            approved=False,
        )


def test_recovery_refuses_a_completed_access_audit(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "test_access_audit.json").write_text('{"status":"completed"}\n', encoding="utf-8")

    with pytest.raises(FinalRecoveryError, match="incomplete authorized"):
        recover_final_registration(
            settings=Settings.load(env_file=None),
            final_config=FinalModelConfig.load(),
            reports_dir=reports,
            model_artifacts_dir=tmp_path / "models",
            project_root=tmp_path,
        )
