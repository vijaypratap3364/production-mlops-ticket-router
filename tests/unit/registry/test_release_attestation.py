"""Tests for release-to-champion lineage attestation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from ticket_router.config import Settings
from ticket_router.data.manifests import GitVersion
from ticket_router.data.split_manifest import OutputFileManifest, SplitManifest
from ticket_router.hashing import sha256_file
from ticket_router.registry import attest_release as release_module
from ticket_router.registry.attest_release import ReleaseAttestationError, attest_release
from ticket_router.registry.config import FinalModelConfig
from ticket_router.registry.service import ModelRegistryService, RegisteredVersion


class FakeModel:
    def predict(self, values: Sequence[str]) -> object:
        return np.asarray(["Queue A"] * len(values), dtype=object)


class FakeRegistry:
    def __init__(self) -> None:
        self.tags = {
            "combined_training_data_sha256": "b" * 64,
            "configuration_sha256": "c" * 64,
            "split_manifest_sha256": "",
            "test_data_sha256": "",
            "promotion_gates_passed": "true",
            "git_commit": "unavailable",
        }
        self.updated: dict[str, str] = {}

    def resolve_alias(self, *, name: str, alias: str) -> RegisteredVersion:
        return RegisteredVersion(name=name, version="1", run_id="d" * 32, source="runs:/model")

    def model_version_tags(self, *, name: str, version: str) -> dict[str, str]:
        return dict(self.tags)

    def load_version(self, *, name: str, version: str) -> FakeModel:
        return FakeModel()

    def set_model_version_tags(
        self,
        *,
        name: str,
        version: str,
        tags: Mapping[str, str],
    ) -> None:
        self.updated.update(tags)


def _manifest(tmp_path: Path) -> tuple[Path, SplitManifest]:
    outputs: dict[str, OutputFileManifest] = {}
    for name in ("train", "validation", "test", "training_reference"):
        path = tmp_path / "data" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        outputs[name] = OutputFileManifest(
            path=path.relative_to(tmp_path).as_posix(),
            sha256=sha256_file(path),
            row_count=1,
        )
    settings = Settings.load(env_file=None)
    manifest = SplitManifest(
        preparation_timestamp_utc="2026-08-13T00:00:00Z",
        data_source_hashes={"normalized_data": "a" * 64},
        configuration_hash="c" * 64,
        preprocessing=settings.project_config.preprocessing,
        splitting=settings.project_config.splitting,
        split_ratios=settings.project_config.split_ratios,
        random_seed=42,
        duplicate_group_column="normalized_text_hash",
        selected_input_rows=3,
        contradictory_group_count=0,
        contradictory_rows_excluded=0,
        final_row_count=3,
        split_counts={"train": 1, "validation": 1, "test": 1},
        split_percentages={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
        per_class_counts={
            "train": {"Queue A": 1},
            "validation": {"Queue A": 1},
            "test": {"Queue A": 1},
        },
        maximum_class_proportion_deviation={"train": 0.0, "validation": 0.0, "test": 0.0},
        most_deviant_class={"train": "Queue A", "validation": "Queue A", "test": "Queue A"},
        label_mapping={"Queue A": 0},
        model_feature_columns=("model_text",),
        target_column="queue",
        output_files=outputs,
        preparation_code_version=None,
    )
    path = tmp_path / "data" / "split_manifest.json"
    manifest.write(path)
    return path, manifest


def test_release_attestation_records_clean_commit_without_moving_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    registry = FakeRegistry()
    registry.tags["split_manifest_sha256"] = sha256_file(manifest_path)
    registry.tags["test_data_sha256"] = manifest.output_files["test"].sha256
    monkeypatch.setattr(
        release_module,
        "get_git_version",
        lambda _path: GitVersion(commit="e" * 40, dirty=False),
    )
    monkeypatch.setattr(release_module, "prediction_contract_passes", lambda *_, **__: True)
    monkeypatch.setattr(release_module, "signature_matches_text_api", lambda *_, **__: True)
    output = tmp_path / "release.json"

    result = attest_release(
        release="v1.0.0",
        settings=Settings.load(env_file=None),
        final_config=FinalModelConfig.load(),
        split_manifest_path=manifest_path,
        output_path=output,
        project_root=tmp_path,
        registry=cast(ModelRegistryService, registry),
        model_info_loader=lambda _uri: SimpleNamespace(signature=object()),
        clock=lambda: datetime(2026, 8, 13, tzinfo=UTC),
    )

    assert result["release_git_commit"] == "e" * 40
    assert result["alias_changed"] is False
    assert result["test_reevaluated"] is False
    assert registry.updated["release_git_commit"] == "e" * 40
    assert output.is_file()


def test_release_attestation_rejects_dirty_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path, _ = _manifest(tmp_path)
    monkeypatch.setattr(
        release_module,
        "get_git_version",
        lambda _path: GitVersion(commit="e" * 40, dirty=True),
    )

    with pytest.raises(ReleaseAttestationError, match="clean Git worktree"):
        attest_release(
            release="v1.0.0",
            settings=Settings.load(env_file=None),
            final_config=FinalModelConfig.load(),
            split_manifest_path=manifest_path,
            output_path=tmp_path / "release.json",
            project_root=tmp_path,
        )
