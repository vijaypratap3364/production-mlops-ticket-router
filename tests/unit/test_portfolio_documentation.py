"""Final portfolio documentation and reviewed-evidence contract tests."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

REQUIRED_DOCUMENTS = (
    "docs/implementation-plan.md",
    "docs/data-source.md",
    "docs/data-card.md",
    "docs/model-card.md",
    "docs/api.md",
    "docs/monitoring.md",
    "docs/retraining.md",
    "docs/deployment.md",
    "docs/privacy.md",
    "docs/demo-script.md",
    "docs/interview-notes.md",
    "docs/resume-bullets.md",
    "docs/adr/README.md",
    "LICENSE",
    "SECURITY.md",
    "CONTRIBUTING.md",
)


def test_required_portfolio_documents_and_architecture_diagrams_exist() -> None:
    missing = [path for path in REQUIRED_DOCUMENTS if not Path(path).is_file()]
    assert missing == []
    readme = Path("README.md").read_text(encoding="utf-8")

    assert readme.count("```mermaid") == 5
    for heading in (
        "## Business problem",
        "## Architecture",
        "## Repository structure",
        "## Data source and license",
        "## Quick start with Docker Compose",
        "## Local non-Docker setup",
        "## Final evaluation and Model Registry",
        "## Monitoring and feedback",
        "## Controlled retraining",
        "## Testing and CI",
        "## Benchmarking",
        "## Limitations",
        "## Future improvements",
    ):
        assert heading in readme


def test_license_and_reviewed_resume_evidence_are_consistent() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    evidence = json.loads(Path("artifacts/resume/project_metrics.json").read_text(encoding="utf-8"))
    bullets = Path("docs/resume-bullets.md").read_text(encoding="utf-8")

    assert pyproject["project"]["license"] == "MIT"
    assert Path("LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    assert evidence["data"]["normalized_english_records"] == 28_190
    assert evidence["data"]["selected_queue_count"] == 10
    assert evidence["model"]["test_macro_f1"] == 0.6960570778151813
    assert evidence["operations"]["load_test_request_count"] == 131
    assert evidence["operations"]["load_test_failure_count"] == 0
    assert evidence["operations"]["reliability_scenarios_passed"] == 9
    assert sum(line.startswith("- ") for line in bullets.splitlines()) == 3
