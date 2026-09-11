from __future__ import annotations

import json
from pathlib import Path


def test_claim_readiness_reports_release_state_blockers(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "artifact/protocol/reviewer").mkdir(parents=True)
    (root / "artifact/evaluation_protocol.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "artifact/evaluation_protocol.json").write_text(
        json.dumps(
            {
                "pairs": {
                    f"transplat/{dataset}": {"sample_count": count}
                    for dataset, count in {"re10k": 2, "acid": 2, "dl3dv": 1}.items()
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "artifact/mechanism_config.json").write_text(
        json.dumps({"status": "preregistered", "selected": None}), encoding="utf-8"
    )
    output = root / "outputs/ae"

    from scripts.claim_readiness import audit

    report = audit(root, output_root=output, profile="full")

    assert report["claim_ready"] is False
    assert any(item.startswith("CALIBRATION_NOT_FROZEN") for item in report["blockers"])
    assert any(item.startswith("TIMING_MANIFEST_MISSING") for item in report["blockers"])
    assert any(item.startswith("RESULT_MISSING") for item in report["blockers"])
    assert report["output_root"] == str(output.resolve())


def test_claim_readiness_uses_reviewer_profile_counts(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    reviewer = root / "artifact/protocol/reviewer"
    reviewer.mkdir(parents=True)
    (reviewer / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "re10k": {"sample_count": 3},
                    "acid": {"sample_count": 4},
                    "dl3dv": {"sample_count": 5},
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "artifact/mechanism_config.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "artifact/mechanism_config.json").write_text(
        json.dumps({"status": "preregistered", "selected": None}), encoding="utf-8"
    )

    from scripts.claim_readiness import audit

    report = audit(root, profile="reviewer")

    assert report["expected_sample_counts"] == {"re10k": 3, "acid": 4, "dl3dv": 5}
