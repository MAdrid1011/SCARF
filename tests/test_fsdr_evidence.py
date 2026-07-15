import pytest


def _provenance(index: int):
    return {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "source_identity": "git",
        "submodules": {
            "transplat": "b" * 40,
            "mvsplat": "c" * 40,
            "depthsplat": "d" * 40,
        },
        "model": "transplat",
        "dataset": {
            "name": "re10k",
            "representation": "re10k-native",
            "tree_sha256": "e" * 64,
            "paper_result_eligible": True,
        },
        "checkpoint": {
            "path": "transplat/checkpoints/re10k.ckpt",
            "sha256": "f" * 64,
            "load": {
                "matched_tensors": 1,
                "matched_checkpoint_numel_fraction": 1.0,
            },
        },
        "environment": {"profile": "classic", "digest_sha256": "1" * 64},
        "runtime_assets": {"asset": {"sha256": "2" * 64}},
        "command": ["python", "scripts/demo.py", "--fsdr-only"],
        "seed": 0,
        "device": {"type": "cuda", "name": "test"},
        "evaluation": {
            "kind": "sample",
            "sample_index": index,
            "execution_index": index,
            "candidate_count": 2,
            "scene": f"scene-{index}",
            "context_indices": [0, 1],
            "target_indices": [2, 3],
        },
    }


def _sample(index: int, total: int, guided: int, in_window: int):
    from scripts.fsdr_evidence import build_fsdr_sample_record

    return build_fsdr_sample_record(
        provenance=_provenance(index),
        total_pixels=total,
        cache_hits=guided,
        cache_misses=total - guided,
        guided_pixels=guided,
        guided_in_window=in_window,
        guided_out_window=guided - in_window,
        guided_top1_covered=guided - 1,
        guided_top1_missed=1,
        depth_inconsistent=0,
        hit_no_guide=0,
        full_depth_candidates=128,
        narrowed_depth_candidates=32,
        candidate_domain="inverse_depth",
        probability_source="pinned_original_depth_head_softmax",
        evidence_height=8,
        evidence_width=8,
    )


def test_fsdr_sample_rejects_zero_or_inconsistent_counts():
    from scripts.fsdr_evidence import build_fsdr_sample_record

    with pytest.raises(ValueError, match="total_pixels"):
        build_fsdr_sample_record(
            provenance=_provenance(0),
            total_pixels=0,
            cache_hits=0,
            cache_misses=0,
            guided_pixels=0,
            guided_in_window=0,
            guided_out_window=0,
            guided_top1_covered=0,
            guided_top1_missed=0,
            depth_inconsistent=0,
            hit_no_guide=0,
            full_depth_candidates=128,
            narrowed_depth_candidates=32,
            candidate_domain="inverse_depth",
            probability_source="pinned_original_depth_head_softmax",
            evidence_height=8,
            evidence_width=8,
        )


def test_fsdr_aggregate_uses_pixel_weighted_rates_and_selection_hash():
    from scripts.fsdr_evidence import aggregate_fsdr_records, validate_fsdr_record

    record = aggregate_fsdr_records(
        [_sample(0, 100, 50, 49), _sample(1, 300, 240, 240)],
        expected_count=2,
    )
    validate_fsdr_record(record)

    assert record["fsdr"]["counts"]["total_pixels"] == 400
    assert record["fsdr"]["counts"]["guided_pixels"] == 290
    assert record["fsdr"]["metrics"]["guided_rate"] == pytest.approx(290 / 400)
    assert record["fsdr"]["metrics"]["top1_coverage"] == pytest.approx(288 / 290)
    assert len(record["provenance"]["evaluation"]["sample_selection_sha256"]) == 64


def test_fsdr_comparison_uses_declared_tolerance_without_overwriting_actuals():
    from scripts.fsdr_evidence import compare_fsdr_aggregate

    record = aggregate = _sample(0, 1000, 720, 719)
    aggregate["kind"] = "fsdr_dataset_aggregate"
    comparison = compare_fsdr_aggregate(
        aggregate,
        {
            "mechanisms": {
                "transplat/re10k": {
                    "guided_rate": 0.721,
                    "top1_coverage": 0.9991,
                }
            },
            "mechanism_tolerance_absolute": 0.02,
        },
    )

    assert comparison["status"] == "PASS"
    assert comparison["checks"]["guided_rate"]["actual"] == pytest.approx(0.72)


def test_fsdr_record_rejects_continuous_window_counts_as_top1_evidence():
    from scripts.fsdr_evidence import validate_fsdr_record

    record = _sample(0, 1000, 720, 720)
    record["fsdr"]["metrics"]["top1_coverage"] = 1.0

    with pytest.raises(ValueError, match="top1_coverage"):
        validate_fsdr_record(record)
