from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _gaussians(count: int = 16):
    generator = torch.Generator().manual_seed(401)
    factors = torch.randn(count, 3, 3, generator=generator)
    return SimpleNamespace(
        means=torch.randn(1, count, 3, generator=generator),
        covariances=(
            factors @ factors.mT + torch.eye(3).reshape(1, 3, 3) * 0.05
        ).unsqueeze(0),
        harmonics=torch.randn(1, count, 3, 25, generator=generator),
        opacities=torch.full((1, count, 1), 0.2),
    )


def _kwargs():
    return {
        "tile_size": 4,
        "gpp": 1,
        "feature_var_threshold": 0.2,
        "depth_std_threshold": 0.1,
        "features": torch.zeros(1, 1, 2, 4, 4),
        "depths": torch.ones(1, 1, 4, 4),
        "cross_check_threshold": 0.02,
        "view_count": 1,
        "materialization": "representative",
        "decision_semantics": "current",
        "beta_x": 0.5,
        "beta_f": 0.1,
        "beta_d": 1.0,
        "num_depth_candidates": 1,
        "context_extrinsics": None,
        "context_intrinsics": None,
        "ray_depth_mode": "euclidean",
        "depth_routing_semantics": "metric-depth-standard-deviation",
        "depth_near": None,
        "depth_far": None,
        "materialization_guard": False,
        "context_safety_guard": False,
    }


def test_dense_adaptor_teacher_packets_are_selected_only_and_target_free():
    from saes.joint_materialization_teacher import (
        PACKET_KIND,
        build_dense_adaptor_teacher_packets,
    )

    record = build_dense_adaptor_teacher_packets(
        _gaussians(), height=4, width=4, saes_kwargs=_kwargs()
    )

    assert record["kind"] == PACKET_KIND
    assert record["route_summary"] == {
        "level0_tiles": 1,
        "level1_tiles": 0,
        "full_tiles": 0,
        "level0_pixels": 12,
        "level1_pixels": 0,
        "l0_representatives": 4,
        "l1_lightweight_anchors": 0,
        "full_stage3_gaussians": 0,
    }
    assert record["controls"]["two_finite_skipped_s3_sentinels_passed"] is True
    assert record["controls"]["route_mask_unchanged"] is True
    assert record["controls"]["retained_counts_unchanged"] is True
    assert record["controls"]["operational_stats_unchanged"] is True
    assert record["controls"]["tile_trace_unchanged"] is True
    assert record["controls"]["ordinary_representative_output_unchanged"] is True
    assert record["controls"]["full_passthrough"] is True
    assert record["controls"]["selected_only_s3"] is True
    assert record["controls"]["sparse_packet_count"] == 4
    assert record["retained_output_mask"].dtype == torch.bool
    assert int(record["retained_output_mask"].sum()) == 4
    assert [packet["anchor_index"] for packet in record["packets"]] == [0, 3, 12, 15]
    for packet in record["packets"]:
        assert set(packet) == {"anchor_index", "level", "sh_degree", "descriptor", "base", "teacher"}
        assert packet["level"] == "L0"
        assert packet["descriptor"].shape == (1, 32)
        assert packet["sh_degree"] == 4
        assert set(packet["base"]) == set(packet["teacher"]) == {
            "means",
            "covariances",
            "harmonics",
            "opacities",
        }
        assert bool(torch.isfinite(packet["descriptor"]).all())
        for name in ("means", "covariances", "harmonics", "opacities"):
            assert not torch.equal(packet["base"][name], packet["teacher"][name])


def test_dense_adaptor_teacher_rejects_nonrepresentative_or_nonfrozen_budget():
    from saes.joint_materialization_teacher import build_dense_adaptor_teacher_packets

    nonrepresentative = _kwargs()
    nonrepresentative["materialization"] = "dense-diagnostic"
    with pytest.raises(ValueError, match="representative"):
        build_dense_adaptor_teacher_packets(
            _gaussians(), height=4, width=4, saes_kwargs=nonrepresentative
        )

    altered_budget = _kwargs()
    altered_budget["gpp"] = 2
    with pytest.raises(ValueError, match="K/2K"):
        build_dense_adaptor_teacher_packets(
            _gaussians(), height=4, width=4, saes_kwargs=altered_budget
        )


def test_dense_adaptor_teacher_accepts_the_registered_context_guard_argument():
    import saes.joint_materialization_teacher as teacher

    kwargs = _kwargs()
    kwargs["context_safety_guard"] = True
    kwargs["materialization_guard"] = True

    validated = teacher._validate_saes_kwargs(kwargs, height=4, width=4)

    assert validated["context_safety_guard"] is True


def test_dense_adaptor_teacher_checks_full_passthrough_in_ordinary_replay():
    from saes.joint_materialization_teacher import build_dense_adaptor_teacher_packets

    kwargs = _kwargs()
    kwargs["features"] = torch.zeros(1, 1, 2, 4, 8)
    kwargs["depths"] = torch.ones(1, 1, 4, 8)
    checkerboard = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    )
    kwargs["features"][0, 0, 0, :, 4:] = checkerboard
    kwargs["depths"][0, 0, :, 4:] = checkerboard + 1.0

    record = build_dense_adaptor_teacher_packets(
        _gaussians(32), height=4, width=8, saes_kwargs=kwargs
    )

    assert record["route_summary"]["level0_tiles"] == 1
    assert record["route_summary"]["full_tiles"] == 1
    assert record["controls"]["full_slot_count"] == 16
    assert record["controls"]["full_passthrough"] is True
    assert record["controls"]["ordinary_representative_output_unchanged"] is True


def test_dense_adaptor_teacher_rejects_an_ordinary_replay_mask_change(monkeypatch):
    import saes.joint_materialization_teacher as teacher

    original = teacher._replay_ordinary_representatives

    def changed_mask(*args, **kwargs):
        output, mask, stats, trace = original(*args, **kwargs)
        return output, ~mask, stats, trace

    monkeypatch.setattr(teacher, "_replay_ordinary_representatives", changed_mask)
    with pytest.raises(RuntimeError, match="route_mask_unchanged"):
        teacher.build_dense_adaptor_teacher_packets(
            _gaussians(), height=4, width=4, saes_kwargs=_kwargs()
        )


def test_dense_adaptor_teacher_rejects_an_ordinary_full_slot_change(monkeypatch):
    import saes.joint_materialization_teacher as teacher

    original = teacher._replay_ordinary_representatives

    def changed_full_slot(*args, **kwargs):
        output, mask, stats, trace = original(*args, **kwargs)
        output.means[0, 4] += 1.0
        return output, mask, stats, trace

    monkeypatch.setattr(teacher, "_replay_ordinary_representatives", changed_full_slot)
    kwargs = _kwargs()
    kwargs["features"] = torch.zeros(1, 1, 2, 4, 8)
    kwargs["depths"] = torch.ones(1, 1, 4, 8)
    checkerboard = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    )
    kwargs["features"][0, 0, 0, :, 4:] = checkerboard
    kwargs["depths"][0, 0, :, 4:] = checkerboard + 1.0

    with pytest.raises(RuntimeError, match="full_passthrough"):
        teacher.build_dense_adaptor_teacher_packets(
            _gaussians(32), height=4, width=8, saes_kwargs=kwargs
        )
