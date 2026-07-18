from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _gaussians(count: int = 8):
    values = torch.arange(count, dtype=torch.float32)
    return SimpleNamespace(
        means=values.reshape(1, count, 1).repeat(1, 1, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, count, 1, 1),
        harmonics=values.reshape(1, count, 1, 1).repeat(1, 1, 3, 2),
        opacities=(values.add(1.0) / 20.0).reshape(1, count, 1),
    )


def _optimized(gaussians, representative_local):
    return {
        name: getattr(gaussians, name)[0, representative_local].clone().add(10.0)
        for name in ("means", "covariances", "harmonics", "opacities")
    }


def test_fixed_parameter_variant_registry_has_exactly_requested_twelve_variants():
    from scripts.saes_same_budget_render_teacher_parameter_attribution import (
        ATTRIBUTION_ID,
        _expected_variant_names,
        _variant_specs,
    )

    assert ATTRIBUTION_ID == "same-budget-render-teacher-parameter-attribution-v2"
    assert _expected_variant_names() == (
        "initial_compact",
        "teacher_means",
        "teacher_covariances",
        "teacher_opacities",
        "teacher_harmonics",
        "teacher_means_covariances",
        "teacher_opacities_harmonics",
        "all_teacher",
        "all_teacher_minus_means",
        "all_teacher_minus_covariances",
        "all_teacher_minus_opacities",
        "all_teacher_minus_harmonics",
    )
    assert len(_variant_specs()) == 12


def test_parameter_variants_change_only_registered_representative_families():
    from scripts.saes_same_budget_render_teacher_parameter_attribution import (
        _build_parameter_variants,
    )

    initial = _gaussians()
    representatives = torch.tensor((1, 6), dtype=torch.long)
    optimized = _optimized(initial, representatives)
    variants = _build_parameter_variants(initial, optimized, representatives)
    frozen = torch.tensor((0, 2, 3, 4, 5, 7), dtype=torch.long)

    for name, entry in variants.items():
        candidate = entry["gaussians"]
        active = set(entry["teacher_families"])
        for family in ("means", "covariances", "harmonics", "opacities"):
            torch.testing.assert_close(
                getattr(candidate, family)[0, frozen], getattr(initial, family)[0, frozen]
            )
            expected = optimized[family] if family in active else getattr(initial, family)[0, representatives]
            torch.testing.assert_close(getattr(candidate, family)[0, representatives], expected)
        if name.startswith("all_teacher_minus_"):
            assert entry["reverted_families"]


def test_l0_l1_masks_and_grouped_changes_cover_each_representative_once():
    from scripts.saes_same_budget_render_teacher_parameter_attribution import (
        _grouped_parameter_changes,
        _representative_level_masks,
    )

    # One L0 tile, one L1 tile, and no Full representatives in this 4x8 grid.
    modified = torch.ones(32, dtype=torch.bool)
    l0_reps = torch.tensor((0, 3, 24, 27))
    l1_reps = torch.tensor((4, 5, 6, 7, 28, 29, 30, 31))
    modified[l0_reps] = False
    modified[l1_reps] = False
    representatives = ~modified
    groups = _representative_level_masks(
        modified, representatives, views=1, height=4, width=8
    )
    assert int(groups["L0"].sum()) == 4
    assert int(groups["L1"].sum()) == 8
    initial = _gaussians(32)
    global_representatives = torch.nonzero(representatives, as_tuple=False).flatten()
    optimized = _optimized(initial, global_representatives)
    report = _grouped_parameter_changes(initial, optimized, global_representatives, groups)
    assert report["L0"]["representative_count"] == 4
    assert report["L1"]["representative_count"] == 8
    assert report["L0"]["families"]["means"]["relative_change"]["maximum"] > 0.0


def test_sanity_match_is_metric_direction_agnostic_and_fail_closed():
    from scripts.saes_same_budget_render_teacher_parameter_attribution import _sanity_match

    expected = {"psnr_db": 53.592, "ssim": 0.998, "lpips": 0.0067}
    assert _sanity_match(expected, expected)["passed"] is True
    changed = {**expected, "psnr_db": 52.0}
    assert _sanity_match(changed, expected)["passed"] is False


def test_selected_only_preflight_rejects_the_posthoc_dense_oracle(monkeypatch):
    import scripts.saes_same_budget_render_teacher_parameter_attribution as attribution

    monkeypatch.setattr(
        attribution,
        "SELECTED_ONLY_PREFLIGHT_MATERIALIZATION",
        attribution.MATERIALIZATION,
    )
    with pytest.raises(RuntimeError, match="post-hoc full-S3 oracle"):
        attribution._run_two_sentinel_selected_only_preflight(
            model=object(), context={}, device=torch.device("cpu")
        )


def test_selected_only_preflight_runs_before_frozen_oracle_reconstruction(monkeypatch):
    import scripts.saes_same_budget_render_teacher_parameter_attribution as attribution

    class Model:
        def eval(self):
            return self

    def blocked_preflight(**_kwargs):
        raise RuntimeError("preflight-blocked")

    monkeypatch.setattr(
        attribution,
        "_load_model_and_inputs",
        lambda _device: (Model(), {}, {}, {}),
    )
    monkeypatch.setattr(
        attribution,
        "_run_two_sentinel_selected_only_preflight",
        blocked_preflight,
    )
    monkeypatch.setattr(
        attribution,
        "_load_frozen_oracle",
        lambda: pytest.fail("oracle reconstruction ran before selected-only preflight"),
    )

    with pytest.raises(RuntimeError, match="preflight-blocked"):
        attribution.collect_parameter_attribution(device=torch.device("cpu"))
