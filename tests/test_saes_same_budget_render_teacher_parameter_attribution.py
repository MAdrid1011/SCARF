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
        FROZEN_PROTOCOL_ID,
        _expected_variant_names,
        _variant_specs,
    )

    assert ATTRIBUTION_ID == "same-budget-render-teacher-parameter-attribution-v3"
    assert FROZEN_PROTOCOL_ID == "same-budget-render-teacher-parameter-attribution-v2"
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


def test_parameter_variants_reject_out_of_range_compact_slots_before_indexing():
    from scripts.saes_same_budget_render_teacher_parameter_attribution import (
        _build_parameter_variants,
    )

    initial = _gaussians(2)
    optimized = _optimized(initial, torch.tensor((0, 1), dtype=torch.long))
    with pytest.raises(ValueError, match="representative local indices"):
        _build_parameter_variants(initial, optimized, torch.tensor((0, 2), dtype=torch.long))


def test_compact_index_map_translates_dense_representatives_before_compact_access():
    from scripts.saes_same_budget_render_teacher_parameter_attribution import (
        _build_compact_index_maps,
    )

    dense = _gaussians(32)
    modified = torch.zeros(32, dtype=torch.bool)
    modified[:10] = True
    compact = _gaussians(22)
    global_indices = torch.tensor((10, 31), dtype=torch.long)
    retained, global_to_local, representative_local = _build_compact_index_maps(
        dense=dense,
        compact=compact,
        modified=modified,
        representative_global=global_indices,
    )
    assert torch.equal(representative_local, torch.tensor((0, 21)))
    assert torch.equal(retained[representative_local], global_indices)
    assert int(global_to_local[31]) == 21

    with pytest.raises(RuntimeError, match="wrong slot count"):
        _build_compact_index_maps(
            dense=dense,
            compact=_gaussians(23),
            modified=modified,
            representative_global=global_indices,
        )


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
    global_representatives = torch.nonzero(representatives, as_tuple=False).flatten()
    representative_local = torch.arange(global_representatives.numel(), dtype=torch.long)
    initial = _gaussians(global_representatives.numel())
    optimized = _optimized(initial, representative_local)
    report = _grouped_parameter_changes(
        initial,
        optimized,
        global_representatives,
        representative_local,
        groups,
    )
    assert report["L0"]["representative_count"] == 4
    assert report["L1"]["representative_count"] == 8
    assert report["L0"]["families"]["means"]["relative_change"]["maximum"] > 0.0


def test_grouped_changes_rejects_a_global_to_local_mapping_outside_compact_slots():
    from scripts.saes_same_budget_render_teacher_parameter_attribution import (
        _grouped_parameter_changes,
    )

    initial = _gaussians(2)
    representatives = torch.tensor((8, 20), dtype=torch.long)
    groups = {"L0": torch.zeros(32, dtype=torch.bool), "L1": torch.zeros(32, dtype=torch.bool)}
    groups["L0"][8] = True
    groups["L1"][20] = True
    optimized = _optimized(initial, torch.tensor((0, 1), dtype=torch.long))
    with pytest.raises(RuntimeError, match="global-to-local"):
        _grouped_parameter_changes(
            initial,
            optimized,
            representatives,
            torch.tensor((0, 2), dtype=torch.long),
            groups,
        )


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
    monkeypatch.setattr(attribution, "_require_clean_source_identity", lambda: {})
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


def test_attribution_rejects_dirty_source_before_model_loading(monkeypatch):
    import scripts.saes_same_budget_render_teacher_parameter_attribution as attribution

    monkeypatch.setattr(
        attribution,
        "_require_clean_source_identity",
        lambda: (_ for _ in ()).throw(RuntimeError("clean-source-blocked")),
    )
    monkeypatch.setattr(
        attribution,
        "_load_model_and_inputs",
        lambda _device: pytest.fail("model loading ran from a dirty source tree"),
    )

    with pytest.raises(RuntimeError, match="clean-source-blocked"):
        attribution.collect_parameter_attribution(device=torch.device("cpu"))
