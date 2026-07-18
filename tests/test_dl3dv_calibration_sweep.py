import hashlib
import itertools
import json
from pathlib import Path

import pytest


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parameters() -> list[dict[str, float]]:
    from scripts.calibration_contract import PARAMETER_GRID

    names = tuple(PARAMETER_GRID)
    return [
        {name: float(value) for name, value in zip(names, values)}
        for values in itertools.product(*(PARAMETER_GRID[name] for name in names))
    ]


def _trace_candidate(parameters: dict[str, float], *, chosen: bool, failed: bool) -> dict:
    if failed:
        quality = {
            "psnr_loss_db": 0.5,
            "ssim_loss": 0.01,
            "lpips_increase": 0.01,
        }
    else:
        quality = {
            "psnr_loss_db": 0.01,
            "ssim_loss": 0.001,
            "lpips_increase": 0.001,
        }
    return {
        "parameters": parameters,
        "quality": quality,
        "work_reduction": 0.9 if chosen else 0.1,
        "compression": 0.1 if chosen else 0.2,
    }


def _build_split_record(
    root: Path,
    *,
    split: str,
    selected: dict[str, float],
    failed: bool = False,
) -> dict:
    from scripts.calibration_contract import (
        candidate_sha256,
        canonical_sha256,
        parameters_sha256,
    )
    from scripts.calibration_sweep import aggregate_candidate_traces

    pair_specs = (
        ("transplat", "re10k"),
        ("mvsplat", "re10k"),
        ("depthsplat", "native"),
    )
    is_holdout = split == "calibration_holdout"
    scope = (
        "exact_committed_train_tuple" if is_holdout else "registered_global_grid"
    )
    scene = f"{split}-scene"
    selection = {
        "sample_index": 0,
        "scene": scene,
        "context_indices": [0, 4],
        "target_indices": [1, 2, 3],
    }
    selection_sha256 = canonical_sha256([selection])
    scene_set_sha256 = canonical_sha256([scene])
    trace_files = []
    traces = []
    bindings = []
    candidate_parameters = [selected] if is_holdout else _parameters()
    for model, representation in pair_specs:
        pair = f"{model}/dl3dv"
        binding = {
            "model": model,
            "dataset": "dl3dv",
            "representation": representation,
            "sample_count": 1,
            "selection_sha256": selection_sha256,
            "scene_set_sha256": scene_set_sha256,
            "calibration_input_root": f"inputs/{split}/{representation}",
            "calibration_input_tree_sha256": _digest(f"{split}-{model}-tree"),
            "calibration_input_manifest_sha256": _digest(
                f"{split}-{model}-manifest"
            ),
            "calibration_input_provenance_sha256": _digest(
                f"{split}-{model}-provenance"
            ),
            "index_file": f"{split}-{representation}.json",
            "index_sha256": _digest(f"{split}-{representation}-index"),
        }
        trace = {
            "schema_version": "1.0",
            "kind": "calibration_sample_trace",
            **selection,
            "model": model,
            "dataset": "dl3dv",
            "seed": 0,
            "calibration_scope": scope,
            "committed_parameters": selected if is_holdout else None,
            "trace": {
                "neural_forward_passes": 1,
                "target_rgb_accessed": False,
                "candidate_count": len(candidate_parameters),
                "candidate_scope": scope,
                "candidate_parameters_sha256": (
                    parameters_sha256(selected) if is_holdout else None
                ),
                "calibration_input": {
                    "tree_sha256": binding["calibration_input_tree_sha256"],
                    "manifest_sha256": binding[
                        "calibration_input_manifest_sha256"
                    ],
                    "input_provenance_sha256": binding[
                        "calibration_input_provenance_sha256"
                    ],
                    "selection_sha256": selection_sha256,
                    "index_sha256": binding["index_sha256"],
                    "index_selection_sha256": selection_sha256,
                    "selected_scene_count": 1,
                    "target_rgb_accessed": False,
                    "target_rgb_included": False,
                },
            },
            "candidates": [
                _trace_candidate(
                    parameters,
                    chosen=parameters == selected,
                    failed=failed and is_holdout,
                )
                for parameters in candidate_parameters
            ],
        }
        path = root / "traces" / split / model / "samples" / "sample_00000" / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace, sort_keys=True), encoding="utf-8")
        traces.append(trace)
        trace_files.append(
            {
                "pair": pair,
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        bindings.append(binding)
    candidates = aggregate_candidate_traces(traces)
    for candidate in candidates:
        candidate["candidate_sha256"] = candidate_sha256(candidate)
    record = {
        "split": split,
        "execution_mode": (
            "exact_train_selected_tuple_no_rerank"
            if is_holdout
            else "registered_global_grid"
        ),
        "sample_count": 1,
        "selection_sha256": selection_sha256,
        "scene_set_sha256": scene_set_sha256,
        "required_pairs": [f"{model}/dl3dv" for model, _ in pair_specs],
        "pair_bindings": bindings,
        "pair_bindings_sha256": canonical_sha256(bindings),
        "trace_count": len(traces),
        "trace_set_sha256": canonical_sha256(traces),
        "trace_files": trace_files,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_set_sha256": canonical_sha256(
            [candidate["candidate_sha256"] for candidate in candidates]
        ),
    }
    if is_holdout:
        record["validated_parameters"] = selected
        record["validated_parameters_sha256"] = parameters_sha256(selected)
    return record


def _candidate_record(root: Path, *, failed_holdout: bool = False) -> Path:
    from scripts.calibration_contract import canonical_parameters, parameters_sha256
    from scripts.calibration_contract import select_global_candidate

    selected = canonical_parameters(
        {"gamma_depth": 0.05, "beta_x": 0.25, "beta_f": 0.05, "beta_d": 0.5}
    )
    train = _build_split_record(root, split="calibration_train", selected=selected)
    selected_candidate = select_global_candidate(train["candidates"])
    assert selected_candidate["parameters"] == selected
    holdout = _build_split_record(
        root,
        split="calibration_holdout",
        selected=selected,
        failed=failed_holdout,
    )
    source = {
        "schema_version": "2.0",
        "kind": "split_calibration_candidate_records",
        "calibration_protocol": "dl3dv_train_holdout_v1",
        "status": "PASS",
        "evaluation_disjoint": True,
        "train_holdout_scene_disjoint": True,
        "calibration_manifest_sha256": _digest("manifest"),
        "train_selection": {
            "parameters": selected,
            "parameters_sha256": parameters_sha256(selected),
            "candidate_sha256": selected_candidate["candidate_sha256"],
            "selection_rule": "quality constraints, maximum event work reduction",
        },
        "splits": {
            "calibration_train": train,
            "calibration_holdout": holdout,
        },
    }
    path = root / "candidates.json"
    path.write_text(json.dumps(source, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_dl3dv_calibration_sweep_maps_each_model_to_its_required_sidecar():
    from scripts.calibration_sweep import _manifest_pair_records

    manifest = {
        "kind": "dl3dv_calibration_protocol",
        "datasets": {
            "dl3dv": {
                "splits": {
                    "calibration_train": {
                        "representations": {
                            "native": {"id": "native-train"},
                            "re10k": {"id": "re10k-train"},
                        }
                    },
                    "calibration_holdout": {
                        "representations": {
                            "native": {"id": "native-holdout"},
                            "re10k": {"id": "re10k-holdout"},
                        }
                    },
                }
            }
        },
    }

    assert _manifest_pair_records(manifest, "calibration_train") == [
        ("transplat", "dl3dv", {"id": "re10k-train"}),
        ("mvsplat", "dl3dv", {"id": "re10k-train"}),
        ("depthsplat", "dl3dv", {"id": "native-train"}),
    ]
    assert _manifest_pair_records(manifest, "calibration_holdout") == [
        ("transplat", "dl3dv", {"id": "re10k-holdout"}),
        ("mvsplat", "dl3dv", {"id": "re10k-holdout"}),
        ("depthsplat", "dl3dv", {"id": "native-holdout"}),
    ]


def test_dl3dv_calibration_sweep_refuses_a_missing_model_representation():
    from scripts.calibration_sweep import _manifest_pair_records

    manifest = {
        "kind": "dl3dv_calibration_protocol",
        "datasets": {"dl3dv": {"representations": {"native": {}}}},
    }
    with pytest.raises(ValueError, match="re10k sidecar"):
        _manifest_pair_records(manifest)


def test_build_plan_binds_disjoint_train_and_holdout_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from scripts.calibration_contract import canonical_sha256
    from scripts.calibration_sweep import build_plan
    from scripts.compile_protocol import canonicalize_index, sha256_file

    protocol = tmp_path / "protocol"
    identities = {}
    splits = {}
    for split, count in (("calibration_train", 24), ("calibration_holdout", 8)):
        index = {
            f"{split}-scene-{ordinal:02d}": {
                "context": [0, 4],
                "target": [1, 2, 3],
            }
            for ordinal in range(count)
        }
        representations = {}
        for representation in ("native", "re10k"):
            index_path = protocol / f"{split}-{representation}.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(index), encoding="utf-8")
            index_sha256 = sha256_file(index_path)
            selections, summary = canonicalize_index(index_path, index_sha256)
            root = protocol / "inputs" / split / representation
            identity = {
                "tree_sha256": _digest(f"{split}-{representation}-tree"),
                "manifest_sha256": _digest(f"{split}-{representation}-manifest"),
                "input_provenance_sha256": _digest(
                    f"{split}-{representation}-provenance"
                ),
                "selection_sha256": summary["sample_selection_sha256"],
                "selected_scene_count": count,
                "target_rgb_accessed": False,
            }
            identities[root.resolve()] = identity
            representations[representation] = {
                "calibration_input_root": root.relative_to(protocol).as_posix(),
                "calibration_input_tree_sha256": identity["tree_sha256"],
                "calibration_input_manifest_sha256": identity["manifest_sha256"],
                "calibration_input_provenance_sha256": identity[
                    "input_provenance_sha256"
                ],
                "target_rgb_included": False,
                "index_file": index_path.name,
                "index_sha256": index_sha256,
                "selection_sha256": summary["sample_selection_sha256"],
                "sample_count": count,
            }
        splits[split] = {
            "scene_count": count,
            "scene_set_sha256": canonical_sha256(sorted(index)),
            "representations": representations,
        }
    manifest = {
        "schema_version": "1.0",
        "kind": "dl3dv_calibration_protocol",
        "status": "PASS",
        "evaluation_disjoint": True,
        "download_plan_sha256": _digest("download-plan"),
        "archive_preparation_sha256": _digest("archive-preparation"),
        "calibration_scene_count": 24,
        "holdout_scene_count": 8,
        "datasets": {"dl3dv": {"splits": splits}},
    }
    manifest["calibration_manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = protocol / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.calibration_sweep.validate_target_free_input_root",
        lambda root, _dataset: identities[Path(root).resolve()],
    )

    plan = build_plan(manifest_path, tmp_path / "sweep")

    assert plan["kind"] == "split_calibration_sweep_plan"
    assert plan["train_holdout_scene_disjoint"] is True
    assert plan["splits"]["calibration_train"]["sample_count"] == 24
    assert plan["splits"]["calibration_holdout"]["sample_count"] == 8
    assert (
        plan["splits"]["calibration_holdout"]["execution_mode"]
        == "deferred_exact_train_selected_tuple"
    )


def test_holdout_validation_cannot_rerank_another_tuple():
    from scripts.calibration_contract import candidate_sha256, canonical_parameters
    from scripts.calibration_sweep import validate_exact_holdout_candidate

    committed = canonical_parameters(
        {"gamma_depth": 0.05, "beta_x": 0.25, "beta_f": 0.05, "beta_d": 0.5}
    )
    alternate = canonical_parameters(
        {"gamma_depth": 0.15, "beta_x": 1.0, "beta_f": 0.2, "beta_d": 2.0}
    )
    candidate = _trace_candidate(committed, chosen=True, failed=False)
    candidate["candidate_sha256"] = candidate_sha256(candidate)
    alternate_candidate = _trace_candidate(alternate, chosen=True, failed=False)
    alternate_candidate["candidate_sha256"] = candidate_sha256(alternate_candidate)
    with pytest.raises(ValueError, match="exactly one committed"):
        validate_exact_holdout_candidate([candidate, alternate_candidate], committed)
    with pytest.raises(ValueError, match="does not match"):
        validate_exact_holdout_candidate([alternate_candidate], committed)


def test_calibrated_config_requires_verified_train_and_holdout_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.calibration_sweep as sweep
    from scripts.calibrate_mechanisms import build_config
    from scripts.mechanism_config import load_mechanism_config

    monkeypatch.setattr(
        sweep,
        "DL3DV_SPLIT_COUNTS",
        {"calibration_train": 1, "calibration_holdout": 1},
    )
    path = _candidate_record(tmp_path)
    config = build_config(path)
    config_path = tmp_path / "mechanism_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _, provenance = load_mechanism_config(config_path)

    assert config["status"] == "calibrated"
    assert config["calibration"]["protocol"] == "dl3dv_train_holdout_v1"
    assert config["calibration"]["holdout"]["validated_candidate_sha256"]
    assert provenance["holdout"]["validated_candidate_sha256"] == config[
        "calibration"
    ]["holdout"]["validated_candidate_sha256"]


def test_missing_or_failed_holdout_blocks_configuration_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.calibration_sweep as sweep
    from scripts.calibrate_mechanisms import build_config

    monkeypatch.setattr(
        sweep,
        "DL3DV_SPLIT_COUNTS",
        {"calibration_train": 1, "calibration_holdout": 1},
    )
    path = _candidate_record(tmp_path / "missing")
    source = json.loads(path.read_text(encoding="utf-8"))
    source["splits"].pop("calibration_holdout")
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="holdout calibration evidence is missing"):
        build_config(path)

    failed_path = _candidate_record(tmp_path / "failed", failed_holdout=True)
    with pytest.raises(ValueError, match="holdout quality gate failed"):
        build_config(failed_path)


def test_trace_or_candidate_tampering_blocks_configuration_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.calibration_sweep as sweep
    from scripts.calibrate_mechanisms import build_config

    monkeypatch.setattr(
        sweep,
        "DL3DV_SPLIT_COUNTS",
        {"calibration_train": 1, "calibration_holdout": 1},
    )
    path = _candidate_record(tmp_path / "candidate")
    source = json.loads(path.read_text(encoding="utf-8"))
    source["splits"]["calibration_holdout"]["candidates"][0]["parameters"][
        "gamma_depth"
    ] = 0.15
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate records do not match"):
        build_config(path)

    trace_path = _candidate_record(tmp_path / "trace")
    source = json.loads(trace_path.read_text(encoding="utf-8"))
    relative = source["splits"]["calibration_train"]["trace_files"][0]["path"]
    artifact = trace_path.parent / relative
    trace = json.loads(artifact.read_text(encoding="utf-8"))
    trace["candidates"][0]["work_reduction"] = 0.77
    artifact.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="trace file hash mismatch"):
        build_config(trace_path)


def test_same_selection_stale_sidecar_trace_is_rejected_after_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Refreshing trace-file hashes cannot make an old sidecar look current."""
    from scripts.calibration_contract import canonical_sha256
    import scripts.calibration_sweep as sweep
    from scripts.calibrate_mechanisms import build_config

    monkeypatch.setattr(
        sweep,
        "DL3DV_SPLIT_COUNTS",
        {"calibration_train": 1, "calibration_holdout": 1},
    )
    path = _candidate_record(tmp_path)
    source = json.loads(path.read_text(encoding="utf-8"))
    split = source["splits"]["calibration_train"]
    first = split["trace_files"][0]
    artifact = path.parent / first["path"]
    trace = json.loads(artifact.read_text(encoding="utf-8"))
    # Keep the scene/index/candidate grid identical, as a stale --resume result
    # would, but prove that its opened target-free sidecar is from another tree.
    trace["trace"]["calibration_input"]["tree_sha256"] = _digest("stale-tree")
    artifact.write_text(json.dumps(trace), encoding="utf-8")

    refreshed_traces = []
    for trace_file in split["trace_files"]:
        trace_artifact = path.parent / trace_file["path"]
        trace_file["sha256"] = hashlib.sha256(trace_artifact.read_bytes()).hexdigest()
        refreshed_traces.append(json.loads(trace_artifact.read_text(encoding="utf-8")))
    split["trace_set_sha256"] = canonical_sha256(refreshed_traces)
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar tree hash mismatch"):
        build_config(path)
