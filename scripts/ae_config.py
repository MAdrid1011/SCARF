"""Pure configuration mapping for SCARF artifact experiments."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


MODELS = ("transplat", "mvsplat", "depthsplat")
DATASETS = ("re10k", "acid", "dl3dv")
CLAIMED_MATRIX = tuple((model, dataset) for model in MODELS for dataset in DATASETS)


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    dataset: str
    experiment: str
    checkpoint: Path
    dataset_root: Path
    dataset_representation: str
    environment_profile: str
    hydra_overrides: tuple[str, ...]


@dataclass(frozen=True)
class ClaimSelection:
    model: str
    dataset: str
    index_path: Path
    source_index_sha256: str
    sample_count: int
    sample_selection_sha256: str


_EXPERIMENTS = {
    ("transplat", "re10k"): ("re10k", "re10k.ckpt", "classic"),
    ("transplat", "acid"): ("acid", "acid.ckpt", "classic"),
    # The converted DL3DV subset uses the Re10K chunk reader and zero-shot model.
    ("transplat", "dl3dv"): ("re10k", "re10k.ckpt", "classic"),
    ("mvsplat", "re10k"): ("re10k", "re10k.ckpt", "classic"),
    ("mvsplat", "acid"): ("acid", "acid.ckpt", "classic"),
    ("mvsplat", "dl3dv"): ("re10k", "re10k.ckpt", "classic"),
    ("depthsplat", "re10k"): ("re10k", "re10k.ckpt", "depthsplat"),
    ("depthsplat", "acid"): ("re10k", "re10k.ckpt", "depthsplat"),
    ("depthsplat", "dl3dv"): ("dl3dv", "dl3dv.ckpt", "depthsplat"),
}

_DATASET_REPRESENTATIONS = {
    ("transplat", "dl3dv"): ("dl3dv/re10k", "re10k-compatible-360x640-v1"),
    ("mvsplat", "dl3dv"): ("dl3dv/re10k", "re10k-compatible-360x640-v1"),
    ("depthsplat", "dl3dv"): ("dl3dv/native", "depthsplat-native-270x480-v1"),
}

_HYDRA_OVERRIDES = {
    # The public Re10K checkpoint is the large ViT-L model. These are the
    # exact evaluation overrides published in the pinned DepthSplat README.
    ("depthsplat", "re10k"): (
        "model.encoder.num_scales=2",
        "model.encoder.upsample_factor=2",
        "model.encoder.lowest_feature_resolution=4",
        "model.encoder.monodepth_vit_type=vitl",
    ),
    ("depthsplat", "acid"): (
        "model.encoder.num_scales=2",
        "model.encoder.upsample_factor=2",
        "model.encoder.lowest_feature_resolution=4",
        "model.encoder.monodepth_vit_type=vitl",
    ),
    # The public DL3DV checkpoint is the base ViT-B model.
    ("depthsplat", "dl3dv"): (
        "model.encoder.num_scales=2",
        "model.encoder.upsample_factor=4",
        "model.encoder.lowest_feature_resolution=8",
        "model.encoder.monodepth_vit_type=vitb",
    ),
}


def resolve_experiment(model: str, dataset: str, root: Path) -> ExperimentConfig:
    if model not in MODELS:
        raise ValueError(f"unsupported model: {model}")
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    experiment, checkpoint_name, profile = _EXPERIMENTS[(model, dataset)]
    root = Path(root)
    dataset_relative, representation = _DATASET_REPRESENTATIONS.get(
        (model, dataset), (dataset, f"{dataset}-native")
    )
    return ExperimentConfig(
        model=model,
        dataset=dataset,
        experiment=experiment,
        checkpoint=root / model / "checkpoints" / checkpoint_name,
        dataset_root=root / "datasets" / dataset_relative,
        dataset_representation=representation,
        environment_profile=profile,
        hydra_overrides=_HYDRA_OVERRIDES.get((model, dataset), ()),
    )


def resolve_claim_selection(model: str, dataset: str, root: Path) -> ClaimSelection:
    if model not in MODELS:
        raise ValueError(f"unsupported model: {model}")
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    root = Path(root).resolve()
    protocol_path = root / "artifact/evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "finalized":
        raise ValueError("evaluation protocol is not finalized")
    record = protocol.get("pairs", {}).get(f"{model}/{dataset}")
    if not isinstance(record, dict):
        raise ValueError(f"evaluation protocol has no pair {model}/{dataset}")
    index_path = (root / record["index_path"]).resolve()
    if root not in index_path.parents or not index_path.is_file():
        raise ValueError(f"invalid claim evaluation index: {record.get('index_path')}")
    return ClaimSelection(
        model=model,
        dataset=dataset,
        index_path=index_path,
        source_index_sha256=record["source_index_sha256"],
        sample_count=int(record["sample_count"]),
        sample_selection_sha256=record["sample_selection_sha256"],
    )
def validate_prepared_dataset(
    config: ExperimentConfig,
    dataset_root: Path,
    manifest_path: Path,
    *,
    allow_functional_fixture: bool = False,
) -> dict:
    """Validate the prepared dataset identity used by one experiment."""
    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_name = config.dataset
    if config.dataset == "dl3dv":
        expected_name = (
            "dl3dv-native"
            if config.dataset_representation == "depthsplat-native-270x480-v1"
            else "dl3dv-re10k"
        )
    if manifest.get("dataset") != expected_name:
        raise ValueError(
            f"dataset manifest identifies {manifest.get('dataset')!r}; "
            f"expected {expected_name!r}"
        )
    functional_fixture = manifest.get("functional_fixture") is True
    if functional_fixture and not allow_functional_fixture:
        raise ValueError("synthetic functional dataset cannot be used for a claim run")
    if functional_fixture and (
        config.dataset != "re10k"
        or manifest.get("representation") != "re10k-synthetic-functional-v1"
        or manifest.get("paper_result_eligible") is not False
    ):
        raise ValueError("synthetic functional dataset manifest is invalid")
    tree_sha256 = manifest.get("tree_sha256")
    if not isinstance(tree_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", tree_sha256
    ):
        raise ValueError("dataset manifest has no valid tree_sha256")

    if config.dataset == "dl3dv":
        conversion_path = dataset_root / "conversion.json"
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
        if conversion.get("representation") != config.dataset_representation:
            raise ValueError(
                "DL3DV representation mismatch: "
                f"{conversion.get('representation')!r} != "
                f"{config.dataset_representation!r}"
            )
        file_record = manifest.get("files", {}).get("conversion.json")
        if not isinstance(file_record, dict):
            raise ValueError("DL3DV manifest does not cover conversion.json")
        conversion_bytes = conversion_path.read_bytes()
        if file_record.get("size") != len(conversion_bytes) or file_record.get(
            "sha256"
        ) != hashlib.sha256(conversion_bytes).hexdigest():
            raise ValueError("DL3DV conversion metadata does not match its manifest")
    return {
        "name": config.dataset,
        "representation": (
            manifest["representation"]
            if functional_fixture
            else config.dataset_representation
        ),
        "tree_sha256": tree_sha256,
        "functional_fixture": functional_fixture,
    }


def validate_claim_dataset_tree(
    model: str, dataset: str, tree_sha256: str, root: Path
) -> None:
    root = Path(root).resolve()
    protocol = json.loads(
        (root / "artifact/evaluation_protocol.json").read_text(encoding="utf-8")
    )
    protocol_tree = protocol.get("pairs", {}).get(f"{model}/{dataset}", {}).get(
        "dataset_tree_sha256"
    )
    datasets = json.loads(
        (root / "artifact/manifests/datasets.json").read_text(encoding="utf-8")
    ).get("datasets", {})
    dataset_contract = datasets.get(dataset, {})
    if dataset == "dl3dv":
        representation = "native" if model == "depthsplat" else "re10k"
        contract_tree = dataset_contract.get("representations", {}).get(
            representation, {}
        ).get("expected_tree_sha256")
    else:
        contract_tree = dataset_contract.get("expected_tree_sha256")
    values = (tree_sha256, protocol_tree, contract_tree)
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in values
    ):
        raise ValueError(f"claim dataset tree contract is unresolved for {model}/{dataset}")
    if len(set(values)) != 1:
        raise ValueError(f"claim dataset tree mismatch for {model}/{dataset}")
