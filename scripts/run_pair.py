#!/usr/bin/env python3
"""Execute one model and dataset pair over a canonical evaluation index."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compile_protocol import canonicalize_index
from scripts.result_record import source_identity
from scripts.validate_result import validate
from data.verify_prepared_dataset import prepared_scene_order


class SampleSession(Protocol):
    def run_sample(self, selection: dict[str, Any], output_dir: Path) -> dict[str, Any]: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_worstcase_evidence(
    pair_dir: Path, result_path: Path, record: dict[str, Any]
) -> None:
    """Keep the strongest FSDR and SAES view while discarding other PNGs."""
    ablation = record.get("ablation")
    evaluation = record.get("provenance", {}).get("evaluation", {})
    if not isinstance(ablation, dict) or not isinstance(evaluation, dict):
        return
    configs = {}
    for name in ("asic", "asic_fsdr", "asic_saes"):
        views = ablation.get(name, {}).get("quality", {}).get("quality_views")
        if not isinstance(views, list) or not views:
            return
        configs[name] = views
    target_indices = evaluation.get("target_indices")
    if not isinstance(target_indices, list) or any(
        [view.get("target_index") for view in configs[name]] != target_indices
        for name in configs
    ):
        raise ValueError("ablation quality views do not match target indices")

    sample_dir = result_path.parent
    source_prefixes = ("gt", "ablation_asic", "ablation_asic_fsdr", "ablation_asic_saes")
    required = [
        sample_dir / f"{prefix}_{view_index:02d}.png"
        for view_index in range(len(target_indices))
        for prefix in source_prefixes
    ]
    if not all(path.is_file() for path in required):
        return

    candidates = {
        "fsdr": [
            float(reference["psnr_db"]) - float(optimized["psnr_db"])
            for reference, optimized in zip(configs["asic"], configs["asic_fsdr"])
        ],
        "saes": [
            float(optimized["lpips"]) - float(reference["lpips"])
            for reference, optimized in zip(configs["asic"], configs["asic_saes"])
        ],
    }
    metrics = {"fsdr": "psnr_loss_db", "saes": "lpips_increase"}
    for kind, values in candidates.items():
        view_index = max(range(len(values)), key=values.__getitem__)
        destination = pair_dir / "worstcase" / kind
        manifest_path = destination / "manifest.json"
        previous = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else None
        )
        if previous is not None and float(
            previous.get("loss_value", float("-inf"))
        ) >= values[view_index]:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for path in destination.glob("*.png"):
            path.unlink()
        selected_sources = {
            "ground_truth": sample_dir / f"gt_{view_index:02d}.png",
            "reference": sample_dir / f"ablation_asic_{view_index:02d}.png",
            "optimized": sample_dir
            / f"ablation_{'asic_fsdr' if kind == 'fsdr' else 'asic_saes'}_{view_index:02d}.png",
        }
        artifacts = {}
        for label, source in selected_sources.items():
            target = destination / f"{label}.png"
            shutil.copy2(source, target)
            artifacts[label] = {
                "path": target.relative_to(destination).as_posix(),
                "sha256": _sha256_file(target),
            }
        manifest = {
            "schema_version": "1.0",
            "kind": f"{kind}_worstcase_view",
            "loss_metric": metrics[kind],
            "loss_value": values[view_index],
            "sample_index": evaluation.get("sample_index"),
            "scene": evaluation.get("scene"),
            "target_index": target_indices[view_index],
            "view_ordinal": view_index,
            "source_result": {
                "path": result_path.relative_to(pair_dir).as_posix(),
                "sha256": _sha256_file(result_path),
            },
            "artifacts": artifacts,
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(manifest_path)

    for path in sample_dir.glob("*.png"):
        path.unlink()


def _validate_record(record: dict[str, Any]) -> None:
    if record.get("kind") in {"fsdr_sample", "fsdr_dataset_aggregate"}:
        from scripts.fsdr_evidence import validate_fsdr_record

        validate_fsdr_record(record)
    else:
        validate(record)


def _append_progress(output_dir: Path, event: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _stable_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        key: selection.get(key)
        for key in ("sample_index", "scene", "context_indices", "target_indices")
    }


def _bind_payload(result_path: Path, payload_path: Path, selection_sha256: str | None) -> dict[str, Any]:
    """Bind the payload produced by this sample to its result record."""
    from scripts.rtl_payload import parse_payload, workload_descriptor

    payload_path = payload_path.resolve()
    parsed = parse_payload(payload_path)
    descriptor = workload_descriptor(parsed)
    record = json.loads(result_path.read_text(encoding="utf-8"))
    provenance = record.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("sample provenance must be an object")
    provenance["rtl_payload"] = {
        "path": str(payload_path),
        "sha256": _sha256_file(payload_path),
        "size_bytes": payload_path.stat().st_size,
        "workload": descriptor,
    }
    if selection_sha256 is not None:
        evaluation = provenance.setdefault("evaluation", {})
        if not isinstance(evaluation, dict):
            raise ValueError("sample evaluation provenance must be an object")
        evaluation["run_selection_sha256"] = selection_sha256
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _bind_selection(result_path: Path, selection_sha256: str | None) -> dict[str, Any]:
    """Bind the reviewer run selection independently of RTL payload export."""
    record = json.loads(result_path.read_text(encoding="utf-8"))
    if selection_sha256 is None:
        return record
    provenance = record.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("sample provenance must be an object")
    evaluation = provenance.setdefault("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("sample evaluation provenance must be an object")
    evaluation["run_selection_sha256"] = selection_sha256
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _selection_from_result(record: dict[str, Any]) -> dict[str, Any] | None:
    evaluation = record.get("provenance", {}).get("evaluation")
    if isinstance(evaluation, dict):
        return {
            "sample_index": evaluation.get("sample_index"),
            "scene": evaluation.get("scene"),
            "context_indices": evaluation.get("context_indices"),
            "target_indices": evaluation.get("target_indices"),
        }
    required = ("sample_index", "scene", "context_indices", "target_indices")
    if all(key in record for key in required):
        return {key: record[key] for key in required}
    return None


def _complete_sample(
    path: Path,
    selection: dict[str, Any],
    identity: dict[str, Any] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    provenance = record.get("provenance")
    if isinstance(provenance, dict):
        try:
            _validate_record(record)
        except (KeyError, TypeError, ValueError):
            return False
        identity = identity or source_identity()
        if (
            provenance.get("git_commit") != identity["git_commit"]
            or provenance.get("git_dirty") is not identity["git_dirty"]
            or provenance.get("source_tree_sha256")
            != identity["source_tree_sha256"]
        ):
            return False
    return _selection_from_result(record) == _stable_selection(selection)


def execute_pair(
    selections: list[dict[str, Any]],
    output_dir: Path,
    session_factory: Callable[[], SampleSession],
    *,
    resume: bool,
    selection_sha256: str | None = None,
    export_rtl_payload: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    samples_root = output_dir / "samples"
    identity = source_identity()
    pending = []
    resumed = 0
    for selection in selections:
        sample_dir = samples_root / f"sample_{selection['sample_index']:05d}"
        result_path = sample_dir / "results.json"
        payload_ready = (not export_rtl_payload) or (sample_dir / "rtl-payload.bin").is_file()
        if resume and payload_ready and _complete_sample(result_path, selection, identity):
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if selection_sha256 is not None and record.get("provenance", {}).get("evaluation", {}).get("run_selection_sha256") != selection_sha256:
                record = _bind_selection(result_path, selection_sha256)
            update_worstcase_evidence(output_dir, result_path, record)
            resumed += 1
        else:
            pending.append((selection, sample_dir))

    _append_progress(
        output_dir,
        {
            "event": "pair_start",
            "sample_count": len(selections),
            "pending": len(pending),
            "resumed": resumed,
            "git_commit": identity["git_commit"],
            "git_dirty": identity["git_dirty"],
        },
    )

    session = session_factory() if pending else None
    executed = 0
    for selection, sample_dir in pending:
        if session is None:
            raise RuntimeError("sample session was not initialized")
        stable_selection = _stable_selection(selection)
        _append_progress(
            output_dir,
            {"event": "sample_start", "selection": stable_selection},
        )
        try:
            sample_record = session.run_sample(selection, sample_dir)
        except Exception as exc:
            _append_progress(
                output_dir,
                {
                    "event": "sample_error",
                    "selection": stable_selection,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        result_path = sample_dir / "results.json"
        payload_path = sample_dir / "rtl-payload.bin"
        if export_rtl_payload:
            if not payload_path.is_file():
                _append_progress(
                    output_dir,
                    {
                        "event": "sample_error",
                        "selection": stable_selection,
                        "error_type": "FileNotFoundError",
                        "error": "quality sample did not produce rtl-payload.bin",
                    },
                )
                raise FileNotFoundError(f"missing RTL payload: {payload_path}")
            sample_record = _bind_payload(result_path, payload_path, selection_sha256)
        elif selection_sha256 is not None:
            sample_record = _bind_selection(result_path, selection_sha256)
        if not _complete_sample(result_path, selection, identity):
            _append_progress(
                output_dir,
                {
                    "event": "sample_error",
                    "selection": stable_selection,
                    "error_type": "RuntimeError",
                    "error": "sample result did not match the run identity",
                },
            )
            raise RuntimeError(
                f"sample {selection['sample_index']} did not produce matching evidence"
            )
        update_worstcase_evidence(
            output_dir, sample_dir / "results.json", sample_record
        )
        executed += 1
        _append_progress(
            output_dir,
            {
                "event": "sample_complete",
                "selection": stable_selection,
                "executed": executed,
                "remaining": len(pending) - executed,
            },
        )
    record = {
        "schema_version": "1.0",
        "kind": "pair_execution",
        "sample_count": len(selections),
        "executed": executed,
        "resumed": resumed,
        "complete": executed + resumed == len(selections),
    }
    _append_progress(output_dir, {"event": "pair_complete", **record})
    return record


class InProcessDemoSession:
    """Run every sample in one model-profile process with a resident model."""

    def __init__(
        self,
        base_command: list[str],
        sample_count: int,
        *,
        export_rtl_payload: bool = False,
    ):
        if len(base_command) < 2:
            raise ValueError("demo command must contain a Python executable and demo.py")
        demo_path = Path(base_command[1]).resolve()
        if demo_path != (ROOT / "scripts/demo.py").resolve():
            raise ValueError(f"pair worker only supports scripts/demo.py: {demo_path}")
        if Path(base_command[0]).resolve() != Path(sys.executable).resolve():
            raise ValueError(
                "run_pair.py and demo.py must use the same locked Python interpreter"
            )
        from scripts import demo

        self.demo = demo
        self.base_args = list(base_command[2:])
        self.sample_count = sample_count
        self.export_rtl_payload = export_rtl_payload

    def run_sample(self, selection: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        argv = [
            *self.base_args,
            "--num-samples",
            str(self.sample_count),
            "--sample-index",
            str(selection.get("execution_index", selection["sample_index"])),
            "--protocol-sample-index",
            str(selection["sample_index"]),
            "--output-dir",
            str(output_dir),
        ]
        if self.export_rtl_payload:
            argv.extend(("--rtl-payload-output", str(output_dir / "rtl-payload.bin")))
        self.demo.main(argv)
        path = output_dir / "results.json"
        return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-index", type=Path, required=True)
    parser.add_argument("--source-index-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="shared scarf-run-selection-v1 consumed by all workflow stages",
    )
    parser.add_argument(
        "--pair",
        help="model/dataset key in --selection-file (for example mvsplat/re10k)",
    )
    parser.add_argument(
        "--export-rtl-payload",
        action="store_true",
        help="write one self-describing payload beside each sample result",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a demo command is required after --")
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        selection_sha256 = None
        if args.selection_file is not None:
            if not args.pair or "/" not in args.pair:
                raise ValueError("--selection-file requires --pair model/dataset")
            from scripts.reviewer_run_config import pair_selection

            selections, selection_sha256 = pair_selection(
                args.selection_file.resolve(),
                args.pair,
                expected_source_index_sha256=args.source_index_sha256,
            )
            summary = {
                "source_index_sha256": args.source_index_sha256,
                "sample_selection_sha256": selection_sha256,
                "sample_count": len(selections),
            }
        else:
            source_selections, _ = canonicalize_index(
                args.evaluation_index.resolve(), args.source_index_sha256
            )
            execution_order = prepared_scene_order(
                args.dataset_root.resolve(),
                {selection["scene"] for selection in source_selections},
            )
            selections, summary = canonicalize_index(
                args.evaluation_index.resolve(),
                args.source_index_sha256,
                execution_scene_order=execution_order,
            )
            if args.num_samples is not None:
                selections = selections[: args.num_samples]
        if args.num_samples is not None and args.selection_file is not None:
            if args.num_samples <= 0:
                raise ValueError("--num-samples must be positive")
            selections = selections[: args.num_samples]
        sample_count = len(selections)
        record = execute_pair(
            selections,
            args.output_dir.resolve(),
            lambda: InProcessDemoSession(
                args.command,
                sample_count,
                export_rtl_payload=args.export_rtl_payload,
            ),
            resume=args.resume,
            selection_sha256=selection_sha256,
            export_rtl_payload=args.export_rtl_payload,
        )
        record["source_index"] = summary
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "pair-execution.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
