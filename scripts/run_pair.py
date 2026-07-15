#!/usr/bin/env python3
"""Execute one model and dataset pair over a canonical evaluation index."""

from __future__ import annotations

import argparse
import json
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
            validate(record)
        except (KeyError, TypeError, ValueError):
            return False
        identity = identity or source_identity()
        if (
            provenance.get("git_commit") != identity["git_commit"]
            or provenance.get("git_dirty") is not identity["git_dirty"]
        ):
            return False
    return _selection_from_result(record) == _stable_selection(selection)


def execute_pair(
    selections: list[dict[str, Any]],
    output_dir: Path,
    session_factory: Callable[[], SampleSession],
    *,
    resume: bool,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    samples_root = output_dir / "samples"
    identity = source_identity()
    pending = []
    resumed = 0
    for selection in selections:
        sample_dir = samples_root / f"sample_{selection['sample_index']:05d}"
        if resume and _complete_sample(
            sample_dir / "results.json", selection, identity
        ):
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
            session.run_sample(selection, sample_dir)
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
        if not _complete_sample(sample_dir / "results.json", selection, identity):
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

    def __init__(self, base_command: list[str], sample_count: int):
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
        sample_count = len(selections)
        record = execute_pair(
            selections,
            args.output_dir.resolve(),
            lambda: InProcessDemoSession(args.command, sample_count),
            resume=args.resume,
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
