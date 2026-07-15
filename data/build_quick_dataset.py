#!/usr/bin/env python3
"""Build a deterministic synthetic Re10K-format functional test fixture."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_manifest import build


SCENE = "scarf-functional-scene"
REPRESENTATION = "re10k-synthetic-functional-v1"


def _encoded_image(frame: int) -> torch.Tensor:
    y, x = np.indices((360, 640), dtype=np.uint16)
    image = np.stack(
        (
            (x + frame * 17) % 256,
            (y * 2 + frame * 29) % 256,
            ((x // 2 + y // 2) + frame * 11) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(
        encoded,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    return torch.from_numpy(np.frombuffer(encoded.getvalue(), dtype=np.uint8).copy())


def _camera(frame: int) -> torch.Tensor:
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[0, 3] = -0.1 * frame
    values = [0.8, 0.8, 0.5, 0.5, 0.0, 0.0]
    values.extend(world_to_camera[:3].reshape(-1).tolist())
    return torch.tensor(values, dtype=torch.float32)


def generate(output: Path) -> dict:
    output = Path(output).resolve()
    generator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if output.exists() and any(output.rglob("*")):
        try:
            manifest = json.loads(
                (output / ".scarf-manifest.json").read_text(encoding="utf-8")
            )
            source = json.loads(
                (output / ".scarf-source.json").read_text(encoding="utf-8")
            )
            rebuilt = build(
                output,
                "re10k",
                "scarf:deterministic-synthetic-functional-fixture",
                f"generator-sha256:{generator_sha256}",
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise FileExistsError(
                f"quick dataset output is incomplete: {output}"
            ) from exc
        if (
            manifest.get("functional_fixture") is True
            and manifest.get("paper_result_eligible") is False
            and manifest.get("representation") == REPRESENTATION
            and source.get("generator_sha256") == generator_sha256
            and manifest.get("tree_sha256") == rebuilt.get("tree_sha256")
            and manifest.get("files") == rebuilt.get("files")
        ):
            return manifest
        raise FileExistsError(f"quick dataset output does not match the generator: {output}")
    stage = output / "test"
    stage.mkdir(parents=True, exist_ok=True)
    example = {
        "key": SCENE,
        "url": "generated-by-scarf",
        "timestamps": torch.arange(5, dtype=torch.int64),
        "cameras": torch.stack([_camera(frame) for frame in range(5)]),
        "images": [_encoded_image(frame) for frame in range(5)],
    }
    chunk = stage / "000000.torch"
    torch.save([example], chunk)
    (stage / "index.json").write_text(
        json.dumps({SCENE: chunk.name}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source = {
        "schema_version": "1.0",
        "functional_fixture": True,
        "representation": REPRESENTATION,
        "scene": SCENE,
        "view_count": 5,
        "generator_sha256": generator_sha256,
        "paper_result_eligible": False,
    }
    (output / ".scarf-source.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = build(
        output,
        "re10k",
        "scarf:deterministic-synthetic-functional-fixture",
        f"generator-sha256:{source['generator_sha256']}",
    )
    manifest.update(
        {
            "functional_fixture": True,
            "representation": REPRESENTATION,
            "paper_result_eligible": False,
        }
    )
    manifest_path = output / ".scarf-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = generate(args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
