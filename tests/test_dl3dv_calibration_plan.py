import random

import pytest


def _archive(scene: int) -> dict:
    name = f"{scene:064x}"
    return {"path": f"{scene // 1000 + 1}K/{name}.zip", "size": 1000 + scene, "oid": name}


def test_dl3dv_calibration_plan_selects_disjoint_24_plus_8_deterministically():
    from data.plan_dl3dv_calibration import build_plan

    evaluation = [f"{scene:064x}" for scene in range(140)]
    tree = [_archive(scene) for scene in range(180)]
    shuffled = list(tree)
    random.Random(7).shuffle(shuffled)

    plan = build_plan(
        shuffled,
        evaluation_scenes=evaluation,
        evaluation_index_sha256="a" * 64,
        revision="b" * 40,
    )

    selection = plan["selection"]
    train = selection["calibration_train"]
    holdout = selection["calibration_holdout"]
    assert plan["status"] == "PLANNED_AWAITING_UPSTREAM_ACCESS"
    assert len(train) == 24
    assert len(holdout) == 8
    assert not ({row["scene"] for row in train} & set(evaluation))
    assert not ({row["scene"] for row in holdout} & set(evaluation))
    assert not ({row["scene"] for row in train} & {row["scene"] for row in holdout})
    assert selection["scene_disjoint"] is True
    assert len(plan["source"]["archive_tree_sha256"]) == 64
    assert plan["source"]["terms_url"].startswith("https://huggingface.co/")


def test_dl3dv_calibration_plan_refuses_insufficient_or_malformed_source_trees():
    from data.plan_dl3dv_calibration import build_plan

    evaluation = [f"{scene:064x}" for scene in range(140)]
    with pytest.raises(ValueError, match="fewer than 24"):
        build_plan(
            [_archive(scene) for scene in range(160)],
            evaluation_scenes=evaluation,
            evaluation_index_sha256="a" * 64,
            revision="b" * 40,
        )
    with pytest.raises(ValueError, match="unsafe scene archive"):
        build_plan(
            [{"path": "../escape.zip", "size": 1, "oid": "x"}],
            evaluation_scenes=["eval"],
            evaluation_index_sha256="a" * 64,
            revision="b" * 40,
        )
