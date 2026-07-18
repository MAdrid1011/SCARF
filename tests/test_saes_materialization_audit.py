from pathlib import Path


def test_target_free_materialization_audit_accepts_adapter_offset_transport():
    from scripts.saes_target_free_materialization_audit import build_parser

    args = build_parser().parse_args(
        [
            "--materialization",
            "conditional-adapter-offset-transport-diagnostic",
            "--output-dir",
            str(Path("outputs") / "new-audit"),
        ]
    )

    assert args.materialization == "conditional-adapter-offset-transport-diagnostic"
