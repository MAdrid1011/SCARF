import pytest


torch = pytest.importorskip("torch")
nn = torch.nn


def test_conv_and_gemm_emit_executed_mmcu_slot_events():
    from encoder import ConvEngine, GEMMUnit
    from encoder.mmcu_events import mmcu_stage, mmcu_stage_records, reset_mmcu_events

    reset_mmcu_events()
    with mmcu_stage("s1"):
        ConvEngine().forward(
            torch.zeros(1, 3, 8, 8),
            torch.zeros(17, 3, 3, 3),
            padding=1,
        )
    with mmcu_stage("s2"):
        GEMMUnit().matmul(torch.zeros(2, 33, 64), torch.zeros(2, 64, 17))

    records = mmcu_stage_records(
        {"feature": 10, "depth": 20, "gaussian": 30, "ggu": 40}
    )

    assert records["s1"]["mmcu_slots_available"] is True
    assert records["s1"]["useful_mmcu_slots"] == 1 * 17 * 8 * 8 * 3 * 3 * 3
    assert records["s1"]["scheduled_mmcu_slots"] >= records["s1"][
        "useful_mmcu_slots"
    ]
    assert records["s2"]["useful_mmcu_slots"] == 2 * 33 * 17 * 64
    assert records["s2"]["scheduled_mmcu_slots"] > records["s2"][
        "useful_mmcu_slots"
    ]
    assert records["s3"]["mmcu_slots_available"] is False
    assert records["s4"]["source"] == "stage_has_no_recorded_mmcu_operation"


def test_accurate_module_hooks_split_depth_and_gaussian_subtrees():
    from encoder.mmcu_events import (
        mmcu_stage_records,
        reset_mmcu_events,
        trace_torch_mmcu_modules,
    )

    class Pipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth = nn.Conv2d(3, 8, 3, padding=1)
            self.gaussian = nn.Sequential(nn.Conv2d(8, 4, 1), nn.GELU())

        def forward(self, value):
            depth = self.depth(value)
            return self.gaussian(depth)

    pipeline = Pipeline()
    gaussian_input = []

    def capture(_module, inputs, _output):
        gaussian_input.append(inputs[0])

    handle = pipeline.gaussian.register_forward_hook(capture)
    reset_mmcu_events()
    with trace_torch_mmcu_modules(
        pipeline, "s2", exclude_modules=[pipeline.gaussian]
    ):
        pipeline(torch.zeros(1, 3, 8, 8))
    handle.remove()
    with trace_torch_mmcu_modules(pipeline.gaussian, "s3"):
        pipeline.gaussian(gaussian_input[0])

    records = mmcu_stage_records(
        {"feature": 1, "depth": 1, "gaussian": 1, "ggu": 1}
    )
    assert records["s2"]["useful_mmcu_slots"] == 1 * 8 * 8 * 8 * 3 * 3 * 3
    assert records["s3"]["useful_mmcu_slots"] == 1 * 4 * 8 * 8 * 8
