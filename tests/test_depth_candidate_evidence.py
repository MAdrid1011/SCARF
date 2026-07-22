import pytest


torch = pytest.importorskip("torch")


class _FakeDepthPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_depth_candidates = 4
        self.depth_head_lowres = torch.nn.Sequential(torch.nn.Identity())

    def forward(self, features, intrinsics, extrinsics, near, far, **kwargs):
        batch, views, _, height, width = features.shape
        logits = torch.zeros(views * batch, 4, height, width)
        logits[:, 2] = 3.0
        probabilities = torch.softmax(self.depth_head_lowres(logits), dim=1)
        coarse = probabilities.argmax(dim=1).float()
        depths = coarse.reshape(views, batch, height, width).permute(1, 0, 2, 3)
        depths = depths.reshape(batch, views, height * width, 1, 1)
        densities = torch.ones_like(depths)
        raw_gaussians = torch.zeros(batch, views, height * width, 1)
        return depths, densities, raw_gaussians


def test_reference_predictor_exposes_executed_depth_head_candidates():
    from depth_predictor import DepthPredictorConfig, HWDepthPredictor

    predictor = HWDepthPredictor(
        DepthPredictorConfig(num_depth_candidates=4),
        torch.device("cpu"),
        model_type="mvsplat",
    )
    predictor.load_from_model(_FakeDepthPredictor())
    predictor.set_use_original(True)
    features = torch.zeros(1, 2, 3, 2, 2)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    near = torch.tensor([[1.0, 2.0]])
    far = torch.tensor([[5.0, 10.0]])

    output = predictor.forward(features, intrinsics, extrinsics, near, far)

    assert output.depth_probs.shape == (1, 2, 4, 2, 2)
    assert torch.equal(output.depth_probs.argmax(dim=2), torch.full((1, 2, 2, 2), 2))
    assert output.depth_candidates.shape == (1, 2, 4, 1, 1)
    assert torch.allclose(output.depth_candidates[0, 0, :, 0, 0], torch.linspace(0.2, 1.0, 4))
    assert torch.allclose(output.depth_candidates[0, 1, :, 0, 0], torch.linspace(0.1, 0.5, 4))
    assert output.candidate_domain == "inverse_depth"
    assert output.probability_source == "pinned_original_depth_head_softmax"


def test_depthsplat_global_candidate_evidence_uses_first_full_search_scale():
    from scripts.fsdr_trace import depthsplat_global_candidate_tensors

    coarse = torch.zeros(2, 8, 2, 3)
    coarse[:, 5] = 1.0
    refined = torch.ones(2, 2, 4, 6) / 2
    near = torch.tensor([[1.0, 2.0]])
    far = torch.tensor([[5.0, 10.0]])

    probabilities, candidates = depthsplat_global_candidate_tensors(
        [coarse, refined], near=near, far=far
    )

    assert probabilities.shape == (1, 2, 8, 2, 3)
    assert torch.equal(probabilities.argmax(dim=2), torch.full((1, 2, 2, 3), 5))
    assert candidates.shape == (1, 2, 8, 1, 1)
    assert torch.allclose(candidates[0, 0, :, 0, 0], torch.linspace(0.2, 1.0, 8))
    assert torch.allclose(candidates[0, 1, :, 0, 0], torch.linspace(0.1, 0.5, 8))
