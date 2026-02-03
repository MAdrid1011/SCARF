"""
Shared test fixtures for SAES unit and integration tests
"""
import pytest
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MockGaussian:
    """Mock Gaussian for testing (decoupled from transplat)"""
    mean: torch.Tensor  # [3] - 3D position
    cov: torch.Tensor   # [3, 3] - covariance matrix
    opacity: float      # scalar - transparency [0, 1]
    harmonics: torch.Tensor  # [C, D_sh] - spherical harmonics


@pytest.fixture
def identical_gaussians():
    """Generate 4 identical Gaussians for high-similarity tests"""
    mean = torch.tensor([1.0, 2.0, 3.0])
    cov = torch.eye(3) * 0.5
    opacity = 0.8
    harmonics = torch.randn(3, 16)  # RGB + higher-order SH
    
    gaussians = [
        MockGaussian(mean.clone(), cov.clone(), opacity, harmonics.clone())
        for _ in range(4)
    ]
    return gaussians


@pytest.fixture
def different_gaussians():
    """Generate 4 very different Gaussians for low-similarity tests"""
    gaussians = []
    for i in range(4):
        mean = torch.tensor([i * 5.0, i * 3.0, i * 2.0])  # Widely spaced
        cov = torch.eye(3) * (0.1 + i * 0.5)  # Different scales
        opacity = 0.3 + i * 0.15  # Different opacities
        harmonics = torch.randn(3, 16) * (i + 1)  # Different colors
        gaussians.append(MockGaussian(mean, cov, opacity, harmonics))
    return gaussians


@pytest.fixture
def moderate_similarity_gaussians():
    """Generate 4 Gaussians with moderate similarity (sparse-continue case)"""
    base_mean = torch.tensor([5.0, 5.0, 5.0])
    base_cov = torch.eye(3) * 0.3
    base_harmonics = torch.randn(3, 16)
    
    gaussians = []
    for i in range(4):
        # Small perturbations
        mean = base_mean + torch.randn(3) * 0.5
        cov = base_cov + torch.randn(3, 3) * 0.05
        cov = (cov + cov.T) / 2  # Ensure symmetric
        opacity = 0.75 + torch.randn(1).item() * 0.05
        harmonics = base_harmonics + torch.randn(3, 16) * 0.1
        gaussians.append(MockGaussian(mean, cov, opacity, harmonics))
    return gaussians


@pytest.fixture
def tile_config():
    """Default SAES tile configuration for testing"""
    from saes.types import TileConfig
    return TileConfig(
        tile_size=4,
        probe_size=4,
        sparse_indices=[4, 7, 11, 15],
        high_similarity_threshold=0.85,
        low_similarity_threshold=0.60,
    )


@pytest.fixture
def mock_depth_predictor_fn():
    """Mock depth predictor function that returns synthetic depths"""
    def depth_fn(features, indices):
        # Return mock depths for requested indices
        # Shape: [len(indices), 1]
        return torch.rand(len(indices), 1) * 5.0 + 1.0  # Depths in [1, 6]
    return depth_fn


@pytest.fixture
def mock_gaussian_adapter_fn():
    """Mock Gaussian adapter function that returns synthetic Gaussians"""
    def gaussian_fn(depths, context):
        # Generate mock Gaussians from depths
        gaussians = []
        for depth in depths:
            mean = torch.randn(3) * depth.item()
            cov = torch.eye(3) * torch.rand(1).item()
            opacity = torch.rand(1).item()
            harmonics = torch.randn(3, 16)
            gaussians.append(MockGaussian(mean, cov, opacity, harmonics))
        return gaussians
    return gaussian_fn


@pytest.fixture
def synthetic_features():
    """Generate synthetic feature maps for testing"""
    # [B=1, V=2, C=128, H=64, W=64]
    return torch.randn(1, 2, 128, 64, 64)


@pytest.fixture
def mock_context():
    """Mock camera context (intrinsics, extrinsics)"""
    B, V = 1, 2
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)
    return {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "near": torch.tensor([[1.0, 1.0]]),
        "far": torch.tensor([[10.0, 10.0]]),
    }


def assert_gaussians_similar(g1: MockGaussian, g2: MockGaussian, tol=1e-3):
    """Helper to assert two Gaussians are similar within tolerance"""
    assert torch.allclose(g1.mean, g2.mean, atol=tol)
    assert torch.allclose(g1.cov, g2.cov, atol=tol)
    assert abs(g1.opacity - g2.opacity) < tol
    assert torch.allclose(g1.harmonics, g2.harmonics, atol=tol)


def create_gaussian_grid(H: int, W: int, smooth=True) -> List[MockGaussian]:
    """
    Create a grid of Gaussians for testing spatial patterns
    
    Args:
        H, W: Grid dimensions
        smooth: If True, create smooth variation; else create random
    
    Returns:
        List of H*W Gaussians in row-major order
    """
    gaussians = []
    for i in range(H):
        for j in range(W):
            if smooth:
                # Smooth spatial variation
                mean = torch.tensor([i * 0.5, j * 0.5, 5.0])
                cov = torch.eye(3) * 0.3
                opacity = 0.8
                harmonics = torch.ones(3, 16) * (i + j) / (H + W)
            else:
                # Random variation
                mean = torch.randn(3) * 5.0
                cov = torch.eye(3) * torch.rand(1).item()
                opacity = torch.rand(1).item()
                harmonics = torch.randn(3, 16)
            
            gaussians.append(MockGaussian(mean, cov, opacity, harmonics))
    return gaussians
