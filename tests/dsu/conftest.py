"""
DSU Test Fixtures

Provides shared test fixtures for DSU (Depth Search Unit) tests.
"""
import pytest
import torch
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class MockDSUConfig:
    """Mock DSU configuration."""
    num_depth_candidates: int = 32
    feature_dim: int = 128
    cost_type: str = 'correlation'  # 'correlation' or 'cost'


# ============================================================================
# Feature Map Fixtures
# ============================================================================

@pytest.fixture
def feature_dim() -> int:
    return 128


@pytest.fixture
def feature_map_size() -> Tuple[int, int]:
    return (64, 64)


@pytest.fixture
def random_feature_map(feature_dim, feature_map_size) -> torch.Tensor:
    """Generate random feature map [C, H, W]."""
    H, W = feature_map_size
    return torch.randn(feature_dim, H, W)


@pytest.fixture
def reference_feature(feature_dim) -> torch.Tensor:
    """Generate reference feature vector [C]."""
    feat = torch.randn(feature_dim)
    return feat / torch.norm(feat)


# ============================================================================
# Depth Candidate Fixtures
# ============================================================================

@pytest.fixture
def num_depth_candidates() -> int:
    return 32


@pytest.fixture
def depth_candidates(num_depth_candidates) -> torch.Tensor:
    """Linear depth candidates."""
    return torch.linspace(0.5, 10.0, num_depth_candidates)


@pytest.fixture
def inverse_depth_candidates(num_depth_candidates) -> torch.Tensor:
    """Inverse depth candidates (for Transplat)."""
    disp = torch.linspace(0.1, 2.0, num_depth_candidates)
    return 1.0 / disp


# ============================================================================
# Camera Fixtures
# ============================================================================

@pytest.fixture
def reference_intrinsics() -> torch.Tensor:
    """Reference camera intrinsics."""
    K = torch.eye(3)
    K[0, 0] = 500.0  # fx
    K[1, 1] = 500.0  # fy
    K[0, 2] = 320.0  # cx
    K[1, 2] = 240.0  # cy
    return K


@pytest.fixture
def reference_extrinsics() -> torch.Tensor:
    """Reference camera extrinsics (identity)."""
    return torch.eye(4)


@pytest.fixture
def target_intrinsics() -> torch.Tensor:
    """Target camera intrinsics (same as reference)."""
    K = torch.eye(3)
    K[0, 0] = 500.0
    K[1, 1] = 500.0
    K[0, 2] = 320.0
    K[1, 2] = 240.0
    return K


@pytest.fixture
def target_extrinsics() -> torch.Tensor:
    """Target camera extrinsics (translated)."""
    E = torch.eye(4)
    E[0, 3] = 0.1  # 10cm translation along x
    return E


@pytest.fixture
def projection_params(
    reference_intrinsics, reference_extrinsics,
    target_intrinsics, target_extrinsics
) -> dict:
    """Complete projection parameters."""
    return {
        'ref_intrinsics': reference_intrinsics,
        'ref_extrinsics': reference_extrinsics,
        'tgt_intrinsics': target_intrinsics,
        'tgt_extrinsics': target_extrinsics,
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def default_dsu_config() -> MockDSUConfig:
    """Default DSU configuration."""
    return MockDSUConfig()


@pytest.fixture
def transplat_config() -> MockDSUConfig:
    """Transplat-specific configuration."""
    return MockDSUConfig(
        num_depth_candidates=32,
        feature_dim=128,
        cost_type='cost',  # Lower is better
    )


@pytest.fixture
def mvsplat_config() -> MockDSUConfig:
    """MVSplat-specific configuration."""
    return MockDSUConfig(
        num_depth_candidates=32,
        feature_dim=64,
        cost_type='correlation',  # Higher is better
    )


# ============================================================================
# Cost Volume Fixtures
# ============================================================================

@pytest.fixture
def mock_cost_volume(num_depth_candidates, feature_map_size) -> torch.Tensor:
    """Mock cost volume [D, H, W]."""
    H, W = feature_map_size
    return torch.randn(num_depth_candidates, H, W)


@pytest.fixture
def peaked_cost_volume(num_depth_candidates, feature_map_size) -> torch.Tensor:
    """Cost volume with clear peak at center depth."""
    H, W = feature_map_size
    D = num_depth_candidates
    
    volume = torch.zeros(D, H, W)
    peak_idx = D // 2
    
    for d in range(D):
        dist = abs(d - peak_idx)
        volume[d] = np.exp(-dist * 0.5)
    
    return volume


# ============================================================================
# Helper Functions
# ============================================================================

def project_point(
    pixel: Tuple[float, float],
    depth: float,
    K_ref: torch.Tensor,
    E_ref: torch.Tensor,
    K_tgt: torch.Tensor,
    E_tgt: torch.Tensor,
) -> Tuple[float, float]:
    """Project a point from reference to target view."""
    u, v = pixel
    
    # Unproject to 3D (camera space)
    uv_homog = torch.tensor([u, v, 1.0])
    ray = torch.linalg.solve(K_ref, uv_homog)
    point_cam = depth * ray
    
    # Transform to world
    point_homog = torch.cat([point_cam, torch.tensor([1.0])])
    point_world = E_ref @ point_homog
    
    # Transform to target camera
    E_tgt_inv = torch.linalg.inv(E_tgt)
    point_tgt = E_tgt_inv @ point_world
    
    # Project to target image
    point_tgt_3d = point_tgt[:3] / point_tgt[2]
    projected = K_tgt @ point_tgt_3d
    
    return float(projected[0]), float(projected[1])


def bilinear_sample(
    feature_map: torch.Tensor,
    u: float,
    v: float,
) -> torch.Tensor:
    """Bilinear sampling helper."""
    C, H, W = feature_map.shape
    
    # Clamp coordinates
    u = max(0, min(W - 1, u))
    v = max(0, min(H - 1, v))
    
    u0, v0 = int(u), int(v)
    u1 = min(u0 + 1, W - 1)
    v1 = min(v0 + 1, H - 1)
    
    du, dv = u - u0, v - v0
    
    f00 = feature_map[:, v0, u0]
    f01 = feature_map[:, v1, u0]
    f10 = feature_map[:, v0, u1]
    f11 = feature_map[:, v1, u1]
    
    result = (
        f00 * (1 - du) * (1 - dv) +
        f10 * du * (1 - dv) +
        f01 * (1 - du) * dv +
        f11 * du * dv
    )
    
    return result


# Export helpers
pytest.project_point = project_point
pytest.bilinear_sample = bilinear_sample
