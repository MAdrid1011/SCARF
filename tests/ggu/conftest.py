"""
GGU Test Fixtures

Provides shared test fixtures for GGU (Gaussian Generation Unit) tests.
"""
import pytest
import torch
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class MockGGUConfig:
    """Mock GGU configuration."""
    scale_min: float = 0.0005
    scale_max: float = 0.5
    sh_degree: int = 3
    feature_dim: int = 128


# ============================================================================
# Camera Fixtures
# ============================================================================

@pytest.fixture
def intrinsics() -> torch.Tensor:
    """Camera intrinsics."""
    K = torch.eye(3)
    K[0, 0] = 500.0  # fx
    K[1, 1] = 500.0  # fy
    K[0, 2] = 320.0  # cx
    K[1, 2] = 240.0  # cy
    return K


@pytest.fixture
def extrinsics_identity() -> torch.Tensor:
    """Identity extrinsics (camera at origin)."""
    return torch.eye(4)


@pytest.fixture
def extrinsics_rotated() -> torch.Tensor:
    """Extrinsics with rotation (45 degrees around y)."""
    E = torch.eye(4)
    angle = np.pi / 4
    c, s = np.cos(angle), np.sin(angle)
    E[0, 0] = c
    E[0, 2] = s
    E[2, 0] = -s
    E[2, 2] = c
    return E


@pytest.fixture
def extrinsics_translated() -> torch.Tensor:
    """Extrinsics with translation."""
    E = torch.eye(4)
    E[0, 3] = 1.0  # x translation
    E[1, 3] = 0.5  # y translation
    E[2, 3] = 2.0  # z translation
    return E


# ============================================================================
# Raw Gaussian Fixtures
# ============================================================================

@pytest.fixture
def raw_scales() -> torch.Tensor:
    """Raw scale values from network (unbounded)."""
    return torch.tensor([0.0, 0.5, -0.5])  # x, y, z


@pytest.fixture
def raw_rotation() -> torch.Tensor:
    """Raw quaternion from network."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity rotation


@pytest.fixture
def raw_sh_coeffs() -> torch.Tensor:
    """Raw spherical harmonics coefficients."""
    # Degree 3: (3+1)^2 = 16 coefficients per channel
    # 3 color channels
    return torch.randn(3, 16)


@pytest.fixture
def raw_gaussian() -> torch.Tensor:
    """Complete raw Gaussian features from network."""
    scales = torch.tensor([0.0, 0.5, -0.5])  # 3
    rotation = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 4
    sh = torch.randn(48)  # 3 * 16 = 48
    return torch.cat([scales, rotation, sh])


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def default_ggu_config() -> MockGGUConfig:
    """Default GGU configuration."""
    return MockGGUConfig()


# ============================================================================
# Helper Functions
# ============================================================================

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = q / torch.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])
    
    return R


def compute_covariance(scales: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Compute covariance matrix from scales and rotation."""
    R = quaternion_to_rotation_matrix(rotation)
    S = torch.diag(scales ** 2)
    return R @ S @ R.T


def compute_ray_direction(
    pixel: Tuple[float, float],
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Compute ray direction for a pixel."""
    u, v = pixel
    uv_homog = torch.tensor([u, v, 1.0])
    ray = torch.linalg.solve(intrinsics, uv_homog)
    return ray / torch.norm(ray)


# Export helpers
pytest.quaternion_to_rotation_matrix = quaternion_to_rotation_matrix
pytest.compute_covariance = compute_covariance
pytest.compute_ray_direction = compute_ray_direction
