"""
FSDR Test Fixtures

Provides shared test fixtures for FSDR unit and integration tests.
"""
import pytest
import torch
import numpy as np
from typing import Callable, Tuple, List
from dataclasses import dataclass


# ============================================================================
# Mock Data Structures (mirror fsdr/types.py for testing without implementation)
# ============================================================================

@dataclass
class MockCacheEntry:
    """Mock cache entry for testing."""
    signature: int           # 16-bit LSH signature
    position: Tuple[int, int]  # (u, v) pixel coordinates
    best_depth: float        # Optimal depth
    best_idx: int            # Index of best depth candidate
    peak_prob: float         # Peak probability (0-1)
    second_offset: int       # Offset to second-best index
    spread: float            # Distribution spread (0-1)
    valid: bool = True       # Entry validity


@dataclass
class MockFSDRConfig:
    """Mock FSDR configuration for testing."""
    cache_size: int = 128
    lsh_dim: int = 16
    feature_dim: int = 128
    hamming_threshold: int = 4
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5
    hamming_direct_reuse: int = 2
    hamming_interpolate: int = 3


# ============================================================================
# Feature Fixtures
# ============================================================================

@pytest.fixture
def feature_dim() -> int:
    """Standard feature dimension."""
    return 128


@pytest.fixture
def num_depth_candidates() -> int:
    """Standard number of depth candidates."""
    return 32


@pytest.fixture
def random_feature(feature_dim) -> torch.Tensor:
    """Generate a random normalized feature vector."""
    feat = torch.randn(feature_dim)
    return feat / torch.norm(feat)


@pytest.fixture
def similar_feature_pair(feature_dim) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two similar feature vectors (cosine similarity > 0.9)."""
    torch.manual_seed(42)  # Deterministic for reliability
    base = torch.randn(feature_dim)
    base = base / torch.norm(base)
    
    # Add very small noise to ensure high similarity
    noise = torch.randn(feature_dim) * 0.05  # Smaller noise for higher similarity
    similar = base + noise
    similar = similar / torch.norm(similar)
    
    # Verify similarity is high enough
    cosine_sim = float(torch.dot(base, similar))
    if cosine_sim < 0.9:
        # If still not high enough, interpolate more towards base
        similar = 0.95 * base + 0.05 * similar
        similar = similar / torch.norm(similar)
    
    return base, similar


@pytest.fixture
def dissimilar_feature_pair(feature_dim) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two dissimilar feature vectors (cosine similarity < 0.5)."""
    feat1 = torch.randn(feature_dim)
    feat1 = feat1 / torch.norm(feat1)
    
    # Generate nearly orthogonal vector
    feat2 = torch.randn(feature_dim)
    # Remove component parallel to feat1
    feat2 = feat2 - torch.dot(feat1, feat2) * feat1
    feat2 = feat2 / torch.norm(feat2)
    
    return feat1, feat2


@pytest.fixture
def batch_features(feature_dim) -> torch.Tensor:
    """Batch of 10 random feature vectors."""
    feats = torch.randn(10, feature_dim)
    return feats / torch.norm(feats, dim=1, keepdim=True)


# ============================================================================
# Depth Fixtures
# ============================================================================

@pytest.fixture
def depth_candidates(num_depth_candidates) -> torch.Tensor:
    """Standard depth candidates (linear spacing)."""
    return torch.linspace(0.5, 10.0, num_depth_candidates)


@pytest.fixture
def inverse_depth_candidates(num_depth_candidates) -> torch.Tensor:
    """Inverse depth (disparity) candidates for Transplat."""
    disp_near = 1.0 / 10.0  # Far
    disp_far = 1.0 / 0.5    # Near
    disparities = torch.linspace(disp_near, disp_far, num_depth_candidates)
    return 1.0 / disparities


@pytest.fixture
def mock_probability_distribution(num_depth_candidates) -> torch.Tensor:
    """Generate mock probability distribution with clear peak."""
    probs = torch.zeros(num_depth_candidates)
    peak_idx = num_depth_candidates // 2
    
    # Create peaked distribution
    for i in range(num_depth_candidates):
        dist = abs(i - peak_idx)
        probs[i] = np.exp(-dist * 0.5)
    
    return probs / probs.sum()


@pytest.fixture
def uniform_probability_distribution(num_depth_candidates) -> torch.Tensor:
    """Uniform probability distribution (low confidence)."""
    return torch.ones(num_depth_candidates) / num_depth_candidates


# ============================================================================
# Cache Entry Fixtures
# ============================================================================

@pytest.fixture
def high_confidence_entry() -> MockCacheEntry:
    """Cache entry with high confidence (suitable for direct reuse)."""
    return MockCacheEntry(
        signature=0xABCD,
        position=(32, 32),
        best_depth=5.0,
        best_idx=16,
        peak_prob=0.9,
        second_offset=2,
        spread=0.1,
        valid=True,
    )


@pytest.fixture
def medium_confidence_entry() -> MockCacheEntry:
    """Cache entry with medium confidence (suitable for interpolation)."""
    return MockCacheEntry(
        signature=0x1234,
        position=(48, 48),
        best_depth=3.5,
        best_idx=10,
        peak_prob=0.65,
        second_offset=-3,
        spread=0.25,
        valid=True,
    )


@pytest.fixture
def low_confidence_entry() -> MockCacheEntry:
    """Cache entry with low confidence (requires verification)."""
    return MockCacheEntry(
        signature=0x5678,
        position=(16, 16),
        best_depth=7.2,
        best_idx=22,
        peak_prob=0.35,
        second_offset=5,
        spread=0.5,
        valid=True,
    )


@pytest.fixture
def invalid_entry() -> MockCacheEntry:
    """Invalid cache entry."""
    return MockCacheEntry(
        signature=0x0000,
        position=(0, 0),
        best_depth=0.0,
        best_idx=0,
        peak_prob=0.0,
        second_offset=0,
        spread=0.0,
        valid=False,
    )


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def default_config() -> MockFSDRConfig:
    """Default FSDR configuration."""
    return MockFSDRConfig()


@pytest.fixture
def aggressive_config() -> MockFSDRConfig:
    """Aggressive configuration (more reuse, less verification)."""
    return MockFSDRConfig(
        hamming_threshold=5,
        high_confidence_threshold=0.7,
        medium_confidence_threshold=0.4,
        hamming_direct_reuse=3,
        hamming_interpolate=4,
    )


@pytest.fixture
def conservative_config() -> MockFSDRConfig:
    """Conservative configuration (more verification, less reuse)."""
    return MockFSDRConfig(
        hamming_threshold=3,
        high_confidence_threshold=0.9,
        medium_confidence_threshold=0.7,
        hamming_direct_reuse=1,
        hamming_interpolate=2,
    )


# ============================================================================
# Mock Functions
# ============================================================================

@pytest.fixture
def mock_cost_fn(depth_candidates) -> Callable:
    """Mock cost function for depth search."""
    def cost_fn(feature: torch.Tensor, depth_idx: int) -> float:
        """
        Mock cost: lower cost near center of depth range.
        """
        center = len(depth_candidates) // 2
        distance = abs(depth_idx - center)
        return float(distance * 0.1 + torch.randn(1).item() * 0.01)
    
    return cost_fn


@pytest.fixture
def mock_prob_fn(depth_candidates, mock_probability_distribution) -> Callable:
    """Mock probability function for full depth search."""
    def prob_fn(feature: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Return mock probability distribution."""
        return mock_probability_distribution
    
    return prob_fn


# ============================================================================
# LSH Fixtures
# ============================================================================

@pytest.fixture
def lsh_projection_matrix(feature_dim) -> torch.Tensor:
    """Random projection matrix for LSH."""
    matrix = torch.randn(16, feature_dim)
    return matrix / torch.norm(matrix, dim=1, keepdim=True)


@pytest.fixture
def known_signature_feature() -> Tuple[torch.Tensor, int]:
    """Feature with known LSH signature for deterministic testing."""
    # Seed for reproducibility
    torch.manual_seed(42)
    feature = torch.randn(128)
    feature = feature / torch.norm(feature)
    
    # Expected signature computed with seed=42
    expected_signature = 0x3A5C  # Example value
    
    return feature, expected_signature


# ============================================================================
# Camera Fixtures (for integration with DSU)
# ============================================================================

@pytest.fixture
def mock_intrinsics() -> torch.Tensor:
    """Mock camera intrinsics."""
    K = torch.eye(3)
    K[0, 0] = 500.0  # fx
    K[1, 1] = 500.0  # fy
    K[0, 2] = 320.0  # cx
    K[1, 2] = 240.0  # cy
    return K


@pytest.fixture
def mock_extrinsics() -> torch.Tensor:
    """Mock camera extrinsics (identity - camera at origin)."""
    return torch.eye(4)


@pytest.fixture
def mock_target_feature_map(feature_dim) -> torch.Tensor:
    """Mock target feature map for sampling."""
    return torch.randn(feature_dim, 64, 64)


# ============================================================================
# Helper Functions
# ============================================================================

def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return float(torch.dot(a, b) / (torch.norm(a) * torch.norm(b)))


def compute_hamming_distance(sig1: int, sig2: int) -> int:
    """Compute Hamming distance between two signatures."""
    return bin(sig1 ^ sig2).count('1')


def create_peaked_distribution(
    num_candidates: int,
    peak_idx: int,
    peak_prob: float = 0.8,
) -> torch.Tensor:
    """Create a probability distribution with specified peak."""
    probs = torch.zeros(num_candidates)
    probs[peak_idx] = peak_prob
    
    # Distribute remaining probability
    remaining = 1.0 - peak_prob
    for i in range(num_candidates):
        if i != peak_idx:
            probs[i] = remaining / (num_candidates - 1)
    
    return probs


# Export helper functions
pytest.compute_cosine_similarity = compute_cosine_similarity
pytest.compute_hamming_distance = compute_hamming_distance
pytest.create_peaked_distribution = create_peaked_distribution
