"""
Benchmark Test Fixtures

Shared fixtures for benchmark testing.
"""
import pytest
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MockCycleEstimate:
    """Mock cycle estimate for testing."""
    component: str
    operation: str
    cycles: int
    memory_accesses: int = 0


@dataclass
class MockQualityMetrics:
    """Mock quality metrics for testing."""
    psnr: float
    ssim: float
    num_samples: int = 1


# ============ Cycle Counter Fixtures ============

@pytest.fixture
def sample_cycle_estimates() -> List[MockCycleEstimate]:
    """Sample cycle estimates for testing."""
    return [
        MockCycleEstimate('fsdr', 'hash', 3, 0),
        MockCycleEstimate('fsdr', 'lookup', 5, 1),
        MockCycleEstimate('dsu', 'project', 8, 0),
        MockCycleEstimate('dsu', 'sample', 4, 32),
        MockCycleEstimate('dsu', 'cost', 2, 0),
        MockCycleEstimate('dsu', 'softmax', 6, 0),
        MockCycleEstimate('ggu', 'position', 2, 0),
        MockCycleEstimate('ggu', 'covariance', 5, 0),
        MockCycleEstimate('ggu', 'sh_rotation', 3, 0),
    ]


@pytest.fixture
def fsdr_only_estimates() -> List[MockCycleEstimate]:
    """FSDR-only cycle estimates for cache hit path."""
    return [
        MockCycleEstimate('fsdr', 'hash', 3, 0),
        MockCycleEstimate('fsdr', 'lookup', 5, 1),
        MockCycleEstimate('fsdr', 'direct_reuse', 1, 0),
    ]


@pytest.fixture
def full_search_estimates() -> List[MockCycleEstimate]:
    """Full DSU search cycle estimates (cache miss)."""
    estimates = [
        MockCycleEstimate('fsdr', 'hash', 3, 0),
        MockCycleEstimate('fsdr', 'lookup', 5, 1),
    ]
    # 32 depth candidates
    for _ in range(32):
        estimates.extend([
            MockCycleEstimate('dsu', 'project', 8, 0),
            MockCycleEstimate('dsu', 'sample', 4, 1),
            MockCycleEstimate('dsu', 'cost', 2, 0),
        ])
    estimates.append(MockCycleEstimate('dsu', 'softmax', 6, 0))
    return estimates


# ============ Quality Validator Fixtures ============

@pytest.fixture
def identical_images() -> tuple:
    """Identical rendered and ground truth images."""
    img = torch.rand(3, 256, 256)
    return img.clone(), img.clone()


@pytest.fixture
def similar_images() -> tuple:
    """Similar images with small noise difference."""
    torch.manual_seed(42)
    gt = torch.rand(3, 256, 256)
    noise = torch.randn_like(gt) * 0.01  # Small noise
    rendered = torch.clamp(gt + noise, 0, 1)
    return rendered, gt


@pytest.fixture
def different_images() -> tuple:
    """Significantly different images."""
    torch.manual_seed(42)
    gt = torch.rand(3, 256, 256)
    torch.manual_seed(123)
    rendered = torch.rand(3, 256, 256)
    return rendered, gt


@pytest.fixture
def baseline_metrics() -> MockQualityMetrics:
    """Baseline quality metrics for comparison."""
    return MockQualityMetrics(psnr=25.5, ssim=0.92, num_samples=100)


# ============ Benchmark Runner Fixtures ============

@pytest.fixture
def mock_model_config() -> Dict:
    """Mock model configuration."""
    return {
        'model_type': 'transplat',
        'checkpoint_path': 'checkpoints/re10k.ckpt',
        'feature_dim': 128,
    }


@pytest.fixture
def mock_scarf_config() -> Dict:
    """Mock SCARF configuration."""
    return {
        'enable_fsdr': True,
        'enable_saes': False,
        'fsdr_config': {
            'hamming_threshold': 4,
            'high_confidence_threshold': 0.8,
        },
    }


@pytest.fixture
def mock_benchmark_config() -> Dict:
    """Mock benchmark configuration."""
    return {
        'num_scenes': 10,
        'warmup_scenes': 2,
        'output_dir': '/tmp/benchmark_test',
    }


# ============ Helper Functions ============

def compute_psnr_reference(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    """Reference PSNR computation for test validation."""
    mse = torch.mean((rendered - gt) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_ssim_reference(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    """Simplified SSIM for test validation (uses mean/variance)."""
    # Convert to numpy for computation
    rendered_np = rendered.numpy()
    gt_np = gt.numpy()
    
    # Compute means
    mu_r = np.mean(rendered_np)
    mu_g = np.mean(gt_np)
    
    # Compute variances and covariance
    sigma_r = np.var(rendered_np)
    sigma_g = np.var(gt_np)
    sigma_rg = np.mean((rendered_np - mu_r) * (gt_np - mu_g))
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim = ((2 * mu_r * mu_g + C1) * (2 * sigma_rg + C2)) / \
           ((mu_r ** 2 + mu_g ** 2 + C1) * (sigma_r + sigma_g + C2))
    
    return float(ssim)


def generate_mock_scene_data(num_pixels: int = 1000) -> Dict:
    """Generate mock scene data for testing."""
    return {
        'features': torch.rand(num_pixels, 128),
        'depths': torch.rand(num_pixels) * 10 + 0.5,
        'pixel_coords': torch.randint(0, 480, (num_pixels, 2)),
    }
