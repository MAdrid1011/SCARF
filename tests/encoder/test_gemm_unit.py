"""
Tests for GEMMUnit - Matrix multiplication hardware simulator.
"""

import pytest
import torch
import sys
from pathlib import Path

SCARF_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from encoder.gemm_unit import GEMMUnit
from encoder.types import GEMMConfig, CycleStats, ResourceEstimate


class TestGEMMUnitInit:
    """Test GEMMUnit initialization."""
    
    def test_default_config(self):
        """Test initialization with default config."""
        unit = GEMMUnit()
        assert unit.config.tile_m == 8
        assert unit.config.tile_n == 16
    
    def test_custom_config(self, small_gemm_config):
        """Test initialization with custom config."""
        unit = GEMMUnit(small_gemm_config)
        assert unit.config.tile_m == 4


class TestGEMMUnitMatmul:
    """Test GEMMUnit matrix multiplication."""
    
    def test_basic_matmul(self):
        """Test basic matrix multiplication."""
        unit = GEMMUnit()
        a = torch.randn(64, 128)
        b = torch.randn(128, 256)
        
        out, cycles = unit.matmul(a, b)
        
        assert out.shape == (64, 256)
        assert cycles.total_cycles > 0
    
    def test_batched_matmul(self):
        """Test batched matrix multiplication."""
        unit = GEMMUnit()
        a = torch.randn(4, 64, 128)
        b = torch.randn(4, 128, 256)
        
        out, cycles = unit.matmul(a, b)
        
        assert out.shape == (4, 64, 256)
        assert cycles.total_cycles > 0
    
    def test_matmul_correctness(self):
        """Test matmul matches PyTorch reference."""
        unit = GEMMUnit()
        a = torch.randn(32, 64)
        b = torch.randn(64, 128)
        
        out, _ = unit.matmul(a, b)
        ref = torch.matmul(a, b)
        
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    
    def test_attention_style_matmul(self):
        """Test attention-style Q@K^T matmul."""
        unit = GEMMUnit()
        # Typical attention: [B, L, d] @ [B, d, L] -> [B, L, L]
        q = torch.randn(1, 1024, 128)
        k = torch.randn(1, 128, 1024)
        
        out, cycles = unit.matmul(q, k)
        
        assert out.shape == (1, 1024, 1024)
        assert cycles.total_cycles > 0


class TestGEMMUnitCycles:
    """Test cycle counting."""
    
    def test_larger_matrices_more_cycles(self):
        """Test that larger matrices take more cycles."""
        unit = GEMMUnit()
        
        a_small = torch.randn(32, 64)
        b_small = torch.randn(64, 128)
        
        a_large = torch.randn(128, 256)
        b_large = torch.randn(256, 512)
        
        _, cycles_small = unit.matmul(a_small, b_small)
        _, cycles_large = unit.matmul(a_large, b_large)
        
        assert cycles_large.total_cycles > cycles_small.total_cycles
    
    def test_inner_dim_affects_cycles(self):
        """Test that inner dimension affects cycle count."""
        unit = GEMMUnit()
        
        # Same output size, different K
        a1 = torch.randn(64, 64)
        b1 = torch.randn(64, 128)
        
        a2 = torch.randn(64, 256)
        b2 = torch.randn(256, 128)
        
        _, cycles1 = unit.matmul(a1, b1)
        _, cycles2 = unit.matmul(a2, b2)
        
        # Larger K should take more cycles
        assert cycles2.total_cycles > cycles1.total_cycles


class TestGEMMUnitResources:
    """Test resource estimation."""
    
    def test_resource_estimate(self):
        """Test resource estimate within bounds."""
        unit = GEMMUnit()
        resources = unit.get_resource_estimate()
        
        assert isinstance(resources, ResourceEstimate)
        assert resources.luts == 20000
        assert resources.dsps == 128
        assert resources.sram_bytes == 32768


class TestGEMMUnitConfigs:
    """Test different configurations."""
    
    def test_small_tiles_more_cycles(self, small_gemm_config):
        """Test that smaller tiles result in more cycles."""
        unit_small = GEMMUnit(small_gemm_config)
        unit_large = GEMMUnit()
        
        a = torch.randn(64, 128)
        b = torch.randn(128, 256)
        
        _, cycles_small = unit_small.matmul(a, b)
        _, cycles_large = unit_large.matmul(a, b)
        
        # Smaller tiles = more iterations = more cycles
        assert cycles_small.total_cycles >= cycles_large.total_cycles
