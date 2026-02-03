"""
Tests for NormalizationUnit - Normalization hardware simulator.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

SCARF_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from encoder.normalization_unit import NormalizationUnit
from encoder.types import NormType, EncoderConfig, CycleStats, ResourceEstimate


class TestNormalizationUnitInit:
    """Test NormalizationUnit initialization."""
    
    def test_layernorm_init(self):
        """Test LayerNorm initialization."""
        unit = NormalizationUnit(NormType.LAYER, dim=128)
        assert unit.norm_type == NormType.LAYER
        assert unit.dim == 128
    
    def test_batchnorm_init(self):
        """Test BatchNorm initialization."""
        unit = NormalizationUnit(NormType.BATCH, dim=64)
        assert unit.norm_type == NormType.BATCH
    
    def test_instancenorm_init(self):
        """Test InstanceNorm initialization."""
        unit = NormalizationUnit(NormType.INSTANCE, dim=64)
        assert unit.norm_type == NormType.INSTANCE
    
    def test_groupnorm_init(self):
        """Test GroupNorm initialization."""
        unit = NormalizationUnit(NormType.GROUP, dim=128, num_groups=8)
        assert unit.norm_type == NormType.GROUP
        assert unit.num_groups == 8


class TestNormalizationUnitForward:
    """Test NormalizationUnit forward pass."""
    
    def test_layernorm_forward(self):
        """Test LayerNorm forward pass."""
        unit = NormalizationUnit(NormType.LAYER, dim=128)
        x = torch.randn(1, 1024, 128)
        
        out, cycles = unit.forward(x)
        
        assert out.shape == x.shape
        assert cycles.total_cycles > 0
    
    def test_batchnorm_forward(self):
        """Test BatchNorm forward pass (2D input)."""
        unit = NormalizationUnit(NormType.BATCH, dim=64)
        x = torch.randn(4, 64, 32, 32)
        
        out, cycles = unit.forward(x)
        
        assert out.shape == x.shape
        assert cycles.total_cycles > 0
    
    def test_instancenorm_forward(self):
        """Test InstanceNorm forward pass."""
        unit = NormalizationUnit(NormType.INSTANCE, dim=64)
        x = torch.randn(4, 64, 32, 32)
        
        out, cycles = unit.forward(x)
        
        assert out.shape == x.shape
        assert cycles.total_cycles > 0
    
    def test_groupnorm_forward(self):
        """Test GroupNorm forward pass."""
        unit = NormalizationUnit(NormType.GROUP, dim=128, num_groups=8)
        x = torch.randn(4, 128, 32, 32)
        
        out, cycles = unit.forward(x)
        
        assert out.shape == x.shape
        assert cycles.total_cycles > 0


class TestNormalizationUnitCorrectness:
    """Test normalization correctness."""
    
    def test_layernorm_correctness(self):
        """Test LayerNorm matches PyTorch."""
        dim = 128
        unit = NormalizationUnit(NormType.LAYER, dim=dim)
        ref_norm = nn.LayerNorm(dim)
        
        # Copy parameters
        unit.weight = ref_norm.weight.clone()
        unit.bias = ref_norm.bias.clone()
        
        x = torch.randn(1, 64, dim)
        
        out, _ = unit.forward(x)
        ref = ref_norm(x)
        
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    
    def test_groupnorm_correctness(self):
        """Test GroupNorm matches PyTorch."""
        dim = 128
        groups = 8
        unit = NormalizationUnit(NormType.GROUP, dim=dim, num_groups=groups)
        ref_norm = nn.GroupNorm(groups, dim)
        
        # Copy parameters
        unit.weight = ref_norm.weight.clone()
        unit.bias = ref_norm.bias.clone()
        
        x = torch.randn(2, dim, 16, 16)
        
        out, _ = unit.forward(x)
        ref = ref_norm(x)
        
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


class TestNormalizationUnitCycles:
    """Test cycle counting."""
    
    def test_larger_dim_more_cycles(self):
        """Test that larger dimension takes more cycles."""
        x_small = torch.randn(1, 64, 64)
        x_large = torch.randn(1, 64, 256)
        
        unit_small = NormalizationUnit(NormType.LAYER, dim=64)
        unit_large = NormalizationUnit(NormType.LAYER, dim=256)
        
        _, cycles_small = unit_small.forward(x_small)
        _, cycles_large = unit_large.forward(x_large)
        
        assert cycles_large.total_cycles > cycles_small.total_cycles
    
    def test_batchnorm_fewer_cycles(self):
        """Test that BatchNorm takes fewer cycles (uses running stats)."""
        dim = 64
        x = torch.randn(4, dim, 16, 16)
        
        unit_bn = NormalizationUnit(NormType.BATCH, dim=dim)
        unit_in = NormalizationUnit(NormType.INSTANCE, dim=dim)
        
        # Set to eval mode to use running stats
        unit_bn.eval()
        
        _, cycles_bn = unit_bn.forward(x)
        _, cycles_in = unit_in.forward(x)
        
        # BatchNorm with running stats should be faster
        assert cycles_bn.total_cycles <= cycles_in.total_cycles


class TestNormalizationUnitResources:
    """Test resource estimation."""
    
    def test_resource_estimate(self):
        """Test resource estimate."""
        unit = NormalizationUnit(NormType.LAYER, dim=128)
        resources = unit.get_resource_estimate()
        
        assert isinstance(resources, ResourceEstimate)
        assert resources.luts == 3000
        assert resources.dsps == 16
