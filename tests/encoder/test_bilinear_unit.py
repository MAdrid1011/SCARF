"""
Tests for BilinearUnit - Bilinear interpolation hardware simulator.
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

SCARF_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from encoder.bilinear_unit import BilinearUnit
from encoder.types import BilinearConfig, CycleStats, ResourceEstimate


class TestBilinearUnitInit:
    """Test BilinearUnit initialization."""
    
    def test_default_config(self):
        """Test initialization with default config."""
        unit = BilinearUnit()
        assert unit.config.parallel_channels == 8
        assert unit.config.align_corners == True
    
    def test_custom_config(self):
        """Test initialization with custom config."""
        config = BilinearConfig(parallel_channels=4, align_corners=False)
        unit = BilinearUnit(config)
        assert unit.config.parallel_channels == 4
        assert unit.config.align_corners == False


class TestBilinearUnitInterpolate:
    """Test BilinearUnit interpolation."""
    
    def test_2x_upscale(self):
        """Test 2x upsampling."""
        unit = BilinearUnit()
        x = torch.randn(1, 64, 32, 32)
        
        out, cycles = unit.interpolate(x, scale_factor=2)
        
        assert out.shape == (1, 64, 64, 64)
        assert cycles.total_cycles > 0
    
    def test_4x_upscale(self):
        """Test 4x upsampling."""
        unit = BilinearUnit()
        x = torch.randn(1, 128, 16, 16)
        
        out, cycles = unit.interpolate(x, scale_factor=4)
        
        assert out.shape == (1, 128, 64, 64)
        assert cycles.total_cycles > 0
    
    def test_target_size(self):
        """Test interpolation to specific size."""
        unit = BilinearUnit()
        x = torch.randn(1, 64, 32, 32)
        
        out, cycles = unit.interpolate(x, size=(64, 64))
        
        assert out.shape == (1, 64, 64, 64)
        assert cycles.total_cycles > 0
    
    def test_non_square(self):
        """Test non-square interpolation."""
        unit = BilinearUnit()
        x = torch.randn(1, 64, 16, 32)
        
        out, cycles = unit.interpolate(x, size=(32, 64))
        
        assert out.shape == (1, 64, 32, 64)
        assert cycles.total_cycles > 0


class TestBilinearUnitCorrectness:
    """Test interpolation correctness."""
    
    def test_2x_upscale_correctness(self):
        """Test 2x upscale matches PyTorch."""
        unit = BilinearUnit(BilinearConfig(align_corners=True))
        x = torch.randn(1, 32, 16, 16)
        
        out, _ = unit.interpolate(x, scale_factor=2)
        ref = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    
    def test_4x_upscale_correctness(self):
        """Test 4x upscale matches PyTorch."""
        unit = BilinearUnit(BilinearConfig(align_corners=True))
        x = torch.randn(1, 32, 8, 8)
        
        out, _ = unit.interpolate(x, scale_factor=4)
        ref = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    
    def test_align_corners_false(self):
        """Test with align_corners=False."""
        unit = BilinearUnit(BilinearConfig(align_corners=False))
        x = torch.randn(1, 32, 16, 16)
        
        out, _ = unit.interpolate(x, scale_factor=2)
        ref = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


class TestBilinearUnitCycles:
    """Test cycle counting."""
    
    def test_larger_output_more_cycles(self):
        """Test that larger output takes more cycles."""
        unit = BilinearUnit()
        x = torch.randn(1, 64, 16, 16)
        
        _, cycles_2x = unit.interpolate(x, scale_factor=2)
        _, cycles_4x = unit.interpolate(x, scale_factor=4)
        
        # 4x has 4x more output pixels
        assert cycles_4x.total_cycles > cycles_2x.total_cycles
    
    def test_more_channels_more_cycles(self):
        """Test that more channels take more cycles."""
        unit = BilinearUnit()
        
        x_small = torch.randn(1, 32, 16, 16)
        x_large = torch.randn(1, 128, 16, 16)
        
        _, cycles_small = unit.interpolate(x_small, scale_factor=2)
        _, cycles_large = unit.interpolate(x_large, scale_factor=2)
        
        assert cycles_large.total_cycles > cycles_small.total_cycles


class TestBilinearUnitResources:
    """Test resource estimation."""
    
    def test_resource_estimate(self):
        """Test resource estimate."""
        unit = BilinearUnit()
        resources = unit.get_resource_estimate()
        
        assert isinstance(resources, ResourceEstimate)
        assert resources.luts == 800
        assert resources.dsps == 8
        assert resources.sram_bytes == 0
