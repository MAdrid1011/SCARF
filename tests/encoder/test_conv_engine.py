"""
Tests for ConvEngine - Convolution hardware simulator.
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

SCARF_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from encoder.conv_engine import ConvEngine
from encoder.types import ConvConfig, CycleStats, ResourceEstimate


class TestConvEngineInit:
    """Test ConvEngine initialization."""
    
    def test_default_config(self):
        """Test initialization with default config."""
        engine = ConvEngine()
        assert engine.config.pe_array_size == 16
    
    def test_custom_config(self, small_conv_config):
        """Test initialization with custom config."""
        engine = ConvEngine(small_conv_config)
        assert engine.config.pe_array_size == 4


class TestConvEngineForward:
    """Test ConvEngine forward pass."""
    
    def test_conv_1x1(self, random_tensor_2d):
        """Test 1x1 convolution."""
        engine = ConvEngine()
        x = random_tensor_2d(1, 64, 32, 32)
        weight = torch.randn(128, 64, 1, 1)
        
        out, cycles = engine.forward(x, weight)
        
        # Check output shape
        assert out.shape == (1, 128, 32, 32)
        # Check cycles are tracked
        assert cycles.total_cycles > 0
    
    def test_conv_3x3(self, random_tensor_2d):
        """Test 3x3 convolution."""
        engine = ConvEngine()
        x = random_tensor_2d(1, 64, 32, 32)
        weight = torch.randn(128, 64, 3, 3)
        
        out, cycles = engine.forward(x, weight, padding=1)
        
        assert out.shape == (1, 128, 32, 32)
        assert cycles.total_cycles > 0
    
    def test_conv_7x7_stride2(self, random_tensor_2d):
        """Test 7x7 convolution with stride 2."""
        engine = ConvEngine()
        x = random_tensor_2d(1, 3, 256, 256)
        weight = torch.randn(64, 3, 7, 7)
        
        out, cycles = engine.forward(x, weight, stride=2, padding=3)
        
        assert out.shape == (1, 64, 128, 128)
        assert cycles.total_cycles > 0
    
    def test_conv_correctness(self, random_tensor_2d):
        """Test convolution matches PyTorch reference."""
        engine = ConvEngine()
        x = random_tensor_2d(1, 64, 16, 16)
        weight = torch.randn(128, 64, 3, 3)
        
        out, _ = engine.forward(x, weight, padding=1)
        ref = F.conv2d(x, weight, padding=1)
        
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


class TestConvEngineCycles:
    """Test cycle counting."""
    
    def test_larger_input_more_cycles(self):
        """Test that larger input takes more cycles."""
        engine = ConvEngine()
        weight = torch.randn(64, 64, 3, 3)
        
        x_small = torch.randn(1, 64, 16, 16)
        x_large = torch.randn(1, 64, 32, 32)
        
        _, cycles_small = engine.forward(x_small, weight, padding=1)
        _, cycles_large = engine.forward(x_large, weight, padding=1)
        
        assert cycles_large.total_cycles > cycles_small.total_cycles
    
    def test_more_channels_more_cycles(self):
        """Test that more channels take more cycles."""
        engine = ConvEngine()
        x = torch.randn(1, 64, 16, 16)
        
        weight_small = torch.randn(64, 64, 3, 3)
        weight_large = torch.randn(256, 64, 3, 3)
        
        _, cycles_small = engine.forward(x, weight_small, padding=1)
        _, cycles_large = engine.forward(x, weight_large, padding=1)
        
        assert cycles_large.total_cycles > cycles_small.total_cycles
    
    def test_cycle_breakdown(self, random_tensor_2d):
        """Test cycle breakdown is provided."""
        engine = ConvEngine()
        x = random_tensor_2d(1, 64, 16, 16)
        weight = torch.randn(128, 64, 3, 3)
        
        _, cycles = engine.forward(x, weight, padding=1)
        
        assert 'compute' in cycles.breakdown or cycles.compute_cycles > 0


class TestConvEngineResources:
    """Test resource estimation."""
    
    def test_resource_estimate(self):
        """Test resource estimate within bounds."""
        engine = ConvEngine()
        resources = engine.get_resource_estimate()
        
        assert isinstance(resources, ResourceEstimate)
        assert resources.luts == 50000
        assert resources.dsps == 256
        assert resources.sram_bytes == 65536


class TestConvEngineConfigs:
    """Test different configurations."""
    
    def test_small_pe_array(self, small_conv_config, random_tensor_2d):
        """Test with smaller PE array (more cycles)."""
        engine_small = ConvEngine(small_conv_config)
        engine_large = ConvEngine(ConvConfig(pe_array_size=16))
        
        x = random_tensor_2d(1, 64, 16, 16)
        weight = torch.randn(128, 64, 3, 3)
        
        _, cycles_small = engine_small.forward(x, weight, padding=1)
        _, cycles_large = engine_large.forward(x, weight, padding=1)
        
        # Smaller array should take more cycles
        assert cycles_small.total_cycles >= cycles_large.total_cycles
