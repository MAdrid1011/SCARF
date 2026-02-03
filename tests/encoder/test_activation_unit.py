"""
Tests for ActivationUnit - Activation function hardware simulator.
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

SCARF_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from encoder.activation_unit import ActivationUnit
from encoder.types import ActivationType, EncoderConfig, CycleStats, ResourceEstimate


class TestActivationUnitInit:
    """Test ActivationUnit initialization."""
    
    def test_relu_init(self):
        """Test ReLU initialization."""
        unit = ActivationUnit(ActivationType.RELU)
        assert unit.activation_type == ActivationType.RELU
    
    def test_gelu_init(self):
        """Test GELU initialization."""
        unit = ActivationUnit(ActivationType.GELU)
        assert unit.activation_type == ActivationType.GELU
    
    def test_silu_init(self):
        """Test SiLU initialization."""
        unit = ActivationUnit(ActivationType.SILU)
        assert unit.activation_type == ActivationType.SILU
    
    def test_sigmoid_init(self):
        """Test Sigmoid initialization."""
        unit = ActivationUnit(ActivationType.SIGMOID)
        assert unit.activation_type == ActivationType.SIGMOID


class TestActivationUnitForward:
    """Test ActivationUnit forward pass."""
    
    def test_relu_forward(self):
        """Test ReLU forward pass."""
        unit = ActivationUnit(ActivationType.RELU)
        x = torch.randn(1, 1024, 128)
        
        out, cycles = unit.forward(x)
        
        assert out.shape == x.shape
        assert cycles.total_cycles > 0
        # ReLU: all negative values should be 0
        assert (out >= 0).all()
    
    def test_gelu_forward(self):
        """Test GELU forward pass."""
        unit = ActivationUnit(ActivationType.GELU)
        x = torch.randn(1, 1024, 128)
        
        out, cycles = unit.forward(x)
        
        assert out.shape == x.shape
        assert cycles.total_cycles > 0
    
    def test_silu_forward(self):
        """Test SiLU forward pass."""
        unit = ActivationUnit(ActivationType.SILU)
        x = torch.randn(1, 1024, 128)
        
        out, cycles = unit.forward(x)
        
        assert out.shape == x.shape
        assert cycles.total_cycles > 0
    
    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        unit = ActivationUnit(ActivationType.SIGMOID)
        x = torch.randn(1, 1024, 128)
        
        out, cycles = unit.forward(x)
        
        assert out.shape == x.shape
        assert cycles.total_cycles > 0
        # Sigmoid: output should be in (0, 1)
        assert (out > 0).all() and (out < 1).all()


class TestActivationUnitCorrectness:
    """Test activation correctness."""
    
    def test_relu_correctness(self):
        """Test ReLU matches PyTorch."""
        unit = ActivationUnit(ActivationType.RELU)
        x = torch.randn(32, 128)
        
        out, _ = unit.forward(x)
        ref = F.relu(x)
        
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    
    def test_gelu_correctness(self):
        """Test GELU approximately matches PyTorch."""
        unit = ActivationUnit(ActivationType.GELU)
        x = torch.randn(32, 128)
        
        out, _ = unit.forward(x)
        ref = F.gelu(x)
        
        # Allow slightly larger tolerance for LUT approximation
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    
    def test_silu_correctness(self):
        """Test SiLU approximately matches PyTorch."""
        unit = ActivationUnit(ActivationType.SILU)
        x = torch.randn(32, 128)
        
        out, _ = unit.forward(x)
        ref = F.silu(x)
        
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    
    def test_sigmoid_correctness(self):
        """Test Sigmoid approximately matches PyTorch."""
        unit = ActivationUnit(ActivationType.SIGMOID)
        x = torch.randn(32, 128)
        
        out, _ = unit.forward(x)
        ref = torch.sigmoid(x)
        
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


class TestActivationUnitCycles:
    """Test cycle counting."""
    
    def test_all_activations_1_cycle(self):
        """Test that all activations take 1 cycle per element (LUT-based)."""
        x = torch.randn(1024)
        
        for act_type in ActivationType:
            unit = ActivationUnit(act_type)
            _, cycles = unit.forward(x)
            # Should be approximately 1 cycle per element
            assert cycles.total_cycles >= len(x.flatten())


class TestActivationUnitResources:
    """Test resource estimation."""
    
    def test_resource_estimate(self):
        """Test resource estimate."""
        unit = ActivationUnit(ActivationType.GELU)
        resources = unit.get_resource_estimate()
        
        assert isinstance(resources, ResourceEstimate)
        assert resources.luts == 2000
        assert resources.dsps == 8
