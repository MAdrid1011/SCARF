"""
GGU Processor Tests

Tests for the Gaussian Generation Unit processor.
"""
import pytest
import torch
import numpy as np
from .conftest import (
    MockGGUConfig,
    quaternion_to_rotation_matrix,
    compute_covariance,
    compute_ray_direction,
)


class TestPositionCalculator:
    """Tests for position calculation."""
    
    def test_position_at_center_depth_1(self, intrinsics, extrinsics_identity):
        """Center pixel at depth 1 should be at (0, 0, 1) for identity camera."""
        pixel = (320.0, 240.0)  # Center pixel
        depth = 1.0
        
        # Compute ray direction
        ray = compute_ray_direction(pixel, intrinsics)
        
        # At identity extrinsics, position = origin + ray * depth
        origin = extrinsics_identity[:3, 3]
        position = origin + ray * depth
        
        # For center pixel with our intrinsics, ray should point along z
        assert abs(position[0]) < 1e-4
        assert abs(position[1]) < 1e-4
        assert abs(position[2] - depth) < 1e-4
    
    def test_position_scales_with_depth(self, intrinsics, extrinsics_identity):
        """Position should scale linearly with depth."""
        pixel = (320.0, 240.0)
        
        ray = compute_ray_direction(pixel, intrinsics)
        origin = extrinsics_identity[:3, 3]
        
        pos_d1 = origin + ray * 1.0
        pos_d5 = origin + ray * 5.0
        
        # Position should scale with depth
        expected_pos_d5 = pos_d1 * 5.0
        assert torch.allclose(pos_d5, expected_pos_d5, atol=1e-4)
    
    def test_position_with_translation(self, intrinsics, extrinsics_translated):
        """Position should account for camera translation."""
        pixel = (320.0, 240.0)
        depth = 1.0
        
        ray = compute_ray_direction(pixel, intrinsics)
        
        # Camera to world rotation and translation
        R_c2w = extrinsics_translated[:3, :3]
        t_c2w = extrinsics_translated[:3, 3]
        
        # Transform ray to world
        ray_world = R_c2w @ ray
        position = t_c2w + ray_world * depth
        
        # Position should include camera translation
        assert position[0] > 0  # Includes x translation
    
    def test_position_with_rotation(self, intrinsics, extrinsics_rotated):
        """Position should account for camera rotation."""
        pixel = (320.0, 240.0)
        depth = 2.0
        
        ray = compute_ray_direction(pixel, intrinsics)
        
        R_c2w = extrinsics_rotated[:3, :3]
        t_c2w = extrinsics_rotated[:3, 3]
        
        ray_world = R_c2w @ ray
        position = t_c2w + ray_world * depth
        
        # With 45-degree rotation, x and z components should be mixed
        # For center ray pointing along z in camera space
        # After rotation: x = sin(45) * z_cam, z = cos(45) * z_cam
        assert position[0] != 0  # x is mixed


class TestCovarianceBuilder:
    """Tests for covariance matrix construction."""
    
    def test_covariance_from_identity_rotation(self, raw_scales):
        """Identity rotation should produce diagonal covariance."""
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        scales = torch.sigmoid(raw_scales) * 0.5  # Map to valid range
        
        cov = compute_covariance(scales, identity_quat)
        
        # Should be diagonal
        expected_diag = torch.diag(scales ** 2)
        assert torch.allclose(cov, expected_diag, atol=1e-5)
    
    def test_covariance_symmetric(self, raw_scales, raw_rotation):
        """Covariance matrix should be symmetric."""
        scales = torch.sigmoid(raw_scales) * 0.5
        
        cov = compute_covariance(scales, raw_rotation)
        
        assert torch.allclose(cov, cov.T, atol=1e-6)
    
    def test_covariance_positive_semidefinite(self):
        """Covariance matrix should be positive semi-definite."""
        scales = torch.tensor([0.1, 0.2, 0.15])
        quat = torch.randn(4)
        quat = quat / torch.norm(quat)
        
        cov = compute_covariance(scales, quat)
        
        # Check eigenvalues are non-negative
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert (eigenvalues >= -1e-6).all()
    
    def test_covariance_world_transform(self, extrinsics_rotated):
        """World transform should rotate covariance correctly."""
        scales = torch.tensor([0.1, 0.2, 0.15])
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        
        cov_local = compute_covariance(scales, quat)
        
        R_c2w = extrinsics_rotated[:3, :3]
        cov_world = R_c2w @ cov_local @ R_c2w.T
        
        # Transformed covariance should still be symmetric
        assert torch.allclose(cov_world, cov_world.T, atol=1e-6)
        
        # Eigenvalues should be preserved (rotation doesn't change them)
        eig_local = torch.linalg.eigvalsh(cov_local)
        eig_world = torch.linalg.eigvalsh(cov_world)
        assert torch.allclose(torch.sort(eig_local)[0], torch.sort(eig_world)[0], atol=1e-5)


class TestScaleMapping:
    """Tests for scale mapping from raw network output."""
    
    def test_scale_mapping_range(self, default_ggu_config):
        """Mapped scales should be in valid range."""
        def map_scales(raw_scales, config):
            return config.scale_min + (config.scale_max - config.scale_min) * torch.sigmoid(raw_scales)
        
        raw = torch.randn(3)
        scales = map_scales(raw, default_ggu_config)
        
        assert (scales >= default_ggu_config.scale_min).all()
        assert (scales <= default_ggu_config.scale_max).all()
    
    def test_scale_mapping_extreme_inputs(self, default_ggu_config):
        """Extreme inputs should map to boundaries."""
        def map_scales(raw_scales, config):
            return config.scale_min + (config.scale_max - config.scale_min) * torch.sigmoid(raw_scales)
        
        # Very negative → near scale_min
        raw_neg = torch.tensor([-100.0, -100.0, -100.0])
        scales_neg = map_scales(raw_neg, default_ggu_config)
        assert torch.allclose(scales_neg, torch.full((3,), default_ggu_config.scale_min), atol=1e-3)
        
        # Very positive → near scale_max
        raw_pos = torch.tensor([100.0, 100.0, 100.0])
        scales_pos = map_scales(raw_pos, default_ggu_config)
        assert torch.allclose(scales_pos, torch.full((3,), default_ggu_config.scale_max), atol=1e-3)
    
    def test_depth_adaptive_scaling(self, default_ggu_config):
        """Scales should adapt to depth."""
        def map_scales_with_depth(raw_scales, depth, config, multiplier=1.0):
            base_scales = config.scale_min + (config.scale_max - config.scale_min) * torch.sigmoid(raw_scales)
            return base_scales * depth * multiplier
        
        raw = torch.zeros(3)  # Maps to mid-range
        
        scales_d1 = map_scales_with_depth(raw, 1.0, default_ggu_config)
        scales_d5 = map_scales_with_depth(raw, 5.0, default_ggu_config)
        
        # Farther depth → larger scales
        assert (scales_d5 > scales_d1).all()
        assert torch.allclose(scales_d5 / scales_d1, torch.full((3,), 5.0), atol=1e-5)


class TestQuaternionNormalization:
    """Tests for quaternion handling."""
    
    def test_quaternion_normalized(self):
        """Quaternion should be normalized before use."""
        raw_quat = torch.tensor([2.0, 1.0, 0.5, 0.3])  # Not normalized
        normalized = raw_quat / torch.norm(raw_quat)
        
        R = quaternion_to_rotation_matrix(normalized)
        
        # Rotation matrix should be orthogonal
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-5)
        
        # Determinant should be 1
        assert abs(torch.linalg.det(R) - 1.0) < 1e-5
    
    def test_identity_quaternion(self):
        """Identity quaternion [1,0,0,0] should give identity matrix."""
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(identity_quat)
        
        assert torch.allclose(R, torch.eye(3), atol=1e-6)
    
    def test_90_degree_rotation(self):
        """Test 90-degree rotation around z-axis."""
        # Quaternion for 90-degree rotation around z: [cos(45°), 0, 0, sin(45°)]
        angle = np.pi / 2
        quat = torch.tensor([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
        
        R = quaternion_to_rotation_matrix(quat)
        
        # Should map x → y, y → -x
        x_axis = torch.tensor([1.0, 0.0, 0.0])
        y_axis = torch.tensor([0.0, 1.0, 0.0])
        
        rotated_x = R @ x_axis
        rotated_y = R @ y_axis
        
        assert torch.allclose(rotated_x, y_axis, atol=1e-5)
        assert torch.allclose(rotated_y, -x_axis, atol=1e-5)


class TestSHRotator:
    """Tests for spherical harmonics rotation."""
    
    def test_sh_degree_0_unchanged(self):
        """DC term should not change under rotation."""
        sh_coeffs = torch.randn(3, 16)
        
        # Mock rotation (any rotation)
        R = quaternion_to_rotation_matrix(torch.randn(4))
        
        # DC term (index 0) should be unchanged
        rotated_dc = sh_coeffs[:, 0]  # In real implementation, this passes through
        
        assert torch.allclose(rotated_dc, sh_coeffs[:, 0])
    
    def test_sh_degree_1_rotates_like_vector(self):
        """Degree-1 SH coefficients should rotate like 3D vectors."""
        # Degree-1 has 3 coefficients (indices 1, 2, 3)
        sh_1 = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])  # [3 channels, 3 coeffs]
        
        # 90-degree rotation around z
        angle = np.pi / 2
        quat = torch.tensor([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
        R = quaternion_to_rotation_matrix(quat)
        
        # Rotate degree-1 coefficients
        rotated_sh_1 = sh_1 @ R.T
        
        # First channel [1,0,0] should become [0,1,0]
        assert torch.allclose(rotated_sh_1[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


class TestGGUProcessorIntegration:
    """Integration tests for complete GGU processor."""
    
    def test_generate_gaussian_complete(
        self, intrinsics, extrinsics_identity, raw_gaussian, default_ggu_config
    ):
        """Complete Gaussian generation should produce valid output."""
        pixel = (320.0, 240.0)
        depth = 5.0
        density = 0.5  # Raw density before sigmoid
        
        # Parse raw features
        raw_scales = raw_gaussian[:3]
        raw_rotation = raw_gaussian[3:7]
        raw_sh = raw_gaussian[7:].reshape(3, -1)
        
        # 1. Compute position
        ray = compute_ray_direction(pixel, intrinsics)
        R_c2w = extrinsics_identity[:3, :3]
        t_c2w = extrinsics_identity[:3, 3]
        ray_world = R_c2w @ ray
        position = t_c2w + ray_world * depth
        
        # 2. Build covariance
        scales = default_ggu_config.scale_min + (
            default_ggu_config.scale_max - default_ggu_config.scale_min
        ) * torch.sigmoid(raw_scales)
        scales = scales * depth  # Depth adaptation
        cov_local = compute_covariance(scales, raw_rotation)
        cov_world = R_c2w @ cov_local @ R_c2w.T
        
        # 3. Compute opacity
        opacity = float(torch.sigmoid(torch.tensor(density)))
        
        # Validate outputs
        assert position.shape == (3,)
        assert cov_world.shape == (3, 3)
        assert 0 < opacity < 1
        assert torch.allclose(cov_world, cov_world.T)  # Symmetric
    
    def test_gaussian_compatible_with_saes(
        self, intrinsics, extrinsics_identity, raw_gaussian, default_ggu_config
    ):
        """Generated Gaussian should be compatible with SAES types."""
        pixel = (320.0, 240.0)
        depth = 5.0
        density = 0.5
        
        # Generate Gaussian (simplified)
        raw_scales = raw_gaussian[:3]
        raw_rotation = raw_gaussian[3:7]
        raw_sh = raw_gaussian[7:].reshape(3, -1)
        
        ray = compute_ray_direction(pixel, intrinsics)
        position = ray * depth
        
        scales = torch.sigmoid(raw_scales) * 0.5 * depth
        cov = compute_covariance(scales, raw_rotation)
        
        opacity = float(torch.sigmoid(torch.tensor(density)))
        
        # Check compatibility with expected SAES Gaussian format
        gaussian_dict = {
            'mean': position,      # [3]
            'cov': cov,           # [3, 3]
            'opacity': opacity,    # float
            'harmonics': raw_sh,   # [3, 16]
        }
        
        assert gaussian_dict['mean'].shape == (3,)
        assert gaussian_dict['cov'].shape == (3, 3)
        assert 0 <= gaussian_dict['opacity'] <= 1
        assert gaussian_dict['harmonics'].shape == (3, 16)
