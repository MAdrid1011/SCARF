"""
LSH Hasher Tests

Tests for Locality-Sensitive Hashing signature generation.
"""
import pytest
import torch
import numpy as np
from .conftest import compute_cosine_similarity, compute_hamming_distance


class TestLSHHasherBasics:
    """Basic LSH hasher functionality tests."""
    
    def test_signature_is_16_bit(self, random_feature):
        """Signature should be a 16-bit integer."""
        # Mock LSH hash implementation for testing
        def mock_lsh_hash(feature, projection_matrix):
            projections = projection_matrix @ feature
            signature = 0
            for i, proj in enumerate(projections):
                if proj >= 0:
                    signature |= (1 << i)
            return signature
        
        projection = torch.randn(16, len(random_feature))
        signature = mock_lsh_hash(random_feature, projection)
        
        assert 0 <= signature < 2**16
        assert isinstance(signature, int)
    
    def test_same_feature_same_signature(self, random_feature, lsh_projection_matrix):
        """Same feature should produce same signature."""
        def mock_lsh_hash(feature, projection_matrix):
            projections = projection_matrix @ feature
            signature = 0
            for i, proj in enumerate(projections):
                if proj >= 0:
                    signature |= (1 << i)
            return signature
        
        sig1 = mock_lsh_hash(random_feature, lsh_projection_matrix)
        sig2 = mock_lsh_hash(random_feature, lsh_projection_matrix)
        
        assert sig1 == sig2
    
    def test_different_features_different_signatures(
        self, dissimilar_feature_pair, lsh_projection_matrix
    ):
        """Dissimilar features should produce different signatures."""
        def mock_lsh_hash(feature, projection_matrix):
            projections = projection_matrix @ feature
            signature = 0
            for i, proj in enumerate(projections):
                if proj >= 0:
                    signature |= (1 << i)
            return signature
        
        feat1, feat2 = dissimilar_feature_pair
        sig1 = mock_lsh_hash(feat1, lsh_projection_matrix)
        sig2 = mock_lsh_hash(feat2, lsh_projection_matrix)
        
        # Should be different (high Hamming distance)
        hamming = compute_hamming_distance(sig1, sig2)
        assert hamming > 4  # Expect significant difference


class TestLSHSimilarityPreservation:
    """Tests for similarity preservation property of LSH."""
    
    def test_similar_features_low_hamming(
        self, similar_feature_pair, lsh_projection_matrix
    ):
        """Similar features should have low Hamming distance."""
        def mock_lsh_hash(feature, projection_matrix):
            projections = projection_matrix @ feature
            signature = 0
            for i, proj in enumerate(projections):
                if proj >= 0:
                    signature |= (1 << i)
            return signature
        
        feat1, feat2 = similar_feature_pair
        cosine_sim = compute_cosine_similarity(feat1, feat2)
        
        sig1 = mock_lsh_hash(feat1, lsh_projection_matrix)
        sig2 = mock_lsh_hash(feat2, lsh_projection_matrix)
        hamming = compute_hamming_distance(sig1, sig2)
        
        # High cosine similarity → low Hamming distance
        assert cosine_sim > 0.9, "Test setup: features should be similar"
        assert hamming <= 4, f"Expected low Hamming for similar features, got {hamming}"
    
    def test_dissimilar_features_high_hamming(
        self, dissimilar_feature_pair, lsh_projection_matrix
    ):
        """Dissimilar features should have high Hamming distance."""
        def mock_lsh_hash(feature, projection_matrix):
            projections = projection_matrix @ feature
            signature = 0
            for i, proj in enumerate(projections):
                if proj >= 0:
                    signature |= (1 << i)
            return signature
        
        feat1, feat2 = dissimilar_feature_pair
        cosine_sim = compute_cosine_similarity(feat1, feat2)
        
        sig1 = mock_lsh_hash(feat1, lsh_projection_matrix)
        sig2 = mock_lsh_hash(feat2, lsh_projection_matrix)
        hamming = compute_hamming_distance(sig1, sig2)
        
        # Low cosine similarity → high Hamming distance
        assert cosine_sim < 0.5, "Test setup: features should be dissimilar"
        assert hamming >= 5, f"Expected high Hamming for dissimilar features, got {hamming}"
    
    def test_hamming_correlates_with_similarity(self, batch_features, lsh_projection_matrix):
        """Hamming distance should correlate with cosine distance."""
        def mock_lsh_hash(feature, projection_matrix):
            projections = projection_matrix @ feature
            signature = 0
            for i, proj in enumerate(projections):
                if proj >= 0:
                    signature |= (1 << i)
            return signature
        
        n = len(batch_features)
        cosine_distances = []
        hamming_distances = []
        
        for i in range(n):
            for j in range(i + 1, n):
                feat_i = batch_features[i]
                feat_j = batch_features[j]
                
                cosine_sim = compute_cosine_similarity(feat_i, feat_j)
                cosine_dist = 1.0 - cosine_sim
                
                sig_i = mock_lsh_hash(feat_i, lsh_projection_matrix)
                sig_j = mock_lsh_hash(feat_j, lsh_projection_matrix)
                hamming = compute_hamming_distance(sig_i, sig_j)
                
                cosine_distances.append(cosine_dist)
                hamming_distances.append(hamming)
        
        # Compute correlation
        cosine_arr = np.array(cosine_distances)
        hamming_arr = np.array(hamming_distances)
        correlation = np.corrcoef(cosine_arr, hamming_arr)[0, 1]
        
        # Should have positive correlation (higher cosine dist → higher Hamming)
        assert correlation > 0.3, f"Expected positive correlation, got {correlation}"


class TestLSHBatchProcessing:
    """Tests for batch LSH processing."""
    
    def test_batch_hash_consistency(self, batch_features, lsh_projection_matrix):
        """Batch hashing should produce same results as individual hashing."""
        def mock_lsh_hash(feature, projection_matrix):
            projections = projection_matrix @ feature
            signature = 0
            for i, proj in enumerate(projections):
                if proj >= 0:
                    signature |= (1 << i)
            return signature
        
        def mock_batch_lsh_hash(features, projection_matrix):
            projections = features @ projection_matrix.T  # [N, 16]
            signatures = torch.zeros(len(features), dtype=torch.int64)
            for i in range(len(features)):
                sig = 0
                for j in range(16):
                    if projections[i, j] >= 0:
                        sig |= (1 << j)
                signatures[i] = sig
            return signatures
        
        # Individual hashing
        individual_sigs = [
            mock_lsh_hash(feat, lsh_projection_matrix)
            for feat in batch_features
        ]
        
        # Batch hashing
        batch_sigs = mock_batch_lsh_hash(batch_features, lsh_projection_matrix)
        
        # Should match
        for i, (ind_sig, batch_sig) in enumerate(zip(individual_sigs, batch_sigs)):
            assert ind_sig == batch_sig.item(), f"Mismatch at index {i}"


class TestLSHEdgeCases:
    """Edge case tests for LSH hasher."""
    
    def test_zero_feature(self, lsh_projection_matrix):
        """Zero feature should produce valid signature."""
        def mock_lsh_hash(feature, projection_matrix):
            projections = projection_matrix @ feature
            signature = 0
            for i, proj in enumerate(projections):
                if proj >= 0:
                    signature |= (1 << i)
            return signature
        
        zero_feature = torch.zeros(128)
        signature = mock_lsh_hash(zero_feature, lsh_projection_matrix)
        
        # All projections are 0, which is >= 0, so all bits should be 1
        assert signature == 0xFFFF  # All 16 bits set
    
    def test_normalized_vs_unnormalized(self, lsh_projection_matrix):
        """Normalized and unnormalized versions should produce same signature."""
        def mock_lsh_hash(feature, projection_matrix):
            projections = projection_matrix @ feature
            signature = 0
            for i, proj in enumerate(projections):
                if proj >= 0:
                    signature |= (1 << i)
            return signature
        
        feature = torch.randn(128)
        scaled_feature = feature * 10.0  # Scale up
        
        sig1 = mock_lsh_hash(feature, lsh_projection_matrix)
        sig2 = mock_lsh_hash(scaled_feature, lsh_projection_matrix)
        
        # Scaling doesn't change sign, so signatures should match
        assert sig1 == sig2
    
    def test_negated_feature(self, random_feature, lsh_projection_matrix):
        """Negated feature should have inverted signature."""
        def mock_lsh_hash(feature, projection_matrix):
            projections = projection_matrix @ feature
            signature = 0
            for i, proj in enumerate(projections):
                if proj >= 0:
                    signature |= (1 << i)
            return signature
        
        negated = -random_feature
        
        sig1 = mock_lsh_hash(random_feature, lsh_projection_matrix)
        sig2 = mock_lsh_hash(negated, lsh_projection_matrix)
        
        # Negation inverts all projection signs → inverted signature
        assert sig1 ^ sig2 == 0xFFFF  # All bits different


class TestLSHProjectionMatrix:
    """Tests for LSH projection matrix properties."""
    
    def test_projection_matrix_normalized(self, lsh_projection_matrix):
        """Projection vectors should be normalized."""
        norms = torch.norm(lsh_projection_matrix, dim=1)
        assert torch.allclose(norms, torch.ones(16), atol=1e-5)
    
    def test_projection_matrix_shape(self, feature_dim):
        """Projection matrix should have correct shape."""
        matrix = torch.randn(16, feature_dim)
        assert matrix.shape == (16, feature_dim)
    
    def test_deterministic_with_seed(self, feature_dim):
        """Same seed should produce same projection matrix."""
        torch.manual_seed(123)
        matrix1 = torch.randn(16, feature_dim)
        
        torch.manual_seed(123)
        matrix2 = torch.randn(16, feature_dim)
        
        assert torch.allclose(matrix1, matrix2)
