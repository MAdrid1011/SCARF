"""
Light Verifier Tests

Tests for FSDR light verification module.
"""
import pytest
import torch
from .conftest import MockCacheEntry, MockFSDRConfig


class TestSearchRangeCalculation:
    """Tests for search range calculation."""
    
    def test_search_range_from_spread(self, num_depth_candidates):
        """Search range should be based on spread."""
        def calculate_search_range(entry, num_depths, max_radius=3):
            spread_idx = max(1, int(entry.spread * num_depths))
            spread_idx = min(spread_idx, max_radius)
            
            start_idx = max(0, entry.best_idx - spread_idx)
            end_idx = min(num_depths, entry.best_idx + spread_idx + 1)
            
            return start_idx, end_idx
        
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.4,
            second_offset=2,
            spread=0.1,  # 10% of range
        )
        
        start, end = calculate_search_range(entry, num_depth_candidates)
        
        # spread=0.1 * 32 = 3.2 → spread_idx = 3
        # start = 16 - 3 = 13
        # end = 16 + 3 + 1 = 20
        assert start == 13
        assert end == 20
        assert end - start == 7  # Search 7 candidates
    
    def test_search_range_capped_at_max_radius(self, num_depth_candidates):
        """Search radius should be capped."""
        def calculate_search_range(entry, num_depths, max_radius=3):
            spread_idx = max(1, int(entry.spread * num_depths))
            spread_idx = min(spread_idx, max_radius)
            
            start_idx = max(0, entry.best_idx - spread_idx)
            end_idx = min(num_depths, entry.best_idx + spread_idx + 1)
            
            return start_idx, end_idx
        
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.4,
            second_offset=2,
            spread=0.5,  # 50% would give 16, but capped at 3
        )
        
        start, end = calculate_search_range(entry, num_depth_candidates)
        
        # Capped at max_radius=3
        assert end - start == 7  # Not 33
    
    def test_search_range_boundary_left(self, num_depth_candidates):
        """Search range should handle left boundary."""
        def calculate_search_range(entry, num_depths, max_radius=3):
            spread_idx = max(1, int(entry.spread * num_depths))
            spread_idx = min(spread_idx, max_radius)
            
            start_idx = max(0, entry.best_idx - spread_idx)
            end_idx = min(num_depths, entry.best_idx + spread_idx + 1)
            
            return start_idx, end_idx
        
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=0.5,
            best_idx=1,  # Near left boundary
            peak_prob=0.4,
            second_offset=1,
            spread=0.1,
        )
        
        start, end = calculate_search_range(entry, num_depth_candidates)
        
        assert start == 0  # Clamped to 0
        assert end == 5  # 1 + 3 + 1
    
    def test_search_range_boundary_right(self, num_depth_candidates):
        """Search range should handle right boundary."""
        def calculate_search_range(entry, num_depths, max_radius=3):
            spread_idx = max(1, int(entry.spread * num_depths))
            spread_idx = min(spread_idx, max_radius)
            
            start_idx = max(0, entry.best_idx - spread_idx)
            end_idx = min(num_depths, entry.best_idx + spread_idx + 1)
            
            return start_idx, end_idx
        
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=10.0,
            best_idx=30,  # Near right boundary
            peak_prob=0.4,
            second_offset=-1,
            spread=0.1,
        )
        
        start, end = calculate_search_range(entry, num_depth_candidates)
        
        assert start == 27  # 30 - 3
        assert end == num_depth_candidates  # Clamped to 32
    
    def test_minimum_search_radius(self, num_depth_candidates):
        """Search should always include at least radius 1."""
        def calculate_search_range(entry, num_depths, max_radius=3):
            spread_idx = max(1, int(entry.spread * num_depths))  # min 1
            spread_idx = min(spread_idx, max_radius)
            
            start_idx = max(0, entry.best_idx - spread_idx)
            end_idx = min(num_depths, entry.best_idx + spread_idx + 1)
            
            return start_idx, end_idx
        
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.4,
            second_offset=1,
            spread=0.01,  # Very low spread
        )
        
        start, end = calculate_search_range(entry, num_depth_candidates)
        
        # Minimum radius = 1
        assert end - start >= 3  # At least 3 candidates (best_idx ± 1)


class TestLocalDepthSearch:
    """Tests for local depth search."""
    
    def test_finds_best_in_range(self, depth_candidates, mock_cost_fn):
        """Should find best depth within search range."""
        def local_search(start_idx, end_idx, depth_candidates, cost_fn):
            costs = [cost_fn(None, i) for i in range(start_idx, end_idx)]
            best_local = costs.index(min(costs))
            best_idx = start_idx + best_local
            return float(depth_candidates[best_idx]), end_idx - start_idx
        
        # Search range [10, 20)
        depth, num_searches = local_search(10, 20, depth_candidates, mock_cost_fn)
        
        assert num_searches == 10
        assert depth > 0
    
    def test_returns_search_count(self, depth_candidates, mock_cost_fn):
        """Should return number of searches performed."""
        def local_search(start_idx, end_idx, depth_candidates, cost_fn):
            costs = [cost_fn(None, i) for i in range(start_idx, end_idx)]
            best_local = costs.index(min(costs))
            best_idx = start_idx + best_local
            return float(depth_candidates[best_idx]), end_idx - start_idx
        
        depth, num_searches = local_search(5, 12, depth_candidates, mock_cost_fn)
        
        assert num_searches == 7
    
    def test_search_with_single_candidate(self, depth_candidates, mock_cost_fn):
        """Should handle single candidate in range."""
        def local_search(start_idx, end_idx, depth_candidates, cost_fn):
            if end_idx <= start_idx:
                return float(depth_candidates[start_idx]), 0
            costs = [cost_fn(None, i) for i in range(start_idx, end_idx)]
            best_local = costs.index(min(costs))
            best_idx = start_idx + best_local
            return float(depth_candidates[best_idx]), end_idx - start_idx
        
        depth, num_searches = local_search(15, 16, depth_candidates, mock_cost_fn)
        
        assert num_searches == 1


class TestVerificationFlow:
    """Tests for complete verification flow."""
    
    def test_verify_returns_depth_and_count(
        self, depth_candidates, mock_cost_fn, low_confidence_entry
    ):
        """Verify should return both depth and search count."""
        def verify(entry, cost_fn, depth_candidates, max_radius=3):
            num_depths = len(depth_candidates)
            spread_idx = max(1, int(entry.spread * num_depths))
            spread_idx = min(spread_idx, max_radius)
            
            start_idx = max(0, entry.best_idx - spread_idx)
            end_idx = min(num_depths, entry.best_idx + spread_idx + 1)
            
            costs = [cost_fn(None, i) for i in range(start_idx, end_idx)]
            best_local = costs.index(min(costs))
            best_idx = start_idx + best_local
            
            return float(depth_candidates[best_idx]), end_idx - start_idx
        
        depth, num_searches = verify(
            low_confidence_entry, mock_cost_fn, depth_candidates
        )
        
        assert depth > 0
        assert 3 <= num_searches <= 7
    
    def test_memory_savings_calculation(self, num_depth_candidates):
        """Calculate memory savings from light verify."""
        num_searches = 5
        full_search = num_depth_candidates
        
        saved = full_search - num_searches
        savings_pct = (saved / full_search) * 100
        
        # 32 - 5 = 27 saved → 84.4% savings
        assert saved == 27
        assert abs(savings_pct - 84.375) < 0.1


class TestVerifierIntegration:
    """Integration tests for verifier with DSU."""
    
    def test_verify_uses_cost_function(self, depth_candidates):
        """Verify should use provided cost function."""
        call_count = [0]
        
        def counting_cost_fn(feature, depth_idx):
            call_count[0] += 1
            return float(depth_idx) * 0.1
        
        def verify(entry, cost_fn, depth_candidates, max_radius=3):
            num_depths = len(depth_candidates)
            spread_idx = max(1, int(entry.spread * num_depths))
            spread_idx = min(spread_idx, max_radius)
            
            start_idx = max(0, entry.best_idx - spread_idx)
            end_idx = min(num_depths, entry.best_idx + spread_idx + 1)
            
            costs = [cost_fn(None, i) for i in range(start_idx, end_idx)]
            best_local = costs.index(min(costs))
            best_idx = start_idx + best_local
            
            return float(depth_candidates[best_idx]), end_idx - start_idx
        
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.4,
            second_offset=2,
            spread=0.1,
        )
        
        depth, num_searches = verify(entry, counting_cost_fn, depth_candidates)
        
        assert call_count[0] == num_searches
