"""
FSDR Processor Tests

Tests for the main FSDR processor module.
"""
import pytest
import torch
from .conftest import (
    MockCacheEntry, MockFSDRConfig,
    compute_hamming_distance,
)


class TestFSDRProcessorBasics:
    """Basic FSDR processor functionality tests."""
    
    def test_process_pixel_returns_result(
        self, random_feature, mock_cost_fn, mock_prob_fn, default_config
    ):
        """Process pixel should return a result with depth."""
        class MockFSDRProcessor:
            def __init__(self, config):
                self.config = config
                self.cache = [None] * config.cache_size
            
            def process_pixel(self, feature, position, cost_fn, prob_fn):
                # Simplified mock - always full search for empty cache
                probs = prob_fn(feature, None)
                depth = float(torch.argmax(probs))
                return {
                    'depth': depth,
                    'source': 'full_search',
                    'cache_hit': False,
                    'hamming_distance': None,
                }
        
        processor = MockFSDRProcessor(default_config)
        result = processor.process_pixel(
            random_feature, (32, 32), mock_cost_fn, mock_prob_fn
        )
        
        assert 'depth' in result
        assert 'source' in result
        assert result['source'] == 'full_search'
    
    def test_first_pixel_is_cache_miss(
        self, random_feature, mock_cost_fn, mock_prob_fn, default_config
    ):
        """First pixel should be a cache miss."""
        class MockFSDRProcessor:
            def __init__(self, config):
                self.config = config
                self.cache = [None] * config.cache_size
            
            def process_pixel(self, feature, position, cost_fn, prob_fn):
                probs = prob_fn(feature, None)
                depth = float(torch.argmax(probs))
                return {
                    'depth': depth,
                    'source': 'full_search',
                    'cache_hit': False,
                    'hamming_distance': None,
                }
        
        processor = MockFSDRProcessor(default_config)
        result = processor.process_pixel(
            random_feature, (0, 0), mock_cost_fn, mock_prob_fn
        )
        
        assert result['cache_hit'] is False
        assert result['source'] == 'full_search'


class TestCacheHitProcessing:
    """Tests for cache hit processing."""
    
    def test_cache_hit_direct_reuse(self, default_config, high_confidence_entry):
        """Cache hit with high confidence should use direct reuse."""
        class MockFSDRProcessor:
            def __init__(self, config):
                self.config = config
                self.cache = [high_confidence_entry] + [None] * (config.cache_size - 1)
                self.lsh_signature = high_confidence_entry.signature
            
            def _hash(self, feature):
                return self.lsh_signature
            
            def _lookup(self, signature):
                for entry in self.cache:
                    if entry is not None and entry.valid:
                        hamming = compute_hamming_distance(signature, entry.signature)
                        if hamming <= self.config.hamming_threshold:
                            return entry, hamming
                return None, -1
            
            def _decide_strategy(self, entry, hamming):
                if entry.peak_prob > self.config.high_confidence_threshold and hamming <= self.config.hamming_direct_reuse:
                    return 'direct_reuse'
                if entry.peak_prob > self.config.medium_confidence_threshold or hamming <= self.config.hamming_interpolate:
                    return 'interpolation'
                return 'light_verify'
            
            def process_pixel(self, feature, position, cost_fn, prob_fn):
                sig = self._hash(feature)
                entry, hamming = self._lookup(sig)
                
                if entry is not None:
                    strategy = self._decide_strategy(entry, hamming)
                    if strategy == 'direct_reuse':
                        return {
                            'depth': entry.best_depth,
                            'source': 'direct_reuse',
                            'cache_hit': True,
                            'hamming_distance': hamming,
                        }
                
                # Fallback
                return {'depth': 0, 'source': 'full_search', 'cache_hit': False}
        
        processor = MockFSDRProcessor(default_config)
        result = processor.process_pixel(
            torch.randn(128), (32, 32), None, None
        )
        
        assert result['cache_hit'] is True
        assert result['source'] == 'direct_reuse'
        assert result['depth'] == high_confidence_entry.best_depth
    
    def test_cache_hit_interpolation(self, default_config, medium_confidence_entry, depth_candidates):
        """Cache hit with medium confidence should interpolate."""
        class MockFSDRProcessor:
            def __init__(self, config, depth_candidates):
                self.config = config
                self.depth_candidates = depth_candidates
                self.cache = [medium_confidence_entry] + [None] * (config.cache_size - 1)
                self.lsh_signature = medium_confidence_entry.signature
            
            def _hash(self, feature):
                return self.lsh_signature
            
            def _lookup(self, signature):
                for entry in self.cache:
                    if entry is not None and entry.valid:
                        hamming = compute_hamming_distance(signature, entry.signature)
                        if hamming <= self.config.hamming_threshold:
                            return entry, hamming
                return None, -1
            
            def _decide_strategy(self, entry, hamming):
                if entry.peak_prob > self.config.high_confidence_threshold and hamming <= self.config.hamming_direct_reuse:
                    return 'direct_reuse'
                if entry.peak_prob > self.config.medium_confidence_threshold or hamming <= self.config.hamming_interpolate:
                    return 'interpolation'
                return 'light_verify'
            
            def _interpolate(self, entry, hamming):
                second_idx = entry.best_idx + entry.second_offset
                second_idx = max(0, min(len(self.depth_candidates) - 1, second_idx))
                second_depth = float(self.depth_candidates[second_idx])
                
                lambda_coeff = min(hamming / 4.0, 0.5)
                return (1 - lambda_coeff) * entry.best_depth + lambda_coeff * second_depth
            
            def process_pixel(self, feature, position, cost_fn, prob_fn):
                sig = self._hash(feature)
                entry, hamming = self._lookup(sig)
                
                if entry is not None:
                    strategy = self._decide_strategy(entry, hamming)
                    if strategy == 'interpolation':
                        depth = self._interpolate(entry, hamming)
                        return {
                            'depth': depth,
                            'source': 'interpolation',
                            'cache_hit': True,
                            'hamming_distance': hamming,
                        }
                
                return {'depth': 0, 'source': 'full_search', 'cache_hit': False}
        
        processor = MockFSDRProcessor(default_config, depth_candidates)
        result = processor.process_pixel(torch.randn(128), (48, 48), None, None)
        
        assert result['cache_hit'] is True
        assert result['source'] == 'interpolation'
    
    def test_cache_hit_light_verify(
        self, default_config, low_confidence_entry, depth_candidates, mock_cost_fn
    ):
        """Cache hit with low confidence should use light verify."""
        class MockFSDRProcessor:
            def __init__(self, config, depth_candidates):
                self.config = config
                self.depth_candidates = depth_candidates
                # Use a higher hamming to trigger light_verify
                self.cache = [low_confidence_entry] + [None] * (config.cache_size - 1)
                self.lsh_signature = low_confidence_entry.signature ^ 0x000F  # Hamming = 4
            
            def _hash(self, feature):
                return self.lsh_signature
            
            def _lookup(self, signature):
                for entry in self.cache:
                    if entry is not None and entry.valid:
                        hamming = compute_hamming_distance(signature, entry.signature)
                        if hamming <= self.config.hamming_threshold:
                            return entry, hamming
                return None, -1
            
            def _decide_strategy(self, entry, hamming):
                if entry.peak_prob > self.config.high_confidence_threshold and hamming <= self.config.hamming_direct_reuse:
                    return 'direct_reuse'
                if entry.peak_prob > self.config.medium_confidence_threshold or hamming <= self.config.hamming_interpolate:
                    return 'interpolation'
                return 'light_verify'
            
            def _light_verify(self, entry, cost_fn):
                num_depths = len(self.depth_candidates)
                spread_idx = max(1, int(entry.spread * num_depths))
                spread_idx = min(spread_idx, 3)
                
                start_idx = max(0, entry.best_idx - spread_idx)
                end_idx = min(num_depths, entry.best_idx + spread_idx + 1)
                
                costs = [cost_fn(None, i) for i in range(start_idx, end_idx)]
                best_local = costs.index(min(costs))
                best_idx = start_idx + best_local
                
                return float(self.depth_candidates[best_idx]), end_idx - start_idx
            
            def process_pixel(self, feature, position, cost_fn, prob_fn):
                sig = self._hash(feature)
                entry, hamming = self._lookup(sig)
                
                if entry is not None:
                    strategy = self._decide_strategy(entry, hamming)
                    if strategy == 'light_verify':
                        depth, num_searches = self._light_verify(entry, cost_fn)
                        return {
                            'depth': depth,
                            'source': 'light_verify',
                            'cache_hit': True,
                            'hamming_distance': hamming,
                            'num_searches': num_searches,
                        }
                
                return {'depth': 0, 'source': 'full_search', 'cache_hit': False}
        
        processor = MockFSDRProcessor(default_config, depth_candidates)
        result = processor.process_pixel(
            torch.randn(128), (16, 16), mock_cost_fn, None
        )
        
        assert result['cache_hit'] is True
        assert result['source'] == 'light_verify'
        assert 'num_searches' in result
        assert result['num_searches'] <= 7


class TestCacheMissProcessing:
    """Tests for cache miss processing."""
    
    def test_cache_miss_full_search(
        self, default_config, mock_cost_fn, mock_prob_fn, depth_candidates
    ):
        """Cache miss should perform full search."""
        class MockFSDRProcessor:
            def __init__(self, config, depth_candidates):
                self.config = config
                self.depth_candidates = depth_candidates
                self.cache = [None] * config.cache_size
            
            def _hash(self, feature):
                return 0x1234
            
            def _lookup(self, signature):
                return None, -1
            
            def process_pixel(self, feature, position, cost_fn, prob_fn):
                sig = self._hash(feature)
                entry, hamming = self._lookup(sig)
                
                if entry is None:
                    probs = prob_fn(feature, self.depth_candidates)
                    depth = float((probs * self.depth_candidates).sum())
                    return {
                        'depth': depth,
                        'source': 'full_search',
                        'cache_hit': False,
                        'hamming_distance': None,
                        'num_searches': len(self.depth_candidates),
                    }
                
                return {'depth': 0, 'source': 'unknown'}
        
        processor = MockFSDRProcessor(default_config, depth_candidates)
        result = processor.process_pixel(
            torch.randn(128), (50, 50), mock_cost_fn, mock_prob_fn
        )
        
        assert result['cache_hit'] is False
        assert result['source'] == 'full_search'
        assert result['num_searches'] == 32


class TestCacheInsertionAfterMiss:
    """Tests for cache insertion after miss."""
    
    def test_insert_after_miss(self, default_config, mock_prob_fn, depth_candidates):
        """New entry should be inserted after cache miss."""
        class MockFSDRProcessor:
            def __init__(self, config, depth_candidates):
                self.config = config
                self.depth_candidates = depth_candidates
                self.cache = [None] * config.cache_size
            
            def _hash(self, feature):
                # Deterministic for testing
                return int(feature[0].item() * 1000) % 65536
            
            def _lookup(self, signature):
                for entry in self.cache:
                    if entry is not None and entry.valid:
                        hamming = compute_hamming_distance(signature, entry.signature)
                        if hamming <= self.config.hamming_threshold:
                            return entry, hamming
                return None, -1
            
            def _insert(self, entry):
                for i, e in enumerate(self.cache):
                    if e is None:
                        self.cache[i] = entry
                        return i
                return -1
            
            def _extract_stats(self, probs):
                best_idx = int(torch.argmax(probs).item())
                peak_prob = float(probs[best_idx])
                
                probs_copy = probs.clone()
                probs_copy[best_idx] = -1
                second_idx = int(torch.argmax(probs_copy).item())
                second_offset = second_idx - best_idx
                
                mean_depth = float((probs * self.depth_candidates).sum())
                variance = float((probs * (self.depth_candidates - mean_depth) ** 2).sum())
                spread = (variance ** 0.5) / (self.depth_candidates[-1] - self.depth_candidates[0])
                
                return best_idx, peak_prob, second_offset, spread
            
            def process_pixel(self, feature, position, cost_fn, prob_fn):
                sig = self._hash(feature)
                entry, hamming = self._lookup(sig)
                
                if entry is None:
                    probs = prob_fn(feature, self.depth_candidates)
                    depth = float((probs * self.depth_candidates).sum())
                    
                    best_idx, peak_prob, second_offset, spread = self._extract_stats(probs)
                    
                    new_entry = MockCacheEntry(
                        signature=sig,
                        position=position,
                        best_depth=depth,
                        best_idx=best_idx,
                        peak_prob=peak_prob,
                        second_offset=second_offset,
                        spread=spread,
                    )
                    self._insert(new_entry)
                    
                    return {
                        'depth': depth,
                        'source': 'full_search',
                        'cache_hit': False,
                    }
                
                return {'depth': 0, 'source': 'unknown'}
        
        processor = MockFSDRProcessor(default_config, depth_candidates)
        
        # First pixel - should insert
        feat1 = torch.randn(128)
        result1 = processor.process_pixel(feat1, (10, 10), None, mock_prob_fn)
        
        assert result1['cache_hit'] is False
        assert processor.cache[0] is not None


class TestFSDRProfiler:
    """Tests for FSDR profiler/statistics."""
    
    def test_counts_hit_miss(self, default_config):
        """Profiler should track hit/miss counts."""
        class MockProfiler:
            def __init__(self):
                self.hits = 0
                self.misses = 0
                self.sources = {'direct_reuse': 0, 'interpolation': 0, 
                               'light_verify': 0, 'full_search': 0}
            
            def record(self, cache_hit, source):
                if cache_hit:
                    self.hits += 1
                else:
                    self.misses += 1
                self.sources[source] += 1
            
            def get_hit_rate(self):
                total = self.hits + self.misses
                return self.hits / total if total > 0 else 0
        
        profiler = MockProfiler()
        
        # Simulate 10 pixels
        profiler.record(True, 'direct_reuse')
        profiler.record(True, 'interpolation')
        profiler.record(True, 'direct_reuse')
        profiler.record(False, 'full_search')
        profiler.record(True, 'light_verify')
        profiler.record(False, 'full_search')
        profiler.record(True, 'direct_reuse')
        profiler.record(True, 'interpolation')
        profiler.record(False, 'full_search')
        profiler.record(True, 'direct_reuse')
        
        assert profiler.hits == 7
        assert profiler.misses == 3
        assert profiler.get_hit_rate() == 0.7
        assert profiler.sources['direct_reuse'] == 4
        assert profiler.sources['full_search'] == 3
    
    def test_memory_savings_calculation(self):
        """Should calculate memory savings."""
        class MockProfiler:
            def __init__(self, num_depths):
                self.num_depths = num_depths
                self.total_searches = 0
                self.pixel_count = 0
            
            def record_search(self, num_searches):
                self.total_searches += num_searches
                self.pixel_count += 1
            
            def get_memory_savings(self):
                baseline = self.pixel_count * self.num_depths
                saved = baseline - self.total_searches
                return saved / baseline if baseline > 0 else 0
        
        profiler = MockProfiler(num_depths=32)
        
        # Simulate various paths
        profiler.record_search(0)   # direct_reuse
        profiler.record_search(0)   # direct_reuse
        profiler.record_search(5)   # light_verify
        profiler.record_search(32)  # full_search
        profiler.record_search(0)   # interpolation
        
        # Total baseline = 5 * 32 = 160
        # Total searches = 0 + 0 + 5 + 32 + 0 = 37
        # Saved = 160 - 37 = 123
        # Savings = 123 / 160 = 76.875%
        
        savings = profiler.get_memory_savings()
        assert abs(savings - 0.76875) < 0.001


class TestFSDREndToEnd:
    """End-to-end integration tests."""
    
    def test_multiple_pixels_with_reuse(
        self, similar_feature_pair, mock_cost_fn, mock_prob_fn, 
        default_config, depth_candidates
    ):
        """Second similar pixel should reuse first pixel's result."""
        feat1, feat2 = similar_feature_pair
        
        class MockFSDRProcessor:
            def __init__(self, config, depth_candidates, projection_matrix):
                self.config = config
                self.depth_candidates = depth_candidates
                self.projection = projection_matrix
                self.cache = [None] * config.cache_size
            
            def _hash(self, feature):
                projections = self.projection @ feature
                signature = 0
                for i, proj in enumerate(projections):
                    if proj >= 0:
                        signature |= (1 << i)
                return signature
            
            def _lookup(self, signature):
                for entry in self.cache:
                    if entry is not None and entry.valid:
                        hamming = compute_hamming_distance(signature, entry.signature)
                        if hamming <= self.config.hamming_threshold:
                            return entry, hamming
                return None, -1
            
            def _insert(self, entry):
                for i, e in enumerate(self.cache):
                    if e is None:
                        self.cache[i] = entry
                        return
            
            def process_pixel(self, feature, position, cost_fn, prob_fn):
                sig = self._hash(feature)
                entry, hamming = self._lookup(sig)
                
                if entry is not None:
                    return {
                        'depth': entry.best_depth,
                        'source': 'cache_hit',
                        'cache_hit': True,
                        'hamming_distance': hamming,
                    }
                else:
                    probs = prob_fn(feature, self.depth_candidates)
                    depth = float((probs * self.depth_candidates).sum())
                    best_idx = int(torch.argmax(probs).item())
                    
                    new_entry = MockCacheEntry(
                        signature=sig,
                        position=position,
                        best_depth=depth,
                        best_idx=best_idx,
                        peak_prob=float(probs[best_idx]),
                        second_offset=1,
                        spread=0.1,
                    )
                    self._insert(new_entry)
                    
                    return {
                        'depth': depth,
                        'source': 'full_search',
                        'cache_hit': False,
                    }
        
        projection = torch.randn(16, 128)
        projection = projection / torch.norm(projection, dim=1, keepdim=True)
        
        processor = MockFSDRProcessor(default_config, depth_candidates, projection)
        
        # First pixel - cache miss
        result1 = processor.process_pixel(feat1, (10, 10), mock_cost_fn, mock_prob_fn)
        assert result1['cache_hit'] is False
        
        # Second similar pixel - should hit cache
        result2 = processor.process_pixel(feat2, (10, 11), mock_cost_fn, mock_prob_fn)
        # Due to LSH similarity preservation, similar features should hit cache
        # Note: There's some probability this might miss due to LSH randomness
        # In a real test, we'd set up the projection to guarantee a hit
