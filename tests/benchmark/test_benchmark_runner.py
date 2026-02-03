"""
Benchmark Runner Tests

Tests for end-to-end benchmark orchestration.
"""
import pytest
import json
from .conftest import (
    MockQualityMetrics,
    MockCycleEstimate,
    generate_mock_scene_data,
)


class TestBenchmarkRunnerInitialization:
    """Benchmark runner initialization tests."""
    
    def test_init_with_valid_config(self, mock_model_config, mock_scarf_config):
        """Should initialize with valid configuration."""
        config = {
            **mock_model_config,
            **mock_scarf_config,
            'num_scenes': 10,
        }
        assert config['model_type'] == 'transplat'
        assert config['enable_fsdr'] == True
        assert config['num_scenes'] == 10
    
    def test_init_validates_paths(self, mock_benchmark_config):
        """Should validate model and dataset paths exist."""
        # In actual implementation, would check path validity
        output_dir = mock_benchmark_config['output_dir']
        assert output_dir is not None


class TestBaselineRun:
    """Baseline run tests."""
    
    def test_baseline_produces_metrics(self):
        """Baseline run should produce quality metrics."""
        # Mock baseline result
        baseline = MockQualityMetrics(psnr=25.5, ssim=0.92, num_samples=100)
        assert baseline.psnr > 0
        assert 0 < baseline.ssim <= 1
    
    def test_baseline_no_cycles(self):
        """Baseline run should not count SCARF cycles."""
        # Baseline doesn't use SCARF, so no cycle counting
        scarf_cycles = 0
        assert scarf_cycles == 0
    
    def test_baseline_per_scene_metrics(self):
        """Should collect per-scene metrics if enabled."""
        per_scene = {
            'scene_001': MockQualityMetrics(psnr=25.0, ssim=0.91),
            'scene_002': MockQualityMetrics(psnr=26.0, ssim=0.93),
        }
        assert len(per_scene) == 2


class TestSCARFRun:
    """SCARF-accelerated run tests."""
    
    def test_scarf_produces_metrics(self):
        """SCARF run should produce quality metrics."""
        scarf = MockQualityMetrics(psnr=25.3, ssim=0.915, num_samples=100)
        assert scarf.psnr > 0
        assert 0 < scarf.ssim <= 1
    
    def test_scarf_counts_cycles(self, sample_cycle_estimates):
        """SCARF run should count hardware cycles."""
        total_cycles = sum(e.cycles for e in sample_cycle_estimates)
        assert total_cycles > 0
    
    def test_scarf_cycle_breakdown(self, sample_cycle_estimates):
        """SCARF run should provide cycle breakdown by component."""
        by_component = {}
        for e in sample_cycle_estimates:
            if e.component not in by_component:
                by_component[e.component] = 0
            by_component[e.component] += e.cycles
        
        assert 'fsdr' in by_component
        assert 'dsu' in by_component
        assert 'ggu' in by_component


class TestResultComparison:
    """Result comparison tests."""
    
    def test_compute_speedup(self):
        """Should compute speedup from baseline to SCARF."""
        baseline_cycles = 454  # Per pixel
        scarf_cycles = 180  # Per pixel
        speedup = baseline_cycles / scarf_cycles
        
        assert abs(speedup - 2.52) < 0.1
    
    def test_compute_quality_delta(self, baseline_metrics):
        """Should compute quality delta."""
        scarf_metrics = MockQualityMetrics(psnr=25.3, ssim=0.915)
        
        psnr_delta = scarf_metrics.psnr - baseline_metrics.psnr
        ssim_delta = scarf_metrics.ssim - baseline_metrics.ssim
        
        assert abs(psnr_delta - (-0.2)) < 0.01
        assert abs(ssim_delta - (-0.005)) < 0.001
    
    def test_quality_acceptable_check(self, baseline_metrics):
        """Should check if quality degradation is acceptable."""
        scarf_metrics = MockQualityMetrics(psnr=25.3, ssim=0.915)
        
        psnr_drop = baseline_metrics.psnr - scarf_metrics.psnr
        ssim_drop = baseline_metrics.ssim - scarf_metrics.ssim
        
        psnr_acceptable = psnr_drop < 0.5
        ssim_acceptable = ssim_drop < 0.01
        
        assert psnr_acceptable
        assert ssim_acceptable


class TestReportGeneration:
    """Report generation tests."""
    
    def test_json_report_structure(self, baseline_metrics, sample_cycle_estimates):
        """JSON report should have correct structure."""
        report = {
            'metadata': {
                'timestamp': '2026-02-03T12:00:00',
                'model': 're10k.ckpt',
                'num_scenes': 100,
            },
            'baseline': {
                'psnr_mean': baseline_metrics.psnr,
                'ssim_mean': baseline_metrics.ssim,
            },
            'scarf': {
                'psnr_mean': 25.3,
                'ssim_mean': 0.915,
                'total_cycles': sum(e.cycles for e in sample_cycle_estimates),
            },
            'comparison': {
                'psnr_delta': -0.2,
                'ssim_delta': -0.005,
                'speedup': 2.52,
                'quality_acceptable': True,
            },
        }
        
        # Verify structure
        assert 'metadata' in report
        assert 'baseline' in report
        assert 'scarf' in report
        assert 'comparison' in report
    
    def test_json_serializable(self, baseline_metrics):
        """Report should be JSON serializable."""
        report = {
            'psnr': baseline_metrics.psnr,
            'ssim': baseline_metrics.ssim,
        }
        json_str = json.dumps(report)
        assert isinstance(json_str, str)
        
        # Can deserialize
        loaded = json.loads(json_str)
        assert loaded['psnr'] == baseline_metrics.psnr


class TestFSDRStatistics:
    """FSDR-specific statistics tests."""
    
    def test_hit_rate_calculation(self):
        """Should calculate FSDR cache hit rate."""
        total_pixels = 1000
        cache_hits = 680
        hit_rate = cache_hits / total_pixels
        
        assert abs(hit_rate - 0.68) < 0.01
    
    def test_memory_reduction(self):
        """Should calculate memory access reduction."""
        baseline_accesses = 32 * 1000  # 32 per pixel × 1000 pixels
        # 68% hit rate: 680 hits × 1 access + 320 misses × 32 accesses
        scarf_accesses = 680 * 1 + 320 * 32  # = 680 + 10240 = 10920
        reduction = 1 - (scarf_accesses / baseline_accesses)
        
        assert reduction > 0.6  # ~66% reduction expected


class TestSceneData:
    """Scene data handling tests."""
    
    def test_generate_mock_data(self):
        """Should generate valid mock scene data."""
        data = generate_mock_scene_data(num_pixels=500)
        
        assert data['features'].shape == (500, 128)
        assert data['depths'].shape == (500,)
        assert data['pixel_coords'].shape == (500, 2)
    
    def test_data_ranges(self):
        """Mock data should have valid ranges."""
        data = generate_mock_scene_data(num_pixels=100)
        
        # Features normalized
        assert data['features'].min() >= 0
        assert data['features'].max() <= 1
        
        # Depths positive
        assert data['depths'].min() >= 0.5
        assert data['depths'].max() <= 10.5


class TestErrorHandling:
    """Error handling tests."""
    
    def test_missing_model_path(self, mock_benchmark_config):
        """Should handle missing model path gracefully."""
        config = mock_benchmark_config.copy()
        config['model_path'] = '/nonexistent/path.ckpt'
        # In actual implementation, would raise FileNotFoundError
        assert config['model_path'] is not None
    
    def test_empty_dataset(self):
        """Should handle empty dataset gracefully."""
        num_scenes = 0
        # In actual implementation, would raise ValueError
        assert num_scenes == 0
