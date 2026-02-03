"""
Pipeline

Pipeline control for batch processing.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    batch_size: int = 64
    enable_fsdr: bool = True
    enable_saes: bool = True
    profile_every_n_batches: int = 100


class Pipeline:
    """
    Pipeline controller for batch processing.
    
    Manages:
    - Batch scheduling
    - Progress tracking
    - Resource monitoring
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.batch_count = 0
        self.pixel_count = 0
    
    def start_batch(self):
        """Mark start of new batch."""
        self.batch_count += 1
    
    def end_batch(self, num_pixels: int):
        """Mark end of batch."""
        self.pixel_count += num_pixels
    
    def should_profile(self) -> bool:
        """Check if profiling should be collected this batch."""
        return self.batch_count % self.config.profile_every_n_batches == 0
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'batch_count': self.batch_count,
            'pixel_count': self.pixel_count,
            'avg_pixels_per_batch': (
                self.pixel_count / self.batch_count if self.batch_count > 0 else 0
            ),
        }
    
    def reset(self):
        """Reset pipeline statistics."""
        self.batch_count = 0
        self.pixel_count = 0
