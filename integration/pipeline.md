# Pipeline

Batch processing and progress tracking.

## External Interface

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    batch_size: int = 64
    enable_fsdr: bool = True
    enable_saes: bool = True
    profile_every_n_batches: int = 100
```

**Fields:**
- `batch_size`: Pixels per batch
- `enable_fsdr`: Toggle FSDR optimization
- `enable_saes`: Toggle SAES optimization
- `profile_every_n_batches`: Profiling frequency

### Pipeline

```python
class Pipeline:
    def __init__(self, config: PipelineConfig)
    
    def start_batch(self)
    def end_batch(self, num_pixels: int)
    def should_profile(self) -> bool
    def get_stats(self) -> Dict
    def reset(self)
```

**Methods:**

#### start_batch()
Mark start of new batch for timing.

#### end_batch(num_pixels)
Mark batch completion with pixel count.

- **Input**: `num_pixels` - pixels processed in this batch

#### should_profile() -> bool
Check if profiling should be collected this batch.

- Returns `True` every `profile_every_n_batches`

#### get_stats() -> Dict
Return pipeline statistics.

```python
{
    'batch_count': int,
    'pixel_count': int,
    'avg_pixels_per_batch': float,
}
```

#### reset()
Reset all statistics for new run.

## Usage Pattern

```python
pipeline = Pipeline(PipelineConfig(batch_size=64))

for batch in batches:
    pipeline.start_batch()
    
    for pixel in batch:
        result = accelerator.process_pixel(...)
    
    pipeline.end_batch(len(batch))
    
    if pipeline.should_profile():
        print(pipeline.get_stats())
```

## Hardware Mapping

Pipeline is software-only for batch scheduling.
Hardware uses streaming with double-buffering.
