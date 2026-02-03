# Cache Table

Semantic cache with LRU replacement for FSDR.

## External Interface

### CacheTable

```python
class CacheTable:
    def __init__(self, config: FSDRConfig, depth_candidates: torch.Tensor)
    
    def lookup(self, signature: int) -> Tuple[Optional[CacheEntry], int]
    def insert(self, entry: CacheEntry)
    def update_confidence(self, signature: int, new_confidence: float)
    def clear(self)
    def get_stats(self) -> Dict
```

**Constructor:**
- `config`: FSDR configuration with `cache_size` and `hamming_threshold`
- `depth_candidates`: [D] depth values for entry deserialization

**Methods:**

#### lookup(signature) -> (entry, hamming_dist)
Find best matching entry within Hamming threshold.

- **Input**: `signature` - 16-bit query signature
- **Output**: `(CacheEntry, hamming_distance)` or `(None, -1)` if miss
- **Complexity**: O(cache_size) with parallel comparison

#### insert(entry)
Insert new entry, evicting lowest-confidence if full.

- **Input**: `entry` - CacheEntry to insert
- **Behavior**: 
  - If cache not full: append
  - If full: replace entry with lowest `confidence × recency`
- **LRU Factor**: Entries lose 10% confidence per lookup cycle

#### update_confidence(signature, new_confidence)
Update confidence for existing entry (after successful reuse).

- **Input**: `signature` - entry identifier, `new_confidence` - new value

#### get_stats() -> Dict
Return cache statistics.

- **Output**: `{'size': int, 'capacity': int, 'fill_rate': float}`

## Internal Helpers

### _find_lru_victim() -> int
Find index of entry to evict.

- Score = `confidence × recency_factor`
- Returns index with lowest score

### _apply_lru_decay()
Apply recency decay to all entries.

- Called after each lookup cycle
- `confidence *= 0.9` for all entries

## Hardware Mapping

- Storage: 256 entries × 70 bits = 2.2KB SRAM
- Lookup: 256 parallel Hamming comparators
- Eviction: Min-finder tree (~400 LUTs)
- Total: 2.2KB SRAM, ~600 LUTs
