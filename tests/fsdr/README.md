# FSDR Tests

Unit tests for Feature-Similarity Depth Reuse components.

## Test Files

| File | Tests |
|------|-------|
| `test_lsh_hasher.py` | LSH signature generation, similarity preservation |
| `test_cache_table.py` | Cache lookup, insertion, LRU replacement |
| `test_depth_corrector.py` | Three-level correction strategy selection |
| `test_light_verifier.py` | Local depth search, range calculation |
| `test_fsdr_processor.py` | End-to-end FSDR workflow |

## Fixtures (`conftest.py`)

- `MockFSDRConfig`: Configuration without validation
- `MockCacheEntry`: Cache entry for testing
- `random_feature`: Random 128-dim feature vector
- `similar_feature_pair`: Two features with high similarity
- `mock_depth_candidates`: Sample depth values

## Running

```bash
# All FSDR tests
pytest tests/fsdr/ -v

# Specific test file
pytest tests/fsdr/test_lsh_hasher.py -v

# Specific test
pytest tests/fsdr/test_cache_table.py::TestCacheLookup -v
```

## Coverage

Tests cover:
- Signature generation and bit width
- Hamming distance correlation with cosine similarity
- Cache hit/miss behavior
- LRU replacement with confidence weighting
- Strategy selection thresholds
- Light verification search ranges
