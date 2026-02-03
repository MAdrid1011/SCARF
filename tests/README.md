# Tests

Test suite for SCARF hardware simulators.

## Structure

```
tests/
├── saes/           # SAES unit and integration tests
├── fsdr/           # FSDR unit tests
├── dsu/            # DSU unit tests
├── ggu/            # GGU unit tests
└── adapters/       # Adapter tests (future)
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/fsdr/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Skip integration tests (require transplat)
pytest tests/ -v -m "not integration"
```

## Test Categories

### Unit Tests
- Located in `tests/{module}/test_*.py`
- Use mock fixtures from `conftest.py`
- No external dependencies

### Integration Tests
- Located in `tests/saes/test_integration_re10k.py`
- Require `transplat` environment
- Marked with `@pytest.mark.skipif`

## Fixtures

Each module has `conftest.py` providing:
- Mock configurations
- Sample data generators
- Helper functions

Example:
```python
@pytest.fixture
def mock_fsdr_config():
    return MockFSDRConfig(feature_dim=128)
```

## Writing New Tests

1. Add test file in appropriate directory
2. Use existing fixtures from `conftest.py`
3. Follow naming: `test_{component}_{behavior}`
4. Add docstring explaining test purpose

```python
def test_cache_hit_returns_cached_depth(mock_config, sample_feature):
    """Cache hit should return depth from cache entry."""
    # Test implementation
```
