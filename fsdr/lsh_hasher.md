# LSH Hasher

Locality-Sensitive Hashing for feature signatures.

## External Interface

### LSHHasher

Generates binary signatures from feature vectors.

```python
class LSHHasher:
    def __init__(self, config: FSDRConfig, seed: int = 42)
    
    def hash(self, feature: torch.Tensor) -> int
    def hash_batch(self, features: torch.Tensor) -> torch.Tensor
```

**Constructor:**
- `config`: FSDR configuration with `feature_dim` and `num_hash_bits`
- `seed`: Random seed for projection matrix initialization

**Methods:**

#### hash(feature) -> int
Generate 16-bit signature from single feature vector.

- **Input**: `feature` - [D] tensor
- **Output**: Integer in range [0, 2^16)
- **Complexity**: O(D × num_bits)

#### hash_batch(features) -> Tensor
Generate signatures for batch of features.

- **Input**: `features` - [N, D] tensor
- **Output**: [N] tensor of integer signatures

### Helper Functions

```python
def hamming_distance(sig1: int, sig2: int) -> int
def expected_hamming_from_cosine(cosine_sim: float, num_bits: int) -> float
```

## Internal Helpers

### _init_projection_matrix()
Initialize random projection matrix with fixed seed.

- Uses `torch.manual_seed(seed)` for reproducibility
- Matrix shape: [num_hash_bits, feature_dim]
- Each row is a random hyperplane normal

### Signature Generation Algorithm
```
for each bit i:
    projection = dot(feature, hyperplane[i])
    bit[i] = 1 if projection >= 0 else 0
signature = pack_bits(bit[0:16])
```

## Hardware Mapping

- Projection: 16 parallel dot products
- Comparison: 16 sign extractors
- Packing: 16-bit shift register
- Total: ~200 LUTs, 0 DSPs (use fixed-point multiply-accumulate)
