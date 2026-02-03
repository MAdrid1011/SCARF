# SCARF Data Directory

This directory contains symlinks or copies of model checkpoints and datasets required for real inference benchmarks.

## Directory Structure

```
data/
├── README.md           # This file
├── checkpoints/        # Model checkpoints
│   └── re10k.ckpt     # Transplat RE10K checkpoint
└── re10k/             # RE10K dataset
    ├── test/          # Test split
    │   └── *.torch    # Scene chunks
    └── index.json     # Scene index
```

## Setup

### Option 1: Symlinks (Recommended)

Link existing files from your transplat installation:

```bash
# From SCARF root directory
mkdir -p data/checkpoints

# Link checkpoint
ln -s /path/to/transplat/checkpoints/re10k.ckpt data/checkpoints/re10k.ckpt

# Link dataset
ln -s /path/to/datasets/re10k data/re10k
```

### Option 2: Copy Files

Copy files directly (requires more disk space):

```bash
# From SCARF root directory
mkdir -p data/checkpoints

# Copy checkpoint (~500MB)
cp /path/to/transplat/checkpoints/re10k.ckpt data/checkpoints/

# Copy dataset (~10GB for test set)
cp -r /path/to/datasets/re10k data/
```

## Verification

Verify your setup:

```bash
# Check checkpoint exists
ls -la data/checkpoints/re10k.ckpt

# Check dataset exists
ls data/re10k/test/*.torch | head -5

# Run quick validation
python scripts/run_re10k_benchmark.py --real \
    --model-path data/checkpoints/re10k.ckpt \
    --dataset-path data/re10k/ \
    --num-scenes 1
```

## Notes

- Files in this directory are ignored by git (see `.gitignore`)
- For CI/CD, use simulation mode which doesn't require real data
- The checkpoint file is typically ~500MB
- The full RE10K test set is ~10GB
