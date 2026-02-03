# SCARF Scripts

Command-line scripts for running SCARF benchmarks and utilities.

## Available Scripts

| Script | Description |
|--------|-------------|
| `run_re10k_benchmark.py` | Run SCARF benchmark on RE10K dataset |

## Usage

All scripts should be run from the SCARF root directory:

```bash
cd /path/to/SCARF
python scripts/run_re10k_benchmark.py [options]
```

## Script Documentation

See individual `.md` files for detailed documentation:
- `run_re10k_benchmark.md` - Benchmark CLI documentation

## Adding New Scripts

When adding new scripts:
1. Place the script in this `scripts/` directory
2. Create a companion `.md` file documenting:
   - Purpose and usage
   - Command-line arguments
   - Example invocations
   - Output format
3. Update this README with the new script entry
