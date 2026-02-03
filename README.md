# SCARF: Generalizable 3DGS Inference Accelerator

SCARF (Scene-adaptive Computation and Access Reduction Framework) is a hardware-software co-design project for accelerating generalizable 3D Gaussian Splatting (3DGS) inference on edge devices.

## Overview

SCARF addresses two major bottlenecks in generalizable 3DGS encoders:
1. **FSDR (Feature-Similarity Depth Reuse)**: Reduces redundant memory access in depth search by exploiting 2D feature similarity
2. **SAES (Scene-Adaptive Early-Stopping)**: Reduces redundant computation by exploiting 3D geometric continuity with feedback-driven tile processing

## Repository Structure

```
SCARF/
├── saes/                  # SAES simulator implementation
│   ├── tile_processor.py
│   ├── similarity_evaluator.py
│   ├── decision_controller.py
│   ├── gaussian_merger.py
│   └── profiler.py
├── fsdr/                  # FSDR simulator (planned)
├── tests/                 # Unit and integration tests
│   ├── saes/
│   └── fsdr/
└── docs/                  # Documentation
    ├── saes-architecture.md
    ├── saes-usage.md
    └── plan-saes-simulator.md

```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.1+
- Access to transplat repository (for integration testing)

### Installation

```bash
# Clone the repository
git clone git@github.com:MAdrid1011/SCARF.git
cd SCARF

# Install dependencies (coming soon)
pip install -r requirements.txt
```

### Running SAES Simulator

```bash
# Run SAES simulation on transplat (integration required)
cd ../transplat
python -m src.main +experiment=re10k \
    checkpointing.load=./checkpoints/re10k.ckpt \
    mode=test \
    test.enable_saes_simulation=true \
    test.saes_tile_size=4
```

## Current Status

### SAES Hardware Simulator
- **Status**: 📝 Planning phase
- **Issue**: [#1 - SAES Hardware Simulator Implementation](https://github.com/MAdrid1011/SCARF/issues/1)
- **Plan**: [`docs/plan-saes-simulator.md`](docs/plan-saes-simulator.md)
- **Complexity**: 2750 LOC (Very Large feature)

### FSDR Hardware Simulator
- **Status**: 🔜 Planned
- **Description**: Feature-similarity depth reuse for memory access optimization

## Documentation

- [SAES Implementation Plan](docs/plan-saes-simulator.md) - Detailed implementation plan for SAES simulator
- [Git Message Tags](docs/git-msg-tags.md) - Commit message tag conventions

## Integration with Transplat

SCARF simulators are designed to integrate seamlessly with the [transplat](https://github.com/MAdrid1011/transplat) repository for evaluation on real 3DGS models and datasets.

Key integration points:
- `transplat/src/model/encoder/encoder_trans.py` - Encoder pipeline hooks
- `transplat/src/model/model_wrapper.py` - Configuration and profiling
- `transplat/scripts/run_all_timing_tests.sh` - Testing infrastructure

## Contributing

See individual issue pages for implementation plans and contribution guidelines.

## License

[License TBD]

## Related Projects

- [transplat](https://github.com/MAdrid1011/transplat) - Generalizable 3D Gaussian Splatting implementation
- Design documents available in `transplat/draft/` directory

## Contact

For questions and discussions, please open an issue in this repository.
