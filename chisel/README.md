# SCARF Chisel RTL

This directory contains the Chisel RTL implementation of SCARF. The RTL mirrors
the Python hardware simulators in `encoder/`, `ggu/`, `depth_predictor/`, and
`fsdr/`.

## Directory Layout

```text
chisel/
|-- build.sbt
|-- project/plugins.sbt
|-- src/
|   |-- main/scala/scarf/
|   |   |-- Config.scala
|   |   |-- Types.scala
|   |   |-- ScarfTop.scala
|   |   |-- VerilogEmitter.scala
|   |   |-- compute/
|   |   |   |-- ConvEngine.scala
|   |   |   |-- GEMMUnit.scala
|   |   |   |-- BilinearUnit.scala
|   |   |   |-- ActivationUnit.scala
|   |   |   |-- NormUnit.scala
|   |   |   |-- VectorALU.scala
|   |   |   |-- SoftmaxUnit.scala
|   |   |   |-- PoolingUnit.scala
|   |   |   |-- PadUnit.scala
|   |   |   `-- LSHHashUnit.scala
|   |   |-- ggu/
|   |   |   |-- GGUArray.scala
|   |   |   |-- PositionCalc.scala
|   |   |   |-- CovBuilder.scala
|   |   |   `-- SHRotator.scala
|   |   |-- memory/
|   |   |   |-- WeightBuffer.scala
|   |   |   |-- FeatureBuffer.scala
|   |   |   |-- TileSPM.scala
|   |   |   |-- DRAMInterface.scala
|   |   |   `-- FSDRCache.scala
|   |   `-- control/
|   |       |-- PipelineController.scala
|   |       |-- SAESController.scala
|   |       |-- FSDRController.scala
|   |       `-- ConfigRegs.scala
|   `-- test/scala/scarf/
|       |-- TestUtils.scala
|       |-- ConvEngineTest.scala
|       |-- GEMMUnitTest.scala
|       `-- VerilogEmitTest.scala
`-- generated/
```

## Requirements

- JDK 11 or newer
- SBT 1.9 or newer
- Verilator for optional generated-Verilog linting

## Common Commands

```bash
cd chisel

# Compile the Chisel project.
sbt compile

# Run all Chisel tests.
sbt test

# Emit per-module Verilog files under generated/.
sbt "runMain scarf.VerilogEmitter"

# Optionally lint the generated top-level Verilog.
cd generated && verilator --lint-only ScarfTop.v
```

## Design Principles

1. Model differences are expressed through `ConfigRegs`, weights, tensor shapes,
   and instruction sequences.
2. Large compute units are instantiated once and reused across pipeline stages.
3. Chisel module boundaries follow the Python simulator modules closely.
4. Verilog is emitted per module to simplify synthesis and inspection.

## Mapping to Python Simulators

| Chisel module | Python reference |
|---------------|------------------|
| `ConvEngine.scala` | `encoder/conv_engine.py` |
| `GEMMUnit.scala` | `encoder/gemm_unit.py` |
| `BilinearUnit.scala` | `encoder/bilinear_unit.py` |
| `VectorALU.scala` | `encoder/softmax_unit.py` |
| `GGUArray.scala` | `ggu/ggu_processor.py` |
| `FSDRController.scala` | `fsdr/narrowed_search_simulator.py` |
| `FSDRCache.scala` | `fsdr/cache_table.py` |

## Related Documentation

- [Architecture Overview](../docs/architecture.md)
- [Pipeline Architecture](../docs/pipeline-architecture.md)
- [FSDR and SAES Mechanisms](../docs/fsdr-saes-mechanisms.md)
