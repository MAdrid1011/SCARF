package scarf.control

import chisel3._
import chisel3.util._
import scarf.{ScarfConfig, ModelConfig, SAESLevel}

/**
 * SAESController — SAES Tile Classification FSM.
 *
 * Corresponds to: saes/progressive_saes.py (ProgressiveSAES.process_all_tiles)
 *
 * For each tile, performs 3-level classification:
 *   L0: Feature variance < threshold → skip entire S2+S3
 *   L1: Depth uniformity (probe depths std < threshold) → skip S2+S3
 *   L2: Gaussian cross-check (leave-one-out cosine error < threshold) → skip S3
 *   Full: No early-stop, run complete pipeline
 *
 * Hardware implementation:
 *   - L0: 4 corner probe feature vectors → compute variance via VectorALU reduction
 *   - L1: 4 probe depths (from lightweight CostVol+SoftArgmax) → compare std
 *   - L2: 4 probe Gaussians → leave-one-out average prediction → cosine distance
 *   - Cross-check: 4 weighted sums + 4 comparisons = 8 cycles per tile
 */
class SAESController extends Module {
  val io = IO(new Bundle {
    // Control
    val start    = Input(Bool())
    val done     = Output(Bool())

    // Configuration thresholds (from ConfigRegs)
    val config   = Input(new ModelConfig)

    // Feature data for L0 check (4 corner probes)
    val probeFeatureVar = Input(UInt(ScarfConfig.DataWidth.W))  // Pre-computed variance

    // Depth data for L1 check (4 probe depths)
    val probeDepthStd   = Input(UInt(ScarfConfig.DataWidth.W))  // Pre-computed std

    // Cross-check error for L2 (max leave-one-out error)
    val crossCheckError = Input(UInt(ScarfConfig.DataWidth.W))

    // Classification output
    val level    = Output(SAESLevel())
  })

  val sIdle :: sCheckL0 :: sCheckL1 :: sCheckL2 :: sResult :: Nil = Enum(5)
  val state = RegInit(sIdle)
  val resultLevel = RegInit(SAESLevel.sFull)

  io.done  := state === sResult
  io.level := resultLevel

  switch(state) {
    is(sIdle) {
      when(io.start) {
        state := Mux(io.config.saesEnabled, sCheckL0, sResult)
        resultLevel := SAESLevel.sFull  // Default: full processing
      }
    }

    is(sCheckL0) {
      // L0: Feature variance check
      // If tile has uniform features → probes + interpolation suffice
      when(io.probeFeatureVar < io.config.saesFeatureVarThresh) {
        resultLevel := SAESLevel.sL0
        state := sResult
      }.otherwise {
        state := sCheckL1
      }
    }

    is(sCheckL1) {
      // L1: Depth uniformity check
      // If 4 corner probe depths are similar → flat surface → safe to interpolate
      when(io.probeDepthStd < io.config.saesDepthStdThresh) {
        // Additional gate: cross-check must also pass (at 1.0x threshold)
        when(io.crossCheckError < io.config.saesCrossCheckThresh) {
          resultLevel := SAESLevel.sL1
          state := sResult
        }.otherwise {
          state := sCheckL2
        }
      }.otherwise {
        state := sCheckL2
      }
    }

    is(sCheckL2) {
      // L2: Gaussian cross-check only (no prior information)
      // Uses base threshold (1.0x) — strictest level
      when(io.crossCheckError < io.config.saesCrossCheckThresh) {
        resultLevel := SAESLevel.sL2
      }.otherwise {
        resultLevel := SAESLevel.sFull
      }
      state := sResult
    }

    is(sResult) {
      // Hold result for 1 cycle, then return to idle
      state := sIdle
    }
  }
}
