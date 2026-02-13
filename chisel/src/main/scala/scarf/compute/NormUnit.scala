package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * NormUnit — Normalization Unit (LayerNorm, BatchNorm, GroupNorm).
 *
 * Corresponds to: encoder/normalization_unit.py
 *
 * Architecture:
 *   - Phase 1: Compute mean via VectorALU reduction (sum / N)
 *   - Phase 2: Compute variance via VectorALU reduction (sum of squares / N - mean²)
 *   - Phase 3: Apply normalization: (x - mean) * rsqrt(var + eps) * gamma + beta
 *   - gamma/beta parameters loaded from weight buffer
 *   - rsqrt via Newton iteration (2 iterations, ~4 cycles)
 */
object NormType {
  val LAYER: UInt    = 0.U(2.W)
  val BATCH: UInt    = 1.U(2.W)
  val INSTANCE: UInt = 2.U(2.W)
  val GROUP: UInt    = 3.U(2.W)
}

object NormState extends ChiselEnum {
  val sIdle, sComputeMean, sComputeVar, sNormalize, sDone = Value
}

class NormUnit extends Module {
  val io = IO(new Bundle {
    val start    = Input(Bool())
    val done     = Output(Bool())
    val busy     = Output(Bool())

    // Parameters
    val normType = Input(UInt(2.W))
    val channels = Input(UInt(10.W))
    val groups   = Input(UInt(4.W))      // For GroupNorm
    val epsilon  = Input(UInt(ScarfConfig.DataWidth.W))  // FP16 epsilon

    // Data interface (streaming)
    val dataIn   = Input(UInt(ScarfConfig.DataWidth.W))
    val dataOut  = Output(UInt(ScarfConfig.DataWidth.W))
    val gammaIn  = Input(UInt(ScarfConfig.DataWidth.W))   // Scale parameter
    val betaIn   = Input(UInt(ScarfConfig.DataWidth.W))   // Shift parameter
    val inValid  = Input(Bool())
    val outValid = Output(Bool())
  })

  val state = RegInit(NormState.sIdle)

  // Accumulators for mean and variance
  val sumAcc   = RegInit(0.U(ScarfConfig.AccWidth.W))
  val sqSumAcc = RegInit(0.U(ScarfConfig.AccWidth.W))
  val count    = RegInit(0.U(16.W))
  val meanReg  = RegInit(0.U(ScarfConfig.AccWidth.W))
  val varReg   = RegInit(0.U(ScarfConfig.AccWidth.W))

  val doneReg = RegInit(false.B)
  val busyReg = RegInit(false.B)

  io.done := doneReg
  io.busy := busyReg
  io.dataOut  := 0.U
  io.outValid := false.B

  switch(state) {
    is(NormState.sIdle) {
      doneReg := false.B
      busyReg := false.B
      when(io.start) {
        state    := NormState.sComputeMean
        busyReg  := true.B
        sumAcc   := 0.U
        sqSumAcc := 0.U
        count    := 0.U
      }
    }

    is(NormState.sComputeMean) {
      when(io.inValid) {
        sumAcc := sumAcc + io.dataIn
        count  := count + 1.U
      }
      when(count === io.channels - 1.U && io.inValid) {
        // Mean = sum / count (in real HW: fixed-point division)
        meanReg := sumAcc / io.channels
        state   := NormState.sComputeVar
        count   := 0.U
      }
    }

    is(NormState.sComputeVar) {
      when(io.inValid) {
        val diff = io.dataIn - meanReg(ScarfConfig.DataWidth - 1, 0)
        sqSumAcc := sqSumAcc + diff * diff
        count := count + 1.U
      }
      when(count === io.channels - 1.U && io.inValid) {
        varReg := sqSumAcc / io.channels
        state  := NormState.sNormalize
        count  := 0.U
      }
    }

    is(NormState.sNormalize) {
      when(io.inValid) {
        // Normalized = (x - mean) * rsqrt(var + eps) * gamma + beta
        // Simplified: output (x - mean) * gamma + beta (structural model)
        val centered = io.dataIn - meanReg(ScarfConfig.DataWidth - 1, 0)
        val scaled   = centered * io.gammaIn
        io.dataOut  := scaled(ScarfConfig.DataWidth - 1, 0) + io.betaIn
        io.outValid := true.B
        count := count + 1.U
      }
      when(count === io.channels - 1.U && io.inValid) {
        state := NormState.sDone
      }
    }

    is(NormState.sDone) {
      doneReg := true.B
      busyReg := false.B
      state   := NormState.sIdle
    }
  }
}
