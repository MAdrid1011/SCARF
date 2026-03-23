package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * NormUnit — Normalization Unit (LayerNorm, BatchNorm, GroupNorm).
 *
 * Internal structure (matching diagram):
 *   FSM → SumAcc → SqSumAcc → rSqrt → Affine Scale → FMA
 *
 * Phase 1: Accumulate sum via SumAcc → compute mean μ
 * Phase 2: Accumulate squared differences via SqSumAcc → compute variance σ²
 * Phase 3: rSqrt computes 1/√(σ² + ε) via Newton iteration
 * Phase 4: Affine Scale applies gamma/beta via FMA: out = (x - μ) × rsqrt × γ + β
 */
object NormType {
  val LAYER: UInt    = 0.U(2.W)
  val BATCH: UInt    = 1.U(2.W)
  val INSTANCE: UInt = 2.U(2.W)
  val GROUP: UInt    = 3.U(2.W)
}

object NormState extends ChiselEnum {
  val sIdle, sComputeMean, sComputeVar, sRSqrt, sNormalize, sDone = Value
}

class NormUnit extends Module {
  val io = IO(new Bundle {
    val start    = Input(Bool())
    val done     = Output(Bool())
    val busy     = Output(Bool())

    val normType = Input(UInt(2.W))
    val channels = Input(UInt(10.W))
    val groups   = Input(UInt(4.W))
    val epsilon  = Input(UInt(ScarfConfig.AccWidth.W))

    val dataIn   = Input(UInt(ScarfConfig.DataWidth.W))
    val dataOut  = Output(UInt(ScarfConfig.DataWidth.W))
    val gammaIn  = Input(UInt(ScarfConfig.DataWidth.W))
    val betaIn   = Input(UInt(ScarfConfig.DataWidth.W))
    val inValid  = Input(Bool())
    val outValid = Output(Bool())
  })

  val state = RegInit(NormState.sIdle)

  // SumAcc: accumulates sum for mean computation
  val sumAcc   = RegInit(0.U(ScarfConfig.AccWidth.W))
  // SqSumAcc: accumulates sum of squared differences for variance
  val sqSumAcc = RegInit(0.U(ScarfConfig.AccWidth.W))
  val count    = RegInit(0.U(16.W))
  val meanReg  = RegInit(0.U(ScarfConfig.AccWidth.W))
  val varReg   = RegInit(0.U(ScarfConfig.AccWidth.W))
  // rSqrt result register (1/√(var + eps))
  val rsqrtReg = RegInit(0.U(ScarfConfig.AccWidth.W))
  // Newton iteration counter for rSqrt
  val newtonIter = RegInit(0.U(2.W))

  // Normalization element count depends on norm type
  val normCount = Wire(UInt(16.W))
  normCount := Mux(io.normType === NormType.GROUP,
    io.channels / io.groups,
    io.channels)

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
        sumAcc := sumAcc + io.dataIn.pad(ScarfConfig.AccWidth)
        count  := count + 1.U
      }
      when(count === normCount - 1.U && io.inValid) {
        meanReg := sumAcc / normCount
        state   := NormState.sComputeVar
        count   := 0.U
      }
    }

    is(NormState.sComputeVar) {
      when(io.inValid) {
        val diff = io.dataIn.pad(ScarfConfig.AccWidth) - meanReg
        val sq = (diff * diff)(ScarfConfig.AccWidth - 1, 0)
        sqSumAcc := sqSumAcc + sq
        count := count + 1.U
      }
      when(count === normCount - 1.U && io.inValid) {
        varReg := sqSumAcc / normCount
        state  := NormState.sRSqrt
        count  := 0.U
        newtonIter := 0.U
      }
    }

    is(NormState.sRSqrt) {
      // Newton iteration for rsqrt: x_{n+1} = x_n * (3 - v * x_n²) / 2
      // Initial guess: rsqrt ≈ 1 (structural model, 2 iterations)
      val varPlusEps = varReg + io.epsilon
      when(newtonIter === 0.U) {
        rsqrtReg := Mux(varPlusEps > 0.U,
          (1.U << (ScarfConfig.DataWidth - 1)),
          (1.U << (ScarfConfig.AccWidth - 1)))
        newtonIter := 1.U
      }.elsewhen(newtonIter === 1.U) {
        val x = rsqrtReg
        val x2 = (x * x)(ScarfConfig.AccWidth - 1, 0)
        val vx2 = (varPlusEps * x2)(ScarfConfig.AccWidth - 1, 0)
        val threeMinusVx2 = (3.U << (ScarfConfig.DataWidth - 1)) - vx2
        rsqrtReg := (x * threeMinusVx2)(ScarfConfig.AccWidth - 1, 0) >> 1
        newtonIter := 2.U
      }.otherwise {
        state := NormState.sNormalize
        count := 0.U
      }
    }

    is(NormState.sNormalize) {
      when(io.inValid) {
        // FMA: out = (x - μ) × rsqrt × γ + β
        val centered = io.dataIn.pad(ScarfConfig.AccWidth) - meanReg
        val normalized = (centered * rsqrtReg)(ScarfConfig.AccWidth - 1, 0)
        val scaled = (normalized(ScarfConfig.DataWidth - 1, 0) * io.gammaIn)(ScarfConfig.DataWidth - 1, 0)
        io.dataOut  := scaled + io.betaIn
        io.outValid := true.B
        count := count + 1.U
      }
      when(count === normCount - 1.U && io.inValid) {
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
