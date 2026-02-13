package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * SoftmaxUnit — Softmax + Depth Regression.
 *
 * Corresponds to: encoder/softmax_unit.py
 *
 * Two-phase operation:
 *   Phase 1: exp(x_i) / sum(exp(x_j)) → probability distribution
 *   Phase 2: sum(prob_i * candidate_i) → expected depth value
 *
 * Uses VectorALU for exp computation and reduction.
 */
object SoftmaxState extends ChiselEnum {
  val sIdle, sExpSum, sRegress, sDone = Value
}

class SoftmaxUnit extends Module {
  val io = IO(new Bundle {
    val start       = Input(Bool())
    val done        = Output(Bool())
    val busy        = Output(Bool())

    // Input logits (streamed)
    val logitIn     = Input(UInt(ScarfConfig.DataWidth.W))
    val candidateIn = Input(UInt(ScarfConfig.DataWidth.W))  // Depth candidate value
    val inValid     = Input(Bool())
    val numElements = Input(UInt(8.W))  // Number of depth candidates (32/64/128)

    // Output
    val probOut     = Output(UInt(ScarfConfig.DataWidth.W))  // Probability
    val depthOut    = Output(UInt(ScarfConfig.AccWidth.W))   // Expected depth
    val outValid    = Output(Bool())
  })

  val state   = RegInit(SoftmaxState.sIdle)
  val doneReg = RegInit(false.B)
  val busyReg = RegInit(false.B)

  // Accumulators
  val expSum    = RegInit(0.U(ScarfConfig.AccWidth.W))
  val depthAcc  = RegInit(0.U(ScarfConfig.AccWidth.W))
  val count     = RegInit(0.U(8.W))

  // exp approximation via shift (structural model: exp(x) ≈ 1 + x for small x)
  val expVal = Wire(UInt(ScarfConfig.DataWidth.W))
  expVal := io.logitIn + (1 << (ScarfConfig.DataWidth - 2)).U  // Shift to positive range

  io.done     := doneReg
  io.busy     := busyReg
  io.probOut  := 0.U
  io.depthOut := depthAcc
  io.outValid := false.B

  switch(state) {
    is(SoftmaxState.sIdle) {
      doneReg := false.B
      busyReg := false.B
      when(io.start) {
        state   := SoftmaxState.sExpSum
        busyReg := true.B
        expSum  := 0.U
        count   := 0.U
      }
    }

    is(SoftmaxState.sExpSum) {
      // Phase 1a: accumulate sum of exp(logits)
      when(io.inValid) {
        expSum := expSum + expVal
        count  := count + 1.U
      }
      when(count === io.numElements - 1.U && io.inValid) {
        state    := SoftmaxState.sRegress
        count    := 0.U
        depthAcc := 0.U
      }
    }

    is(SoftmaxState.sRegress) {
      // Phase 2: prob = exp(x_i) / sum, depth += prob * candidate
      when(io.inValid) {
        val prob = (expVal << 16) / (expSum + 1.U)  // Fixed-point division
        val contribution = prob * io.candidateIn
        depthAcc := depthAcc + contribution(ScarfConfig.AccWidth - 1, 0)
        io.probOut  := prob(ScarfConfig.DataWidth - 1, 0)
        io.outValid := true.B
        count := count + 1.U
      }
      when(count === io.numElements - 1.U && io.inValid) {
        state := SoftmaxState.sDone
      }
    }

    is(SoftmaxState.sDone) {
      doneReg := true.B
      busyReg := false.B
      io.outValid := true.B  // Final depth value available
      state := SoftmaxState.sIdle
    }
  }
}
