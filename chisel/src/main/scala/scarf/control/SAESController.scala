package scarf.control

import chisel3._
import chisel3.util._
import scarf.{ScarfConfig, ModelConfig, SAESLevel}

/**
 * SAESController — SAES Tile Classification FSM.
 *
 * First-hit classification: L0 (feature) → L1 (depth) → Full.
 */
class SAESController extends Module {
  val io = IO(new Bundle {
    val start    = Input(Bool())
    val done     = Output(Bool())
    val config   = Input(new ModelConfig)

    val probeFeatureVar = Input(UInt(ScarfConfig.DataWidth.W))
    val probeDepthStd   = Input(UInt(ScarfConfig.DataWidth.W))
    val crossCheckError = Input(UInt(ScarfConfig.DataWidth.W))
    // Probe-only materialization checks are Control inputs. Their producer
    // reads only selected native descriptors; an L0 rejection still permits
    // the existing L1 depth decision, while an L1 rejection is fail-closed.
    val l0MaterializationValid = Input(Bool())
    val l1MaterializationValid = Input(Bool())

    val level          = Output(SAESLevel())
    // Valid only with ``done``.  This traces the accepted-start-to-result
    // controller latency so the software SAES event ledger can check its
    // L0 versus L1/Full classification counts against real RTL behavior.
    val decisionCycles = Output(UInt(2.W))
  })

  val sIdle :: sCheckL0 :: sCheckL1 :: sResult :: Nil = Enum(4)
  val state = RegInit(sIdle)
  val resultLevel = RegInit(SAESLevel.sFull)
  val decisionCycles = RegInit(0.U(2.W))

  io.done := state === sResult
  io.level := resultLevel
  io.decisionCycles := Mux(io.done, decisionCycles, 0.U)

  switch(state) {
    is(sIdle) {
      when(io.start) {
        state := Mux(io.config.saesEnabled, sCheckL0, sResult)
        resultLevel := SAESLevel.sFull
        decisionCycles := 1.U
      }
    }
    is(sCheckL0) {
      decisionCycles := decisionCycles + 1.U
      when(
        io.probeFeatureVar < io.config.saesFeatureVarThresh &&
          io.l0MaterializationValid
      ) {
        resultLevel := SAESLevel.sL0
        state := sResult
      }.otherwise {
        state := sCheckL1
      }
    }
    is(sCheckL1) {
      decisionCycles := decisionCycles + 1.U
      when(
        io.probeDepthStd < io.config.saesDepthStdThresh &&
          io.l1MaterializationValid
      ) {
        resultLevel := SAESLevel.sL1
      }.otherwise {
        resultLevel := SAESLevel.sFull
      }
      state := sResult
    }
    is(sResult) {
      state := sIdle
    }
  }
}
