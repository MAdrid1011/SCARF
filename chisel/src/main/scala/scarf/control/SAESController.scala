package scarf.control

import chisel3._
import chisel3.util._
import scarf.{ScarfConfig, ModelConfig, SAESLevel}

/**
 * SAESController — SAES Tile Classification FSM.
 *
 * 3-level progressive classification: L0 (feature) → L1 (depth) → L2/Full.
 */
class SAESController extends Module {
  val io = IO(new Bundle {
    val start    = Input(Bool())
    val done     = Output(Bool())
    val config   = Input(new ModelConfig)

    val probeFeatureVar = Input(UInt(ScarfConfig.DataWidth.W))
    val probeDepthStd   = Input(UInt(ScarfConfig.DataWidth.W))
    val crossCheckError = Input(UInt(ScarfConfig.DataWidth.W))

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
        resultLevel := SAESLevel.sFull
      }
    }
    is(sCheckL0) {
      when(io.probeFeatureVar < io.config.saesFeatureVarThresh) {
        resultLevel := SAESLevel.sL0
        state := sResult
      }.otherwise {
        state := sCheckL1
      }
    }
    is(sCheckL1) {
      when(io.probeDepthStd < io.config.saesDepthStdThresh) {
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
      when(io.crossCheckError < io.config.saesCrossCheckThresh) {
        resultLevel := SAESLevel.sL2
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
