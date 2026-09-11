package scarf.fsdr

import chisel3._
import chisel3.util._
import scarf.{ScarfConfig, ModelConfig}

/**
 * FSDRController — Feature-Similarity Depth Reuse Control FSM.
 *
 * Per-pixel within a tile: hash → lookup → decide → narrow/full → insert → next.
 * Narrowing reduces depth candidates from D to D/4.
 */

object FSDRState extends ChiselEnum {
  val sIdle, sHash, sLookup, sDecide, sNarrow, sFull, sInsert, sNextPixel, sDone = Value
}

class FSDRController extends Module {
  val io = IO(new Bundle {
    val start       = Input(Bool())
    val done        = Output(Bool())
    val busy        = Output(Bool())

    val config      = Input(new ModelConfig)

    // LSH Hash Unit interface
    val hashStart   = Output(Bool())
    val hashDone    = Input(Bool())
    val hashResult  = Input(UInt(ScarfConfig.LSHDim.W))

    // Cache interface
    val cacheLookupEn  = Output(Bool())
    val cacheLookupSig = Output(UInt(ScarfConfig.LSHDim.W))
    val cacheHit       = Input(Bool())
    val cacheHitDepth  = Input(UInt(ScarfConfig.DataWidth.W))
    val localDepthMean = Input(UInt(ScarfConfig.DataWidth.W))
    val localDepthCount = Input(UInt(8.W))

    val cacheInsertEn    = Output(Bool())
    val cacheInsertSig   = Output(UInt(ScarfConfig.LSHDim.W))
    val cacheInsertDepth = Output(UInt(ScarfConfig.DataWidth.W))

    val useNarrowSearch    = Output(Bool())
    val narrowCenterDepth  = Output(UInt(ScarfConfig.DataWidth.W))
    val narrowCandidates   = Output(UInt(8.W))

    val computedDepth      = Input(UInt(ScarfConfig.DataWidth.W))
    // A constant feature vector cannot establish feature-space locality.
    // It must not turn a cache hit into a narrowed depth search.
    val featureInformative = Input(Bool())
    val pixelIdx    = Output(UInt(8.W))
    val totalPixels = Input(UInt(8.W))
  })

  val state = RegInit(FSDRState.sIdle)
  val pixelCounter = RegInit(0.U(8.W))
  val currentSig   = RegInit(0.U(ScarfConfig.LSHDim.W))
  val isNarrow     = RegInit(false.B)
  val cachedDepth  = RegInit(0.U(ScarfConfig.DataWidth.W))
  val localValidity = Module(new FSDRLocalDepthValidity)
  localValidity.io.cachedDepth := io.cacheHitDepth
  localValidity.io.localDepthMean := io.localDepthMean
  localValidity.io.localDepthCount := io.localDepthCount
  localValidity.io.gammaQ10 := io.config.fsdrDepthValidThresh

  io.done := state === FSDRState.sDone
  io.busy := state =/= FSDRState.sIdle
  io.pixelIdx := pixelCounter

  io.hashStart       := false.B
  io.cacheLookupEn   := false.B
  io.cacheLookupSig  := currentSig
  io.cacheInsertEn   := false.B
  io.cacheInsertSig  := currentSig
  io.cacheInsertDepth := io.computedDepth
  io.useNarrowSearch   := isNarrow
  io.narrowCenterDepth := cachedDepth
  io.narrowCandidates  := io.config.numDepthCandidates >> 2

  switch(state) {
    is(FSDRState.sIdle) {
      when(io.start && io.config.fsdrEnabled) {
        state := FSDRState.sHash
        pixelCounter := 0.U
        isNarrow := false.B
      }.elsewhen(io.start && !io.config.fsdrEnabled) {
        state := FSDRState.sDone
      }
    }
    is(FSDRState.sHash) {
      io.hashStart := true.B
      when(io.hashDone) {
        currentSig := io.hashResult
        state := FSDRState.sLookup
      }
    }
    is(FSDRState.sLookup) {
      io.cacheLookupEn  := true.B
      io.cacheLookupSig := currentSig
      state := FSDRState.sDecide
    }
    is(FSDRState.sDecide) {
      when(io.cacheHit && localValidity.io.valid && io.featureInformative) {
        isNarrow    := true.B
        cachedDepth := io.cacheHitDepth
        state       := FSDRState.sNarrow
      }.otherwise {
        isNarrow := false.B
        state    := FSDRState.sFull
      }
    }
    is(FSDRState.sNarrow) {
      state := FSDRState.sInsert
    }
    is(FSDRState.sFull) {
      state := FSDRState.sInsert
    }
    is(FSDRState.sInsert) {
      io.cacheInsertEn    := true.B
      io.cacheInsertSig   := currentSig
      io.cacheInsertDepth := io.computedDepth
      state := FSDRState.sNextPixel
    }
    is(FSDRState.sNextPixel) {
      pixelCounter := pixelCounter + 1.U
      when(pixelCounter >= io.totalPixels - 1.U) {
        state := FSDRState.sDone
      }.otherwise {
        state := FSDRState.sHash
      }
    }
    is(FSDRState.sDone) {
      state := FSDRState.sIdle
      isNarrow := false.B
    }
  }
}
