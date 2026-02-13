package scarf.control

import chisel3._
import chisel3.util._
import scarf.{ScarfConfig, ModelConfig}

/**
 * FSGRController — Feature-Similarity Gaussian Reuse Control FSM.
 *
 * Corresponds to: fsgr/narrowed_search_simulator.py (FSGRSimulator)
 *
 * Integrates with PipelineController's S2_CostVol state:
 *   Before CostVol: hash current pixel's feature → query FSGRCache
 *   If hit:  narrow depth search from D candidates to D/4 (centered on cached depth)
 *   If miss: run full D-candidate CostVol, then insert result into cache
 *
 * Design: operates per-pixel within a tile. For a 4×4 tile (16 pixels),
 * FSGR processes each pixel before its CostVol computation.
 *
 * FSM states:
 *   IDLE → HASH → LOOKUP → DECIDE → NARROW/FULL → INSERT → NEXT_PIXEL → DONE
 *
 * The narrowing only affects CostVol (BilinearUnit): reduces depth candidates
 * from `numDepthCandidates` to `numDepthCandidates/4`. UNet, DepthHead, and
 * Regression still process the full image — FSGR savings are CostVol-only.
 */

object FSGRState extends ChiselEnum {
  val sIdle, sHash, sLookup, sDecide, sNarrow, sFull, sInsert, sNextPixel, sDone = Value
}

class FSGRController extends Module {
  val io = IO(new Bundle {
    // Control
    val start       = Input(Bool())
    val done        = Output(Bool())
    val busy        = Output(Bool())

    // Configuration (from ConfigRegs)
    val config      = Input(new ModelConfig)

    // LSH Hash Unit interface
    val hashStart   = Output(Bool())
    val hashDone    = Input(Bool())
    val hashResult  = Input(UInt(16.W))  // LSH signature

    // Cache interface
    val cacheLookupEn  = Output(Bool())
    val cacheLookupSig = Output(UInt(16.W))
    val cacheHit       = Input(Bool())
    val cacheHitDepth  = Input(UInt(ScarfConfig.DataWidth.W))

    val cacheInsertEn    = Output(Bool())
    val cacheInsertSig   = Output(UInt(16.W))
    val cacheInsertDepth = Output(UInt(ScarfConfig.DataWidth.W))

    // Depth search control (to PipelineController / BilinearUnit)
    val useNarrowSearch    = Output(Bool())        // True = use D/4 candidates
    val narrowCenterDepth  = Output(UInt(ScarfConfig.DataWidth.W))  // Center of narrow window
    val narrowCandidates   = Output(UInt(8.W))     // Number of candidates (D/4)

    // Current pixel's computed depth (from S2 regression output, for cache insert)
    val computedDepth      = Input(UInt(ScarfConfig.DataWidth.W))

    // Pixel counter
    val pixelIdx    = Output(UInt(8.W))            // Current pixel within tile
    val totalPixels = Input(UInt(8.W))             // Pixels in current tile (16 for 4×4)
  })

  val state = RegInit(FSGRState.sIdle)
  val pixelCounter = RegInit(0.U(8.W))
  val currentSig   = RegInit(0.U(16.W))
  val isNarrow     = RegInit(false.B)
  val cachedDepth  = RegInit(0.U(ScarfConfig.DataWidth.W))

  // Default outputs
  io.done := state === FSGRState.sDone
  io.busy := state =/= FSGRState.sIdle
  io.pixelIdx := pixelCounter

  io.hashStart       := false.B
  io.cacheLookupEn   := false.B
  io.cacheLookupSig  := currentSig
  io.cacheInsertEn   := false.B
  io.cacheInsertSig  := currentSig
  io.cacheInsertDepth := io.computedDepth
  io.useNarrowSearch   := isNarrow
  io.narrowCenterDepth := cachedDepth
  io.narrowCandidates  := io.config.numDepthCandidates >> 2  // D/4

  switch(state) {
    is(FSGRState.sIdle) {
      when(io.start && io.config.fsgrEnabled) {
        state := FSGRState.sHash
        pixelCounter := 0.U
        isNarrow := false.B
      }.elsewhen(io.start && !io.config.fsgrEnabled) {
        // FSGR disabled: skip directly to done
        state := FSGRState.sDone
      }
    }

    is(FSGRState.sHash) {
      // Trigger LSH hash computation for current pixel's features
      io.hashStart := true.B
      when(io.hashDone) {
        currentSig := io.hashResult
        state := FSGRState.sLookup
      }
    }

    is(FSGRState.sLookup) {
      // Query cache with the computed signature
      io.cacheLookupEn  := true.B
      io.cacheLookupSig := currentSig
      state := FSGRState.sDecide  // Result available next cycle
    }

    is(FSGRState.sDecide) {
      // Check cache hit/miss
      when(io.cacheHit) {
        // Hit: use narrowed search centered on cached depth
        isNarrow    := true.B
        cachedDepth := io.cacheHitDepth
        state       := FSGRState.sNarrow
      }.otherwise {
        // Miss: run full search, then insert into cache
        isNarrow := false.B
        state    := FSGRState.sFull
      }
    }

    is(FSGRState.sNarrow) {
      // Signal narrowed search to PipelineController
      // PipelineController will use io.narrowCandidates instead of full D
      // This state persists for 1 cycle (signal is latched by pipeline)
      state := FSGRState.sInsert
    }

    is(FSGRState.sFull) {
      // Full search — signal is already isNarrow=false
      // After S2 completes for this pixel, insert into cache
      state := FSGRState.sInsert
    }

    is(FSGRState.sInsert) {
      // Insert computed depth into cache for future reuse
      io.cacheInsertEn    := true.B
      io.cacheInsertSig   := currentSig
      io.cacheInsertDepth := io.computedDepth
      state := FSGRState.sNextPixel
    }

    is(FSGRState.sNextPixel) {
      pixelCounter := pixelCounter + 1.U
      when(pixelCounter >= io.totalPixels - 1.U) {
        state := FSGRState.sDone
      }.otherwise {
        state := FSGRState.sHash  // Process next pixel
      }
    }

    is(FSGRState.sDone) {
      state := FSGRState.sIdle
      isNarrow := false.B
    }
  }
}
