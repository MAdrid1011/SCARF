package scarf.control

import chisel3._
import chisel3.util._
import scarf.{ScarfConfig, ModelConfig, PipeState, SAESLevel}

/**
 * PipelineController — Top-level FSM for SCARF pipeline orchestration.
 *
 * Corresponds to: depth_predictor/hw_depth_predictor.py (HWDepthPredictor)
 *
 * Manages the full inference pipeline:
 *   IDLE → LOAD_CONFIG → S1_CNN → S1_TRANSFORMER [→ S1_DINOV2]
 *        → S2S3_TILE_LOOP (per tile: SAES_CLASSIFY → S2/S3 or PROBE_ONLY)
 *        → GGU → DONE
 *
 * Key design: single FSM controls all compute unit allocation.
 * No model-specific paths — all branching is based on numeric config values.
 */
class PipelineController extends Module {
  val io = IO(new Bundle {
    // External control
    val start     = Input(Bool())
    val done      = Output(Bool())
    val busy      = Output(Bool())

    // Configuration input (from ConfigRegs)
    val config    = Input(new ModelConfig)
    val configValid = Input(Bool())

    // Compute unit control signals (active-high enable for each unit)
    val convEngineStart = Output(Bool())
    val convEngineDone  = Input(Bool())
    val gemmStart       = Output(Bool())
    val gemmDone        = Input(Bool())
    val bilinearStart   = Output(Bool())
    val bilinearDone    = Input(Bool())
    val gguStart        = Output(Bool())
    val gguDone         = Input(Bool())

    // SAES classification result (from SAESController)
    val saesLevel       = Input(SAESLevel())
    val saesClassifyDone = Input(Bool())
    val saesClassifyStart = Output(Bool())

    // Tile tracking
    val currentTileRow  = Output(UInt(8.W))
    val currentTileCol  = Output(UInt(8.W))
    val totalTileRows   = Output(UInt(8.W))
    val totalTileCols   = Output(UInt(8.W))

    // Current FSM state (for debug/monitoring)
    val state           = Output(PipeState())
  })

  // FSM state register
  val state = RegInit(PipeState.sIdle)

  // Tile counters
  val tileRow = RegInit(0.U(8.W))
  val tileCol = RegInit(0.U(8.W))
  val numTileRows = Wire(UInt(8.W))
  val numTileCols = Wire(UInt(8.W))
  numTileRows := io.config.imageH / io.config.tileSize
  numTileCols := io.config.imageW / io.config.tileSize

  // CNN layer counter (for S1)
  val cnnLayer = RegInit(0.U(8.W))
  // Transformer layer counter
  val txLayer = RegInit(0.U(4.W))

  // SAES classification result latch
  val saesResult = RegInit(SAESLevel.sFull)

  // Default outputs
  io.done := state === PipeState.sDone
  io.busy := state =/= PipeState.sIdle
  io.state := state
  io.currentTileRow := tileRow
  io.currentTileCol := tileCol
  io.totalTileRows  := numTileRows
  io.totalTileCols  := numTileCols

  // ── Edge-sensitive start pulse logic ──
  // Each compute unit receives a 1-cycle start pulse when entering
  // its state, preventing spurious re-triggering when the unit
  // returns to idle while the controller is still in the same state.
  val prevState = RegNext(state, PipeState.sIdle)
  val stateEntry = state =/= prevState  // True on first cycle of new state

  io.convEngineStart   := false.B
  io.gemmStart         := false.B
  io.bilinearStart     := false.B
  io.gguStart          := false.B
  io.saesClassifyStart := false.B

  switch(state) {
    // ──── IDLE: Wait for start signal ────
    is(PipeState.sIdle) {
      when(io.start && io.configValid) {
        state := PipeState.sLoadConfig
      }
    }

    // ──── LOAD_CONFIG: Latch configuration ────
    is(PipeState.sLoadConfig) {
      cnnLayer := 0.U
      txLayer  := 0.U
      tileRow  := 0.U
      tileCol  := 0.U
      state    := PipeState.sS1_CNN
    }

    // ──── S1: Feature Extraction — CNN backbone ────
    // ConvEngine processes all CNN layers sequentially
    is(PipeState.sS1_CNN) {
      io.convEngineStart := stateEntry  // 1-cycle pulse on state entry
      when(io.convEngineDone) {
        cnnLayer := cnnLayer + 1.U
        when(cnnLayer >= io.config.cnnLayers - 1.U) {
          state  := PipeState.sS1_Transformer
          txLayer := 0.U
        }.otherwise {
          // Re-trigger for next layer: transition to self forces stateEntry
          state := PipeState.sS1_CNN
        }
      }
    }

    // ──── S1: Feature Extraction — Transformer encoder ────
    // GEMM Unit processes QKV projections + attention
    is(PipeState.sS1_Transformer) {
      io.gemmStart := stateEntry  // 1-cycle pulse
      when(io.gemmDone) {
        txLayer := txLayer + 1.U
        when(txLayer >= io.config.transformerLayers - 1.U) {
          state := Mux(io.config.hasDINOv2, PipeState.sS1_DINOv2, PipeState.sS2S3_TileLoad)
        }.otherwise {
          state := PipeState.sS1_Transformer  // Re-trigger for next layer
        }
      }
    }

    // ──── S1: DINOv2 ViT (DepthSplat only) ────
    // Uses GEMM Unit for ViT self-attention layers
    // Controlled by hasDINOv2 config — NOT a model-specific branch
    is(PipeState.sS1_DINOv2) {
      io.gemmStart := stateEntry
      when(io.gemmDone) {
        state := PipeState.sS2S3_TileLoad
      }
    }

    // ──── S2+S3 Fused Tile Loop: Load tile ────
    is(PipeState.sS2S3_TileLoad) {
      // Load tile features from FeatureBuffer into TileSPM
      // (1-2 cycles, handled by memory controller)
      state := Mux(io.config.saesEnabled, PipeState.sS2S3_SAESClassify, PipeState.sS2_CostVol)
    }

    // ──── SAES: Classify tile (L0/L1/L2/Full) ────
    is(PipeState.sS2S3_SAESClassify) {
      io.saesClassifyStart := stateEntry
      when(io.saesClassifyDone) {
        saesResult := io.saesLevel
        state := Mux(
          io.saesLevel === SAESLevel.sFull,
          PipeState.sS2_CostVol,
          PipeState.sS2S3_ProbeOnly,
        )
      }
    }

    // ──── S2: Cost Volume (BilinearUnit + ConvEngine) ────
    is(PipeState.sS2_CostVol) {
      io.bilinearStart   := stateEntry
      io.convEngineStart := stateEntry
      when(io.bilinearDone && io.convEngineDone) {
        state := PipeState.sS2_UNet
      }
    }

    // ──── S2: U-Net refinement (ConvEngine) ────
    is(PipeState.sS2_UNet) {
      io.convEngineStart := stateEntry
      when(io.convEngineDone) {
        state := PipeState.sS2_DepthHead
      }
    }

    // ──── S2: Depth Head (ConvEngine 1×1 conv) ────
    is(PipeState.sS2_DepthHead) {
      io.convEngineStart := stateEntry
      when(io.convEngineDone) {
        state := PipeState.sS2_Regression
      }
    }

    // ──── S2: Depth Regression (GEMM + VectorALU softmax) ────
    is(PipeState.sS2_Regression) {
      io.gemmStart := stateEntry
      when(io.gemmDone) {
        state := PipeState.sS3_Refine
      }
    }

    // ──── S3: Refine U-Net (ConvEngine) ────
    is(PipeState.sS3_Refine) {
      io.convEngineStart := stateEntry
      when(io.convEngineDone) {
        state := PipeState.sS3_GaussHead
      }
    }

    // ──── S3: Gaussian Head (ConvEngine 1×1 conv) ────
    is(PipeState.sS3_GaussHead) {
      io.convEngineStart := stateEntry
      when(io.convEngineDone) {
        state := PipeState.sGGU
      }
    }

    // ──── Lightweight Probe Path (SAES L0/L1/L2) ────
    is(PipeState.sS2S3_ProbeOnly) {
      io.bilinearStart := stateEntry
      when(io.bilinearDone) {
        state := PipeState.sGGU
      }
    }

    // ──── GGU: Gaussian post-processing (32 PEs) ────
    is(PipeState.sGGU) {
      io.gguStart := stateEntry
      when(io.gguDone) {
        state := PipeState.sS2S3_NextTile
      }
    }

    // ──── Next Tile: advance tile counters ────
    is(PipeState.sS2S3_NextTile) {
      tileCol := tileCol + 1.U
      when(tileCol >= numTileCols - 1.U) {
        tileCol := 0.U
        tileRow := tileRow + 1.U
        when(tileRow >= numTileRows - 1.U) {
          state := PipeState.sDone  // All tiles processed
        }.otherwise {
          state := PipeState.sS2S3_TileLoad  // Next row
        }
      }.otherwise {
        state := PipeState.sS2S3_TileLoad  // Next column
      }
    }

    // ──── DONE: Inference complete ────
    is(PipeState.sDone) {
      when(!io.start) {
        state := PipeState.sIdle  // Return to idle when start deasserted
      }
    }
  }
}
