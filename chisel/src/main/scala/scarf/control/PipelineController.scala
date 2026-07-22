package scarf.control

import chisel3._
import chisel3.util._
import scarf.{ScarfConfig, ModelConfig, PipeState, SAESLevel}

/**
 * PipelineController — Top-level FSM for SCARF pipeline.
 *
 * Manages: IDLE → LOAD_CONFIG → S1_CNN → S1_TRANSFORMER [→ S1_DINOV2]
 *          → S2S3_TILE_LOOP (per tile: SAES decision → S2/S3)
 *          → GGU → DONE
 *
 * MMCU is the single compute unit for Conv/GEMM/Attention (replaces
 * separate ConvEngine + GEMMUnit).
 */
class PipelineController extends Module {
  val io = IO(new Bundle {
    val start     = Input(Bool())
    val done      = Output(Bool())
    val busy      = Output(Bool())
    val config    = Input(new ModelConfig)
    val configValid = Input(Bool())

    // MMCU control (unified)
    val mmcuStart       = Output(Bool())
    val mmcuDone        = Input(Bool())
    val bilinearStart   = Output(Bool())
    val bilinearDone    = Input(Bool())
    val gguStart        = Output(Bool())
    val gguDone         = Input(Bool())

    // SAES
    val saesLevel         = Input(SAESLevel())
    val saesClassifyDone  = Input(Bool())
    val saesClassifyStart = Output(Bool())

    // FSDR
    val fsdrStart         = Output(Bool())
    val fsdrDone          = Input(Bool())
    val fsdrUseNarrow     = Input(Bool())
    val fsdrNarrowCands   = Input(UInt(8.W))

    val costVolCandidates = Output(UInt(8.W))

    val currentTileRow  = Output(UInt(8.W))
    val currentTileCol  = Output(UInt(8.W))
    val totalTileRows   = Output(UInt(8.W))
    val totalTileCols   = Output(UInt(8.W))

    val state = Output(PipeState())
  })

  val state = RegInit(PipeState.sIdle)

  val tileRow = RegInit(0.U(8.W))
  val tileCol = RegInit(0.U(8.W))
  val numTileRows = Wire(UInt(8.W))
  val numTileCols = Wire(UInt(8.W))
  numTileRows := io.config.imageH / io.config.tileSize
  numTileCols := io.config.imageW / io.config.tileSize

  val cnnLayer = RegInit(0.U(8.W))
  val txLayer  = RegInit(0.U(4.W))
  val saesResult = RegInit(SAESLevel.sFull)

  io.done := state === PipeState.sDone
  io.busy := state =/= PipeState.sIdle
  io.state := state
  io.currentTileRow := tileRow
  io.currentTileCol := tileCol
  io.totalTileRows  := numTileRows
  io.totalTileCols  := numTileCols

  val prevState = RegNext(state, PipeState.sIdle)
  val stateEntry = state =/= prevState

  io.mmcuStart         := false.B
  io.bilinearStart     := false.B
  io.gguStart          := false.B
  io.saesClassifyStart := false.B
  io.fsdrStart         := false.B
  io.costVolCandidates := Mux(io.fsdrUseNarrow,
    io.fsdrNarrowCands,
    io.config.numDepthCandidates)

  switch(state) {
    is(PipeState.sIdle) {
      when(io.start && io.configValid) {
        state := PipeState.sLoadConfig
      }
    }
    is(PipeState.sLoadConfig) {
      cnnLayer := 0.U
      txLayer  := 0.U
      tileRow  := 0.U
      tileCol  := 0.U
      state    := PipeState.sS1_CNN
    }

    // S1: CNN backbone (MMCU Conv mode)
    is(PipeState.sS1_CNN) {
      io.mmcuStart := stateEntry
      when(io.mmcuDone) {
        cnnLayer := cnnLayer + 1.U
        when(cnnLayer >= io.config.cnnLayers - 1.U) {
          state   := PipeState.sS1_Transformer
          txLayer := 0.U
        }.otherwise {
          state := PipeState.sS1_CNN
        }
      }
    }

    // S1: Transformer encoder (MMCU GEMM/Attention mode)
    is(PipeState.sS1_Transformer) {
      io.mmcuStart := stateEntry
      when(io.mmcuDone) {
        txLayer := txLayer + 1.U
        when(txLayer >= io.config.transformerLayers - 1.U) {
          state := Mux(io.config.hasDINOv2, PipeState.sS1_DINOv2, PipeState.sS2S3_TileLoad)
        }.otherwise {
          state := PipeState.sS1_Transformer
        }
      }
    }

    // S1: DINOv2 ViT (DepthSplat only, MMCU Attention mode)
    is(PipeState.sS1_DINOv2) {
      io.mmcuStart := stateEntry
      when(io.mmcuDone) {
        state := PipeState.sS2S3_TileLoad
      }
    }

    // S2+S3 Tile Loop
    is(PipeState.sS2S3_TileLoad) {
      state := Mux(io.config.saesEnabled, PipeState.sS2S3_SAESClassify, PipeState.sS2_CostVol)
    }
    is(PipeState.sS2S3_SAESClassify) {
      io.saesClassifyStart := stateEntry
      when(io.saesClassifyDone) {
        saesResult := io.saesLevel
        // The controller cannot legally bypass S2/S3 merely because a tile
        // classifies as L0/L1.  Probe execution, assignment/moment matching,
        // and retained-descriptor buffering need a real datapath; until that
        // datapath is implemented, every level follows the baseline S2/S3
        // route.  This preserves functional semantics and prevents the RTL
        // control skeleton from implying unimplemented SAES savings.
        state := Mux(
          io.config.fsdrEnabled,
          PipeState.sS2_FSDRLookup,
          PipeState.sS2_CostVol,
        )
      }
    }
    is(PipeState.sS2_FSDRLookup) {
      io.fsdrStart := stateEntry
      when(io.fsdrDone) {
        state := PipeState.sS2_CostVol
      }
    }

    // S2: Cost Volume (BilinearUnit + MMCU)
    is(PipeState.sS2_CostVol) {
      io.bilinearStart := stateEntry
      io.mmcuStart     := stateEntry
      when(io.bilinearDone && io.mmcuDone) {
        state := PipeState.sS2_UNet
      }
    }
    is(PipeState.sS2_UNet) {
      io.mmcuStart := stateEntry
      when(io.mmcuDone) { state := PipeState.sS2_DepthHead }
    }
    is(PipeState.sS2_DepthHead) {
      io.mmcuStart := stateEntry
      when(io.mmcuDone) { state := PipeState.sS2_Regression }
    }
    is(PipeState.sS2_Regression) {
      io.mmcuStart := stateEntry
      when(io.mmcuDone) { state := PipeState.sS3_Refine }
    }

    // S3
    is(PipeState.sS3_Refine) {
      io.mmcuStart := stateEntry
      when(io.mmcuDone) { state := PipeState.sS3_GaussHead }
    }
    is(PipeState.sS3_GaussHead) {
      io.mmcuStart := stateEntry
      when(io.mmcuDone) { state := PipeState.sGGU }
    }

    // GGU
    is(PipeState.sGGU) {
      io.gguStart := stateEntry
      when(io.gguDone) { state := PipeState.sS2S3_NextTile }
    }

    // Next tile
    is(PipeState.sS2S3_NextTile) {
      tileCol := tileCol + 1.U
      when(tileCol >= numTileCols - 1.U) {
        tileCol := 0.U
        tileRow := tileRow + 1.U
        when(tileRow >= numTileRows - 1.U) {
          state := PipeState.sDone
        }.otherwise {
          state := PipeState.sS2S3_TileLoad
        }
      }.otherwise {
        state := PipeState.sS2S3_TileLoad
      }
    }

    is(PipeState.sDone) {
      when(!io.start) { state := PipeState.sIdle }
    }
  }
}
