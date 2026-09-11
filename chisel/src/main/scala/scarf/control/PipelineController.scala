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

    // Payload binding. A TileLoad does not issue a mechanism decision until
    // the source-bound vector and depth probe for this work group are present.
    val tilePayloadStart = Output(Bool())
    val tilePayloadReady = Input(Bool())
    val workGroupIndex = Output(UInt(32.W))

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
    // Number of native T=4 outputs retained by the accepted SAES route.  The
    // S3 Gaussian head uses this count to issue representative sparse work.
    val saesRetainedDescriptors = Output(UInt(5.W))

    val currentTileRow  = Output(UInt(8.W))
    val currentTileCol  = Output(UInt(8.W))
    val totalTileRows   = Output(UInt(8.W))
    val totalTileCols   = Output(UInt(8.W))

    val state = Output(PipeState())
  })

  val state = RegInit(PipeState.sIdle)

  // S2 CostVol launches two independent units once per physical candidate
  // group. Their completion signals are pulses and need not occur in the same
  // cycle, so retain each handshake until both units have completed.
  val costVolMmcuDoneSeen = RegInit(false.B)
  val costVolBilinearDoneSeen = RegInit(false.B)
  val costVolBilinearLaunchPending = RegInit(false.B)
  val costVolPass = RegInit(0.U(8.W))

  val tileRow = RegInit(0.U(8.W))
  val tileCol = RegInit(0.U(8.W))
  val gaussianBatch = RegInit(0.U(32.W))
  val numTileRows = Wire(UInt(8.W))
  val numTileCols = Wire(UInt(8.W))
  // Preserve the complete image domain for ordinary runs, while claim
  // payloads use a bounded work-group model that still depends on every
  // registered workload dimension.  This keeps large real payloads
  // simulatable without reverting to fixed 1/4/4/4 literals.
  numTileRows := (io.config.imageH + io.config.tileSize - 1.U) / io.config.tileSize
  numTileCols := (io.config.imageW + io.config.tileSize - 1.U) / io.config.tileSize
  val imageTileCount = (numTileRows * numTileCols).pad(32)
  val payloadGaussianBatches =
    ((io.config.numGaussians + (ScarfConfig.GGUPECount - 1).U) >> 5).pad(32)
  val payloadTileGroups = ((imageTileCount + 15.U) >> 4).pad(32)
  val payloadDepthGroups = ((io.config.numDepthCandidates + 3.U) >> 2).pad(32)
  val payloadFeatureGroups = ((io.config.featureDim + 3.U) >> 2).pad(32)
  val payloadDepthWork =
    (((payloadTileGroups * payloadDepthGroups * 3.U) + 7.U) >> 3).pad(32) +
      payloadDepthGroups + payloadTileGroups
  val payloadGaussianWork =
    (((payloadGaussianBatches * 3.U) + 7.U) >> 3).pad(32) +
      Mux(payloadGaussianBatches < 8.U, payloadGaussianBatches, 8.U)
  val payloadFeatureWork = (payloadFeatureGroups * 22.U).pad(32)
  val payloadBaseWork = (payloadDepthWork + payloadGaussianWork + payloadFeatureWork).pad(32)
  val totalWorkItems = Mux(
    io.config.payloadValid && io.config.numGaussians =/= 0.U,
    Mux(payloadBaseWork === 0.U, 1.U, payloadBaseWork),
    imageTileCount,
  )

  val cnnLayer = RegInit(0.U(8.W))
  val txLayer  = RegInit(0.U(4.W))
  val dinov2Layer = RegInit(0.U(6.W))
  val saesResult = RegInit(SAESLevel.sFull)
  // The FSDR result is valid only in the completion cycle. Latch the selected
  // candidate count so every following S2 cost-volume pass sees the decision
  // that was actually accepted for this tile.
  val fsdrTileCandidates = RegInit(0.U(8.W))
  // TileLoad ignores the previous group's completion level on its entry
  // cycle. This register owns the request/response lifetime for exactly one
  // bound feature-vector transfer.
  val tilePayloadAwaiting = RegInit(false.B)
  // MMCU keeps `done` asserted until its next launch. A state-entry pulse
  // alone is therefore insufficient: the next pipeline state could consume
  // that stale completion before its own command reaches the idle MMCU.
  val mmcuLaunchPending = RegInit(false.B)
  val mmcuAwaitingDone = RegInit(false.B)

  io.done := state === PipeState.sDone
  io.busy := state =/= PipeState.sIdle
  io.state := state
  io.currentTileRow := tileRow
  io.currentTileCol := tileCol
  io.totalTileRows  := numTileRows
  io.totalTileCols  := numTileCols
  io.workGroupIndex := gaussianBatch

  val prevState = RegNext(state, PipeState.sIdle)
  val stateEntry = state =/= prevState
  val mmcuTaskState = state === PipeState.sS1_CNN ||
    state === PipeState.sS1_Transformer ||
    state === PipeState.sS1_DINOv2 ||
    state === PipeState.sS2_CostVol ||
    state === PipeState.sS2_UNet ||
    state === PipeState.sS2_DepthHead ||
    state === PipeState.sS2_Regression ||
    state === PipeState.sS3_Refine ||
    state === PipeState.sS3_GaussHead
  val mmcuCompletion = !stateEntry && mmcuAwaitingDone && io.mmcuDone
  val costVolCandidatesPerPass = 32.U(8.W)
  val costVolPassCount = Mux(
    fsdrTileCandidates === 0.U,
    1.U(8.W),
    (fsdrTileCandidates + (costVolCandidatesPerPass - 1.U)) >> 5,
  )

  io.mmcuStart         := mmcuTaskState && mmcuLaunchPending
  io.bilinearStart     := state === PipeState.sS2_CostVol && costVolBilinearLaunchPending
  io.gguStart          := false.B
  io.tilePayloadStart  := false.B
  io.saesClassifyStart := false.B
  io.fsdrStart         := false.B
  io.costVolCandidates := fsdrTileCandidates
  io.saesRetainedDescriptors := MuxLookup(saesResult.asUInt, 16.U(5.W))(Seq(
    SAESLevel.sL0.asUInt -> 4.U(5.W),
    SAESLevel.sL1.asUInt -> 12.U(5.W),
    SAESLevel.sFull.asUInt -> 16.U(5.W),
  ))

  when(!mmcuTaskState) {
    mmcuLaunchPending := false.B
    mmcuAwaitingDone := false.B
  }.elsewhen(stateEntry) {
    mmcuLaunchPending := true.B
    mmcuAwaitingDone := false.B
  }.elsewhen(mmcuLaunchPending) {
    mmcuLaunchPending := false.B
    mmcuAwaitingDone := true.B
  }

  when(state =/= PipeState.sS2_CostVol) {
    costVolMmcuDoneSeen := false.B
    costVolBilinearDoneSeen := false.B
    costVolBilinearLaunchPending := false.B
    costVolPass := 0.U
  }.elsewhen(stateEntry) {
    costVolMmcuDoneSeen := false.B
    costVolBilinearDoneSeen := false.B
    costVolBilinearLaunchPending := true.B
    costVolPass := 0.U
  }.otherwise {
    when(costVolBilinearLaunchPending) {
      costVolBilinearLaunchPending := false.B
    }
    when(mmcuCompletion) { costVolMmcuDoneSeen := true.B }
    when(io.bilinearDone) { costVolBilinearDoneSeen := true.B }
  }

  when(state =/= PipeState.sS2S3_TileLoad) {
    tilePayloadAwaiting := false.B
  }.elsewhen(stateEntry) {
    tilePayloadAwaiting := true.B
  }.elsewhen(tilePayloadAwaiting && io.tilePayloadReady) {
    tilePayloadAwaiting := false.B
  }

  switch(state) {
    is(PipeState.sIdle) {
      when(io.start && io.configValid) {
        state := PipeState.sLoadConfig
      }
    }
    is(PipeState.sLoadConfig) {
      cnnLayer := 0.U
      txLayer  := 0.U
      dinov2Layer := 0.U
      tileRow  := 0.U
      tileCol  := 0.U
      gaussianBatch := 0.U
      fsdrTileCandidates := io.config.numDepthCandidates
      state    := PipeState.sS1_CNN
    }

    // S1: CNN backbone (MMCU Conv mode)
    is(PipeState.sS1_CNN) {
      when(mmcuCompletion) {
        cnnLayer := cnnLayer + 1.U
        when(cnnLayer >= io.config.cnnLayers - 1.U) {
          state   := PipeState.sS1_Transformer
          txLayer := 0.U
        }.otherwise {
          mmcuLaunchPending := true.B
          mmcuAwaitingDone := false.B
        }
      }
    }

    // S1: Transformer encoder (MMCU GEMM/Attention mode)
    is(PipeState.sS1_Transformer) {
      when(mmcuCompletion) {
        txLayer := txLayer + 1.U
        when(txLayer >= io.config.transformerLayers - 1.U) {
          state := Mux(
            io.config.hasDINOv2 && io.config.dinov2Layers =/= 0.U,
            PipeState.sS1_DINOv2,
            PipeState.sS2S3_TileLoad,
          )
        }.otherwise {
          mmcuLaunchPending := true.B
          mmcuAwaitingDone := false.B
        }
      }
    }

    // S1: DINOv2 ViT (DepthSplat only, MMCU Attention mode)
    is(PipeState.sS1_DINOv2) {
      when(mmcuCompletion) {
        dinov2Layer := dinov2Layer + 1.U
        when(dinov2Layer >= io.config.dinov2Layers - 1.U) {
          state := PipeState.sS2S3_TileLoad
        }.otherwise {
          mmcuLaunchPending := true.B
          mmcuAwaitingDone := false.B
        }
      }
    }

    // S2+S3 Tile Loop
    is(PipeState.sS2S3_TileLoad) {
      fsdrTileCandidates := io.config.numDepthCandidates
      io.tilePayloadStart := stateEntry
      when(!stateEntry && tilePayloadAwaiting && io.tilePayloadReady) {
        state := Mux(
          io.config.saesEnabled,
          PipeState.sS2S3_SAESClassify,
          Mux(io.config.fsdrEnabled, PipeState.sS2_FSDRLookup, PipeState.sS2_CostVol),
        )
      }
    }
    is(PipeState.sS2S3_SAESClassify) {
      io.saesClassifyStart := stateEntry
      when(io.saesClassifyDone) {
        saesResult := io.saesLevel
        // Without FSDR, each regular L0/L1 route skips one terminal candidate
        // group. Full remains dense.
        when(!io.config.fsdrEnabled && io.saesLevel =/= SAESLevel.sFull &&
          io.config.numDepthCandidates > 32.U) {
          fsdrTileCandidates := io.config.numDepthCandidates - 32.U
        }
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
        val fsdrCandidates = Mux(
          io.fsdrUseNarrow,
          io.fsdrNarrowCands,
          io.config.numDepthCandidates,
        )
        // FSDR owns the depth-search choice in the combined route. Only the
        // representative L0 path can reduce a Full FSDR miss; L1 keeps its
        // S3 retained-output work without applying a second S2 reduction.
        fsdrTileCandidates := Mux(
          saesResult === SAESLevel.sL0 && fsdrCandidates > 32.U,
          fsdrCandidates - 32.U,
          fsdrCandidates,
        )
        state := PipeState.sS2_CostVol
      }
    }

    // S2: Cost Volume (BilinearUnit + MMCU)
    is(PipeState.sS2_CostVol) {
      when((costVolBilinearDoneSeen || io.bilinearDone) &&
           (costVolMmcuDoneSeen || mmcuCompletion)) {
        when(costVolPass + 1.U >= costVolPassCount) {
          state := PipeState.sS2_UNet
        }.otherwise {
          costVolPass := costVolPass + 1.U
          costVolMmcuDoneSeen := false.B
          costVolBilinearDoneSeen := false.B
          costVolBilinearLaunchPending := true.B
          mmcuLaunchPending := true.B
          mmcuAwaitingDone := false.B
        }
      }
    }
    is(PipeState.sS2_UNet) {
      when(mmcuCompletion) { state := PipeState.sS2_DepthHead }
    }
    is(PipeState.sS2_DepthHead) {
      when(mmcuCompletion) { state := PipeState.sS2_Regression }
    }
    is(PipeState.sS2_Regression) {
      when(mmcuCompletion) { state := PipeState.sS3_Refine }
    }

    // S3
    is(PipeState.sS3_Refine) {
      when(mmcuCompletion) { state := PipeState.sS3_GaussHead }
    }
    is(PipeState.sS3_GaussHead) {
      when(mmcuCompletion) { state := PipeState.sGGU }
    }

    // GGU
    is(PipeState.sGGU) {
      io.gguStart := stateEntry
      when(io.gguDone) { state := PipeState.sS2S3_NextTile }
    }

    // Next tile
    is(PipeState.sS2S3_NextTile) {
      // Candidate reduction is charged in sS2_CostVol. SAES cannot change
      // completed work until retained-descriptor materialization is wired.
      gaussianBatch := gaussianBatch + 1.U
      when(tileCol >= numTileCols - 1.U) {
        tileCol := 0.U
        when(tileRow >= numTileRows - 1.U) {
          tileRow := 0.U
        }.otherwise {
          tileRow := tileRow + 1.U
        }
      }.otherwise {
        tileCol := tileCol + 1.U
      }
      when(gaussianBatch + 1.U >= totalWorkItems) {
        state := PipeState.sDone
      }.otherwise {
        state := PipeState.sS2S3_TileLoad
      }
    }

    is(PipeState.sDone) {
      when(!io.start) { state := PipeState.sIdle }
    }
  }
}
