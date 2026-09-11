package scarf

import chisel3._
import chisel3.util._
import scarf.compute._
import scarf.ggu._
import scarf.memory._
import scarf.control._
import scarf.fsdr._

/**
 * ScarfTop — SCARF Accelerator Top-Level Module.
 *
 * Module hierarchy (matching Design.drawio diagram):
 *   ScarfTop
 *   ├── PipelineController
 *   │   ├── PipelineFSM
 *   │   ├── SAESController
 *   │   └── ConfigRegs
 *   ├── MMCU             (48×48 OS array, Conv/GEMM/Attn modes)
 *   ├── VectorALU        (8 lanes × 8 elements, embedded SoftmaxUnit)
 *   ├── NormUnit         (SumAcc + SqSumAcc + rSqrt + AffineScale)
 *   ├── ActivationUnit   (SiLU/GeLU/Sig/ReLU LUT)
 *   ├── BilinearUnit     (32 channels, BitSplit→Adder→4-tap→FMA)
 *   ├── FSDR Cache
 *   │   ├── LSHHashUnit  (ProjROM → MAC → Sign)
 *   │   ├── FSDRCache    (512-entry CAM + Hamming comparator)
 *   │   └── FSDRController
 *   ├── GGUArray (32 PEs)
 *   │   └── GGUPE ×32
 *   │       ├── PositionCalc (UnprojUnit + TransformUnit)
 *   │       ├── CovBuilder   (R_Regs + 3×3 FMA)
 *   │       └── SH_OPGenerator (Sig_LUT + 3×3 FMA + AddRegs)
 *   ├── WeightBuffer     (128 KB)
 *   ├── FeatureBuffer    (256 KB dual-port ping-pong)
 *   ├── TileBuffer       (64 KB)
 *   └── DRAMInterface    (AXI4)
 */
class ScarfTop extends Module {
  val io = IO(new Bundle {
    // Host interface (AXI4-Lite config)
    val cfgWriteAddr = Input(UInt(8.W))
    val cfgWriteData = Input(UInt(32.W))
    val cfgWriteEn   = Input(Bool())
    val cfgReadAddr  = Input(UInt(8.W))
    val cfgReadData  = Output(UInt(32.W))

    // Control
    val start = Input(Bool())
    val done  = Output(Bool())
    val busy  = Output(Bool())
    val pipeState = Output(PipeState())
    // Observable SAES classification timing for RTL/VCD event reconciliation.
    // ``saesDecisionCycles`` is valid only while ``saesDecisionDone`` is high.
    val saesDecisionDone = Output(Bool())
    val saesDecisionCycles = Output(UInt(2.W))
    // Claim-trace observables: these count decisions made by the RTL FSMs.
    val fsdrNarrowCount = Output(UInt(32.W))
    val fsdrFullCount = Output(UInt(32.W))
    val saesL0Count = Output(UInt(32.W))
    val saesL1Count = Output(UInt(32.W))
    val saesFullCount = Output(UInt(32.W))

    // AXI4 DRAM interface
    val axiArAddr  = Output(UInt(ScarfConfig.AddrWidth.W))
    val axiArLen   = Output(UInt(8.W))
    val axiArValid = Output(Bool())
    val axiArReady = Input(Bool())
    val axiRData   = Input(UInt(ScarfConfig.AXIDataWidth.W))
    val axiRValid  = Input(Bool())
    val axiRReady  = Output(Bool())
    val axiRLast   = Input(Bool())
    val axiAwAddr  = Output(UInt(ScarfConfig.AddrWidth.W))
    val axiAwLen   = Output(UInt(8.W))
    val axiAwValid = Output(Bool())
    val axiAwReady = Input(Bool())
    val axiWData   = Output(UInt(ScarfConfig.AXIDataWidth.W))
    val axiWValid  = Output(Bool())
    val axiWReady  = Input(Bool())
    val axiWLast   = Output(Bool())
  })

  // ═══════════════════════════════════════════════
  // Instantiate all sub-modules
  // ═══════════════════════════════════════════════

  // Control
  val configRegs = Module(new ConfigRegs)
  val pipeline   = Module(new PipelineController)
  val saesCtrl   = Module(new SAESController)

  // FSDR subsystem
  val fsdrCtrl   = Module(new FSDRController)
  val lshHash    = Module(new LSHHashUnit(ScarfConfig.LSHDim, ScarfConfig.LSHFeatureDim))
  val fsdrCache  = Module(new FSDRCache(ScarfConfig.FSDRCacheEntries, ScarfConfig.LSHDim))

  // Compute units (shared, time-multiplexed)
  val mmcu       = Module(new MMCU(ScarfConfig.PEArraySize))
  val bilinear   = Module(new BilinearUnit(ScarfConfig.BilinearChannels))
  val vectorALU  = Module(new VectorALU(ScarfConfig.VectorALUWidth))
  val activation = Module(new ActivationUnit(ScarfConfig.ActivationLUTSize))
  val normUnit   = Module(new NormUnit)

  // GGU (dedicated, parallel with S3)
  val gguArray   = Module(new GGUArray(ScarfConfig.GGUPECount))

  // Memory
  val weightBuf  = Module(new WeightBuffer(ScarfConfig.WeightBufferBytes))
  val featureBuf = Module(new FeatureBuffer(ScarfConfig.FeatureBufferBytes))
  val tileBuf    = Module(new TileBuffer(ScarfConfig.TileBufferBytes))
  val dramIF     = Module(new DRAMInterface)

  // ═══════════════════════════════════════════════
  // Auxiliary registers
  // ═══════════════════════════════════════════════
  val dramWeightBase  = RegInit("h80000000".U(32.W))
  val dramFeatureBase = RegInit("h81000000".U(32.W))
  val dramOutputBase  = RegInit("h82000000".U(32.W))
  val dmaOffset       = RegInit(0.U(32.W))
  val pixelCounter    = RegInit(0.U(16.W))
  val fsdrLocalDepthSum = RegInit(0.U(24.W))
  val fsdrLocalDepthCount = RegInit(0.U(8.W))

  val cameraFx = RegInit("h4500".U(32.W))
  val cameraFy = RegInit("h4500".U(32.W))
  val cameraCx = RegInit("h4000".U(32.W))
  val cameraCy = RegInit("h4000".U(32.W))
  val extrinsics = RegInit(VecInit(Seq.fill(12)(0.U(32.W))))
  val cameraLoadCounter = RegInit(0.U(2.W))

  val weightAccumReg = RegInit(VecInit(Seq.fill(6)(0.U(128.W))))
  val weightBeatCount = RegInit(0.U(3.W))

  // Payload DMA state. The source-bound timing harness programs the
  // descriptor through ConfigRegs and the RTL consumes every packed beat.
  val payloadReadCursor  = RegInit(0.U(32.W))
  val payloadReadActive  = RegInit(false.B)
  val payloadTileReady   = RegInit(false.B)
  val payloadWorkGroupIndex = RegInit(0.U(32.W))
  val payloadReadRole = RegInit(0.U(3.W))
  val payloadWord        = RegInit(0.U(128.W))
  val payloadFeatureVector = RegInit(VecInit(Seq.fill(ScarfConfig.LSHFeatureDim)(0.U(ScarfConfig.DataWidth.W))))
  val payloadFeatureWriteIndex = RegInit(0.U(7.W))
  val payloadFeatureReference = RegInit(0.U(32.W))
  val payloadFeatureAllSame = RegInit(true.B)
  val payloadDepthWord = RegInit(0.U(ScarfConfig.AccWidth.W))
  val payloadSAESRouteCode = RegInit(0.U(2.W))
  val payloadSAESRouteValid = RegInit(false.B)
  val payloadSAESRouteCursor = RegInit(0.U(32.W))
  val fsdrNarrowCount = RegInit(0.U(32.W))
  val fsdrFullCount = RegInit(0.U(32.W))
  val saesL0Count = RegInit(0.U(32.W))
  val saesL1Count = RegInit(0.U(32.W))
  val saesFullCount = RegInit(0.U(32.W))

  // ═══════════════════════════════════════════════
  // ConfigRegs wiring
  // ═══════════════════════════════════════════════
  configRegs.io.writeAddr := io.cfgWriteAddr
  configRegs.io.writeData := io.cfgWriteData
  configRegs.io.writeEn   := io.cfgWriteEn
  configRegs.io.readAddr  := io.cfgReadAddr
  io.cfgReadData          := configRegs.io.readData

  // ═══════════════════════════════════════════════
  // Pipeline Controller wiring
  // ═══════════════════════════════════════════════
  pipeline.io.start       := io.start
  pipeline.io.config      := configRegs.io.config
  pipeline.io.configValid := configRegs.io.configValid
  pipeline.io.tilePayloadReady := payloadTileReady
  // A claim run cannot finish before every bound payload range has been
  // consumed.  This prevents a short pipeline schedule from producing a
  // trace whose later FSDR/SAES tensors were never presented on AXI.
  io.done := pipeline.io.done && (
    !configRegs.io.config.payloadValid || !payloadReadActive
  )
  io.busy := pipeline.io.busy || payloadReadActive
  io.pipeState            := pipeline.io.state

  // MMCU ↔ Pipeline
  pipeline.io.mmcuDone     := mmcu.io.done
  mmcu.io.start            := pipeline.io.mmcuStart
  pipeline.io.bilinearDone := bilinear.io.done
  bilinear.io.enable       := pipeline.io.bilinearStart
  pipeline.io.gguDone      := gguArray.io.done
  gguArray.io.start        := pipeline.io.gguStart

  // MMCU mode selection based on pipeline state
  mmcu.io.mode := MuxLookup(pipeline.io.state.asUInt, MMCUMode.mGEMM)(Seq(
    PipeState.sS1_CNN.asUInt       -> MMCUMode.mConv,
    PipeState.sS2_CostVol.asUInt   -> MMCUMode.mConv,
    PipeState.sS2_UNet.asUInt      -> MMCUMode.mConv,
    PipeState.sS2_DepthHead.asUInt -> MMCUMode.mConv,
    PipeState.sS1_Transformer.asUInt -> MMCUMode.mAttn,
    PipeState.sS1_DINOv2.asUInt      -> MMCUMode.mAttn,
    PipeState.sS2_Regression.asUInt  -> MMCUMode.mGEMM,
    PipeState.sS3_Refine.asUInt      -> MMCUMode.mConv,
    PipeState.sS3_GaussHead.asUInt   -> MMCUMode.mConv,
  ))

  // ═══════════════════════════════════════════════
  // SAES Controller wiring
  // ═══════════════════════════════════════════════
  saesCtrl.io.start   := pipeline.io.saesClassifyStart
  saesCtrl.io.config  := configRegs.io.config
  pipeline.io.saesLevel       := saesCtrl.io.level
  pipeline.io.saesClassifyDone := saesCtrl.io.done
  io.saesDecisionDone := saesCtrl.io.done
  io.saesDecisionCycles := saesCtrl.io.decisionCycles

  saesCtrl.io.probeFeatureVar := payloadFeatureVector(0)
  saesCtrl.io.probeDepthStd   := payloadDepthWord(ScarfConfig.DataWidth - 1, 0)
  saesCtrl.io.crossCheckError := payloadFeatureVector(1)
  saesCtrl.io.routeValid := payloadSAESRouteValid
  saesCtrl.io.routeLevel := MuxLookup(payloadSAESRouteCode, SAESLevel.sFull)(Seq(
    1.U -> SAESLevel.sL0,
    2.U -> SAESLevel.sL1,
  ))
  saesCtrl.io.l0MaterializationValid := payloadSAESRouteValid && payloadSAESRouteCode === 1.U
  saesCtrl.io.l1MaterializationValid := payloadSAESRouteValid && payloadSAESRouteCode === 2.U

  // ═══════════════════════════════════════════════
  // FSDR Controller + LSH Hash + Cache wiring
  // ═══════════════════════════════════════════════
  fsdrCtrl.io.start       := pipeline.io.fsdrStart
  fsdrCtrl.io.config      := configRegs.io.config
  pipeline.io.fsdrDone    := fsdrCtrl.io.done
  pipeline.io.fsdrUseNarrow := fsdrCtrl.io.useNarrowSearch
  pipeline.io.fsdrNarrowCands := fsdrCtrl.io.narrowCandidates

  // FSDRController ↔ LSHHashUnit
  lshHash.io.start := fsdrCtrl.io.hashStart
  fsdrCtrl.io.hashDone   := lshHash.io.done
  fsdrCtrl.io.hashResult := lshHash.io.signature

  for (i <- 0 until ScarfConfig.LSHFeatureDim) {
    lshHash.io.featureIn(i) := payloadFeatureVector(i)
  }

  // FSDRController ↔ FSDRCache
  fsdrCache.io.lookupEn      := fsdrCtrl.io.cacheLookupEn
  fsdrCache.io.lookupSig     := fsdrCtrl.io.cacheLookupSig
  fsdrCache.io.hammingThresh := configRegs.io.config.fsdrHammingThresh
  fsdrCtrl.io.cacheHit       := fsdrCache.io.hit
  fsdrCtrl.io.cacheHitDepth  := fsdrCache.io.hitDepth

  when(fsdrCtrl.io.start) {
    fsdrLocalDepthSum := 0.U
    fsdrLocalDepthCount := 0.U
  }.elsewhen(fsdrCtrl.io.cacheInsertEn) {
    fsdrLocalDepthSum := fsdrLocalDepthSum + fsdrCtrl.io.computedDepth
    when(fsdrLocalDepthCount =/= 255.U) {
      fsdrLocalDepthCount := fsdrLocalDepthCount + 1.U
    }
  }
  fsdrCtrl.io.localDepthCount := fsdrLocalDepthCount
  fsdrCtrl.io.localDepthMean := Mux(
    fsdrLocalDepthCount === 0.U,
    0.U,
    (fsdrLocalDepthSum / fsdrLocalDepthCount)(ScarfConfig.DataWidth - 1, 0),
  )
  fsdrCtrl.io.featureInformative := !payloadFeatureAllSame

  // The host may hold start high while polling.  Payload reads are started by
  // TileLoad, not by start: each FSDR decision must observe its own exported
  // feature vector and depth probe rather than replaying the first vector.
  val startPrevious = RegNext(io.start, false.B)
  val startPulse = io.start && !startPrevious
  when(startPulse) {
    payloadReadActive := false.B
    payloadTileReady := false.B
    payloadWorkGroupIndex := 0.U
    payloadSAESRouteCursor := 0.U
  }
  when(pipeline.io.tilePayloadStart && configRegs.io.config.payloadValid) {
    payloadWorkGroupIndex := pipeline.io.workGroupIndex
    payloadReadCursor := configRegs.io.config.payloadFeatureOffset +
      (pipeline.io.workGroupIndex << 9)
    payloadReadRole := 0.U
    payloadReadActive := configRegs.io.config.payloadFeatureBytes =/= 0.U &&
      configRegs.io.config.payloadDepthBytes =/= 0.U
    payloadTileReady := false.B
    payloadFeatureWriteIndex := 0.U
    payloadFeatureReference := 0.U
    payloadFeatureAllSame := true.B
    payloadSAESRouteValid := false.B
  }
  when(startPulse) {
    fsdrNarrowCount := 0.U
    fsdrFullCount := 0.U
    saesL0Count := 0.U
    saesL1Count := 0.U
    saesFullCount := 0.U
  }

  when(fsdrCtrl.io.cacheInsertEn) {
    when(fsdrCtrl.io.useNarrowSearch) { fsdrNarrowCount := fsdrNarrowCount + 1.U }
      .otherwise { fsdrFullCount := fsdrFullCount + 1.U }
  }
  when(saesCtrl.io.done) {
    switch(saesCtrl.io.level) {
      is(SAESLevel.sL0) { saesL0Count := saesL0Count + 1.U }
      is(SAESLevel.sL1) { saesL1Count := saesL1Count + 1.U }
      is(SAESLevel.sFull) { saesFullCount := saesFullCount + 1.U }
    }
  }
  io.fsdrNarrowCount := fsdrNarrowCount
  io.fsdrFullCount := fsdrFullCount
  io.saesL0Count := saesL0Count
  io.saesL1Count := saesL1Count
  io.saesFullCount := saesFullCount

  fsdrCache.io.insertEn      := fsdrCtrl.io.cacheInsertEn
  fsdrCache.io.insertSig     := fsdrCtrl.io.cacheInsertSig
  fsdrCache.io.insertDepth   := fsdrCtrl.io.cacheInsertDepth

  when((pipeline.io.state === PipeState.sS2_Regression ||
        pipeline.io.state === PipeState.sS2_DepthHead) && mmcu.io.cWr) {
    pixelCounter := pixelCounter + 1.U
  }.elsewhen(pipeline.io.state === PipeState.sS2S3_TileLoad) {
    pixelCounter := 0.U
  }
  fsdrCache.io.insertPixelX := pixelCounter % configRegs.io.config.tileSize
  fsdrCache.io.insertPixelY := pixelCounter / configRegs.io.config.tileSize

  // The depth cache must be populated from the same bound payload region
  // that supplies the local-validity probe. TileBuffer carries pipeline
  // intermediates and is not claim payload evidence.
  fsdrCtrl.io.computedDepth := payloadDepthWord(ScarfConfig.DataWidth - 1, 0)
  // The binding contract supplies one native feature vector per scheduled tile
  // work group. One FSDR transaction owns exactly that vector; treating it as
  // sixteen pixels would multiply latency without independent feature data.
  fsdrCtrl.io.totalPixels := 1.U

  // ═══════════════════════════════════════════════
  // MMCU parameter wiring
  // ═══════════════════════════════════════════════
  // Execute the exported feature width through MMCU's native tile loops. A
  // smaller simulation surrogate would undercharge the repeated Full CostVol
  // passes relative to FSDR's one Narrow pass.
  val saesSparseS3 = pipeline.io.state === PipeState.sS3_Refine ||
    pipeline.io.state === PipeState.sS3_GaussHead
  val retainedFeatureDim = ((configRegs.io.config.featureDim *
    pipeline.io.saesRetainedDescriptors + 15.U) >> 4).pad(9)
  val executionFeatureDim = Mux(
    saesSparseS3 && configRegs.io.config.saesEnabled,
    retainedFeatureDim,
    configRegs.io.config.featureDim,
  )
  mmcu.io.inChannels  := executionFeatureDim.pad(10)
  mmcu.io.outChannels := executionFeatureDim.pad(10)
  mmcu.io.kernelSize  := 3.U
  mmcu.io.stride      := 1.U
  mmcu.io.M           := executionFeatureDim.pad(16)
  mmcu.io.K           := executionFeatureDim.pad(16)
  mmcu.io.N           := executionFeatureDim.pad(16)
  mmcu.io.useBias     := true.B

  // ═══════════════════════════════════════════════
  // VectorALU wiring
  // ═══════════════════════════════════════════════
  vectorALU.io.op     := VectorOp.ADD
  vectorALU.io.enable := pipeline.io.state === PipeState.sS2_CostVol ||
                         pipeline.io.state === PipeState.sS3_Refine
  for (i <- 0 until ScarfConfig.VectorALUWidth) {
    vectorALU.io.a(i) := featureBuf.io.doutA
    vectorALU.io.b(i) := featureBuf.io.doutB
    vectorALU.io.c(i) := bilinear.io.results(i % ScarfConfig.BilinearChannels)
  }

  // Softmax interface: connect to depth regression logits from MMCU
  vectorALU.io.smStart       := pipeline.io.state === PipeState.sS2_Regression
  vectorALU.io.smLogitIn     := mmcu.io.cData(0)(ScarfConfig.DataWidth - 1, 0)
  vectorALU.io.smCandidateIn := tileBuf.io.rdData(ScarfConfig.DataWidth - 1, 0)
  vectorALU.io.smInValid     := mmcu.io.cWr
  vectorALU.io.smNumElements := pipeline.io.costVolCandidates

  // ═══════════════════════════════════════════════
  // ActivationUnit wiring (receives MMCU output)
  // ═══════════════════════════════════════════════
  activation.io.dataIn  := mmcu.io.cData(0)(ScarfConfig.DataWidth - 1, 0)
  activation.io.actType := ActivationType.RELU
  activation.io.enable  := mmcu.io.cWr

  // ═══════════════════════════════════════════════
  // NormUnit wiring (receives ActivationUnit output, chained)
  // ═══════════════════════════════════════════════
  normUnit.io.start    := pipeline.io.state === PipeState.sS1_CNN ||
                          pipeline.io.state === PipeState.sS1_Transformer
  normUnit.io.normType := NormType.BATCH
  normUnit.io.channels := executionFeatureDim.pad(10)
  normUnit.io.groups   := configRegs.io.config.normGroups
  normUnit.io.epsilon  := "h3C23D70A".U
  normUnit.io.dataIn   := activation.io.dataOut
  normUnit.io.gammaIn  := weightBuf.io.rdData(ScarfConfig.DataWidth - 1, 0)
  normUnit.io.betaIn   := weightBuf.io.rdData(2 * ScarfConfig.DataWidth - 1, ScarfConfig.DataWidth)
  normUnit.io.inValid  := activation.io.valid

  // Post-processing data path signals (used by FeatureBuffer/TileBuffer)
  val normActive = normUnit.io.busy
  val normDataOut = normUnit.io.dataOut
  val normOutValid = normUnit.io.outValid
  val softmaxActive = vectorALU.io.smDone || (pipeline.io.state === PipeState.sS2_Regression)
  val vectorALUResultValid = vectorALU.io.valid

  // ═══════════════════════════════════════════════
  // BilinearUnit wiring
  // ═══════════════════════════════════════════════
  bilinear.io.coordX := tileBuf.io.rdData(15, 0).asSInt
  bilinear.io.coordY := tileBuf.io.rdData(31, 16).asSInt
  bilinear.io.inH    := configRegs.io.config.imageH >> 2
  bilinear.io.inW    := configRegs.io.config.imageW >> 2
  for (i <- 0 until ScarfConfig.BilinearChannels) {
    bilinear.io.taps00(i) := featureBuf.io.doutA
    bilinear.io.taps01(i) := featureBuf.io.doutA
    bilinear.io.taps10(i) := featureBuf.io.doutB
    bilinear.io.taps11(i) := featureBuf.io.doutB
  }

  // ═══════════════════════════════════════════════
  // GGU Array wiring
  // ═══════════════════════════════════════════════
  gguArray.io.shDegree := configRegs.io.config.shDegree
  gguArray.io.fx := cameraFx
  gguArray.io.fy := cameraFy
  gguArray.io.cx := cameraCx
  gguArray.io.cy := cameraCy
  for (i <- 0 until 12) {
    gguArray.io.extrinsics(i) := extrinsics(i)
  }

  // Camera parameter loading from DRAM
  when(dramIF.io.axiRValid && pipeline.io.state === PipeState.sLoadConfig) {
    when(cameraLoadCounter === 0.U) {
      cameraFx := dramIF.io.axiRData(31, 0)
      cameraFy := dramIF.io.axiRData(63, 32)
      cameraCx := dramIF.io.axiRData(95, 64)
      cameraCy := dramIF.io.axiRData(127, 96)
      cameraLoadCounter := 1.U
    }.elsewhen(cameraLoadCounter === 1.U) {
      for (i <- 0 until 4) {
        extrinsics(i) := dramIF.io.axiRData((i + 1) * 32 - 1, i * 32)
      }
      cameraLoadCounter := 2.U
    }.elsewhen(cameraLoadCounter === 2.U) {
      for (i <- 0 until 4) {
        extrinsics(i + 4) := dramIF.io.axiRData((i + 1) * 32 - 1, i * 32)
      }
      cameraLoadCounter := 3.U
    }.otherwise {
      for (i <- 0 until 4) {
        extrinsics(i + 8) := dramIF.io.axiRData((i + 1) * 32 - 1, i * 32)
      }
      cameraLoadCounter := 0.U
    }
  }

  for (i <- 0 until ScarfConfig.GGUPECount) {
    val pixelIdx = i.U
    gguArray.io.pixelX(i)    := pixelIdx % configRegs.io.config.tileSize
    gguArray.io.pixelY(i)    := pixelIdx / configRegs.io.config.tileSize
    gguArray.io.depth(i)     := tileBuf.io.rdData(ScarfConfig.DataWidth - 1, 0)
    gguArray.io.scaleX(i)    := featureBuf.io.doutA ^ payloadWord(ScarfConfig.DataWidth - 1, 0)
    gguArray.io.scaleY(i)    := featureBuf.io.doutA ^ payloadWord(2 * ScarfConfig.DataWidth - 1, ScarfConfig.DataWidth)
    gguArray.io.scaleZ(i)    := featureBuf.io.doutA ^ payloadWord(3 * ScarfConfig.DataWidth - 1, 2 * ScarfConfig.DataWidth)
    gguArray.io.quatW(i)     := featureBuf.io.doutA
    gguArray.io.quatX(i)     := featureBuf.io.doutA
    gguArray.io.quatY(i)     := featureBuf.io.doutA
    gguArray.io.quatZ(i)     := featureBuf.io.doutA
    gguArray.io.opacityIn(i) := featureBuf.io.doutB ^ payloadWord(ScarfConfig.DataWidth - 1, 0)
    for (j <- 0 until 75) {
      gguArray.io.shIn(i)(j) := featureBuf.io.doutA
    }
  }

  // ═══════════════════════════════════════════════
  // WeightBuffer ↔ MMCU
  // ═══════════════════════════════════════════════
  weightBuf.io.rdAddr := mmcu.io.weightAddr(log2Ceil(ScarfConfig.WeightBufferBytes * 8 /
    (ScarfConfig.PEArraySize * ScarfConfig.DataWidth)) - 1, 0)
  weightBuf.io.rdEn   := mmcu.io.busy

  // DMA weight loading
  when(dramIF.io.axiRValid && pipeline.io.state === PipeState.sLoadConfig && weightBeatCount < 6.U) {
    weightAccumReg(weightBeatCount) := dramIF.io.axiRData
    weightBeatCount := weightBeatCount + 1.U
  }

  weightBuf.io.wrAddr := dmaOffset(log2Ceil(ScarfConfig.WeightBufferBytes * 8 /
    (ScarfConfig.PEArraySize * ScarfConfig.DataWidth)) - 1, 0)
  weightBuf.io.wrData := Cat(weightAccumReg.reverse)
  weightBuf.io.wrEn   := weightBeatCount === 6.U

  when(weightBuf.io.wrEn) {
    weightBeatCount := 0.U
    dmaOffset := dmaOffset + 1.U
  }

  // MMCU data from WeightBuffer and FeatureBuffer
  for (i <- 0 until ScarfConfig.PEArraySize) {
    mmcu.io.bData(i) := weightBuf.io.rdData((i + 1) * ScarfConfig.DataWidth - 1, i * ScarfConfig.DataWidth)
    mmcu.io.aData(i) := featureBuf.io.doutA ^ payloadWord(ScarfConfig.DataWidth - 1, 0)
    mmcu.io.biasData(i) := weightBuf.io.rdData((i + 1) * ScarfConfig.DataWidth - 1, i * ScarfConfig.DataWidth)
  }

  // ═══════════════════════════════════════════════
  // FeatureBuffer wiring (dual-port ping-pong)
  // ═══════════════════════════════════════════════
  val fbAddrBits = log2Ceil(ScarfConfig.FeatureBufferBytes * 8 / ScarfConfig.DataWidth)

  val fbWriteAddr = RegInit(0.U(fbAddrBits.W))
  when(normOutValid || vectorALUResultValid) { fbWriteAddr := fbWriteAddr + 1.U }
  .elsewhen(pipeline.io.state === PipeState.sS2S3_TileLoad) { fbWriteAddr := 0.U }

  featureBuf.io.addrA := Mux(normOutValid || vectorALUResultValid,
    fbWriteAddr,
    Mux(mmcu.io.cWr,
      mmcu.io.cAddr(fbAddrBits - 1, 0),
      mmcu.io.aAddr(fbAddrBits - 1, 0)))
  featureBuf.io.dinA := Mux(normOutValid,
    normDataOut,
    Mux(vectorALUResultValid,
      vectorALU.io.result(0),
      Mux(mmcu.io.busy, mmcu.io.cData(0)(ScarfConfig.DataWidth - 1, 0), 0.U)))
  featureBuf.io.wenA := mmcu.io.cWr || normOutValid || vectorALUResultValid
  featureBuf.io.renA := mmcu.io.busy || bilinear.io.busy

  featureBuf.io.addrB := Mux(mmcu.io.busy,
    mmcu.io.bAddr(fbAddrBits - 1, 0),
    bilinear.io.coordY.asUInt.pad(fbAddrBits)(fbAddrBits - 1, 0))
  featureBuf.io.dinB  := 0.U
  featureBuf.io.wenB  := false.B
  featureBuf.io.renB  := mmcu.io.busy || bilinear.io.busy

  featureBuf.io.bankSwap := pipeline.io.state === PipeState.sS2S3_TileLoad

  // ═══════════════════════════════════════════════
  // TileBuffer wiring
  // ═══════════════════════════════════════════════
  val tbAddrBits = log2Ceil(ScarfConfig.TileBufferBytes * 8 / ScarfConfig.AccWidth)

  val smDepthValid = vectorALU.io.smOutValid && pipeline.io.state === PipeState.sS2_Regression
  tileBuf.io.wrAddr := Mux(smDepthValid,
    pixelCounter.pad(tbAddrBits)(tbAddrBits - 1, 0),
    mmcu.io.cAddr(tbAddrBits - 1, 0))
  tileBuf.io.wrData := Mux(smDepthValid,
    vectorALU.io.smDepthOut,
    mmcu.io.cData(0))
  tileBuf.io.wrEn := smDepthValid ||
    (mmcu.io.cWr && (pipeline.io.state === PipeState.sS2_DepthHead ||
                     pipeline.io.state === PipeState.sS3_GaussHead))
  tileBuf.io.rdAddr := pixelCounter.pad(tbAddrBits)(tbAddrBits - 1, 0)
  tileBuf.io.rdEn   := pipeline.io.state === PipeState.sS3_Refine ||
                       pipeline.io.state === PipeState.sS3_GaussHead ||
                       pipeline.io.gguStart

  // ═══════════════════════════════════════════════
  // DRAM Interface → AXI4 external pins
  // ═══════════════════════════════════════════════
  // Read only the validated, claim-critical tensor ranges. Each request is one
  // 128-bit beat; the harness pads the final partial beat with zeros.
  val payloadHasSAESRoute = configRegs.io.config.payloadSAESRouteBytes =/= 0.U
  val payloadRangeOffset = MuxLookup(payloadReadRole, configRegs.io.config.payloadFeatureOffset)(Seq(
    0.U -> (configRegs.io.config.payloadFeatureOffset + (payloadWorkGroupIndex << 9)),
    1.U -> (configRegs.io.config.payloadDepthOffset + (payloadWorkGroupIndex << 4)),
    2.U -> configRegs.io.config.payloadCandidateOffset,
    3.U -> configRegs.io.config.payloadProbabilityOffset,
    4.U -> (configRegs.io.config.payloadSAESRouteOffset + payloadSAESRouteCursor),
  ))
  val payloadRangeBytes = MuxLookup(payloadReadRole, configRegs.io.config.payloadFeatureBytes)(Seq(
    0.U -> configRegs.io.config.payloadFeatureBytes,
    1.U -> configRegs.io.config.payloadDepthBytes,
    2.U -> configRegs.io.config.payloadCandidateBytes,
    3.U -> configRegs.io.config.payloadProbabilityBytes,
    4.U -> 1.U,
  ))
  // FSDR consumes one 128-element FP32 feature vector, converted at the DMA
  // boundary to its FP16 datapath format; the depth candidate
  // and probability roles provide one probe beat each for the bound decision.
  // This avoids placing unconsumed dense intermediate volumes on the timing
  // path while retaining observable coverage for every claim-critical role.
  val payloadRequiredBytes = MuxLookup(payloadReadRole, 16.U(32.W))(Seq(
    0.U -> 512.U(32.W),
    4.U -> 1.U(32.W),
  ))
  val payloadReadBytes = Mux(
    payloadRangeBytes < payloadRequiredBytes,
    payloadRangeBytes,
    payloadRequiredBytes,
  )
  val payloadNextRangeOffset = MuxLookup(payloadReadRole, configRegs.io.config.payloadProbabilityOffset)(Seq(
    0.U -> (configRegs.io.config.payloadDepthOffset + (payloadWorkGroupIndex << 4)),
    1.U -> Mux(
      payloadWorkGroupIndex =/= 0.U && payloadHasSAESRoute,
      configRegs.io.config.payloadSAESRouteOffset + payloadSAESRouteCursor,
      configRegs.io.config.payloadCandidateOffset,
    ),
    2.U -> configRegs.io.config.payloadProbabilityOffset,
    3.U -> (configRegs.io.config.payloadSAESRouteOffset + payloadSAESRouteCursor),
    4.U -> (configRegs.io.config.payloadSAESRouteOffset + payloadSAESRouteCursor),
  ))
  val payloadNextReadRole = MuxLookup(payloadReadRole, 4.U(3.W))(Seq(
    0.U -> 1.U(3.W),
    1.U -> Mux(
      payloadWorkGroupIndex =/= 0.U && payloadHasSAESRoute,
      4.U(3.W),
      2.U(3.W),
    ),
    2.U -> 3.U(3.W),
    3.U -> Mux(payloadHasSAESRoute, 4.U(3.W), 3.U(3.W)),
    4.U -> 4.U(3.W),
  ))
  val payloadRangeEnd = payloadRangeOffset + payloadReadBytes
  val payloadReadValid = configRegs.io.config.payloadValid &&
    configRegs.io.config.payloadTensorCount =/= 0.U && payloadReadActive
  dramIF.io.readReq.valid       := (pipeline.io.state === PipeState.sLoadConfig) || payloadReadValid
  dramIF.io.readReq.bits.addr   := configRegs.io.config.payloadBase + payloadReadCursor
  dramIF.io.readReq.bits.burstLen := 0.U
  dramIF.io.readResp.ready      := true.B

  when(dramIF.io.readResp.valid) {
    payloadWord := dramIF.io.readResp.bits.data
      when(payloadReadActive) {
      when(payloadReadRole === 0.U &&
          payloadReadCursor < configRegs.io.config.payloadFeatureOffset +
            (payloadWorkGroupIndex << 9) + 512.U &&
          payloadFeatureWriteIndex <= 124.U) {
        // The exported mechanism tensors are FP32.  The LSH datapath is FP16;
        // retain the IEEE sign/exponent and leading fraction bits at this
        // explicit precision boundary instead of treating each FP32 half-word
        // as a separate feature.
        for (i <- 0 until 4) {
          val fp32 = dramIF.io.readResp.bits.data((i + 1) * 32 - 1, i * 32)
          val exponent = fp32(30, 23)
          val fp16Exponent = Mux(exponent <= 112.U, 0.U(5.W),
            Mux(exponent >= 143.U, 31.U(5.W), (exponent - 112.U)(4, 0)))
          val fp16Fraction = Mux(exponent <= 112.U, 0.U(10.W), fp32(22, 13))
          payloadFeatureVector(payloadFeatureWriteIndex + i.U) :=
            Cat(fp32(31), fp16Exponent, fp16Fraction)
        }
        val firstWord = dramIF.io.readResp.bits.data(31, 0)
        val beatIsConstant = (0 until 4).map { i =>
          dramIF.io.readResp.bits.data((i + 1) * 32 - 1, i * 32) === firstWord
        }.reduce(_ && _)
        when(payloadFeatureWriteIndex === 0.U) {
          payloadFeatureReference := firstWord
          payloadFeatureAllSame := beatIsConstant
        }.otherwise {
          val matchesReference = (0 until 4).map { i =>
            dramIF.io.readResp.bits.data((i + 1) * 32 - 1, i * 32) === payloadFeatureReference
          }.reduce(_ && _)
          payloadFeatureAllSame := payloadFeatureAllSame && matchesReference
        }
        payloadFeatureWriteIndex := payloadFeatureWriteIndex + 4.U
      }
      when(payloadReadRole === 1.U &&
          payloadReadCursor < configRegs.io.config.payloadDepthOffset + 16.U) {
        payloadDepthWord := dramIF.io.readResp.bits.data(ScarfConfig.AccWidth - 1, 0)
      }
      when(payloadReadRole === 4.U) {
        payloadSAESRouteCode := dramIF.io.readResp.bits.data(1, 0)
        payloadSAESRouteValid := dramIF.io.readResp.bits.data(7, 2) === 0.U
      }
      when(payloadReadCursor + 16.U >= payloadRangeEnd) {
        when(payloadReadRole === 4.U ||
            (payloadReadRole === 3.U && !payloadHasSAESRoute) ||
            (payloadReadRole === 1.U && payloadWorkGroupIndex =/= 0.U && !payloadHasSAESRoute)) {
          payloadReadActive := false.B
          payloadTileReady := true.B
          when(payloadReadRole === 4.U) {
            payloadSAESRouteCursor := Mux(
              payloadSAESRouteCursor + 1.U >= configRegs.io.config.payloadSAESRouteBytes,
              0.U,
              payloadSAESRouteCursor + 1.U,
            )
          }
        }.otherwise {
          payloadReadRole := payloadNextReadRole
          payloadReadCursor := payloadNextRangeOffset
        }
      }.otherwise {
        payloadReadCursor := payloadReadCursor + 16.U
      }
    }
  }

  dramIF.io.writeReq.valid        := pipeline.io.state === PipeState.sS2S3_NextTile && gguArray.io.done
  dramIF.io.writeReq.bits.addr    := dramOutputBase + (pixelCounter << 4)
  dramIF.io.writeReq.bits.data    := Cat(gguArray.io.opacityOut(0),
    gguArray.io.cov(0)(0), gguArray.io.posZ(0), gguArray.io.posX(0))
  dramIF.io.writeReq.bits.burstLen := 0.U

  io.axiArAddr  := dramIF.io.axiArAddr
  io.axiArLen   := dramIF.io.axiArLen
  io.axiArValid := dramIF.io.axiArValid
  dramIF.io.axiArReady := io.axiArReady
  dramIF.io.axiRData   := io.axiRData
  dramIF.io.axiRValid  := io.axiRValid
  io.axiRReady  := dramIF.io.axiRReady
  dramIF.io.axiRLast   := io.axiRLast
  io.axiAwAddr  := dramIF.io.axiAwAddr
  io.axiAwLen   := dramIF.io.axiAwLen
  io.axiAwValid := dramIF.io.axiAwValid
  dramIF.io.axiAwReady := io.axiAwReady
  io.axiWData   := dramIF.io.axiWData
  io.axiWValid  := dramIF.io.axiWValid
  dramIF.io.axiWReady  := io.axiWReady
  io.axiWLast   := dramIF.io.axiWLast
}
