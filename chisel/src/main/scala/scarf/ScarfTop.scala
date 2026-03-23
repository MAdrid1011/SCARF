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

  val cameraFx = RegInit("h4500".U(32.W))
  val cameraFy = RegInit("h4500".U(32.W))
  val cameraCx = RegInit("h4000".U(32.W))
  val cameraCy = RegInit("h4000".U(32.W))
  val extrinsics = RegInit(VecInit(Seq.fill(12)(0.U(32.W))))
  val cameraLoadCounter = RegInit(0.U(2.W))

  val weightAccumReg = RegInit(VecInit(Seq.fill(6)(0.U(128.W))))
  val weightBeatCount = RegInit(0.U(3.W))

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
  io.done                 := pipeline.io.done
  io.busy                 := pipeline.io.busy
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

  saesCtrl.io.probeFeatureVar := featureBuf.io.doutA
  saesCtrl.io.probeDepthStd   := tileBuf.io.rdData
  saesCtrl.io.crossCheckError := featureBuf.io.doutB

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
    lshHash.io.featureIn(i) := featureBuf.io.doutA
  }

  // FSDRController ↔ FSDRCache
  fsdrCache.io.lookupEn      := fsdrCtrl.io.cacheLookupEn
  fsdrCache.io.lookupSig     := fsdrCtrl.io.cacheLookupSig
  fsdrCache.io.hammingThresh := configRegs.io.config.fsdrHammingThresh
  fsdrCtrl.io.cacheHit       := fsdrCache.io.hit
  fsdrCtrl.io.cacheHitDepth  := fsdrCache.io.hitDepth

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

  fsdrCtrl.io.computedDepth := tileBuf.io.rdData
  fsdrCtrl.io.totalPixels   := configRegs.io.config.tileSize * configRegs.io.config.tileSize

  // ═══════════════════════════════════════════════
  // MMCU parameter wiring
  // ═══════════════════════════════════════════════
  mmcu.io.inChannels  := configRegs.io.config.featureDim.pad(10)
  mmcu.io.outChannels := configRegs.io.config.featureDim.pad(10)
  mmcu.io.kernelSize  := 3.U
  mmcu.io.stride      := 1.U
  mmcu.io.M           := configRegs.io.config.featureDim.pad(16)
  mmcu.io.K           := configRegs.io.config.featureDim.pad(16)
  mmcu.io.N           := configRegs.io.config.featureDim.pad(16)
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
  vectorALU.io.smNumElements := configRegs.io.config.numDepthCandidates

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
  normUnit.io.channels := configRegs.io.config.featureDim.pad(10)
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
    gguArray.io.scaleX(i)    := featureBuf.io.doutA
    gguArray.io.scaleY(i)    := featureBuf.io.doutA
    gguArray.io.scaleZ(i)    := featureBuf.io.doutA
    gguArray.io.quatW(i)     := featureBuf.io.doutA
    gguArray.io.quatX(i)     := featureBuf.io.doutA
    gguArray.io.quatY(i)     := featureBuf.io.doutA
    gguArray.io.quatZ(i)     := featureBuf.io.doutA
    gguArray.io.opacityIn(i) := featureBuf.io.doutB
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
    mmcu.io.aData(i) := featureBuf.io.doutA
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
  dramIF.io.readReq.valid       := pipeline.io.state === PipeState.sLoadConfig
  dramIF.io.readReq.bits.addr   := dramWeightBase + dmaOffset
  dramIF.io.readReq.bits.burstLen := 15.U
  dramIF.io.readResp.ready      := true.B

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
