package scarf

import chisel3._
import chisel3.util._
import scarf.compute._
import scarf.ggu._
import scarf.memory._
import scarf.control._

/**
 * ScarfTop — SCARF Accelerator Top-Level Module.
 *
 * Integrates all compute units, memory subsystem, GGU array,
 * and pipeline controller into a single ASIC-ready top module.
 *
 * Key design principle: SINGLE INSTANCE of each compute unit,
 * time-multiplexed across pipeline stages via PipelineController FSM.
 * No model-specific data paths.
 *
 * Module hierarchy:
 *   ScarfTop
 *   ├── ConfigRegs          (MMIO registers)
 *   ├── PipelineController  (top-level FSM)
 *   ├── SAESController      (tile classification)
 *   ├── ConvEngine          (48×48 systolic array × 1)
 *   ├── GEMMUnit            (48×48 OS array × 1)
 *   ├── BilinearUnit        (32 samplers × 1)
 *   ├── VectorALU           (64-wide SIMD × 1)
 *   ├── ActivationUnit      (LUT × 1)
 *   ├── NormUnit            (× 1)
 *   ├── SoftmaxUnit         (× 1)
 *   ├── GGUArray            (32 PEs)
 *   ├── WeightBuffer        (128 KB)
 *   ├── FeatureBuffer       (256 KB)
 *   ├── TileSPM             (64 KB)
 *   └── DRAMInterface       (AXI4)
 */
class ScarfTop extends Module {
  val io = IO(new Bundle {
    // Host interface (AXI4-Lite for config)
    val cfgWriteAddr = Input(UInt(8.W))
    val cfgWriteData = Input(UInt(32.W))
    val cfgWriteEn   = Input(Bool())
    val cfgReadAddr  = Input(UInt(8.W))
    val cfgReadData  = Output(UInt(32.W))

    // Control
    val start = Input(Bool())
    val done  = Output(Bool())
    val busy  = Output(Bool())

    // Current state (debug)
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

  // ════════════════════════════════════════════════════════
  // Instantiate all sub-modules (each exactly ONCE)
  // ════════════════════════════════════════════════════════

  // Control
  val configRegs = Module(new ConfigRegs)
  val pipeline   = Module(new PipelineController)
  val saesCtrl   = Module(new SAESController)
  val fsgrCtrl   = Module(new FSGRController)

  // FSGR hardware (LSH hashing + semantic cache)
  val lshHash    = Module(new scarf.compute.LSHHashUnit(lshDim = 16, featureDim = 128))
  val fsgrCache  = Module(new scarf.memory.FSGRCache(numEntries = 512, sigWidth = 16))

  // Compute units (shared, time-multiplexed)
  val convEngine = Module(new ConvEngine(ScarfConfig.PEArraySize))
  val gemmUnit   = Module(new GEMMUnit(ScarfConfig.PEArraySize))
  val bilinear   = Module(new BilinearUnit(ScarfConfig.BilinearChannels))
  val vectorALU  = Module(new VectorALU(ScarfConfig.VectorALUWidth))
  val activation = Module(new ActivationUnit(ScarfConfig.ActivationLUTSize))
  val normUnit   = Module(new NormUnit)
  val softmax    = Module(new SoftmaxUnit)

  // GGU (dedicated, runs parallel with S3)
  val gguArray   = Module(new GGUArray(ScarfConfig.GGUPECount))

  // Memory
  val weightBuf  = Module(new WeightBuffer(ScarfConfig.WeightBufferBytes))
  val featureBuf = Module(new FeatureBuffer(ScarfConfig.FeatureBufferBytes))
  val tileSPM    = Module(new TileSPM(ScarfConfig.TileSPMBytes))
  val dramIF     = Module(new DRAMInterface)

  // ════════════════════════════════════════════════════════
  // Auxiliary registers for data path management
  // ════════════════════════════════════════════════════════
  // DRAM base addresses (loaded via ConfigRegs write in real system)
  val dramWeightBase  = RegInit("h80000000".U(32.W))  // Default: 2GB offset for weights
  val dramFeatureBase = RegInit("h81000000".U(32.W))  // Default: 2GB + 16MB for features
  val dramOutputBase  = RegInit("h82000000".U(32.W))  // Default: 2GB + 32MB for output
  val dmaOffset       = RegInit(0.U(32.W))             // Current transfer offset

  // Pixel counter for tile processing
  val pixelCounter = RegInit(0.U(16.W))

  // Camera parameters (loaded from DRAM at runtime, held in dedicated registers)
  val cameraFx = RegInit("h4500".U(16.W))  // Default: fx ≈ 450 pixels
  val cameraFy = RegInit("h4500".U(16.W))  // Default: fy ≈ 450 pixels
  val cameraCx = RegInit("h4000".U(16.W))  // Default: cx = 256/2 = 128
  val cameraCy = RegInit("h4000".U(16.W))  // Default: cy = 256/2 = 128
  val extrinsics = RegInit(VecInit(Seq.fill(12)(0.U(16.W))))
  val cameraLoadCounter = RegInit(0.U(2.W))

  // Weight buffer DMA accumulation (6x128-bit beats = 768 bits)
  val weightAccumReg = RegInit(VecInit(Seq.fill(6)(0.U(128.W))))
  val weightBeatCount = RegInit(0.U(3.W))

  // ════════════════════════════════════════════════════════
  // ConfigRegs wiring
  // ════════════════════════════════════════════════════════
  configRegs.io.writeAddr := io.cfgWriteAddr
  configRegs.io.writeData := io.cfgWriteData
  configRegs.io.writeEn   := io.cfgWriteEn
  configRegs.io.readAddr  := io.cfgReadAddr
  io.cfgReadData          := configRegs.io.readData

  // ════════════════════════════════════════════════════════
  // Pipeline Controller wiring
  // ════════════════════════════════════════════════════════
  pipeline.io.start       := io.start
  pipeline.io.config      := configRegs.io.config
  pipeline.io.configValid := configRegs.io.configValid
  io.done                 := pipeline.io.done
  io.busy                 := pipeline.io.busy
  io.pipeState            := pipeline.io.state

  // Compute unit done signals → pipeline
  pipeline.io.convEngineDone := convEngine.io.done
  pipeline.io.gemmDone       := gemmUnit.io.done
  pipeline.io.bilinearDone   := bilinear.io.done
  pipeline.io.gguDone        := gguArray.io.done

  // Pipeline → compute unit start signals
  convEngine.io.start := pipeline.io.convEngineStart
  gemmUnit.io.start   := pipeline.io.gemmStart
  bilinear.io.enable  := pipeline.io.bilinearStart
  gguArray.io.start   := pipeline.io.gguStart

  // ════════════════════════════════════════════════════════
  // SAES Controller wiring
  // ════════════════════════════════════════════════════════
  saesCtrl.io.start   := pipeline.io.saesClassifyStart
  saesCtrl.io.config  := configRegs.io.config
  pipeline.io.saesLevel       := saesCtrl.io.level
  pipeline.io.saesClassifyDone := saesCtrl.io.done

  // ════════════════════════════════════════════════════════
  // SAES data inputs (probe statistics from FeatureBuffer/TileSPM)
  // ════════════════════════════════════════════════════════
  saesCtrl.io.probeFeatureVar := featureBuf.io.doutA  // Feature variance from probe window
  saesCtrl.io.probeDepthStd   := tileSPM.io.rdData    // Depth std dev from S2
  saesCtrl.io.crossCheckError := featureBuf.io.doutB  // Cross-check error metric

  // ════════════════════════════════════════════════════════
  // FSGR Controller + LSH Hash + Cache wiring
  // ════════════════════════════════════════════════════════
  fsgrCtrl.io.start       := pipeline.io.fsgrStart
  fsgrCtrl.io.config      := configRegs.io.config
  pipeline.io.fsgrDone    := fsgrCtrl.io.done
  pipeline.io.fsgrUseNarrow := fsgrCtrl.io.useNarrowSearch
  pipeline.io.fsgrNarrowCands := fsgrCtrl.io.narrowCandidates

  // ════════════════════════════════════════════════════════
  // FSGRController ↔ LSH Hash Unit
  // ════════════════════════════════════════════════════════
  lshHash.io.start := fsgrCtrl.io.hashStart
  fsgrCtrl.io.hashDone   := lshHash.io.done
  fsgrCtrl.io.hashResult := lshHash.io.signature
  
  // LSH input features from FeatureBuffer (128-channel features sequentially read)
  for (i <- 0 until 128) { 
    lshHash.io.featureIn(i) := featureBuf.io.doutA  // Sequenced reads over 128 cycles
  }

  // FSGRController ↔ FSGR Cache
  fsgrCache.io.lookupEn      := fsgrCtrl.io.cacheLookupEn
  fsgrCache.io.lookupSig     := fsgrCtrl.io.cacheLookupSig
  fsgrCache.io.hammingThresh := configRegs.io.config.fsgrHammingThresh
  fsgrCtrl.io.cacheHit       := fsgrCache.io.hit
  fsgrCtrl.io.cacheHitDepth  := fsgrCache.io.hitDepth

  fsgrCache.io.insertEn      := fsgrCtrl.io.cacheInsertEn
  fsgrCache.io.insertSig     := fsgrCtrl.io.cacheInsertSig
  fsgrCache.io.insertDepth   := fsgrCtrl.io.cacheInsertDepth
  
  // Update pixel counter during S2 depth output
  when((pipeline.io.state === PipeState.sS2_Regression || pipeline.io.state === PipeState.sS2_DepthHead) && gemmUnit.io.cWr) {
    pixelCounter := pixelCounter + 1.U
  }.elsewhen(pipeline.io.state === PipeState.sS2S3_TileLoad) {
    pixelCounter := 0.U
  }
  fsgrCache.io.insertPixelX  := pixelCounter % configRegs.io.config.tileSize
  fsgrCache.io.insertPixelY  := pixelCounter / configRegs.io.config.tileSize

  // FSGR computed depth input (from S2 regression output in TileSPM)
  fsgrCtrl.io.computedDepth  := tileSPM.io.rdData
  fsgrCtrl.io.totalPixels    := configRegs.io.config.tileSize * configRegs.io.config.tileSize

  // ════════════════════════════════════════════════════════
  // Compute unit parameter wiring (from ConfigRegs)
  // ════════════════════════════════════════════════════════

  // ConvEngine parameters (overridden per-state by controller in real design)
  convEngine.io.inChannels  := configRegs.io.config.featureDim
  convEngine.io.outChannels := configRegs.io.config.featureDim
  convEngine.io.kernelSize  := 3.U
  convEngine.io.stride      := 1.U
  convEngine.io.padding     := 1.U
  convEngine.io.fuseReLU    := true.B

  // GEMMUnit parameters
  gemmUnit.io.M       := configRegs.io.config.featureDim
  gemmUnit.io.K       := configRegs.io.config.featureDim
  gemmUnit.io.N       := configRegs.io.config.featureDim
  gemmUnit.io.useBias := true.B

  // BilinearUnit parameters
  bilinear.io.coordX := tileSPM.io.rdData(15, 0).asSInt  // Sample coord from TileSPM
  bilinear.io.coordY := tileSPM.io.rdData(31, 16).asSInt
  bilinear.io.inH    := configRegs.io.config.imageH >> 2  // H/4 for feature maps
  bilinear.io.inW    := configRegs.io.config.imageW >> 2

  // ════════════════════════════════════════════════════════
  // VectorALU wiring (element-wise operations on feature vectors)
  // ════════════════════════════════════════════════════════
  vectorALU.io.op     := VectorOp.ADD  // Controlled by pipeline for add/sub/mul operations
  vectorALU.io.enable := pipeline.io.state === PipeState.sS2_CostVol || 
                         pipeline.io.state === PipeState.sS3_Refine
  // Connect to FeatureBuffer output for vector operations (single word broadcast)
  for (i <- 0 until ScarfConfig.VectorALUWidth) {
    vectorALU.io.a(i) := featureBuf.io.doutA  // Broadcast from port A
    vectorALU.io.b(i) := featureBuf.io.doutB  // Broadcast from port B
    vectorALU.io.c(i) := bilinear.io.results(i % ScarfConfig.BilinearChannels)  // For warping operations
  }

  // ════════════════════════════════════════════════════════
  // ActivationUnit wiring (post-conv/GEMM activation)
  // ════════════════════════════════════════════════════════
  activation.io.dataIn  := convEngine.io.outputData(0)  // First output channel
  activation.io.actType := Mux(convEngine.io.fuseReLU, ActivationType.RELU, ActivationType.GELU)
  activation.io.enable  := convEngine.io.outputWr || gemmUnit.io.cWr

  // ════════════════════════════════════════════════════════
  // NormUnit wiring (BatchNorm/GroupNorm)
  // ════════════════════════════════════════════════════════
  normUnit.io.start    := pipeline.io.state === PipeState.sS1_CNN || 
                          pipeline.io.state === PipeState.sS1_Transformer
  normUnit.io.normType := NormType.BATCH
  normUnit.io.channels := configRegs.io.config.featureDim
  normUnit.io.groups   := configRegs.io.config.normGroups
  normUnit.io.epsilon  := "h3C23D70A".U  // 1e-5 in FP32 hex
  normUnit.io.dataIn   := convEngine.io.outputData(0)
  normUnit.io.gammaIn  := weightBuf.io.rdData(ScarfConfig.DataWidth - 1, 0)  // Read BN gamma from weight buffer
  normUnit.io.betaIn   := weightBuf.io.rdData(2 * ScarfConfig.DataWidth - 1, ScarfConfig.DataWidth)  // Read BN beta
  normUnit.io.inValid  := convEngine.io.outputWr

  // ════════════════════════════════════════════════════════
  // SoftmaxUnit wiring (depth candidate selection)
  // ════════════════════════════════════════════════════════
  softmax.io.start       := pipeline.io.state === PipeState.sS2_Regression
  softmax.io.logitIn     := gemmUnit.io.cData(0)  // Depth logits from regression head
  softmax.io.candidateIn := tileSPM.io.rdData  // Candidate depth values
  softmax.io.inValid     := gemmUnit.io.cWr
  softmax.io.numElements := configRegs.io.config.numDepthCandidates

  // ════════════════════════════════════════════════════════
  // GGU Array wiring (Gaussian rasterization)
  // ════════════════════════════════════════════════════════
  gguArray.io.shDegree := configRegs.io.config.shDegree
  
  // Connect camera parameters to GGU
  gguArray.io.fx := cameraFx
  gguArray.io.fy := cameraFy
  gguArray.io.cx := cameraCx
  gguArray.io.cy := cameraCy
  
  for (i <- 0 until 12) { 
    gguArray.io.extrinsics(i) := extrinsics(i)
  }
  
  // Update camera parameters from DRAM during config load phase
  // AXI data bus is 128 bits wide, can transfer 8x16-bit values per beat
  when(dramIF.io.axiRValid && pipeline.io.state === PipeState.sLoadConfig) {
    when(cameraLoadCounter === 0.U) {
      // First beat: camera intrinsics (4 values)
      cameraFx := dramIF.io.axiRData(15, 0)
      cameraFy := dramIF.io.axiRData(31, 16)
      cameraCx := dramIF.io.axiRData(47, 32)
      cameraCy := dramIF.io.axiRData(63, 48)
      cameraLoadCounter := 1.U
    }.elsewhen(cameraLoadCounter === 1.U) {
      // Second beat: extrinsics[0-7]
      for (i <- 0 until 8) {
        extrinsics(i) := dramIF.io.axiRData((i + 1) * 16 - 1, i * 16)
      }
      cameraLoadCounter := 2.U
    }.elsewhen(cameraLoadCounter === 2.U) {
      // Third beat: extrinsics[8-11]
      for (i <- 0 until 4) {
        extrinsics(i + 8) := dramIF.io.axiRData((i + 1) * 16 - 1, i * 16)
      }
      cameraLoadCounter := 0.U
    }
  }
  
  // Per-Gaussian parameters from TileSPM (depth) and FeatureBuffer (SH coefficients)
  for (i <- 0 until ScarfConfig.GGUPECount) {
    // Pixel coordinates - sequential processing across tile
    val pixelIdx = i.U
    gguArray.io.pixelX(i)    := pixelIdx % configRegs.io.config.tileSize
    gguArray.io.pixelY(i)    := pixelIdx / configRegs.io.config.tileSize
    
    // Depth from TileSPM (S2 output) - broadcast to all PEs
    gguArray.io.depth(i)     := tileSPM.io.rdData
    
    // Gaussian parameters from S3 GaussHead output (stored in FeatureBuffer)
    // Pipeline controller sequences reads for each parameter
    // All PEs receive the same parameter value per cycle, but process different pixels
    gguArray.io.scaleX(i)    := featureBuf.io.doutA
    gguArray.io.scaleY(i)    := featureBuf.io.doutA
    gguArray.io.scaleZ(i)    := featureBuf.io.doutA
    gguArray.io.quatW(i)     := featureBuf.io.doutA
    gguArray.io.quatX(i)     := featureBuf.io.doutA
    gguArray.io.quatY(i)     := featureBuf.io.doutA
    gguArray.io.quatZ(i)     := featureBuf.io.doutA
    gguArray.io.opacityIn(i) := featureBuf.io.doutB
  }

  // ════════════════════════════════════════════════════════
  // Memory interface wiring
  // ════════════════════════════════════════════════════════

  // ════════════════════════════════════════════════════════
  // WeightBuffer ↔ Compute Units
  // ════════════════════════════════════════════════════════
  weightBuf.io.rdAddr := convEngine.io.weightAddr(log2Ceil(ScarfConfig.WeightBufferBytes * 8 / (ScarfConfig.PEArraySize * ScarfConfig.DataWidth)) - 1, 0)
  weightBuf.io.rdEn   := convEngine.io.busy
  
  // DMA write from DRAM interface (during config load phase)
  // WeightBuffer word width is 768 bits (48x16), but AXI is 128 bits
  // Accumulate AXI beats into a full weight buffer word (6 beats required)
  when(dramIF.io.axiRValid && pipeline.io.state === PipeState.sLoadConfig && weightBeatCount < 6.U) {
    weightAccumReg(weightBeatCount) := dramIF.io.axiRData
    weightBeatCount := weightBeatCount + 1.U
  }
  
  // Write to WeightBuffer when all 6 beats accumulated
  weightBuf.io.wrAddr := dmaOffset(log2Ceil(ScarfConfig.WeightBufferBytes * 8 / (ScarfConfig.PEArraySize * ScarfConfig.DataWidth)) - 1, 0)
  weightBuf.io.wrData := Cat(weightAccumReg.reverse)  // Concatenate 6x128-bit = 768 bits
  weightBuf.io.wrEn   := weightBeatCount === 6.U
  
  when(weightBuf.io.wrEn) {
    weightBeatCount := 0.U
    dmaOffset := dmaOffset + 1.U
  }
  
  // Connect weight data back to ConvEngine
  // WeightBuffer outputs packed 48x16-bit words, extract each PE's weight
  for (i <- 0 until ScarfConfig.PEArraySize) {
    convEngine.io.weightData(i) := weightBuf.io.rdData((i + 1) * ScarfConfig.DataWidth - 1, i * ScarfConfig.DataWidth)
  }
  
  // Connect input data from FeatureBuffer port A (single word per cycle)
  // Address sequencing provides one row at a time to systolic array
  for (i <- 0 until ScarfConfig.PEArraySize) {
    convEngine.io.inputData(i)  := featureBuf.io.doutA  // Broadcast same value to all PEs
  }

  // ════════════════════════════════════════════════════════
  // GEMMUnit data connections
  // ════════════════════════════════════════════════════════
  // GEMM reads A, B matrices from FeatureBuffer dual ports (single word per cycle)
  for (i <- 0 until ScarfConfig.PEArraySize) {
    gemmUnit.io.aData(i)    := featureBuf.io.doutA  // Broadcast to row
    gemmUnit.io.bData(i)    := featureBuf.io.doutB  // Broadcast to column
    gemmUnit.io.biasData(i) := weightBuf.io.rdData((i + 1) * ScarfConfig.DataWidth - 1, i * ScarfConfig.DataWidth)
  }

  // ════════════════════════════════════════════════════════
  // BilinearUnit taps from FeatureBuffer
  // ════════════════════════════════════════════════════════
  // 4-tap bilinear sampling: each tap reads from a separate FeatureBuffer address
  // The addressing logic in FeatureBuffer.addrB cycles through tap positions
  for (i <- 0 until ScarfConfig.BilinearChannels) {
    // Simple connection: all taps read from the same output port
    // Address sequencing handled by pipeline controller
    bilinear.io.taps00(i) := featureBuf.io.doutA
    bilinear.io.taps01(i) := featureBuf.io.doutA
    bilinear.io.taps10(i) := featureBuf.io.doutB
    bilinear.io.taps11(i) := featureBuf.io.doutB
  }

  // ════════════════════════════════════════════════════════
  // FeatureBuffer wiring (dual-port ping-pong buffer)
  // ════════════════════════════════════════════════════════
  // Port A: Write from ConvEngine/GEMM output, Read for compute units
  featureBuf.io.addrA := Mux(convEngine.io.outputWr, 
    convEngine.io.outputAddr(log2Ceil(ScarfConfig.FeatureBufferBytes * 8 / ScarfConfig.DataWidth) - 1, 0),
    convEngine.io.inputAddr(log2Ceil(ScarfConfig.FeatureBufferBytes * 8 / ScarfConfig.DataWidth) - 1, 0))
  
  // Write data from compute outputs (single word per cycle)
  featureBuf.io.dinA  := Mux(convEngine.io.busy, convEngine.io.outputData(0), 
                             Mux(gemmUnit.io.busy, gemmUnit.io.cData(0), 0.U))
  featureBuf.io.wenA  := convEngine.io.outputWr || gemmUnit.io.cWr
  featureBuf.io.renA  := convEngine.io.busy || gemmUnit.io.busy || bilinear.io.busy
  
  // Port B: Read for bilinear/GEMM secondary operand
  featureBuf.io.addrB := Mux(gemmUnit.io.busy, 
    gemmUnit.io.bAddr(log2Ceil(ScarfConfig.FeatureBufferBytes * 8 / ScarfConfig.DataWidth) - 1, 0),
    bilinear.io.coordY.asUInt)  // Use coord as address for tap access
  featureBuf.io.dinB  := 0.U  // Port B is read-only in most operations
  featureBuf.io.wenB  := false.B
  featureBuf.io.renB  := gemmUnit.io.busy || bilinear.io.busy
  
  // Ping-pong bank swap controlled by pipeline (swap between S1 and S2 stages)
  featureBuf.io.bankSwap := pipeline.io.state === PipeState.sS2S3_TileLoad

  // ════════════════════════════════════════════════════════
  // TileSPM wiring (stores S2 depth predictions for S3)
  // ════════════════════════════════════════════════════════
  // Write port: S2 depth output
  tileSPM.io.wrAddr := gemmUnit.io.cAddr(log2Ceil(ScarfConfig.TileSPMBytes * 8 / ScarfConfig.AccWidth) - 1, 0)
  tileSPM.io.wrData := gemmUnit.io.cData(0)  // Depth prediction from regression head
  tileSPM.io.wrEn   := gemmUnit.io.cWr && (pipeline.io.state === PipeState.sS2_Regression || 
                                             pipeline.io.state === PipeState.sS2_DepthHead)
  
  // Read port: S3/GGU reads depth + coordinates
  tileSPM.io.rdAddr := gguArray.io.done.asUInt  // Use simple counter-based addressing
  tileSPM.io.rdEn   := pipeline.io.state === PipeState.sS3_Refine || 
                       pipeline.io.state === PipeState.sS3_GaussHead || 
                       pipeline.io.gguStart

  // ════════════════════════════════════════════════════════
  // DRAM interface → AXI4 external pins
  // ════════════════════════════════════════════════════════
  // Read request: load weights, features, or model parameters during config load
  dramIF.io.readReq.valid       := pipeline.io.state === PipeState.sLoadConfig
  dramIF.io.readReq.bits.addr   := dramWeightBase + dmaOffset
  dramIF.io.readReq.bits.burstLen := 15.U  // Burst of 16 transfers (indices 0-15)
  dramIF.io.readResp.ready      := true.B
  
  // Write request: write back GGU output to DRAM after tile completion
  // Pack GGU outputs: position (3x32), covariance (6x32), SH coeffs, opacity
  dramIF.io.writeReq.valid        := pipeline.io.state === PipeState.sS2S3_NextTile && gguArray.io.done
  dramIF.io.writeReq.bits.addr    := dramOutputBase + (pixelCounter << 4)  // 16 bytes per Gaussian
  dramIF.io.writeReq.bits.data    := Cat(gguArray.io.opacityOut(0), gguArray.io.cov(0)(0), gguArray.io.posZ(0), gguArray.io.posX(0))
  dramIF.io.writeReq.bits.burstLen := 0.U  // Single transfer per cycle

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
