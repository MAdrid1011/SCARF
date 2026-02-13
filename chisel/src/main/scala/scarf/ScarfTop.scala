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

  // SAES data inputs (placeholder — real implementation reads from FeatureBuffer)
  saesCtrl.io.probeFeatureVar := 0.U
  saesCtrl.io.probeDepthStd   := 0.U
  saesCtrl.io.crossCheckError := 0.U

  // ════════════════════════════════════════════════════════
  // Compute unit parameter wiring (from ConfigRegs)
  // ════════════════════════════════════════════════════════

  // ConvEngine defaults (overridden per-state by controller in real design)
  convEngine.io.inChannels  := configRegs.io.config.featureDim
  convEngine.io.outChannels := configRegs.io.config.featureDim
  convEngine.io.kernelSize  := 3.U
  convEngine.io.stride      := 1.U
  convEngine.io.padding     := 1.U
  convEngine.io.fuseReLU    := true.B

  // GEMMUnit defaults
  gemmUnit.io.M       := configRegs.io.config.featureDim
  gemmUnit.io.K       := configRegs.io.config.featureDim
  gemmUnit.io.N       := configRegs.io.config.featureDim
  gemmUnit.io.useBias := false.B

  // BilinearUnit defaults
  bilinear.io.coordX := 0.S
  bilinear.io.coordY := 0.S
  bilinear.io.inH    := configRegs.io.config.imageH >> 2  // H/4 for feature maps
  bilinear.io.inW    := configRegs.io.config.imageW >> 2

  // VectorALU defaults (unused ports driven to 0)
  vectorALU.io.op     := VectorOp.ADD
  vectorALU.io.enable := false.B
  for (i <- 0 until ScarfConfig.VectorALUWidth) {
    vectorALU.io.a(i) := 0.U
    vectorALU.io.b(i) := 0.U
    vectorALU.io.c(i) := 0.U
  }

  // Activation defaults
  activation.io.dataIn  := 0.U
  activation.io.actType := ActivationType.RELU
  activation.io.enable  := false.B

  // NormUnit defaults
  normUnit.io.start    := false.B
  normUnit.io.normType := NormType.BATCH
  normUnit.io.channels := configRegs.io.config.featureDim
  normUnit.io.groups   := configRegs.io.config.normGroups
  normUnit.io.epsilon  := 0.U
  normUnit.io.dataIn   := 0.U
  normUnit.io.gammaIn  := 0.U
  normUnit.io.betaIn   := 0.U
  normUnit.io.inValid  := false.B

  // SoftmaxUnit defaults
  softmax.io.start       := false.B
  softmax.io.logitIn     := 0.U
  softmax.io.candidateIn := 0.U
  softmax.io.inValid     := false.B
  softmax.io.numElements := configRegs.io.config.numDepthCandidates

  // ════════════════════════════════════════════════════════
  // GGU Array wiring
  // ════════════════════════════════════════════════════════
  gguArray.io.shDegree := configRegs.io.config.shDegree
  gguArray.io.fx := 0.U  // Loaded from DRAM at runtime
  gguArray.io.fy := 0.U
  gguArray.io.cx := 0.U
  gguArray.io.cy := 0.U
  for (i <- 0 until 12) { gguArray.io.extrinsics(i) := 0.U }
  for (i <- 0 until ScarfConfig.GGUPECount) {
    gguArray.io.pixelX(i)    := 0.U
    gguArray.io.pixelY(i)    := 0.U
    gguArray.io.depth(i)     := 0.U
    gguArray.io.scaleX(i)    := 0.U
    gguArray.io.scaleY(i)    := 0.U
    gguArray.io.scaleZ(i)    := 0.U
    gguArray.io.quatW(i)     := 0.U
    gguArray.io.quatX(i)     := 0.U
    gguArray.io.quatY(i)     := 0.U
    gguArray.io.quatZ(i)     := 0.U
    gguArray.io.opacityIn(i) := 0.U
  }

  // ════════════════════════════════════════════════════════
  // Memory interface wiring
  // ════════════════════════════════════════════════════════

  // WeightBuffer ↔ ConvEngine
  weightBuf.io.rdAddr := convEngine.io.weightAddr(log2Ceil(ScarfConfig.WeightBufferBytes * 8 / (ScarfConfig.PEArraySize * ScarfConfig.DataWidth)) - 1, 0)
  weightBuf.io.rdEn   := pipeline.io.convEngineStart
  weightBuf.io.wrAddr := 0.U  // Written by DMA
  weightBuf.io.wrData := 0.U
  weightBuf.io.wrEn   := false.B
  // Connect weight data back to ConvEngine
  for (i <- 0 until ScarfConfig.PEArraySize) {
    convEngine.io.weightData(i) := weightBuf.io.rdData((i + 1) * ScarfConfig.DataWidth - 1, i * ScarfConfig.DataWidth)
    convEngine.io.inputData(i)  := 0.U  // Connected to FeatureBuffer in real design
  }

  // GEMM data connections (placeholder)
  for (i <- 0 until ScarfConfig.PEArraySize) {
    gemmUnit.io.aData(i)    := 0.U
    gemmUnit.io.bData(i)    := 0.U
    gemmUnit.io.biasData(i) := 0.U
  }

  // BilinearUnit taps (placeholder — connected to FeatureBuffer in real design)
  for (i <- 0 until ScarfConfig.BilinearChannels) {
    bilinear.io.taps00(i) := 0.U
    bilinear.io.taps01(i) := 0.U
    bilinear.io.taps10(i) := 0.U
    bilinear.io.taps11(i) := 0.U
  }

  // FeatureBuffer defaults
  featureBuf.io.addrA := 0.U
  featureBuf.io.dinA  := 0.U
  featureBuf.io.wenA  := false.B
  featureBuf.io.renA  := false.B
  featureBuf.io.addrB := 0.U
  featureBuf.io.dinB  := 0.U
  featureBuf.io.wenB  := false.B
  featureBuf.io.renB  := false.B
  featureBuf.io.bankSwap := false.B

  // TileSPM defaults
  tileSPM.io.wrAddr := 0.U
  tileSPM.io.wrData := 0.U
  tileSPM.io.wrEn   := false.B
  tileSPM.io.rdAddr := 0.U
  tileSPM.io.rdEn   := false.B

  // DRAM interface → AXI4 external pins
  dramIF.io.readReq.valid  := false.B
  dramIF.io.readReq.bits   := DontCare
  dramIF.io.readResp.ready := false.B
  dramIF.io.writeReq.valid := false.B
  dramIF.io.writeReq.bits  := DontCare

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
