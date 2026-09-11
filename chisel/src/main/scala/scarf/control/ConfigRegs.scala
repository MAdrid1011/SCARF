package scarf.control

import chisel3._
import chisel3.util._
import scarf.{ScarfConfig, ModelConfig}

/**
 * ConfigRegs — MMIO-mapped Configuration Register File.
 *
 * All model differences (TranSplat/MVSplat/DepthSplat) captured here.
 * Written by host CPU via AXI4-Lite before inference starts.
 */
class ConfigRegs extends Module {
  val io = IO(new Bundle {
    val writeAddr  = Input(UInt(8.W))
    val writeData  = Input(UInt(32.W))
    val writeEn    = Input(Bool())
    val readAddr   = Input(UInt(8.W))
    val readData   = Output(UInt(32.W))
    val config     = Output(new ModelConfig)
    val configValid = Output(Bool())
  })

  val numDepthCandidates = RegInit(64.U(8.W))
  val featureDim         = RegInit(256.U(9.W))
  val imageH             = RegInit(256.U(10.W))
  val imageW             = RegInit(256.U(10.W))
  val tileSize           = RegInit(4.U(4.W))
  val cnnLayers          = RegInit(6.U(8.W))
  val transformerLayers  = RegInit(6.U(4.W))
  val normGroups         = RegInit(8.U(4.W))
  val shDegree           = RegInit(4.U(3.W))
  val hasDINOv2          = RegInit(false.B)
  val dinov2Layers       = RegInit(0.U(6.W))
  val saesFeatureVar     = RegInit(0.U(16.W))
  val saesCrossCheck     = RegInit(0.U(16.W))
  val saesDepthStd       = RegInit(0.U(16.W))
  val saesEnabled        = RegInit(true.B)
  val fsdrEnabled        = RegInit(true.B)
  val fsdrCacheSize      = RegInit(512.U(10.W))
  val fsdrHammingThresh  = RegInit(3.U(4.W))
  val fsdrDepthValidThresh = RegInit(102.U(10.W))
  val payloadBase        = RegInit("h80000000".U(32.W))
  val payloadBytes       = RegInit(0.U(32.W))
  val payloadTensorCount = RegInit(0.U(16.W))
  val numGaussians       = RegInit(0.U(32.W))
  val payloadFeatureOffset = RegInit(0.U(32.W))
  val payloadDepthOffset = RegInit(0.U(32.W))
  val payloadCandidateOffset = RegInit(0.U(32.W))
  val payloadProbabilityOffset = RegInit(0.U(32.W))
  val payloadFeatureBytes = RegInit(0.U(32.W))
  val payloadDepthBytes = RegInit(0.U(32.W))
  val payloadCandidateBytes = RegInit(0.U(32.W))
  val payloadProbabilityBytes = RegInit(0.U(32.W))
  val payloadSAESRouteOffset = RegInit(0.U(32.W))
  val payloadSAESRouteBytes = RegInit(0.U(32.W))
  val payloadValid       = RegInit(false.B)
  val configValidReg     = RegInit(false.B)

  when(io.writeEn) {
    switch(io.writeAddr) {
      is(0x00.U) { numDepthCandidates := io.writeData(7, 0) }
      is(0x04.U) { featureDim := io.writeData(8, 0) }
      is(0x08.U) { imageH := io.writeData(9, 0) }
      is(0x0C.U) { imageW := io.writeData(9, 0) }
      is(0x10.U) { tileSize := io.writeData(3, 0) }
      is(0x14.U) { cnnLayers := io.writeData(7, 0) }
      is(0x18.U) { transformerLayers := io.writeData(3, 0) }
      is(0x1C.U) { normGroups := io.writeData(3, 0) }
      is(0x20.U) { shDegree := io.writeData(2, 0) }
      is(0x24.U) { hasDINOv2 := io.writeData(0) }
      is(0x28.U) { saesFeatureVar := io.writeData(15, 0) }
      is(0x2C.U) { saesCrossCheck := io.writeData(15, 0) }
      is(0x30.U) { saesDepthStd := io.writeData(15, 0) }
      is(0x34.U) { saesEnabled := io.writeData(0) }
      is(0x38.U) { fsdrEnabled := io.writeData(0) }
      is(0x3C.U) { fsdrCacheSize := io.writeData(9, 0) }
      is(0x40.U) { fsdrHammingThresh := io.writeData(3, 0) }
      is(0x44.U) { configValidReg := io.writeData(0) }
      is(0x48.U) { fsdrDepthValidThresh := io.writeData(9, 0) }
      is(0x4C.U) { payloadBase := io.writeData }
      is(0x50.U) { payloadBytes := io.writeData }
      is(0x54.U) { payloadTensorCount := io.writeData(15, 0) }
      is(0x58.U) { numGaussians := io.writeData }
      is(0x60.U) { payloadValid := io.writeData(0) }
      is(0x64.U) { payloadFeatureOffset := io.writeData }
      is(0x68.U) { payloadDepthOffset := io.writeData }
      is(0x6C.U) { payloadProbabilityOffset := io.writeData }
      is(0x70.U) { payloadCandidateOffset := io.writeData }
      is(0x74.U) { payloadFeatureBytes := io.writeData }
      is(0x78.U) { payloadDepthBytes := io.writeData }
      is(0x7C.U) { payloadCandidateBytes := io.writeData }
      is(0x80.U) { payloadProbabilityBytes := io.writeData }
      is(0x84.U) { dinov2Layers := io.writeData(5, 0) }
      is(0x88.U) { payloadSAESRouteOffset := io.writeData }
      is(0x8C.U) { payloadSAESRouteBytes := io.writeData }
    }
  }

  io.readData := 0.U(32.W)
  switch(io.readAddr) {
    is(0x00.U) { io.readData := Cat(0.U(24.W), numDepthCandidates) }
    is(0x04.U) { io.readData := Cat(0.U(23.W), featureDim) }
    is(0x08.U) { io.readData := Cat(0.U(22.W), imageH) }
    is(0x0C.U) { io.readData := Cat(0.U(22.W), imageW) }
    is(0x10.U) { io.readData := Cat(0.U(28.W), tileSize) }
    is(0x14.U) { io.readData := Cat(0.U(24.W), cnnLayers) }
    is(0x18.U) { io.readData := Cat(0.U(28.W), transformerLayers) }
    is(0x1C.U) { io.readData := Cat(0.U(28.W), normGroups) }
    is(0x20.U) { io.readData := Cat(0.U(29.W), shDegree) }
    is(0x24.U) { io.readData := Cat(0.U(31.W), hasDINOv2.asUInt) }
    is(0x28.U) { io.readData := Cat(0.U(16.W), saesFeatureVar) }
    is(0x2C.U) { io.readData := Cat(0.U(16.W), saesCrossCheck) }
    is(0x30.U) { io.readData := Cat(0.U(16.W), saesDepthStd) }
    is(0x34.U) { io.readData := Cat(0.U(31.W), saesEnabled.asUInt) }
    is(0x38.U) { io.readData := Cat(0.U(31.W), fsdrEnabled.asUInt) }
    is(0x3C.U) { io.readData := Cat(0.U(22.W), fsdrCacheSize) }
    is(0x40.U) { io.readData := Cat(0.U(28.W), fsdrHammingThresh) }
    is(0x44.U) { io.readData := Cat(0.U(31.W), configValidReg.asUInt) }
    is(0x48.U) { io.readData := Cat(0.U(22.W), fsdrDepthValidThresh) }
    is(0x4C.U) { io.readData := payloadBase }
    is(0x50.U) { io.readData := payloadBytes }
    is(0x54.U) { io.readData := Cat(0.U(16.W), payloadTensorCount) }
    is(0x58.U) { io.readData := numGaussians }
    is(0x60.U) { io.readData := Cat(0.U(31.W), payloadValid.asUInt) }
    is(0x64.U) { io.readData := payloadFeatureOffset }
    is(0x68.U) { io.readData := payloadDepthOffset }
    is(0x6C.U) { io.readData := payloadProbabilityOffset }
    is(0x70.U) { io.readData := payloadCandidateOffset }
    is(0x74.U) { io.readData := payloadFeatureBytes }
    is(0x78.U) { io.readData := payloadDepthBytes }
    is(0x7C.U) { io.readData := payloadCandidateBytes }
    is(0x80.U) { io.readData := payloadProbabilityBytes }
    is(0x84.U) { io.readData := Cat(0.U(26.W), dinov2Layers) }
    is(0x88.U) { io.readData := payloadSAESRouteOffset }
    is(0x8C.U) { io.readData := payloadSAESRouteBytes }
  }

  io.config.numDepthCandidates := numDepthCandidates
  io.config.featureDim         := featureDim
  io.config.imageH             := imageH
  io.config.imageW             := imageW
  io.config.tileSize           := tileSize
  io.config.cnnLayers          := cnnLayers
  io.config.transformerLayers  := transformerLayers
  io.config.normGroups         := normGroups
  io.config.shDegree           := shDegree
  io.config.hasDINOv2          := hasDINOv2
  io.config.dinov2Layers       := dinov2Layers
  io.config.saesFeatureVarThresh := saesFeatureVar
  io.config.saesCrossCheckThresh := saesCrossCheck
  io.config.saesDepthStdThresh   := saesDepthStd
  io.config.saesEnabled        := saesEnabled
  io.config.fsdrEnabled        := fsdrEnabled
  io.config.fsdrCacheSize      := fsdrCacheSize
  io.config.fsdrHammingThresh  := fsdrHammingThresh
  io.config.fsdrDepthValidThresh := fsdrDepthValidThresh
  io.config.payloadBase        := payloadBase
  io.config.payloadBytes       := payloadBytes
  io.config.payloadTensorCount := payloadTensorCount
  io.config.numGaussians       := numGaussians
  io.config.payloadFeatureOffset := payloadFeatureOffset
  io.config.payloadDepthOffset := payloadDepthOffset
  io.config.payloadCandidateOffset := payloadCandidateOffset
  io.config.payloadProbabilityOffset := payloadProbabilityOffset
  io.config.payloadFeatureBytes := payloadFeatureBytes
  io.config.payloadDepthBytes := payloadDepthBytes
  io.config.payloadCandidateBytes := payloadCandidateBytes
  io.config.payloadProbabilityBytes := payloadProbabilityBytes
  io.config.payloadSAESRouteOffset := payloadSAESRouteOffset
  io.config.payloadSAESRouteBytes := payloadSAESRouteBytes
  io.config.payloadValid       := payloadValid
  io.configValid               := configValidReg
}
