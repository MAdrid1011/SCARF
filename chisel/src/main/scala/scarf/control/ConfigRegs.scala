package scarf.control

import chisel3._
import chisel3.util._
import scarf.{ScarfConfig, ModelConfig}

/**
 * ConfigRegs — MMIO-mapped Configuration Register File.
 *
 * Holds model parameters that differentiate Transplat/MVSplat/DepthSplat.
 * Written by host CPU via AXI4-Lite before inference starts.
 * Read by PipelineController and compute units during execution.
 *
 * Design principle: ALL model differences are captured here.
 * The hardware data path has zero model-specific branches.
 */
class ConfigRegs extends Module {
  val io = IO(new Bundle {
    // AXI4-Lite write interface (host CPU)
    val writeAddr  = Input(UInt(8.W))  // Register address (byte-aligned)
    val writeData  = Input(UInt(32.W))
    val writeEn    = Input(Bool())

    // AXI4-Lite read interface
    val readAddr   = Input(UInt(8.W))
    val readData   = Output(UInt(32.W))

    // Configuration output (to pipeline controller and compute units)
    val config     = Output(new ModelConfig)

    // Status
    val configValid = Output(Bool())  // Set after all registers written
  })

  // Register storage
  val numDepthCandidates = RegInit(128.U(8.W))
  val featureDim         = RegInit(128.U(8.W))
  val imageH             = RegInit(256.U(10.W))
  val imageW             = RegInit(256.U(10.W))
  val tileSize           = RegInit(4.U(4.W))
  val cnnLayers          = RegInit(6.U(8.W))
  val transformerLayers  = RegInit(6.U(4.W))
  val normGroups         = RegInit(8.U(4.W))
  val shDegree           = RegInit(4.U(3.W))
  val hasDINOv2          = RegInit(false.B)
  val saesFeatureVar     = RegInit(0.U(16.W))
  val saesCrossCheck     = RegInit(0.U(16.W))
  val saesDepthStd       = RegInit(0.U(16.W))
  val saesEnabled        = RegInit(true.B)
  val fsgrEnabled        = RegInit(true.B)
  val fsgrCacheSize      = RegInit(512.U(10.W))
  val fsgrHammingThresh  = RegInit(4.U(4.W))
  val configValidReg     = RegInit(false.B)

  // Write decoder
  when(io.writeEn) {
    switch(io.writeAddr) {
      is(0x00.U)  { numDepthCandidates := io.writeData(7, 0) }
      is(0x04.U)  { featureDim := io.writeData(7, 0) }
      is(0x08.U)  { imageH := io.writeData(9, 0) }
      is(0x0C.U)  { imageW := io.writeData(9, 0) }
      is(0x10.U)  { tileSize := io.writeData(3, 0) }
      is(0x14.U)  { cnnLayers := io.writeData(7, 0) }
      is(0x18.U)  { transformerLayers := io.writeData(3, 0) }
      is(0x1C.U)  { normGroups := io.writeData(3, 0) }
      is(0x20.U)  { shDegree := io.writeData(2, 0) }
      is(0x24.U)  { hasDINOv2 := io.writeData(0) }
      is(0x28.U)  { saesFeatureVar := io.writeData(15, 0) }
      is(0x2C.U)  { saesCrossCheck := io.writeData(15, 0) }
      is(0x30.U)  { saesDepthStd := io.writeData(15, 0) }
      is(0x34.U)  { saesEnabled := io.writeData(0) }
      is(0x38.U)  { fsgrEnabled := io.writeData(0) }
      is(0x3C.U)  { fsgrCacheSize := io.writeData(9, 0) }
      is(0x40.U)  { fsgrHammingThresh := io.writeData(3, 0) }
      is(0x44.U)  { configValidReg := io.writeData(0) }  // "Go" bit
    }
  }

  // Read decoder
  io.readData := MuxLookup(io.readAddr, 0.U)(Seq(
    0x00.U -> numDepthCandidates,
    0x04.U -> featureDim,
    0x08.U -> imageH,
    0x0C.U -> imageW,
    0x10.U -> tileSize,
    0x14.U -> cnnLayers,
    0x18.U -> transformerLayers,
    0x1C.U -> normGroups,
    0x20.U -> shDegree,
    0x24.U -> hasDINOv2.asUInt,
    0x34.U -> saesEnabled.asUInt,
    0x38.U -> fsgrEnabled.asUInt,
    0x44.U -> configValidReg.asUInt,
  ))

  // Wire config output
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
  io.config.saesFeatureVarThresh := saesFeatureVar
  io.config.saesCrossCheckThresh := saesCrossCheck
  io.config.saesDepthStdThresh   := saesDepthStd
  io.config.saesEnabled        := saesEnabled
  io.config.fsgrEnabled        := fsgrEnabled
  io.config.fsgrCacheSize      := fsgrCacheSize
  io.config.fsgrHammingThresh  := fsgrHammingThresh
  io.configValid               := configValidReg
}
