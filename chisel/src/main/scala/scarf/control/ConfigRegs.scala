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
  val saesFeatureVar     = RegInit(0.U(16.W))
  val saesCrossCheck     = RegInit(0.U(16.W))
  val saesDepthStd       = RegInit(0.U(16.W))
  val saesEnabled        = RegInit(true.B)
  val fsdrEnabled        = RegInit(true.B)
  val fsdrCacheSize      = RegInit(512.U(10.W))
  val fsdrHammingThresh  = RegInit(3.U(4.W))
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
  io.config.saesFeatureVarThresh := saesFeatureVar
  io.config.saesCrossCheckThresh := saesCrossCheck
  io.config.saesDepthStdThresh   := saesDepthStd
  io.config.saesEnabled        := saesEnabled
  io.config.fsdrEnabled        := fsdrEnabled
  io.config.fsdrCacheSize      := fsdrCacheSize
  io.config.fsdrHammingThresh  := fsdrHammingThresh
  io.configValid               := configValidReg
}
