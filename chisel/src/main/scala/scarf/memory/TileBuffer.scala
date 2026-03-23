package scarf.memory

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * TileBuffer — 64 KB Tile Scratchpad Memory.
 *
 * Holds intermediate S2→S3 data within fused tile execution:
 * - Depth predictions from S2
 * - Raw Gaussian parameters for S3/GGU consumption
 * - SAES probe data for cross-check
 *
 * Dual-port for concurrent S2 write + S3 read.
 */
class TileBuffer(
  val sizeBytes: Int = ScarfConfig.TileBufferBytes,
) extends Module {
  val wordWidth = ScarfConfig.AccWidth // 32 bits (FP32 depth precision)
  val numWords  = sizeBytes * 8 / wordWidth

  val io = IO(new Bundle {
    val wrAddr = Input(UInt(log2Ceil(numWords).W))
    val wrData = Input(UInt(wordWidth.W))
    val wrEn   = Input(Bool())

    val rdAddr = Input(UInt(log2Ceil(numWords).W))
    val rdData = Output(UInt(wordWidth.W))
    val rdEn   = Input(Bool())
  })

  val mem = SyncReadMem(numWords, UInt(wordWidth.W))

  io.rdData := 0.U
  when(io.wrEn) {
    mem.write(io.wrAddr, io.wrData)
  }
  when(io.rdEn) {
    io.rdData := mem.read(io.rdAddr)
  }
}
