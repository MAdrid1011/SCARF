package scarf.memory

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * TileSPM — 64 KB Tile Scratchpad Memory.
 *
 * Holds intermediate S2→S3 data within fused tile execution:
 * - Depth predictions from S2 (stays in SPM, never written to DRAM)
 * - Raw Gaussian parameters for S3 consumption
 * - SAES probe data for cross-check validation
 *
 * Dual-port for concurrent S2 write + S3 read within the same tile.
 */
class TileSPM(
  val sizeBytes: Int = ScarfConfig.TileSPMBytes,
) extends Module {
  val wordWidth = ScarfConfig.AccWidth // 32 bits (FP32 for depth precision)
  val numWords  = sizeBytes * 8 / wordWidth

  val io = IO(new Bundle {
    // Write port (S2 depth output)
    val wrAddr = Input(UInt(log2Ceil(numWords).W))
    val wrData = Input(UInt(wordWidth.W))
    val wrEn   = Input(Bool())

    // Read port (S3 input / GGU input)
    val rdAddr = Input(UInt(log2Ceil(numWords).W))
    val rdData = Output(UInt(wordWidth.W))
    val rdEn   = Input(Bool())
  })

  val mem = SyncReadMem(numWords, UInt(wordWidth.W))

  io.rdData := DontCare
  when(io.wrEn) {
    mem.write(io.wrAddr, io.wrData)
  }
  when(io.rdEn) {
    io.rdData := mem.read(io.rdAddr)
  }
}
