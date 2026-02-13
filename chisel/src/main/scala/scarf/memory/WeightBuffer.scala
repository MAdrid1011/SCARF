package scarf.memory

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * WeightBuffer — 128 KB single-port SRAM for ConvEngine weights.
 *
 * Supports shadow-bank prefetching: while ConvEngine reads current layer
 * weights from bank A, next layer weights are DMA'd into bank B.
 * Bank swap is a 1-cycle pointer flip.
 *
 * Interface: word-addressed, each word = PEArraySize × DataWidth bits.
 */
class WeightBuffer(
  val sizeBytes: Int = ScarfConfig.WeightBufferBytes,
) extends Module {
  val wordWidth = ScarfConfig.PEArraySize * ScarfConfig.DataWidth // 48 × 16 = 768 bits
  val numWords  = sizeBytes * 8 / wordWidth

  val io = IO(new Bundle {
    // Read port (ConvEngine)
    val rdAddr  = Input(UInt(log2Ceil(numWords).W))
    val rdData  = Output(UInt(wordWidth.W))
    val rdEn    = Input(Bool())

    // Write port (DMA prefetch)
    val wrAddr  = Input(UInt(log2Ceil(numWords).W))
    val wrData  = Input(UInt(wordWidth.W))
    val wrEn    = Input(Bool())
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
