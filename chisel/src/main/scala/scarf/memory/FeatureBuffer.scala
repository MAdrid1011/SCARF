package scarf.memory

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * FeatureBuffer — 256 KB dual-port SRAM for feature maps.
 *
 * Ping-pong buffering: S1 writes to bank A while S2 reads from bank B,
 * then banks swap. Enables pipeline overlap between S1 and S2.
 *
 * Two independent read/write ports for simultaneous access.
 */
class FeatureBuffer(
  val sizeBytes: Int = ScarfConfig.FeatureBufferBytes,
) extends Module {
  val wordWidth = ScarfConfig.DataWidth // 16 bits per element
  val numWords  = sizeBytes * 8 / wordWidth

  val io = IO(new Bundle {
    // Port A (typically: S1 write / S2 read)
    val addrA   = Input(UInt(log2Ceil(numWords).W))
    val dinA    = Input(UInt(wordWidth.W))
    val doutA   = Output(UInt(wordWidth.W))
    val wenA    = Input(Bool())
    val renA    = Input(Bool())

    // Port B (typically: S2 read / DMA write)
    val addrB   = Input(UInt(log2Ceil(numWords).W))
    val dinB    = Input(UInt(wordWidth.W))
    val doutB   = Output(UInt(wordWidth.W))
    val wenB    = Input(Bool())
    val renB    = Input(Bool())

    // Bank select for ping-pong (0 = normal, 1 = swapped)
    val bankSwap = Input(Bool())
  })

  // Two banks for ping-pong
  val bankSize = numWords / 2
  val bank0 = SyncReadMem(bankSize, UInt(wordWidth.W))
  val bank1 = SyncReadMem(bankSize, UInt(wordWidth.W))

  // Address mapping with bank swap
  val addrA_bank = Mux(io.bankSwap, io.addrA % bankSize.U, io.addrA % bankSize.U)
  val addrB_bank = Mux(io.bankSwap, io.addrB % bankSize.U, io.addrB % bankSize.U)
  val selA = Mux(io.bankSwap, io.addrA >= bankSize.U, io.addrA < bankSize.U)
  val selB = Mux(io.bankSwap, io.addrB >= bankSize.U, io.addrB < bankSize.U)

  io.doutA := DontCare
  io.doutB := DontCare

  // Port A
  when(io.wenA) {
    when(selA) { bank0.write(addrA_bank, io.dinA) }
      .otherwise { bank1.write(addrA_bank, io.dinA) }
  }
  when(io.renA) {
    io.doutA := Mux(selA, bank0.read(addrA_bank), bank1.read(addrA_bank))
  }

  // Port B
  when(io.wenB) {
    when(selB) { bank0.write(addrB_bank, io.dinB) }
      .otherwise { bank1.write(addrB_bank, io.dinB) }
  }
  when(io.renB) {
    io.doutB := Mux(selB, bank0.read(addrB_bank), bank1.read(addrB_bank))
  }
}
