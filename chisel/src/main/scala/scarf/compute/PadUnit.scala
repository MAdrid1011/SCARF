package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * PadUnit — Zero/Replicate Padding.
 *
 * Corresponds to: encoder/pad_unit.py
 *
 * Adds padding around feature maps before convolution.
 * Two modes:
 *   - Zero padding: border pixels = 0
 *   - Replicate padding: border pixels = nearest valid pixel
 */
object PadMode {
  val ZERO: UInt      = 0.U(1.W)
  val REPLICATE: UInt = 1.U(1.W)
}

class PadUnit extends Module {
  val io = IO(new Bundle {
    val dataIn   = Input(UInt(ScarfConfig.DataWidth.W))
    val dataOut  = Output(UInt(ScarfConfig.DataWidth.W))
    val padMode  = Input(UInt(1.W))
    val padSize  = Input(UInt(4.W))        // Padding on each side
    val row      = Input(UInt(10.W))       // Current row in padded output
    val col      = Input(UInt(10.W))       // Current col in padded output
    val origH    = Input(UInt(10.W))       // Original height
    val origW    = Input(UInt(10.W))       // Original width
    val enable   = Input(Bool())
    val valid    = Output(Bool())
  })

  val inPadRegion = (io.row < io.padSize) ||
                    (io.row >= io.origH + io.padSize) ||
                    (io.col < io.padSize) ||
                    (io.col >= io.origW + io.padSize)

  io.dataOut := Mux(inPadRegion,
    Mux(io.padMode === PadMode.ZERO,
      0.U,             // Zero padding
      io.dataIn),      // Replicate: pass through edge value
    io.dataIn)         // Inside valid region: pass through

  io.valid := io.enable
}
