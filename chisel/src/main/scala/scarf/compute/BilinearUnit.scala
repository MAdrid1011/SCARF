package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * BilinearUnit — Parallel Bilinear Interpolation Sampler.
 *
 * Corresponds to: encoder/bilinear_unit.py
 *
 * Architecture:
 *   - 32 parallel samplers, each handling one channel
 *   - 3-stage pipeline: address generation → SRAM read (4-tap) → interpolation
 *   - Fixed-point coordinate arithmetic (8-bit fractional)
 *   - Used exclusively in S2 Cost Volume construction
 *
 * For 128-channel features: 128/32 = 4 cycles per output pixel.
 */

/** Single bilinear sampler — computes one channel's interpolated value. */
class BilinearSampler extends Module {
  val io = IO(new Bundle {
    // Coordinate input (fixed-point: integer + 8-bit fraction)
    val coordX   = Input(SInt(16.W))  // Source x coordinate
    val coordY   = Input(SInt(16.W))  // Source y coordinate
    val inH      = Input(UInt(10.W))  // Source height
    val inW      = Input(UInt(10.W))  // Source width

    // 4-tap values (read from feature SRAM)
    val tap00    = Input(UInt(ScarfConfig.DataWidth.W))  // top-left
    val tap01    = Input(UInt(ScarfConfig.DataWidth.W))  // top-right
    val tap10    = Input(UInt(ScarfConfig.DataWidth.W))  // bottom-left
    val tap11    = Input(UInt(ScarfConfig.DataWidth.W))  // bottom-right

    // Interpolated output
    val result   = Output(UInt(ScarfConfig.DataWidth.W))
    val valid    = Output(Bool())

    val enable   = Input(Bool())
  })

  // Pipeline stage 1: Compute integer coordinates and fractional weights
  val fracBits = ScarfConfig.CoordFracBits
  val fracMask = ((1 << fracBits) - 1).U(fracBits.W)

  val x0 = (io.coordX >> fracBits).asUInt  // Floor(x)
  val y0 = (io.coordY >> fracBits).asUInt  // Floor(y)
  val fx = io.coordX(fracBits - 1, 0).asUInt  // Fractional x (0..255)
  val fy = io.coordY(fracBits - 1, 0).asUInt  // Fractional y (0..255)

  // Weights: w = frac/256, (1-w) = (256-frac)/256
  val wx1 = fx                               // Weight for right
  val wx0 = ((1 << fracBits).U - fx)         // Weight for left
  val wy1 = fy                               // Weight for bottom
  val wy0 = ((1 << fracBits).U - fy)         // Weight for top

  // Pipeline stage 2: registered tap values (from SRAM)
  val tapReg00 = RegNext(io.tap00)
  val tapReg01 = RegNext(io.tap01)
  val tapReg10 = RegNext(io.tap10)
  val tapReg11 = RegNext(io.tap11)
  val wx0Reg = RegNext(wx0)
  val wx1Reg = RegNext(wx1)
  val wy0Reg = RegNext(wy0)
  val wy1Reg = RegNext(wy1)
  val validP1 = RegNext(io.enable)

  // Pipeline stage 3: Bilinear interpolation
  // result = wy0*(wx0*tap00 + wx1*tap01) + wy1*(wx0*tap10 + wx1*tap11)
  // All arithmetic in extended precision, then truncate
  val top    = wx0Reg * tapReg00 + wx1Reg * tapReg01
  val bottom = wx0Reg * tapReg10 + wx1Reg * tapReg11
  val interp = wy0Reg * top(ScarfConfig.DataWidth + fracBits - 1, fracBits) +
               wy1Reg * bottom(ScarfConfig.DataWidth + fracBits - 1, fracBits)

  val resultReg = RegNext(interp(ScarfConfig.DataWidth + fracBits - 1, fracBits))
  val validP2   = RegNext(validP1)

  io.result := resultReg(ScarfConfig.DataWidth - 1, 0)
  io.valid  := validP2
}

/**
 * BilinearUnit: 32 parallel samplers with shared control.
 *
 * Processes `BilinearChannels` channels simultaneously.
 * For a 128-channel feature map, requires 4 cycles per output pixel.
 */
class BilinearUnit(
  val numSamplers: Int = ScarfConfig.BilinearChannels,
) extends Module {
  val io = IO(new Bundle {
    val done     = Output(Bool())
    val busy     = Output(Bool())

    // Sampling coordinates
    val coordX   = Input(SInt(16.W))
    val coordY   = Input(SInt(16.W))
    val inH      = Input(UInt(10.W))
    val inW      = Input(UInt(10.W))

    // 4-tap data for all parallel channels
    val taps00   = Input(Vec(numSamplers, UInt(ScarfConfig.DataWidth.W)))
    val taps01   = Input(Vec(numSamplers, UInt(ScarfConfig.DataWidth.W)))
    val taps10   = Input(Vec(numSamplers, UInt(ScarfConfig.DataWidth.W)))
    val taps11   = Input(Vec(numSamplers, UInt(ScarfConfig.DataWidth.W)))

    // Results
    val results  = Output(Vec(numSamplers, UInt(ScarfConfig.DataWidth.W)))
    val valid    = Output(Bool())

    val enable   = Input(Bool())
  })

  val samplers = Seq.fill(numSamplers)(Module(new BilinearSampler))
  val anyValid = Wire(Bool())

  for (i <- 0 until numSamplers) {
    samplers(i).io.coordX := io.coordX
    samplers(i).io.coordY := io.coordY
    samplers(i).io.inH    := io.inH
    samplers(i).io.inW    := io.inW
    samplers(i).io.tap00  := io.taps00(i)
    samplers(i).io.tap01  := io.taps01(i)
    samplers(i).io.tap10  := io.taps10(i)
    samplers(i).io.tap11  := io.taps11(i)
    samplers(i).io.enable := io.enable

    io.results(i) := samplers(i).io.result
  }

  anyValid := samplers.head.io.valid
  io.valid := anyValid
  io.done  := RegNext(!io.enable && anyValid)
  io.busy  := io.enable || anyValid
}
