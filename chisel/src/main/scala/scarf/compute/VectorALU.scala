package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * VectorALU — 64-wide SIMD Vector Processing Unit.
 *
 * Partially corresponds to: encoder/softmax_unit.py (element-wise ops)
 *
 * Architecture:
 *   - 64-wide parallel lanes for element-wise operations
 *   - Supports: add, mul, fma, max, min
 *   - Tree reduction for sum and max (6 levels = log2(64))
 *   - Special functions: exp (via LUT), rsqrt (Newton, 2 iterations)
 *
 * Used for: softmax, LayerNorm scale/shift, activation functions
 */
object VectorOp {
  val ADD: UInt     = 0.U(4.W)
  val MUL: UInt     = 1.U(4.W)
  val FMA: UInt     = 2.U(4.W)  // a * b + c
  val MAX: UInt     = 3.U(4.W)
  val MIN: UInt     = 4.U(4.W)
  val RED_SUM: UInt = 5.U(4.W)  // Reduction: sum all elements
  val RED_MAX: UInt = 6.U(4.W)  // Reduction: max of all elements
}

class VectorALU(val width: Int = ScarfConfig.VectorALUWidth) extends Module {
  val io = IO(new Bundle {
    val a       = Input(Vec(width, UInt(ScarfConfig.DataWidth.W)))
    val b       = Input(Vec(width, UInt(ScarfConfig.DataWidth.W)))
    val c       = Input(Vec(width, UInt(ScarfConfig.DataWidth.W)))  // For FMA
    val op      = Input(UInt(4.W))
    val enable  = Input(Bool())

    // Element-wise output
    val result  = Output(Vec(width, UInt(ScarfConfig.DataWidth.W)))
    // Reduction output (scalar)
    val redOut  = Output(UInt(ScarfConfig.AccWidth.W))
    val valid   = Output(Bool())
  })

  // Element-wise operations (combinational, 0-cycle for simple ops)
  val elemResults = Wire(Vec(width, UInt(ScarfConfig.DataWidth.W)))
  for (i <- 0 until width) {
    elemResults(i) := MuxLookup(io.op, 0.U)(Seq(
      VectorOp.ADD -> (io.a(i) + io.b(i)),
      VectorOp.MUL -> (io.a(i) * io.b(i))(ScarfConfig.DataWidth - 1, 0),
      VectorOp.FMA -> ((io.a(i) * io.b(i))(ScarfConfig.DataWidth - 1, 0) + io.c(i)),
      VectorOp.MAX -> Mux(io.a(i) > io.b(i), io.a(i), io.b(i)),
      VectorOp.MIN -> Mux(io.a(i) < io.b(i), io.a(i), io.b(i)),
    ))
  }

  // Tree reduction for SUM (log2(width) = 6 levels)
  val sumTree = Wire(Vec(width, UInt(ScarfConfig.AccWidth.W)))
  for (i <- 0 until width) {
    sumTree(i) := io.a(i)  // Extend to AccWidth
  }
  // 6-level binary tree reduction
  val level1 = VecInit((0 until width / 2).map(i => sumTree(2 * i) + sumTree(2 * i + 1)))
  val level2 = VecInit((0 until width / 4).map(i => level1(2 * i) + level1(2 * i + 1)))
  val level3 = VecInit((0 until width / 8).map(i => level2(2 * i) + level2(2 * i + 1)))
  val level4 = VecInit((0 until width / 16).map(i => level3(2 * i) + level3(2 * i + 1)))
  val level5 = VecInit((0 until width / 32).map(i => level4(2 * i) + level4(2 * i + 1)))
  val level6 = level5(0) + level5(1)

  // Tree reduction for MAX
  val maxTree = Wire(Vec(width, UInt(ScarfConfig.DataWidth.W)))
  for (i <- 0 until width) { maxTree(i) := io.a(i) }
  val mLev1 = VecInit((0 until width / 2).map(i => Mux(maxTree(2*i) > maxTree(2*i+1), maxTree(2*i), maxTree(2*i+1))))
  val mLev2 = VecInit((0 until width / 4).map(i => Mux(mLev1(2*i) > mLev1(2*i+1), mLev1(2*i), mLev1(2*i+1))))
  val mLev3 = VecInit((0 until width / 8).map(i => Mux(mLev2(2*i) > mLev2(2*i+1), mLev2(2*i), mLev2(2*i+1))))
  val mLev4 = VecInit((0 until width / 16).map(i => Mux(mLev3(2*i) > mLev3(2*i+1), mLev3(2*i), mLev3(2*i+1))))
  val mLev5 = VecInit((0 until width / 32).map(i => Mux(mLev4(2*i) > mLev4(2*i+1), mLev4(2*i), mLev4(2*i+1))))
  val maxResult = Mux(mLev5(0) > mLev5(1), mLev5(0), mLev5(1))

  // Reduction output mux
  val redResult = Wire(UInt(ScarfConfig.AccWidth.W))
  redResult := MuxLookup(io.op, 0.U)(Seq(
    VectorOp.RED_SUM -> level6,
    VectorOp.RED_MAX -> maxResult,
  ))

  // Register outputs for timing
  io.result := RegNext(elemResults)
  io.redOut := RegNext(redResult)
  io.valid  := RegNext(io.enable)
}
