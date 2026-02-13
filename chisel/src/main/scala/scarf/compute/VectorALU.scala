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

  // Generic tree reduction (works for any power-of-2 width)
  private def treeReduce[T <: Data](
    elems: Seq[T],
    op: (T, T) => T,
  ): T = {
    require(elems.nonEmpty, "Cannot reduce empty sequence")
    if (elems.length == 1) elems.head
    else {
      val pairs = elems.grouped(2).map {
        case Seq(a, b) => op(a, b)
        case Seq(a)    => a  // Odd element passes through
      }.toSeq
      treeReduce(pairs, op)
    }
  }

  // Tree reduction for SUM
  val sumElems = (0 until width).map(i => io.a(i).pad(ScarfConfig.AccWidth))
  val sumResult = treeReduce(sumElems, (a: UInt, b: UInt) => a + b)

  // Tree reduction for MAX
  val maxElems = (0 until width).map(i => io.a(i))
  val maxResult = treeReduce(maxElems, (a: UInt, b: UInt) => Mux(a > b, a, b))

  // Reduction output mux
  val redResult = Wire(UInt(ScarfConfig.AccWidth.W))
  redResult := MuxLookup(io.op, 0.U)(Seq(
    VectorOp.RED_SUM -> sumResult,
    VectorOp.RED_MAX -> maxResult,
  ))

  // Register outputs for timing
  io.result := RegNext(elemResults)
  io.redOut := RegNext(redResult)
  io.valid  := RegNext(io.enable)
}
