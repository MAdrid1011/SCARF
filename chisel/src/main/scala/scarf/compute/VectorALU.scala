package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * VectorALU — 64-wide SIMD unit organized as 8 lanes × 8 elements.
 *
 * Each lane provides: ALU (add/sub), DIV, MUL, EXP functional units.
 * Embeds SoftmaxUnit for in-place softmax computation.
 * Tree reduction for sum/max (log2(64) = 6 levels).
 */
object VectorOp {
  val ADD: UInt     = 0.U(4.W)
  val MUL: UInt     = 1.U(4.W)
  val FMA: UInt     = 2.U(4.W)
  val MAX: UInt     = 3.U(4.W)
  val MIN: UInt     = 4.U(4.W)
  val RED_SUM: UInt = 5.U(4.W)
  val RED_MAX: UInt = 6.U(4.W)
  val EXP: UInt     = 7.U(4.W)
  val DIV: UInt     = 8.U(4.W)
  val SUB: UInt     = 9.U(4.W)
  val SOFTMAX: UInt = 10.U(4.W)
}

object SoftmaxPhase extends ChiselEnum {
  val sIdle, sExpSum, sNormalize, sDone = Value
}

class VectorALU(val width: Int = ScarfConfig.VectorALUWidth) extends Module {
  val numLanes = ScarfConfig.VectorALULanes
  val elemsPerLane = width / numLanes

  val io = IO(new Bundle {
    val a       = Input(Vec(width, UInt(ScarfConfig.DataWidth.W)))
    val b       = Input(Vec(width, UInt(ScarfConfig.DataWidth.W)))
    val c       = Input(Vec(width, UInt(ScarfConfig.DataWidth.W)))
    val op      = Input(UInt(4.W))
    val enable  = Input(Bool())

    val result  = Output(Vec(width, UInt(ScarfConfig.DataWidth.W)))
    val redOut  = Output(UInt(ScarfConfig.AccWidth.W))
    val valid   = Output(Bool())

    // Softmax interface
    val smStart      = Input(Bool())
    val smLogitIn    = Input(UInt(ScarfConfig.DataWidth.W))
    val smCandidateIn = Input(UInt(ScarfConfig.DataWidth.W))
    val smInValid    = Input(Bool())
    val smNumElements = Input(UInt(8.W))
    val smProbOut    = Output(UInt(ScarfConfig.DataWidth.W))
    val smDepthOut   = Output(UInt(ScarfConfig.AccWidth.W))
    val smOutValid   = Output(Bool())
    val smDone       = Output(Bool())
  })

  // ── Per-lane EXP LUT (shared across elements in lane) ──
  val expLUT = VecInit(Seq.fill(256)(0.U(ScarfConfig.DataWidth.W)))

  // ── Element-wise operations ──
  val elemResults = Wire(Vec(width, UInt(ScarfConfig.DataWidth.W)))
  for (lane <- 0 until numLanes) {
    for (elem <- 0 until elemsPerLane) {
      val i = lane * elemsPerLane + elem
      val expOut = expLUT(io.a(i)(ScarfConfig.DataWidth - 2, ScarfConfig.DataWidth - 9))
      val divOut = Mux(io.b(i) =/= 0.U, (io.a(i) << 8) / io.b(i), 0.U)
      elemResults(i) := MuxLookup(io.op, 0.U)(Seq(
        VectorOp.ADD -> (io.a(i) + io.b(i)),
        VectorOp.SUB -> (io.a(i) - io.b(i)),
        VectorOp.MUL -> (io.a(i) * io.b(i))(ScarfConfig.DataWidth - 1, 0),
        VectorOp.FMA -> ((io.a(i) * io.b(i))(ScarfConfig.DataWidth - 1, 0) + io.c(i)),
        VectorOp.MAX -> Mux(io.a(i) > io.b(i), io.a(i), io.b(i)),
        VectorOp.MIN -> Mux(io.a(i) < io.b(i), io.a(i), io.b(i)),
        VectorOp.EXP -> expOut,
        VectorOp.DIV -> divOut(ScarfConfig.DataWidth - 1, 0),
      ))
    }
  }

  // ── Tree reduction ──
  private def treeReduce[T <: Data](elems: Seq[T], op: (T, T) => T): T = {
    require(elems.nonEmpty)
    if (elems.length == 1) elems.head
    else {
      val pairs = elems.grouped(2).map {
        case Seq(a, b) => op(a, b)
        case Seq(a)    => a
      }.toSeq
      treeReduce(pairs, op)
    }
  }

  val sumElems = (0 until width).map(i => io.a(i).pad(ScarfConfig.AccWidth))
  val sumResult = treeReduce(sumElems, (a: UInt, b: UInt) => a + b)

  val maxElems = (0 until width).map(i => io.a(i))
  val maxResult = treeReduce(maxElems, (a: UInt, b: UInt) => Mux(a > b, a, b))

  val redResult = MuxLookup(io.op, 0.U)(Seq(
    VectorOp.RED_SUM -> sumResult,
    VectorOp.RED_MAX -> maxResult,
  ))

  io.result := RegNext(elemResults)
  io.redOut := RegNext(redResult)
  io.valid  := RegNext(io.enable)

  // ── Embedded SoftmaxUnit ──
  val smState   = RegInit(SoftmaxPhase.sIdle)
  val expSum    = RegInit(0.U(ScarfConfig.AccWidth.W))
  val depthAcc  = RegInit(0.U(ScarfConfig.AccWidth.W))
  val smCount   = RegInit(0.U(8.W))

  val expVal = io.smLogitIn + (1 << (ScarfConfig.DataWidth - 2)).U

  io.smProbOut  := 0.U
  io.smDepthOut := depthAcc
  io.smOutValid := false.B
  io.smDone     := smState === SoftmaxPhase.sDone

  switch(smState) {
    is(SoftmaxPhase.sIdle) {
      when(io.smStart) {
        smState  := SoftmaxPhase.sExpSum
        expSum   := 0.U
        smCount  := 0.U
        depthAcc := 0.U
      }
    }
    is(SoftmaxPhase.sExpSum) {
      when(io.smInValid) {
        expSum  := expSum + expVal
        smCount := smCount + 1.U
      }
      when(smCount === io.smNumElements - 1.U && io.smInValid) {
        smState := SoftmaxPhase.sNormalize
        smCount := 0.U
      }
    }
    is(SoftmaxPhase.sNormalize) {
      when(io.smInValid) {
        val prob = (expVal << 16) / (expSum + 1.U)
        val contribution = prob * io.smCandidateIn
        depthAcc := depthAcc + contribution(ScarfConfig.AccWidth - 1, 0)
        io.smProbOut  := prob(ScarfConfig.DataWidth - 1, 0)
        io.smOutValid := true.B
        smCount := smCount + 1.U
      }
      when(smCount === io.smNumElements - 1.U && io.smInValid) {
        smState := SoftmaxPhase.sDone
      }
    }
    is(SoftmaxPhase.sDone) {
      io.smOutValid := true.B
      smState := SoftmaxPhase.sIdle
    }
  }
}
