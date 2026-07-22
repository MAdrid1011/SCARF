package scarf.compute

import chisel3._
import chisel3.util._

/** One exact-integer sufficient-statistics lane for SAES moment matching.
  *
  * The surrounding descriptor wrapper supplies already-selected pseudo
  * descriptors and assignment weights. This lane intentionally has no feature,
  * depth, tile, image, or target-RGB input, so it cannot change SAES routing.
  * Means are signed fixed-point values, variances use squared units, and all
  * weights use one shared nonnegative integer unit.
  */
class SAESScalarMomentAccumulator(
    val valueWidth: Int = 16,
    val weightWidth: Int = 16,
    val accumulatorWidth: Int = 64,
) extends Module {
  require(valueWidth > 0)
  require(weightWidth > 0)
  require(accumulatorWidth >= 2 * valueWidth + weightWidth + 2)

  val io = IO(new Bundle {
    val start = Input(Bool())
    val baseWeight = Input(UInt(weightWidth.W))
    val baseMean = Input(SInt(valueWidth.W))
    val baseVariance = Input(UInt(valueWidth.W))
    val startAccepted = Output(Bool())

    val updateValid = Input(Bool())
    val updateWeight = Input(UInt(weightWidth.W))
    val updateMean = Input(SInt(valueWidth.W))
    val updateVariance = Input(UInt(valueWidth.W))
    val updateAccepted = Output(Bool())

    val finish = Input(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val inputError = Output(Bool())

    val totalWeight = Output(UInt(accumulatorWidth.W))
    val meanOut = Output(SInt(accumulatorWidth.W))
    val varianceOut = Output(UInt(accumulatorWidth.W))
    val acceptedUpdates = Output(UInt(16.W))
  })

  val sIdle :: sAccumulate :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val weightSum = RegInit(0.S(accumulatorWidth.W))
  val firstMoment = RegInit(0.S(accumulatorWidth.W))
  val secondMoment = RegInit(0.S(accumulatorWidth.W))
  val meanResult = RegInit(0.S(accumulatorWidth.W))
  val varianceResult = RegInit(0.U(accumulatorWidth.W))
  val updateCount = RegInit(0.U(16.W))

  val startAccepted = state === sIdle && io.start && io.baseWeight =/= 0.U
  val updateAccepted = state === sAccumulate && io.updateValid && !io.finish
  io.startAccepted := startAccepted
  io.updateAccepted := updateAccepted
  io.busy := state === sAccumulate
  io.done := state === sDone
  io.inputError := (state === sIdle && io.start && io.baseWeight === 0.U) ||
    (state === sAccumulate && io.updateValid && io.finish)
  io.totalWeight := weightSum.asUInt
  io.meanOut := meanResult
  io.varianceOut := varianceResult
  io.acceptedUpdates := updateCount

  def weightedFirst(weight: UInt, mean: SInt): SInt =
    (weight.zext * mean).pad(accumulatorWidth)

  def weightedSecond(weight: UInt, mean: SInt, variance: UInt): SInt = {
    val rawSecond = variance.zext +& (mean * mean)
    (weight.zext * rawSecond).pad(accumulatorWidth)
  }

  switch(state) {
    is(sIdle) {
      when(startAccepted) {
        weightSum := io.baseWeight.zext.pad(accumulatorWidth)
        firstMoment := weightedFirst(io.baseWeight, io.baseMean)
        secondMoment := weightedSecond(
          io.baseWeight, io.baseMean, io.baseVariance
        )
        updateCount := 0.U
        state := sAccumulate
      }
    }
    is(sAccumulate) {
      when(io.updateValid && io.finish) {
        // Reject the ambiguous control beat without consuming descriptor data
        // or finalizing a partial accumulator. The caller must retry one
        // unambiguous update or finish command on a later cycle.
      }.elsewhen(updateAccepted) {
        weightSum := weightSum + io.updateWeight.zext.pad(accumulatorWidth)
        firstMoment := firstMoment + weightedFirst(io.updateWeight, io.updateMean)
        secondMoment := secondMoment + weightedSecond(
          io.updateWeight, io.updateMean, io.updateVariance
        )
        updateCount := updateCount + 1.U
      }.elsewhen(io.finish) {
        val finalMean = firstMoment / weightSum
        val finalSecond = secondMoment / weightSum
        val finalVariance = finalSecond - (finalMean * finalMean)
        meanResult := finalMean
        varianceResult := Mux(
          finalVariance > 0.S,
          finalVariance.asUInt,
          0.U(accumulatorWidth.W),
        )
        state := sDone
      }
    }
    is(sDone) {
      state := sIdle
    }
  }
}
