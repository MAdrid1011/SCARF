package scarf.compute

import chisel3._
import chisel3.util._

/** Deterministic Q0.fractionBits normalization for SAES bilateral scores.
  *
  * The score producer remains responsible for the paper's spatial, feature,
  * and L1 depth-reliability terms. This module only normalizes already computed
  * nonnegative scores and assigns finite-precision residual mass to the lowest
  * index with the maximum active score.
  */
class SAESAssignmentNormalizer(
    val scoreWidth: Int = 16,
    val fractionBits: Int = 16,
    val maxAnchors: Int = 8,
) extends Module {
  require(scoreWidth > 0)
  require(fractionBits > 0)
  require(maxAnchors > 0)

  private val countWidth = math.max(1, log2Ceil(maxAnchors + 1))
  private val indexWidth = math.max(1, log2Ceil(maxAnchors))
  private val outputWidth = fractionBits + 1
  private val scoreSumWidth = scoreWidth + log2Ceil(maxAnchors + 1)
  private val weightSumWidth = outputWidth + log2Ceil(maxAnchors + 1)

  val io = IO(new Bundle {
    val start = Input(Bool())
    val anchorCount = Input(UInt(countWidth.W))
    val scores = Input(Vec(maxAnchors, UInt(scoreWidth.W)))
    val startAccepted = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val inputError = Output(Bool())
    val weights = Output(Vec(maxAnchors, UInt(outputWidth.W)))
    val weightSum = Output(UInt(weightSumWidth.W))
    val residualAnchor = Output(UInt(indexWidth.W))
  })

  val sIdle :: sDone :: Nil = Enum(2)
  val state = RegInit(sIdle)
  val weightsReg = RegInit(VecInit(Seq.fill(maxAnchors)(0.U(outputWidth.W))))
  val residualAnchorReg = RegInit(0.U(indexWidth.W))

  val activeScores = (0 until maxAnchors).map { index =>
    Mux(index.U < io.anchorCount, io.scores(index).pad(scoreSumWidth), 0.U(scoreSumWidth.W))
  }
  val scoreSum = Wire(UInt(scoreSumWidth.W))
  scoreSum := activeScores.reduce(_ +& _)
  val countValid = io.anchorCount >= 1.U && io.anchorCount <= maxAnchors.U

  val runningMaxScore = Wire(Vec(maxAnchors + 1, UInt(scoreWidth.W)))
  val runningMaxIndex = Wire(Vec(maxAnchors + 1, UInt(indexWidth.W)))
  runningMaxScore(0) := 0.U
  runningMaxIndex(0) := 0.U
  for (index <- 0 until maxAnchors) {
    val replace = index.U < io.anchorCount && io.scores(index) > runningMaxScore(index)
    runningMaxScore(index + 1) := Mux(replace, io.scores(index), runningMaxScore(index))
    runningMaxIndex(index + 1) := Mux(replace, index.U, runningMaxIndex(index))
  }

  val scale = (BigInt(1) << fractionBits).U(outputWidth.W)
  val safeScoreSum = Mux(scoreSum === 0.U, 1.U(scoreSumWidth.W), scoreSum)
  val rawWeights = Wire(Vec(maxAnchors, UInt(outputWidth.W)))
  for (index <- 0 until maxAnchors) {
    val scaledScore = io.scores(index) * scale
    val quotient = scaledScore / safeScoreSum
    rawWeights(index) := Mux(
      index.U < io.anchorCount,
      quotient(outputWidth - 1, 0),
      0.U(outputWidth.W),
    )
  }
  val rawWeightSum = Wire(UInt(weightSumWidth.W))
  rawWeightSum := rawWeights.map(_.pad(weightSumWidth)).reduce(_ +& _)
  val residual = (scale.pad(weightSumWidth) - rawWeightSum)(outputWidth - 1, 0)
  val plannedWeights = Wire(Vec(maxAnchors, UInt(outputWidth.W)))
  for (index <- 0 until maxAnchors) {
    plannedWeights(index) := Mux(
      index.U < io.anchorCount && runningMaxIndex(maxAnchors) === index.U,
      (rawWeights(index) + residual)(outputWidth - 1, 0),
      rawWeights(index),
    )
  }

  val startAccepted = state === sIdle && io.start && countValid && scoreSum =/= 0.U
  io.startAccepted := startAccepted
  io.busy := false.B
  io.done := state === sDone
  io.inputError := state === sIdle && io.start && !startAccepted
  io.weights := weightsReg
  io.weightSum := weightsReg.map(_.pad(weightSumWidth)).reduce(_ +& _)
  io.residualAnchor := residualAnchorReg

  switch(state) {
    is(sIdle) {
      when(startAccepted) {
        weightsReg := plannedWeights
        residualAnchorReg := runningMaxIndex(maxAnchors)
        state := sDone
      }
    }
    is(sDone) {
      state := sIdle
    }
  }
}
