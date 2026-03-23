package scarf.fsdr

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * LSHHashUnit — Locality-Sensitive Hashing for feature signatures.
 *
 * Projects feature vector onto K random hyperplanes (ProjROM),
 * accumulates via MAC units, extracts sign bits, packs into K-bit signature.
 *
 * Latency: ceil(D / VectorALUWidth) + 1 = 3 cycles for D=128, ALU=64.
 */
class LSHHashUnit(
  val lshDim: Int = ScarfConfig.LSHDim,
  val featureDim: Int = ScarfConfig.LSHFeatureDim,
) extends Module {
  val io = IO(new Bundle {
    val featureIn   = Input(Vec(featureDim, UInt(ScarfConfig.DataWidth.W)))
    val start       = Input(Bool())
    val done        = Output(Bool())
    val signature   = Output(UInt(lshDim.W))
  })

  // ProjROM: K rows × D columns (FP16), inferred as ROM by synthesis
  val projMatrix = RegInit(VecInit(Seq.fill(lshDim)(
    VecInit(Seq.fill(featureDim)(0.U(ScarfConfig.DataWidth.W)))
  )))

  val sIdle :: sProject :: sPack :: sDone :: Nil = Enum(4)
  val state = RegInit(sIdle)

  // MAC accumulators (one per hyperplane)
  val accum = RegInit(VecInit(Seq.fill(lshDim)(0.U(ScarfConfig.AccWidth.W))))

  val elemIdx = RegInit(0.U(log2Ceil(featureDim + 1).W))
  val chunkSize = ScarfConfig.VectorALUWidth

  val sigReg = RegInit(0.U(lshDim.W))

  io.done      := state === sDone
  io.signature := sigReg

  switch(state) {
    is(sIdle) {
      when(io.start) {
        state := sProject
        elemIdx := 0.U
        for (k <- 0 until lshDim) { accum(k) := 0.U }
      }
    }
    is(sProject) {
      for (k <- 0 until lshDim) {
        var partialSum = 0.U(ScarfConfig.AccWidth.W)
        for (j <- 0 until chunkSize) {
          val idx = elemIdx + j.U
          val idxTrunc = idx(log2Ceil(featureDim) - 1, 0)
          val product = Mux(idx < featureDim.U,
            (projMatrix(k)(idxTrunc) * io.featureIn(idxTrunc))(ScarfConfig.AccWidth - 1, 0),
            0.U)
          partialSum = partialSum + product
        }
        accum(k) := accum(k) + partialSum
      }
      elemIdx := elemIdx + chunkSize.U
      when(elemIdx + chunkSize.U >= featureDim.U) {
        state := sPack
      }
    }
    is(sPack) {
      // Sign extraction + bit packing
      val sigBits = Wire(Vec(lshDim, Bool()))
      for (k <- 0 until lshDim) {
        sigBits(k) := !accum(k)(ScarfConfig.AccWidth - 1)
      }
      sigReg := sigBits.asUInt
      state := sDone
    }
    is(sDone) {
      state := sIdle
    }
  }
}

object HammingDistance {
  def apply(a: UInt, b: UInt, width: Int): UInt = {
    val xorBits = a ^ b
    val bits = (0 until width).map(i => xorBits(i))
    bits.map(_.asUInt).reduce(_ +& _)
  }
}
