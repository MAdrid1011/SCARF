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
 * Uses one signed MAC per hash bit and streams one feature per cycle.
 * Latency: D projection cycles + 1 pack cycle.
 */
class LSHHashUnit(
  val lshDim: Int = ScarfConfig.LSHDim,
  val featureDim: Int = ScarfConfig.LSHFeatureDim,
) extends Module {
  require(lshDim == LSHProjectionROM.Rows)
  require(featureDim == LSHProjectionROM.Columns)

  val io = IO(new Bundle {
    val featureIn   = Input(Vec(featureDim, UInt(ScarfConfig.DataWidth.W)))
    val start       = Input(Bool())
    val done        = Output(Bool())
    val signature   = Output(UInt(lshDim.W))
  })

  // FP16 operands are normalized to [-1, 1]. Every such binary16 value can
  // be decoded exactly as a signed Q1.24 integer before the MAC.
  private def fp16ToQ24(bits: UInt): SInt = {
    val exponent = bits(14, 10)
    val fraction = bits(9, 0)
    val mantissa = Mux(exponent === 0.U, fraction.pad(11), Cat(1.U(1.W), fraction))
    val shift = Mux(exponent === 0.U, 0.U, exponent - 1.U)
    val magnitude = Wire(UInt(32.W))
    magnitude := (mantissa << shift)(31, 0)
    Mux(bits(15), -magnitude.asSInt, magnitude.asSInt)
  }

  // Seed-42 normalized hyperplanes, stored as IEEE-754 binary16 bit patterns.
  val projMatrix = VecInit(LSHProjectionROM.fp16Bits.map(row =>
    VecInit(row.map(value => value.U(ScarfConfig.DataWidth.W)))
  ))

  val sIdle :: sProject :: sPack :: sDone :: Nil = Enum(4)
  val state = RegInit(sIdle)

  // Q2.48 products need 55 signed bits for a 128-term dot product. Keep
  // additional headroom so chunk additions cannot truncate the sign.
  val accumWidth = 72
  val accum = RegInit(VecInit(Seq.fill(lshDim)(0.S(accumWidth.W))))

  val elemIdx = RegInit(0.U(log2Ceil(featureDim + 1).W))
  val chunkSize = 1

  val sigReg = RegInit(0.U(lshDim.W))

  io.done      := state === sDone
  io.signature := sigReg

  switch(state) {
    is(sIdle) {
      when(io.start) {
        state := sProject
        elemIdx := 0.U
        for (k <- 0 until lshDim) { accum(k) := 0.S }
      }
    }
    is(sProject) {
      for (k <- 0 until lshDim) {
        val products = (0 until chunkSize).map { j =>
          val idx = elemIdx + j.U
          val idxTrunc = idx(log2Ceil(featureDim) - 1, 0)
          val feature = fp16ToQ24(io.featureIn(idxTrunc))
          val projection = fp16ToQ24(projMatrix(k)(idxTrunc))
          Mux(idx < featureDim.U, feature * projection, 0.S(64.W))
        }
        val partialSum = products.reduce(_ +& _)
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
        sigBits(k) := accum(k) >= 0.S
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
