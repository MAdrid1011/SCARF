package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * LSHHashUnit — Locality-Sensitive Hashing for feature signatures.
 *
 * Corresponds to: fsgr/lsh_hasher.py (LSHHasher)
 *
 * Algorithm:
 *   1. Project feature vector onto K random hyperplanes (stored in ROM)
 *   2. Extract sign of each projection → 1-bit per hyperplane
 *   3. Pack K sign bits into K-bit integer signature
 *
 * Property: P(sign(r·x) = sign(r·y)) = 1 - arccos(cos(x,y))/π
 *   → High cosine similarity → Low Hamming distance between signatures
 *
 * Architecture:
 *   - K parallel dot-product units (each = D MACs using VectorALU)
 *   - K sign extractors (comparator, combinational)
 *   - K-bit pack (wire concatenation, 0 cycles)
 *   - Projection matrix stored in on-chip ROM (K × D × 16 bits)
 *
 * Latency: ceil(D / VectorALUWidth) cycles for dot products + 1 cycle sign pack
 * For D=128, VectorALU=64: 2 + 1 = 3 cycles per signature
 */
class LSHHashUnit(
  val lshDim: Int = 16,           // K: number of hash bits (hyperplanes)
  val featureDim: Int = 128,      // D: input feature dimension
) extends Module {
  val io = IO(new Bundle {
    // Feature input (streamed, featureDim elements)
    val featureIn   = Input(Vec(featureDim, UInt(ScarfConfig.DataWidth.W)))
    val start       = Input(Bool())
    val done        = Output(Bool())

    // Hash output: K-bit signature
    val signature   = Output(UInt(lshDim.W))
  })

  // Projection matrix ROM: K rows × D columns (FP16)
  // In real implementation, loaded from DRAM at initialization.
  // Structurally modeled as registers (synthesis tool infers ROM).
  val projMatrix = RegInit(VecInit(Seq.fill(lshDim)(
    VecInit(Seq.fill(featureDim)(0.U(ScarfConfig.DataWidth.W)))
  )))

  // FSM
  val sIdle :: sProject :: sPack :: sDone :: Nil = Enum(4)
  val state = RegInit(sIdle)

  // Dot product accumulators (one per hyperplane)
  val accum = RegInit(VecInit(Seq.fill(lshDim)(0.U(ScarfConfig.AccWidth.W))))

  // Streaming counter (processes VectorALUWidth elements per cycle)
  val elemIdx = RegInit(0.U(log2Ceil(featureDim + 1).W))
  val chunkSize = ScarfConfig.VectorALUWidth  // 64 elements per cycle

  // Output signature register
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
      // Compute partial dot products: accum[k] += sum(proj[k][i] * feature[i])
      // Process `chunkSize` elements per cycle
      for (k <- 0 until lshDim) {
        var partialSum = 0.U(ScarfConfig.AccWidth.W)
        for (j <- 0 until chunkSize) {
          val idx = elemIdx + j.U
          // Guard against out-of-bounds
          val product = Mux(idx < featureDim.U,
            (projMatrix(k)(idx) * io.featureIn(idx))(ScarfConfig.AccWidth - 1, 0),
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
      // Extract sign bits and pack into signature
      // Sign bit = MSB of accumulator (0 = positive → hash bit 1, 1 = negative → hash bit 0)
      val sigBits = Wire(Vec(lshDim, Bool()))
      for (k <- 0 until lshDim) {
        sigBits(k) := !accum(k)(ScarfConfig.AccWidth - 1)  // Positive → 1
      }
      sigReg := sigBits.asUInt
      state := sDone
    }

    is(sDone) {
      state := sIdle
    }
  }
}

/**
 * HammingDistance — Combinational Hamming distance calculator.
 *
 * Computes popcount(a XOR b) — the number of differing bits.
 * Used by FSGRCache for parallel cache lookup.
 *
 * Hardware: XOR gate array + popcount tree (combinational, 0 cycle latency)
 */
object HammingDistance {
  def apply(a: UInt, b: UInt, width: Int): UInt = {
    val xorBits = a ^ b
    // Popcount via binary tree addition
    val bits = (0 until width).map(i => xorBits(i))
    bits.map(_.asUInt).reduce(_ +& _)
  }
}
