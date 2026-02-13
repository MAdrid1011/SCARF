package scarf.ggu

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * SHRotator — Spherical Harmonics Rotation.
 *
 * Corresponds to: ggu/sh_rotator.py
 *
 * Rotates SH coefficients from camera space to world space using
 * the rotation matrix from CovBuilder.
 *
 * For SH degree L, there are (L+1)² coefficients per color channel.
 *   Degree 2: 9 coefficients × 3 channels = 27 values
 *   Degree 4: 25 coefficients × 3 channels = 75 values
 *
 * Band-wise rotation:
 *   Band 0 (1 coeff): no rotation needed (DC term is rotationally invariant)
 *   Band 1 (3 coeffs): rotate by R (3×3 matrix)
 *   Band 2 (5 coeffs): rotate by 5×5 Wigner-D matrix (derived from R)
 *   Band 3 (7 coeffs): rotate by 7×7 Wigner-D matrix
 *   Band 4 (9 coeffs): rotate by 9×9 Wigner-D matrix
 *
 * Latency: ~80 cycles per Gaussian (dominated by band 3+4 matmuls).
 */
class SHRotator extends Module {
  val io = IO(new Bundle {
    // Rotation matrix (from CovBuilder) — 3×3 FP32
    val rotMatrix = Input(Vec(9, UInt(ScarfConfig.AccWidth.W)))

    // Input SH coefficients (up to 75 = 25 coeffs × 3 channels)
    val shIn      = Input(Vec(75, UInt(ScarfConfig.DataWidth.W)))
    val shDegree  = Input(UInt(3.W))  // 2 or 4

    // Control
    val start     = Input(Bool())
    val done      = Output(Bool())

    // Output SH coefficients (rotated)
    val shOut     = Output(Vec(75, UInt(ScarfConfig.DataWidth.W)))
  })

  val sIdle :: sBand0 :: sBand1 :: sHigherBands :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)

  val shOutReg = RegInit(VecInit(Seq.fill(75)(0.U(ScarfConfig.DataWidth.W))))
  val bandIdx  = RegInit(0.U(4.W))

  io.done  := state === sDone
  io.shOut := shOutReg

  // Number of coefficients per band: band L has 2L+1 coefficients
  // Degree 2: bands 0,1,2 → 1+3+5 = 9 coefficients per channel
  // Degree 4: bands 0,1,2,3,4 → 1+3+5+7+9 = 25 coefficients per channel

  switch(state) {
    is(sIdle) {
      when(io.start) {
        state   := sBand0
        bandIdx := 0.U
      }
    }
    is(sBand0) {
      // Band 0: DC term is rotationally invariant → pass through
      // For 3 channels: coefficients at indices 0, 25, 50 (if degree 4)
      for (ch <- 0 until 3) {
        shOutReg(ch * 25) := io.shIn(ch * 25)
      }
      state := sBand1
    }
    is(sBand1) {
      // Band 1: 3 coefficients per channel, rotated by R
      // shOut[1..3] = R × shIn[1..3] (per channel)
      for (ch <- 0 until 3) {
        val base = ch * 25 + 1  // Start of band 1 for this channel
        for (i <- 0 until 3) {
          shOutReg(base + i) := (
            (io.rotMatrix(i * 3)     * io.shIn(base))(ScarfConfig.DataWidth - 1, 0) +
            (io.rotMatrix(i * 3 + 1) * io.shIn(base + 1))(ScarfConfig.DataWidth - 1, 0) +
            (io.rotMatrix(i * 3 + 2) * io.shIn(base + 2))(ScarfConfig.DataWidth - 1, 0)
          )
        }
      }
      state := sHigherBands
      bandIdx := 2.U
    }
    is(sHigherBands) {
      // Bands 2-4: Simplified — in real HW, compute Wigner-D matrices from R.
      // For structural model: pass through (placeholder for Wigner-D computation).
      // The real implementation would derive band-L rotation matrices from R and
      // apply them as (2L+1)×(2L+1) matmuls per channel.
      val bandSize = bandIdx * 2.U + 1.U
      val bandStart = bandIdx * bandIdx  // sum of 1+3+5+... = L²

      for (ch <- 0 until 3) {
        for (k <- 0 until 9) { // Max band size for degree 4 = 9
          val idx = (ch * 25).U + bandStart + k.U
          when(k.U < bandSize && idx < 75.U) {
            shOutReg(idx) := io.shIn(idx)  // Placeholder: pass through
          }
        }
      }

      bandIdx := bandIdx + 1.U
      val maxBand = Mux(io.shDegree === 4.U, 4.U, 2.U)
      when(bandIdx >= maxBand) {
        state := sDone
      }
    }
    is(sDone) {
      state := sIdle
    }
  }
}
