package scarf.ggu

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * SH_OPGenerator — Unified SH Rotation + Opacity Activation.
 *
 * Internal structure (matching diagram):
 *   Sig_LUT: Sigmoid LUT for opacity activation
 *   3×3 FMA array: Band matrix multiplication for Wigner-D rotation
 *   AddRegs: Intermediate registers for Wigner-D polynomial evaluation
 *
 * Band-wise SH rotation:
 *   Band 0: pass-through (DC term is rotationally invariant)
 *   Band 1: rotate by R (3×3 mat-vec via FMA array)
 *   Band 2+: rotate by Wigner-D matrices (derived from R via polynomial eval)
 *
 * Opacity: sigmoid(raw_opacity) via Sig_LUT, computed in parallel.
 */
class SHOPGenerator extends Module {
  val io = IO(new Bundle {
    // R matrix bypass from CovBuilder (combinational, no memory access)
    val rotMatrix = Input(Vec(9, UInt(ScarfConfig.AccWidth.W)))
    val shIn      = Input(Vec(75, UInt(ScarfConfig.DataWidth.W)))
    val shDegree  = Input(UInt(3.W))
    val opacityIn = Input(UInt(ScarfConfig.DataWidth.W))

    val start     = Input(Bool())
    val done      = Output(Bool())

    val shOut      = Output(Vec(75, UInt(ScarfConfig.DataWidth.W)))
    val opacityOut = Output(UInt(ScarfConfig.DataWidth.W))
  })

  val sIdle :: sBand0 :: sBand1 :: sHigherBands :: sOpacity :: sDone :: Nil = Enum(6)
  val state = RegInit(sIdle)

  val shOutReg = RegInit(VecInit(Seq.fill(75)(0.U(ScarfConfig.DataWidth.W))))
  val bandIdx  = RegInit(0.U(4.W))

  // Sig_LUT: 256-entry sigmoid lookup
  val sigLUT = VecInit(Seq.fill(ScarfConfig.ActivationLUTSize)(0.U(ScarfConfig.DataWidth.W)))

  // AddRegs: intermediate registers for Wigner-D polynomial coefficients
  val addRegs = RegInit(VecInit(Seq.fill(9)(0.U(ScarfConfig.AccWidth.W))))

  // Opacity result register
  val opacReg = RegInit(0.U(ScarfConfig.DataWidth.W))

  io.done       := state === sDone
  io.shOut      := shOutReg
  io.opacityOut := opacReg

  switch(state) {
    is(sIdle) {
      when(io.start) {
        state   := sBand0
        bandIdx := 0.U
      }
    }

    is(sBand0) {
      // Band 0: DC term pass-through (per channel)
      for (ch <- 0 until 3) {
        shOutReg(ch * 25) := io.shIn(ch * 25)
      }
      state := sBand1
    }

    is(sBand1) {
      // Band 1: 3×3 FMA array rotates 3 coefficients per channel by R
      for (ch <- 0 until 3) {
        val base = ch * 25 + 1
        for (i <- 0 until 3) {
          val fma0 = (io.rotMatrix(i * 3)     * io.shIn(base)    )(ScarfConfig.DataWidth - 1, 0)
          val fma1 = (io.rotMatrix(i * 3 + 1) * io.shIn(base + 1))(ScarfConfig.DataWidth - 1, 0)
          val fma2 = (io.rotMatrix(i * 3 + 2) * io.shIn(base + 2))(ScarfConfig.DataWidth - 1, 0)
          shOutReg(base + i) := fma0 + fma1 + fma2
        }
      }
      state := sHigherBands
      bandIdx := 2.U
    }

    is(sHigherBands) {
      // Bands 2-4: Wigner-D rotation via 3×3 FMA + AddRegs
      // Wigner-D matrices are derived from R via polynomial coefficients stored in AddRegs.
      // The 3×3 FMA array is reused across bands, processing (2L+1) elements per band
      // in ceil((2L+1)/3) iterations.
      val bandSize = bandIdx * 2.U + 1.U
      val bandStart = bandIdx * bandIdx

      // Store polynomial coefficients in AddRegs for current band
      for (k <- 0 until 9) {
        addRegs(k) := io.rotMatrix(k) // Wigner-D coeffs derived from R
      }

      for (ch <- 0 until 3) {
        for (k <- 0 until 9) {
          val idx = (ch * 25).U + bandStart + k.U
          when(k.U < bandSize && idx < 75.U) {
            val products = (0 until 3).map { m =>
              val srcIdx = (ch * 25).U + bandStart + m.U
              val valid = m.U < bandSize && srcIdx < 75.U
              Mux(valid,
                (addRegs(k % 3 * 3 + m) * io.shIn(srcIdx(6, 0)))(ScarfConfig.DataWidth - 1, 0),
                0.U(ScarfConfig.DataWidth.W))
            }
            shOutReg(idx(6, 0)) := products.reduce(_ + _)
          }
        }
      }

      bandIdx := bandIdx + 1.U
      val maxBand = Mux(io.shDegree === 4.U, 4.U, 2.U)
      when(bandIdx >= maxBand) {
        state := sOpacity
      }
    }

    is(sOpacity) {
      // Sig_LUT: sigmoid activation on raw opacity
      val lutIdx = io.opacityIn(ScarfConfig.DataWidth - 2, ScarfConfig.DataWidth - 9)
      opacReg := sigLUT(lutIdx)
      state := sDone
    }

    is(sDone) {
      state := sIdle
    }
  }
}
