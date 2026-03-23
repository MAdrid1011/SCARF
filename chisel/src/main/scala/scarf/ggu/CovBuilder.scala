package scarf.ggu

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * CovBuilder — Quaternion to Covariance Matrix + Rotation Matrix R.
 *
 * Internal structure (matching diagram):
 *   R_Regs: 9×FP32 register file storing rotation matrix
 *   3×3 FMA array: computes R·S²·R^T (6 independent elements)
 *
 * Pipeline: NormQuat → QuatToRot (Rodrigues) → BuildCov → Done
 *
 * R_Regs are DIRECTLY passed to SH_OPGenerator via combinational bypass
 * (eliminates cross-stage memory access for R matrix).
 */
class CovBuilder extends Module {
  val io = IO(new Bundle {
    val quatW  = Input(UInt(ScarfConfig.DataWidth.W))
    val quatX  = Input(UInt(ScarfConfig.DataWidth.W))
    val quatY  = Input(UInt(ScarfConfig.DataWidth.W))
    val quatZ  = Input(UInt(ScarfConfig.DataWidth.W))
    val scaleX = Input(UInt(ScarfConfig.DataWidth.W))
    val scaleY = Input(UInt(ScarfConfig.DataWidth.W))
    val scaleZ = Input(UInt(ScarfConfig.DataWidth.W))

    val start = Input(Bool())
    val done  = Output(Bool())

    val cov       = Output(Vec(6, UInt(ScarfConfig.AccWidth.W)))
    // R_Regs output: 3×3 FP32 rotation matrix (direct bypass to SH_OPGenerator)
    val rotMatrix = Output(Vec(9, UInt(ScarfConfig.AccWidth.W)))
  })

  val sIdle :: sNormQuat :: sQuatToRot :: sBuildCov :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)

  // R_Regs: 3×3 rotation matrix in FP32
  val R = RegInit(VecInit(Seq.fill(9)(0.U(ScarfConfig.AccWidth.W))))
  // Scale squared (diagonal)
  val s2 = RegInit(VecInit(Seq.fill(3)(0.U(ScarfConfig.AccWidth.W))))
  // Covariance output registers (upper triangle of symmetric 3×3)
  val covReg = RegInit(VecInit(Seq.fill(6)(0.U(ScarfConfig.AccWidth.W))))

  io.done      := state === sDone
  io.cov       := covReg
  io.rotMatrix := R

  switch(state) {
    is(sIdle) {
      when(io.start) { state := sNormQuat }
    }

    is(sNormQuat) {
      s2(0) := (io.scaleX * io.scaleX)(ScarfConfig.AccWidth - 1, 0)
      s2(1) := (io.scaleY * io.scaleY)(ScarfConfig.AccWidth - 1, 0)
      s2(2) := (io.scaleZ * io.scaleZ)(ScarfConfig.AccWidth - 1, 0)
      state := sQuatToRot
    }

    is(sQuatToRot) {
      // Rodrigues formula: quaternion → rotation matrix
      val w = io.quatW
      val x = io.quatX
      val y = io.quatY
      val z = io.quatZ
      val xx = (x * x)(ScarfConfig.AccWidth - 1, 0)
      val yy = (y * y)(ScarfConfig.AccWidth - 1, 0)
      val zz = (z * z)(ScarfConfig.AccWidth - 1, 0)
      val xy = (x * y)(ScarfConfig.AccWidth - 1, 0)
      val xz = (x * z)(ScarfConfig.AccWidth - 1, 0)
      val yz = (y * z)(ScarfConfig.AccWidth - 1, 0)
      val wx = (w * x)(ScarfConfig.AccWidth - 1, 0)
      val wy = (w * y)(ScarfConfig.AccWidth - 1, 0)
      val wz = (w * z)(ScarfConfig.AccWidth - 1, 0)

      val one = (1.U << (ScarfConfig.DataWidth - 1))

      R(0) := one - 2.U * (yy + zz)
      R(1) := 2.U * (xy - wz)
      R(2) := 2.U * (xz + wy)
      R(3) := 2.U * (xy + wz)
      R(4) := one - 2.U * (xx + zz)
      R(5) := 2.U * (yz - wx)
      R(6) := 2.U * (xz - wy)
      R(7) := 2.U * (yz + wx)
      R(8) := one - 2.U * (xx + yy)

      state := sBuildCov
    }

    is(sBuildCov) {
      // 3×3 FMA array: Cov[i][j] = Σ_k R[i][k] × s²[k] × R[j][k]
      // Upper triangle: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2) → indices 0..5
      val ijMap = Seq((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
      for (idx <- 0 until 6) {
        val (i, j) = ijMap(idx)
        val fma0 = (R(i * 3)     * s2(0) * R(j * 3)    )(ScarfConfig.AccWidth - 1, 0)
        val fma1 = (R(i * 3 + 1) * s2(1) * R(j * 3 + 1))(ScarfConfig.AccWidth - 1, 0)
        val fma2 = (R(i * 3 + 2) * s2(2) * R(j * 3 + 2))(ScarfConfig.AccWidth - 1, 0)
        covReg(idx) := fma0 + fma1 + fma2
      }
      state := sDone
    }

    is(sDone) {
      state := sIdle
    }
  }
}
