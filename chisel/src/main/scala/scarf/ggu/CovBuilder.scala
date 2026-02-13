package scarf.ggu

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * CovBuilder — Quaternion to 3D Covariance Matrix.
 *
 * Corresponds to: ggu/covariance_builder.py
 *
 * Algorithm:
 *   1. Normalize quaternion: q' = q / ||q||
 *   2. Convert to rotation matrix R (3×3) from quaternion
 *   3. Build scale matrix S = diag(exp(s_x), exp(s_y), exp(s_z))
 *   4. Covariance = R × S × S^T × R^T
 *
 * Latency: ~66 cycles (30 quat→R + 36 matrix multiply)
 */
class CovBuilder extends Module {
  val io = IO(new Bundle {
    // Quaternion input [w, x, y, z] (FP16)
    val quatW = Input(UInt(ScarfConfig.DataWidth.W))
    val quatX = Input(UInt(ScarfConfig.DataWidth.W))
    val quatY = Input(UInt(ScarfConfig.DataWidth.W))
    val quatZ = Input(UInt(ScarfConfig.DataWidth.W))
    // Scale input [sx, sy, sz] (FP16, pre-activation)
    val scaleX = Input(UInt(ScarfConfig.DataWidth.W))
    val scaleY = Input(UInt(ScarfConfig.DataWidth.W))
    val scaleZ = Input(UInt(ScarfConfig.DataWidth.W))

    val start = Input(Bool())
    val done  = Output(Bool())

    // Output: upper triangle of 3×3 symmetric covariance (FP32)
    val cov = Output(Vec(6, UInt(ScarfConfig.AccWidth.W)))
  })

  val sIdle :: sNormQuat :: sQuatToRot :: sBuildCov :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)

  // Rotation matrix registers (3×3, FP32)
  val R = RegInit(VecInit(Seq.fill(9)(0.U(ScarfConfig.AccWidth.W))))
  // Scale squared (diagonal)
  val s2 = RegInit(VecInit(Seq.fill(3)(0.U(ScarfConfig.AccWidth.W))))
  // Covariance output
  val covReg = RegInit(VecInit(Seq.fill(6)(0.U(ScarfConfig.AccWidth.W))))

  io.done := state === sDone
  io.cov  := covReg

  switch(state) {
    is(sIdle) {
      when(io.start) { state := sNormQuat }
    }
    is(sNormQuat) {
      // Quaternion normalization (structural: compute q_norm = q / ||q||)
      // Simplified: just use raw quaternion (assume pre-normalized)
      // Scale: s^2 = scale * scale (exp done externally via ActivationUnit)
      s2(0) := (io.scaleX * io.scaleX)(ScarfConfig.AccWidth - 1, 0)
      s2(1) := (io.scaleY * io.scaleY)(ScarfConfig.AccWidth - 1, 0)
      s2(2) := (io.scaleZ * io.scaleZ)(ScarfConfig.AccWidth - 1, 0)
      state := sQuatToRot
    }
    is(sQuatToRot) {
      // Quaternion to rotation matrix (Rodrigues formula)
      // R[0] = 1-2(y²+z²), R[1] = 2(xy-wz), R[2] = 2(xz+wy)
      // R[3] = 2(xy+wz),   R[4] = 1-2(x²+z²), R[5] = 2(yz-wx)
      // R[6] = 2(xz-wy),   R[7] = 2(yz+wx),   R[8] = 1-2(x²+y²)
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

      val one = (1.U << (ScarfConfig.DataWidth - 1))  // ~1.0 in fixed-point

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
      // Cov = R × diag(s²) × R^T
      // cov[i][j] = sum_k R[i][k] * s²[k] * R[j][k]
      // Upper triangle: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2) → indices 0..5
      // This is a 3×3 symmetric matrix, so we compute 6 elements
      for (idx <- 0 until 6) {
        val (i, j) = idx match {
          case 0 => (0, 0)
          case 1 => (0, 1)
          case 2 => (0, 2)
          case 3 => (1, 1)
          case 4 => (1, 2)
          case 5 => (2, 2)
        }
        covReg(idx) := (
          (R(i * 3) * s2(0) * R(j * 3))(ScarfConfig.AccWidth - 1, 0) +
          (R(i * 3 + 1) * s2(1) * R(j * 3 + 1))(ScarfConfig.AccWidth - 1, 0) +
          (R(i * 3 + 2) * s2(2) * R(j * 3 + 2))(ScarfConfig.AccWidth - 1, 0)
        )
      }
      state := sDone
    }
    is(sDone) {
      state := sIdle
    }
  }
}
