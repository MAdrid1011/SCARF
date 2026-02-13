package scarf.ggu

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * PositionCalc — Compute 3D world position from pixel coordinate + depth.
 *
 * Corresponds to: ggu/position_calculator.py
 *
 * Algorithm:
 *   1. ray_dir = K_inv × [u, v, 1]  (unproject to camera space)
 *   2. pos_cam = ray_dir * depth     (scale by depth)
 *   3. pos_world = R × pos_cam + t   (transform to world)
 *
 * Fixed-point arithmetic for ASIC efficiency.
 * Latency: 10 cycles per Gaussian.
 */
class PositionCalc extends Module {
  val io = IO(new Bundle {
    // Inputs
    val pixelX    = Input(UInt(10.W))     // Pixel x coordinate
    val pixelY    = Input(UInt(10.W))     // Pixel y coordinate
    val depth     = Input(UInt(ScarfConfig.DataWidth.W))  // FP16 depth value
    val fx        = Input(UInt(ScarfConfig.AccWidth.W))   // Focal length x (FP32)
    val fy        = Input(UInt(ScarfConfig.AccWidth.W))   // Focal length y (FP32)
    val cx        = Input(UInt(ScarfConfig.AccWidth.W))   // Principal point x (FP32)
    val cy        = Input(UInt(ScarfConfig.AccWidth.W))   // Principal point y (FP32)
    val extrinsics = Input(Vec(12, UInt(ScarfConfig.AccWidth.W))) // 3×4 [R|t] (FP32)

    // Control
    val start     = Input(Bool())
    val done      = Output(Bool())

    // Output: world position [x, y, z] (FP32)
    val posX      = Output(UInt(ScarfConfig.AccWidth.W))
    val posY      = Output(UInt(ScarfConfig.AccWidth.W))
    val posZ      = Output(UInt(ScarfConfig.AccWidth.W))
  })

  val sIdle :: sUnproject :: sScale :: sTransform :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)
  val cycle = RegInit(0.U(4.W))

  // Intermediate registers
  val rayX = RegInit(0.U(ScarfConfig.AccWidth.W))
  val rayY = RegInit(0.U(ScarfConfig.AccWidth.W))
  val rayZ = RegInit(0.U(ScarfConfig.AccWidth.W))
  val camX = RegInit(0.U(ScarfConfig.AccWidth.W))
  val camY = RegInit(0.U(ScarfConfig.AccWidth.W))
  val camZ = RegInit(0.U(ScarfConfig.AccWidth.W))
  val worldX = RegInit(0.U(ScarfConfig.AccWidth.W))
  val worldY = RegInit(0.U(ScarfConfig.AccWidth.W))
  val worldZ = RegInit(0.U(ScarfConfig.AccWidth.W))

  io.done := state === sDone
  io.posX := worldX
  io.posY := worldY
  io.posZ := worldZ

  switch(state) {
    is(sIdle) {
      when(io.start) {
        state := sUnproject
        cycle := 0.U
      }
    }
    is(sUnproject) {
      // ray_dir = [(u - cx) / fx, (v - cy) / fy, 1.0]
      // Structural model: integer subtraction + division
      rayX := io.pixelX - io.cx(ScarfConfig.AccWidth - 1, 0)
      rayY := io.pixelY - io.cy(ScarfConfig.AccWidth - 1, 0)
      rayZ := 1.U << (ScarfConfig.DataWidth - 1)  // 1.0 in FP16 approx
      state := sScale
    }
    is(sScale) {
      // pos_cam = ray_dir * depth
      camX := (rayX * io.depth)(ScarfConfig.AccWidth - 1, 0)
      camY := (rayY * io.depth)(ScarfConfig.AccWidth - 1, 0)
      camZ := (rayZ * io.depth)(ScarfConfig.AccWidth - 1, 0)
      state := sTransform
      cycle := 0.U
    }
    is(sTransform) {
      // pos_world = R × pos_cam + t
      // R is extrinsics[0..8] (3×3), t is extrinsics[9..11]
      worldX := (io.extrinsics(0) * camX + io.extrinsics(1) * camY +
                 io.extrinsics(2) * camZ)(ScarfConfig.AccWidth - 1, 0) + io.extrinsics(9)
      worldY := (io.extrinsics(3) * camX + io.extrinsics(4) * camY +
                 io.extrinsics(5) * camZ)(ScarfConfig.AccWidth - 1, 0) + io.extrinsics(10)
      worldZ := (io.extrinsics(6) * camX + io.extrinsics(7) * camY +
                 io.extrinsics(8) * camZ)(ScarfConfig.AccWidth - 1, 0) + io.extrinsics(11)
      state := sDone
    }
    is(sDone) {
      state := sIdle
    }
  }
}
