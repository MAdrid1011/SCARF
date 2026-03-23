package scarf.ggu

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * PositionCalc — Compute 3D world position from pixel + depth.
 *
 * Contains two sub-units (matching diagram):
 *   UnprojUnit: ray_dir = K_inv × [u, v, 1], pos_cam = ray_dir × depth
 *   TransformUnit: pos_world = R × pos_cam + t  (3×3 FMA array)
 *
 * CovBuilder's R is NOT the same as extrinsic R here;
 * extrinsics = camera-to-world transform [R_ext | t_ext].
 */

class UnprojUnit extends Module {
  val io = IO(new Bundle {
    val pixelX = Input(UInt(10.W))
    val pixelY = Input(UInt(10.W))
    val depth  = Input(UInt(ScarfConfig.DataWidth.W))
    val fx     = Input(UInt(ScarfConfig.AccWidth.W))
    val fy     = Input(UInt(ScarfConfig.AccWidth.W))
    val cx     = Input(UInt(ScarfConfig.AccWidth.W))
    val cy     = Input(UInt(ScarfConfig.AccWidth.W))

    val start  = Input(Bool())
    val done   = Output(Bool())
    val camX   = Output(UInt(ScarfConfig.AccWidth.W))
    val camY   = Output(UInt(ScarfConfig.AccWidth.W))
    val camZ   = Output(UInt(ScarfConfig.AccWidth.W))
  })

  val sIdle :: sUnproject :: sScale :: sDone :: Nil = Enum(4)
  val state = RegInit(sIdle)

  val rayX = RegInit(0.U(ScarfConfig.AccWidth.W))
  val rayY = RegInit(0.U(ScarfConfig.AccWidth.W))
  val rayZ = RegInit(0.U(ScarfConfig.AccWidth.W))
  val camXR = RegInit(0.U(ScarfConfig.AccWidth.W))
  val camYR = RegInit(0.U(ScarfConfig.AccWidth.W))
  val camZR = RegInit(0.U(ScarfConfig.AccWidth.W))

  io.done := state === sDone
  io.camX := camXR
  io.camY := camYR
  io.camZ := camZR

  switch(state) {
    is(sIdle) {
      when(io.start) { state := sUnproject }
    }
    is(sUnproject) {
      // ray = [(u - cx)/fx, (v - cy)/fy, 1.0]
      rayX := io.pixelX.pad(ScarfConfig.AccWidth) - io.cx
      rayY := io.pixelY.pad(ScarfConfig.AccWidth) - io.cy
      rayZ := 1.U << (ScarfConfig.DataWidth - 1)
      state := sScale
    }
    is(sScale) {
      // pos_cam = ray × depth
      camXR := (rayX * io.depth)(ScarfConfig.AccWidth - 1, 0)
      camYR := (rayY * io.depth)(ScarfConfig.AccWidth - 1, 0)
      camZR := (rayZ * io.depth)(ScarfConfig.AccWidth - 1, 0)
      state := sDone
    }
    is(sDone) {
      state := sIdle
    }
  }
}

/**
 * TransformUnit — 3×3 FMA array for extrinsic transform.
 * pos_world = R_ext × pos_cam + t_ext
 */
class TransformUnit extends Module {
  val io = IO(new Bundle {
    val camPos     = Input(Vec(3, UInt(ScarfConfig.AccWidth.W)))
    val extrinsics = Input(Vec(12, UInt(ScarfConfig.AccWidth.W))) // [R(9) | t(3)]
    val start      = Input(Bool())
    val done       = Output(Bool())
    val worldPos   = Output(Vec(3, UInt(ScarfConfig.AccWidth.W)))
  })

  val sIdle :: sCompute :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val result = RegInit(VecInit(Seq.fill(3)(0.U(ScarfConfig.AccWidth.W))))

  io.done := state === sDone
  io.worldPos := result

  switch(state) {
    is(sIdle) {
      when(io.start) { state := sCompute }
    }
    is(sCompute) {
      // 3×3 FMA: world[i] = R[i,0]*cam[0] + R[i,1]*cam[1] + R[i,2]*cam[2] + t[i]
      for (i <- 0 until 3) {
        val mac0 = (io.extrinsics(i * 3)     * io.camPos(0))(ScarfConfig.AccWidth - 1, 0)
        val mac1 = (io.extrinsics(i * 3 + 1) * io.camPos(1))(ScarfConfig.AccWidth - 1, 0)
        val mac2 = (io.extrinsics(i * 3 + 2) * io.camPos(2))(ScarfConfig.AccWidth - 1, 0)
        result(i) := mac0 + mac1 + mac2 + io.extrinsics(9 + i)
      }
      state := sDone
    }
    is(sDone) {
      state := sIdle
    }
  }
}

/**
 * PositionCalc — top-level combining UnprojUnit + TransformUnit.
 */
class PositionCalc extends Module {
  val io = IO(new Bundle {
    val pixelX     = Input(UInt(10.W))
    val pixelY     = Input(UInt(10.W))
    val depth      = Input(UInt(ScarfConfig.DataWidth.W))
    val fx         = Input(UInt(ScarfConfig.AccWidth.W))
    val fy         = Input(UInt(ScarfConfig.AccWidth.W))
    val cx         = Input(UInt(ScarfConfig.AccWidth.W))
    val cy         = Input(UInt(ScarfConfig.AccWidth.W))
    val extrinsics = Input(Vec(12, UInt(ScarfConfig.AccWidth.W)))
    val start      = Input(Bool())
    val done       = Output(Bool())
    val posX       = Output(UInt(ScarfConfig.AccWidth.W))
    val posY       = Output(UInt(ScarfConfig.AccWidth.W))
    val posZ       = Output(UInt(ScarfConfig.AccWidth.W))
  })

  val unproj    = Module(new UnprojUnit)
  val transform = Module(new TransformUnit)

  val sIdle :: sUnproj :: sTransform :: sDone :: Nil = Enum(4)
  val state = RegInit(sIdle)

  unproj.io.pixelX := io.pixelX
  unproj.io.pixelY := io.pixelY
  unproj.io.depth  := io.depth
  unproj.io.fx     := io.fx
  unproj.io.fy     := io.fy
  unproj.io.cx     := io.cx
  unproj.io.cy     := io.cy
  unproj.io.start  := state === sUnproj

  val camPos = Wire(Vec(3, UInt(ScarfConfig.AccWidth.W)))
  camPos(0) := unproj.io.camX
  camPos(1) := unproj.io.camY
  camPos(2) := unproj.io.camZ

  transform.io.camPos     := camPos
  transform.io.extrinsics := io.extrinsics
  transform.io.start      := state === sTransform

  io.done := state === sDone
  io.posX := transform.io.worldPos(0)
  io.posY := transform.io.worldPos(1)
  io.posZ := transform.io.worldPos(2)

  switch(state) {
    is(sIdle) {
      when(io.start) { state := sUnproj }
    }
    is(sUnproj) {
      when(unproj.io.done) { state := sTransform }
    }
    is(sTransform) {
      when(transform.io.done) { state := sDone }
    }
    is(sDone) {
      state := sIdle
    }
  }
}
