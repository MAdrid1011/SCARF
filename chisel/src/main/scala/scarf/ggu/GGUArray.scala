package scarf.ggu

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * GGU PE — Single Gaussian Generation Processing Element.
 *
 * Composes: PositionCalc → CovBuilder → SHRotator → Opacity sigmoid
 * Total latency: ~187 cycles per Gaussian (matching ggu/ggu_processor.py)
 */
class GGUPE extends Module {
  val io = IO(new Bundle {
    // Pixel coordinate + depth
    val pixelX    = Input(UInt(10.W))
    val pixelY    = Input(UInt(10.W))
    val depth     = Input(UInt(ScarfConfig.DataWidth.W))

    // Raw Gaussian parameters from S3 neural network
    val scaleX    = Input(UInt(ScarfConfig.DataWidth.W))
    val scaleY    = Input(UInt(ScarfConfig.DataWidth.W))
    val scaleZ    = Input(UInt(ScarfConfig.DataWidth.W))
    val quatW     = Input(UInt(ScarfConfig.DataWidth.W))
    val quatX     = Input(UInt(ScarfConfig.DataWidth.W))
    val quatY     = Input(UInt(ScarfConfig.DataWidth.W))
    val quatZ     = Input(UInt(ScarfConfig.DataWidth.W))
    val shIn      = Input(Vec(75, UInt(ScarfConfig.DataWidth.W)))
    val opacityIn = Input(UInt(ScarfConfig.DataWidth.W))
    val shDegree  = Input(UInt(3.W))

    // Camera parameters
    val fx = Input(UInt(ScarfConfig.AccWidth.W))
    val fy = Input(UInt(ScarfConfig.AccWidth.W))
    val cx = Input(UInt(ScarfConfig.AccWidth.W))
    val cy = Input(UInt(ScarfConfig.AccWidth.W))
    val extrinsics = Input(Vec(12, UInt(ScarfConfig.AccWidth.W)))

    // Control
    val start = Input(Bool())
    val done  = Output(Bool())
    val busy  = Output(Bool())

    // Outputs
    val posX   = Output(UInt(ScarfConfig.AccWidth.W))
    val posY   = Output(UInt(ScarfConfig.AccWidth.W))
    val posZ   = Output(UInt(ScarfConfig.AccWidth.W))
    val cov    = Output(Vec(6, UInt(ScarfConfig.AccWidth.W)))
    val shOut  = Output(Vec(75, UInt(ScarfConfig.DataWidth.W)))
    val opacityOut = Output(UInt(ScarfConfig.DataWidth.W))
  })

  // Sub-modules
  val posCalc = Module(new PositionCalc)
  val covBld  = Module(new CovBuilder)
  val shRot   = Module(new SHRotator)

  // 4-state sequential pipeline: Position → Covariance → SH Rotation → Done
  val sIdle :: sPosition :: sCovariance :: sSHRotation :: sOpacity :: sDone :: Nil = Enum(6)
  val state = RegInit(sIdle)

  // Wire up sub-modules
  posCalc.io.pixelX := io.pixelX
  posCalc.io.pixelY := io.pixelY
  posCalc.io.depth  := io.depth
  posCalc.io.fx := io.fx
  posCalc.io.fy := io.fy
  posCalc.io.cx := io.cx
  posCalc.io.cy := io.cy
  posCalc.io.extrinsics := io.extrinsics
  posCalc.io.start := state === sPosition

  covBld.io.quatW  := io.quatW
  covBld.io.quatX  := io.quatX
  covBld.io.quatY  := io.quatY
  covBld.io.quatZ  := io.quatZ
  covBld.io.scaleX := io.scaleX
  covBld.io.scaleY := io.scaleY
  covBld.io.scaleZ := io.scaleZ
  covBld.io.start  := state === sCovariance

  // Wire CovBuilder's rotation matrix directly to SHRotator
  shRot.io.rotMatrix := covBld.io.rotMatrix
  shRot.io.shIn     := io.shIn
  shRot.io.shDegree := io.shDegree
  shRot.io.start    := state === sSHRotation

  // Opacity: sigmoid LUT (simplified: passthrough for structural model)
  val opacityReg = RegInit(0.U(ScarfConfig.DataWidth.W))

  // Outputs
  io.posX := posCalc.io.posX
  io.posY := posCalc.io.posY
  io.posZ := posCalc.io.posZ
  io.cov  := covBld.io.cov
  io.shOut := shRot.io.shOut
  io.opacityOut := opacityReg
  io.done := state === sDone
  io.busy := state =/= sIdle

  switch(state) {
    is(sIdle) {
      when(io.start) { state := sPosition }
    }
    is(sPosition) {
      when(posCalc.io.done) { state := sCovariance }
    }
    is(sCovariance) {
      when(covBld.io.done) { state := sSHRotation }
    }
    is(sSHRotation) {
      when(shRot.io.done) { state := sOpacity }
    }
    is(sOpacity) {
      // Sigmoid activation on opacity (LUT-based, 2 cycles)
      // Simplified: passthrough
      opacityReg := io.opacityIn
      state := sDone
    }
    is(sDone) {
      state := sIdle
    }
  }
}

/**
 * GGUArray — 32 Parallel Gaussian Generation PEs.
 *
 * Corresponds to: ggu/ggu_processor.py
 *
 * Processes 32 Gaussians simultaneously. For 131,072 total Gaussians:
 *   131,072 / 32 = 4,096 batches × ~187 cycles = ~766K cycles.
 *
 * The array runs in parallel with S3 ConvEngine work (hidden behind pipeline).
 */
class GGUArray(val numPEs: Int = ScarfConfig.GGUPECount) extends Module {
  val io = IO(new Bundle {
    // Per-PE inputs (batched)
    val pixelX    = Input(Vec(numPEs, UInt(10.W)))
    val pixelY    = Input(Vec(numPEs, UInt(10.W)))
    val depth     = Input(Vec(numPEs, UInt(ScarfConfig.DataWidth.W)))
    val scaleX    = Input(Vec(numPEs, UInt(ScarfConfig.DataWidth.W)))
    val scaleY    = Input(Vec(numPEs, UInt(ScarfConfig.DataWidth.W)))
    val scaleZ    = Input(Vec(numPEs, UInt(ScarfConfig.DataWidth.W)))
    val quatW     = Input(Vec(numPEs, UInt(ScarfConfig.DataWidth.W)))
    val quatX     = Input(Vec(numPEs, UInt(ScarfConfig.DataWidth.W)))
    val quatY     = Input(Vec(numPEs, UInt(ScarfConfig.DataWidth.W)))
    val quatZ     = Input(Vec(numPEs, UInt(ScarfConfig.DataWidth.W)))
    val opacityIn = Input(Vec(numPEs, UInt(ScarfConfig.DataWidth.W)))
    val shDegree  = Input(UInt(3.W))

    // Shared camera parameters
    val fx = Input(UInt(ScarfConfig.AccWidth.W))
    val fy = Input(UInt(ScarfConfig.AccWidth.W))
    val cx = Input(UInt(ScarfConfig.AccWidth.W))
    val cy = Input(UInt(ScarfConfig.AccWidth.W))
    val extrinsics = Input(Vec(12, UInt(ScarfConfig.AccWidth.W)))

    // Control
    val start = Input(Bool())
    val done  = Output(Bool())
    val busy  = Output(Bool())

    // Per-PE outputs
    val posX = Output(Vec(numPEs, UInt(ScarfConfig.AccWidth.W)))
    val posY = Output(Vec(numPEs, UInt(ScarfConfig.AccWidth.W)))
    val posZ = Output(Vec(numPEs, UInt(ScarfConfig.AccWidth.W)))
    val cov  = Output(Vec(numPEs, Vec(6, UInt(ScarfConfig.AccWidth.W))))
    val opacityOut = Output(Vec(numPEs, UInt(ScarfConfig.DataWidth.W)))
  })

  val pes = Seq.fill(numPEs)(Module(new GGUPE))

  // Wire up all PEs with shared camera params and per-PE data
  for (i <- 0 until numPEs) {
    pes(i).io.pixelX    := io.pixelX(i)
    pes(i).io.pixelY    := io.pixelY(i)
    pes(i).io.depth     := io.depth(i)
    pes(i).io.scaleX    := io.scaleX(i)
    pes(i).io.scaleY    := io.scaleY(i)
    pes(i).io.scaleZ    := io.scaleZ(i)
    pes(i).io.quatW     := io.quatW(i)
    pes(i).io.quatX     := io.quatX(i)
    pes(i).io.quatY     := io.quatY(i)
    pes(i).io.quatZ     := io.quatZ(i)
    pes(i).io.opacityIn := io.opacityIn(i)
    pes(i).io.shDegree  := io.shDegree
    pes(i).io.shIn      := VecInit(Seq.fill(75)(0.U(ScarfConfig.DataWidth.W)))  // SH placeholder
    pes(i).io.fx := io.fx
    pes(i).io.fy := io.fy
    pes(i).io.cx := io.cx
    pes(i).io.cy := io.cy
    pes(i).io.extrinsics := io.extrinsics
    pes(i).io.start := io.start

    io.posX(i) := pes(i).io.posX
    io.posY(i) := pes(i).io.posY
    io.posZ(i) := pes(i).io.posZ
    io.cov(i)  := pes(i).io.cov
    io.opacityOut(i) := pes(i).io.opacityOut
  }

  // Array is done when ALL PEs are done
  io.done := pes.map(_.io.done).reduce(_ && _)
  io.busy := pes.map(_.io.busy).reduce(_ || _)
}
