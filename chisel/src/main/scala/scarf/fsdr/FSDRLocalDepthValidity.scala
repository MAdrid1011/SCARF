package scarf.fsdr

import chisel3._
import scarf.ScarfConfig

/** Tile-local depth validity guard shared by the FSDR claim controller.
  *
  * Fewer than three completed probe/neighbor depths conservatively allow the
  * Hamming hit. Afterwards, |cached-mean| / mean is compared with gamma in
  * unsigned Q0.10. Depth payloads use the existing positive FP16/fixed-point
  * representation; only relative integer magnitudes are required here.
  */
class FSDRLocalDepthValidity extends Module {
  val io = IO(new Bundle {
    val cachedDepth = Input(UInt(ScarfConfig.DataWidth.W))
    val localDepthMean = Input(UInt(ScarfConfig.DataWidth.W))
    val localDepthCount = Input(UInt(8.W))
    val gammaQ10 = Input(UInt(10.W))
    val valid = Output(Bool())
  })

  val difference = Mux(
    io.cachedDepth >= io.localDepthMean,
    io.cachedDepth - io.localDepthMean,
    io.localDepthMean - io.cachedDepth,
  )
  val scaledDifference = difference * 1024.U
  val allowedDifference = io.localDepthMean * io.gammaQ10
  io.valid := io.localDepthCount < 3.U ||
    io.localDepthMean === 0.U ||
    scaledDifference <= allowedDifference
}
