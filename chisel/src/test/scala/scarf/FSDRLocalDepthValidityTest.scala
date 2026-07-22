package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.fsdr.FSDRLocalDepthValidity

class FSDRLocalDepthValidityTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "FSDRLocalDepthValidity"

  it should "match the tile-local Q0.10 software guard" in {
    test(new FSDRLocalDepthValidity) { dut =>
      dut.io.gammaQ10.poke(102.U)
      dut.io.cachedDepth.poke(1100.U)
      dut.io.localDepthMean.poke(1000.U)

      dut.io.localDepthCount.poke(2.U)
      dut.io.valid.expect(true.B)

      dut.io.localDepthCount.poke(3.U)
      dut.io.valid.expect(false.B)

      dut.io.cachedDepth.poke(1090.U)
      dut.io.valid.expect(true.B)

      dut.io.localDepthMean.poke(0.U)
      dut.io.valid.expect(true.B)
    }
  }
}
