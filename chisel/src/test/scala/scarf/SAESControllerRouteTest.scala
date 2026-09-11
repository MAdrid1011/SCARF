package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.control.SAESController

class SAESControllerRouteTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SAESController"

  it should "use a valid model-exported L1 route before scalar fallback" in {
    test(new SAESController) { dut =>
      dut.io.config.saesEnabled.poke(true.B)
      dut.io.config.saesFeatureVarThresh.poke(1.U)
      dut.io.config.saesDepthStdThresh.poke(1.U)
      dut.io.probeFeatureVar.poke(100.U)
      dut.io.probeDepthStd.poke(100.U)
      dut.io.crossCheckError.poke(0.U)
      dut.io.l0MaterializationValid.poke(false.B)
      dut.io.l1MaterializationValid.poke(false.B)
      dut.io.routeValid.poke(true.B)
      dut.io.routeLevel.poke(SAESLevel.sL1)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.clock.step()
      dut.io.done.expect(true.B)
      dut.io.level.expect(SAESLevel.sL1)
      dut.io.decisionCycles.expect(2.U)
    }
  }
}
