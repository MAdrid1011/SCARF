package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.compute.SAESScalarMomentAccumulator

/**
 * Exact-integer first/second-moment tests for one SAES numeric lane.
 *
 * Inputs are synthetic fixed-point integers, never model or target data.  The
 * expected values match ``saes.rtl_moment_reference`` and establish the
 * rounding/clamp contract before this lane is replicated across descriptors.
 */
class SAESScalarMomentAccumulatorTest
    extends AnyFlatSpec
    with ChiselScalatestTester {
  behavior of "SAESScalarMomentAccumulator"

  private def idle(dut: SAESScalarMomentAccumulator): Unit = {
    dut.io.start.poke(false.B)
    dut.io.baseWeight.poke(0.U)
    dut.io.baseMean.poke(0.S)
    dut.io.baseVariance.poke(0.U)
    dut.io.updateValid.poke(false.B)
    dut.io.updateWeight.poke(0.U)
    dut.io.updateMean.poke(0.S)
    dut.io.updateVariance.poke(0.U)
    dut.io.finish.poke(false.B)
  }

  it should "match an exact first- and second-moment merge" in {
    test(new SAESScalarMomentAccumulator(valueWidth = 16, weightWidth = 8)) { dut =>
      idle(dut)
      dut.io.start.poke(true.B)
      dut.io.baseWeight.poke(8.U)
      dut.io.baseMean.poke(10.S)
      dut.io.baseVariance.poke(4.U)
      dut.io.startAccepted.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      dut.io.updateValid.poke(true.B)
      dut.io.updateWeight.poke(2.U)
      dut.io.updateMean.poke(16.S)
      dut.io.updateVariance.poke(9.U)
      dut.io.updateAccepted.expect(true.B)
      dut.clock.step()

      dut.io.updateWeight.poke(6.U)
      dut.io.updateMean.poke((-2).S)
      dut.io.updateVariance.poke(1.U)
      dut.io.updateAccepted.expect(true.B)
      dut.clock.step()
      dut.io.updateValid.poke(false.B)

      dut.io.finish.poke(true.B)
      dut.clock.step()
      dut.io.finish.poke(false.B)
      dut.io.done.expect(true.B)
      dut.io.totalWeight.expect(16.U)
      dut.io.meanOut.expect(6.S)
      dut.io.varianceOut.expect(51.U)
      dut.io.acceptedUpdates.expect(2.U)
    }
  }

  it should "preserve a constant descriptor and fail closed on zero base mass" in {
    test(new SAESScalarMomentAccumulator(valueWidth = 16, weightWidth = 8)) { dut =>
      idle(dut)
      dut.io.start.poke(true.B)
      dut.io.baseWeight.poke(4.U)
      dut.io.baseMean.poke((-7).S)
      dut.io.baseVariance.poke(3.U)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.io.updateValid.poke(true.B)
      dut.io.updateWeight.poke(12.U)
      dut.io.updateMean.poke((-7).S)
      dut.io.updateVariance.poke(3.U)
      dut.clock.step()
      dut.io.updateValid.poke(false.B)
      dut.io.finish.poke(true.B)
      dut.clock.step()
      dut.io.finish.poke(false.B)
      dut.io.done.expect(true.B)
      dut.io.meanOut.expect((-7).S)
      dut.io.varianceOut.expect(3.U)

      dut.clock.step()
      dut.io.start.poke(true.B)
      dut.io.baseWeight.poke(0.U)
      dut.io.baseMean.poke(1.S)
      dut.io.baseVariance.poke(1.U)
      dut.io.startAccepted.expect(false.B)
      dut.io.inputError.expect(true.B)
      dut.clock.step()
      dut.io.done.expect(false.B)
    }
  }

  it should "reject a simultaneous update and finish without consuming either" in {
    test(new SAESScalarMomentAccumulator(valueWidth = 16, weightWidth = 8)) { dut =>
      idle(dut)
      dut.io.start.poke(true.B)
      dut.io.baseWeight.poke(4.U)
      dut.io.baseMean.poke(2.S)
      dut.io.baseVariance.poke(1.U)
      dut.clock.step()
      dut.io.start.poke(false.B)

      dut.io.updateValid.poke(true.B)
      dut.io.updateWeight.poke(4.U)
      dut.io.updateMean.poke(10.S)
      dut.io.updateVariance.poke(5.U)
      dut.io.finish.poke(true.B)
      dut.io.inputError.expect(true.B)
      dut.io.updateAccepted.expect(false.B)
      dut.clock.step()
      dut.io.done.expect(false.B)
      dut.io.busy.expect(true.B)
      dut.io.acceptedUpdates.expect(0.U)

      dut.io.finish.poke(false.B)
      dut.io.updateAccepted.expect(true.B)
      dut.clock.step()
      dut.io.updateValid.poke(false.B)
      dut.io.finish.poke(true.B)
      dut.clock.step()
      dut.io.finish.poke(false.B)
      dut.io.done.expect(true.B)
      dut.io.totalWeight.expect(8.U)
      dut.io.meanOut.expect(6.S)
      dut.io.varianceOut.expect(19.U)
      dut.io.acceptedUpdates.expect(1.U)
    }
  }
}
