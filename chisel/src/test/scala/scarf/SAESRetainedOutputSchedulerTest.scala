package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.control.SAESRetainedOutputScheduler

/** Fixed T=4 request-order tests for the staged retained-output scheduler.
  *
  * A request is held until an upstream native S2/S3 descriptor is confirmed;
  * no synthetic descriptor data is injected by this control-only test.
  */
class SAESRetainedOutputSchedulerTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SAESRetainedOutputScheduler"

  private def idle(dut: SAESRetainedOutputScheduler): Unit = {
    dut.io.start.poke(false.B)
    dut.io.level.poke(SAESLevel.sFull)
    dut.io.upstreamDescriptorValid.poke(false.B)
  }

  private def runRoute(
      dut: SAESRetainedOutputScheduler,
      level: SAESLevel.Type,
      expected: Seq[Int],
  ): Unit = {
    dut.io.level.poke(level)
    dut.io.start.poke(true.B)
    dut.io.startAccepted.expect(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)

    dut.io.requestValid.expect(true.B)
    dut.io.expectedDescriptorCount.expect(expected.length.U)
    dut.io.requestOrdinal.expect(0.U)
    dut.io.requestPixelIndex.expect(expected.head.U)
    dut.io.upstreamDescriptorValid.poke(false.B)
    dut.clock.step()
    dut.io.requestOrdinal.expect(0.U)
    dut.io.requestPixelIndex.expect(expected.head.U)

    for ((pixel, ordinal) <- expected.zipWithIndex) {
      dut.io.requestValid.expect(true.B)
      dut.io.requestOrdinal.expect(ordinal.U)
      dut.io.requestPixelIndex.expect(pixel.U)
      dut.io.upstreamDescriptorValid.poke(true.B)
      dut.io.descriptorAccepted.expect(true.B)
      dut.clock.step()
      dut.io.upstreamDescriptorValid.poke(false.B)
    }
    dut.io.done.expect(true.B)
    dut.clock.step()
    dut.io.busy.expect(false.B)
  }

  it should "request exactly the four L0 primary descriptors in software order" in {
    test(new SAESRetainedOutputScheduler) { dut =>
      idle(dut)
      runRoute(dut, SAESLevel.sL0, Seq(0, 3, 12, 15))
    }
  }

  it should "retain the L0 prefix and request eight boundary L1 descriptors" in {
    test(new SAESRetainedOutputScheduler) { dut =>
      idle(dut)
      runRoute(dut, SAESLevel.sL1, Seq(0, 3, 12, 15, 1, 2, 4, 7, 8, 11, 13, 14))
    }
  }

  it should "reject full-path starts and overlapping route launches" in {
    test(new SAESRetainedOutputScheduler) { dut =>
      idle(dut)
      dut.io.start.poke(true.B)
      dut.io.level.poke(SAESLevel.sFull)
      dut.io.startAccepted.expect(false.B)
      dut.io.inputError.expect(true.B)
      dut.clock.step()

      dut.io.level.poke(SAESLevel.sL0)
      dut.io.startAccepted.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(true.B)
      dut.io.level.poke(SAESLevel.sL1)
      dut.io.startAccepted.expect(false.B)
      dut.io.inputError.expect(true.B)
    }
  }
}
