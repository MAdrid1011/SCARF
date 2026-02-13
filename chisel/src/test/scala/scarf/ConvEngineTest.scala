package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.compute._

class ConvEngineTest extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "PE"

  it should "pass data through with 1-cycle delay" in {
    test(new PE) { dut =>
      // Load weight = 2
      dut.io.weightIn.poke(2.U)
      dut.io.weightLoad.poke(true.B)
      dut.io.enable.poke(false.B)
      dut.io.clear.poke(false.B)
      dut.io.dataIn.poke(0.U)
      dut.io.psumIn.poke(0.U)
      dut.clock.step(1)
      dut.io.weightLoad.poke(false.B)

      // Send data = 3, psum_in = 0 → psum_out = 0 + 2*3 = 6
      dut.io.enable.poke(true.B)
      dut.io.dataIn.poke(3.U)
      dut.io.psumIn.poke(0.U)
      dut.clock.step(1)

      // After 1 cycle: dataOut = 3 (passed through), psumOut = 6
      dut.io.dataOut.expect(3.U, "data should pass through with 1-cycle delay")
      dut.io.psumOut.expect(6.U, "psum should be weight * data + psum_in = 2*3+0 = 6")
    }
  }

  it should "accumulate partial sums across cycles" in {
    test(new PE) { dut =>
      dut.io.weightIn.poke(1.U)
      dut.io.weightLoad.poke(true.B)
      dut.io.enable.poke(false.B)
      dut.io.clear.poke(false.B)
      dut.io.dataIn.poke(0.U)
      dut.io.psumIn.poke(0.U)
      dut.clock.step(1)
      dut.io.weightLoad.poke(false.B)

      // Cycle 1: data=5, psum_in=10 → psum_out = 10 + 1*5 = 15
      dut.io.enable.poke(true.B)
      dut.io.dataIn.poke(5.U)
      dut.io.psumIn.poke(10.U)
      dut.clock.step(1)
      dut.io.psumOut.expect(15.U)
    }
  }

  it should "clear accumulated state" in {
    test(new PE) { dut =>
      dut.io.weightIn.poke(3.U)
      dut.io.weightLoad.poke(true.B)
      dut.io.enable.poke(false.B)
      dut.io.clear.poke(false.B)
      dut.io.dataIn.poke(0.U)
      dut.io.psumIn.poke(0.U)
      dut.clock.step(1)
      dut.io.weightLoad.poke(false.B)

      // Compute one cycle
      dut.io.enable.poke(true.B)
      dut.io.dataIn.poke(4.U)
      dut.io.psumIn.poke(0.U)
      dut.clock.step(1)
      dut.io.psumOut.expect(12.U) // 3*4 = 12

      // Clear
      dut.io.enable.poke(false.B)
      dut.io.clear.poke(true.B)
      dut.clock.step(1)
      dut.io.psumOut.expect(0.U, "psum should be 0 after clear")
      dut.io.dataOut.expect(0.U, "data should be 0 after clear")
    }
  }

  behavior of "SystolicArray"

  it should "instantiate without error for small size" in {
    // Use size=2 for fast test (48×48 is too large for simulation)
    test(new SystolicArray(size = 2)) { dut =>
      dut.io.clear.poke(true.B)
      dut.io.enable.poke(false.B)
      dut.io.weightLoad.poke(false.B)
      dut.clock.step(1)
      dut.io.clear.poke(false.B)

      // Load identity-like weights: col 0 = [1, 0], col 1 = [0, 1]
      dut.io.weightLoad.poke(true.B)
      dut.io.weightCol.poke(0.U)
      dut.io.weightData(0).poke(1.U)
      dut.io.weightData(1).poke(0.U)
      dut.clock.step(1)

      dut.io.weightCol.poke(1.U)
      dut.io.weightData(0).poke(0.U)
      dut.io.weightData(1).poke(1.U)
      dut.clock.step(1)
      dut.io.weightLoad.poke(false.B)

      // Stream input [5, 7] through
      dut.io.enable.poke(true.B)
      dut.io.dataIn(0).poke(5.U)
      dut.io.dataIn(1).poke(7.U)
      dut.clock.step(2) // Need 2 cycles for data to propagate through 2-row array

      // With identity weights, output should be [5, 7]
      // (after systolic delay accounting)
      dut.io.psumOut(0).peek()
      dut.io.psumOut(1).peek()
    }
  }

  behavior of "ConvEngine"

  it should "start in idle and transition to done" in {
    test(new ConvEngine(arraySize = 4)) { dut =>
      dut.io.busy.expect(false.B, "should start idle")
      dut.io.done.expect(false.B)

      // Configure: 4 in_channels, 4 out_channels, kernel 1×1
      dut.io.inChannels.poke(4.U)
      dut.io.outChannels.poke(4.U)
      dut.io.kernelSize.poke(1.U)
      dut.io.stride.poke(1.U)
      dut.io.padding.poke(0.U)
      dut.io.fuseReLU.poke(false.B)

      // Provide dummy data
      for (i <- 0 until 4) {
        dut.io.weightData(i).poke(1.U)
        dut.io.inputData(i).poke(1.U)
      }

      // Start
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B, "should be busy after start")

      // Run until done (max 100 cycles to prevent hang)
      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 100) {
        dut.clock.step(1)
        cycles += 1
      }
      assert(cycles < 100, s"ConvEngine did not finish within 100 cycles (ran $cycles)")
      dut.io.done.expect(true.B)
    }
  }
}
