package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.compute._

class GEMMUnitTest extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "OSPE"

  it should "accumulate products correctly" in {
    test(new OSPE) { dut =>
      dut.io.clear.poke(true.B)
      dut.io.enable.poke(false.B)
      dut.clock.step(1)
      dut.io.clear.poke(false.B)
      dut.io.accOut.expect(0.U, "should be 0 after clear")

      // Accumulate: 3*4 = 12
      dut.io.enable.poke(true.B)
      dut.io.aIn.poke(3.U)
      dut.io.bIn.poke(4.U)
      dut.clock.step(1)
      dut.io.accOut.expect(12.U, "should be 3*4 = 12")

      // Accumulate more: 12 + 2*5 = 22
      dut.io.aIn.poke(2.U)
      dut.io.bIn.poke(5.U)
      dut.clock.step(1)
      dut.io.accOut.expect(22.U, "should be 12 + 2*5 = 22")
    }
  }

  it should "clear accumulator" in {
    test(new OSPE) { dut =>
      dut.io.enable.poke(true.B)
      dut.io.clear.poke(false.B)
      dut.io.aIn.poke(10.U)
      dut.io.bIn.poke(10.U)
      dut.clock.step(1)
      dut.io.accOut.expect(100.U)

      // Clear
      dut.io.enable.poke(false.B)
      dut.io.clear.poke(true.B)
      dut.clock.step(1)
      dut.io.accOut.expect(0.U, "should be 0 after clear")
    }
  }

  behavior of "OutputStationaryArray"

  it should "instantiate for small size" in {
    test(new OutputStationaryArray(size = 2)) { dut =>
      dut.io.clear.poke(true.B)
      dut.io.enable.poke(false.B)
      dut.clock.step(1)
      dut.io.clear.poke(false.B)

      // Compute A=[1,2; 3,4] × B=[5,6; 7,8]
      // C[0,0] = 1*5 + 2*7 = 19
      // C[0,1] = 1*6 + 2*8 = 22
      // C[1,0] = 3*5 + 4*7 = 43
      // C[1,1] = 3*6 + 4*8 = 50

      // K-step 0: A col 0 = [1,3], B row 0 = [5,6]
      dut.io.enable.poke(true.B)
      dut.io.aRow(0).poke(1.U)
      dut.io.aRow(1).poke(3.U)
      dut.io.bCol(0).poke(5.U)
      dut.io.bCol(1).poke(6.U)
      dut.clock.step(1)

      // K-step 1: A col 1 = [2,4], B row 1 = [7,8]
      dut.io.aRow(0).poke(2.U)
      dut.io.aRow(1).poke(4.U)
      dut.io.bCol(0).poke(7.U)
      dut.io.bCol(1).poke(8.U)
      dut.clock.step(1)

      dut.io.enable.poke(false.B)

      // Verify results
      dut.io.results(0)(0).expect(19.U, "C[0,0] = 1*5+2*7 = 19")
      dut.io.results(0)(1).expect(22.U, "C[0,1] = 1*6+2*8 = 22")
      dut.io.results(1)(0).expect(43.U, "C[1,0] = 3*5+4*7 = 43")
      dut.io.results(1)(1).expect(50.U, "C[1,1] = 3*6+4*8 = 50")
    }
  }

  behavior of "GEMMUnit"

  it should "start idle and complete a small matmul" in {
    test(new GEMMUnit(arraySize = 2)) { dut =>
      dut.io.busy.expect(false.B)
      dut.io.done.expect(false.B)

      // 2×2 × 2×2 matmul
      dut.io.M.poke(2.U)
      dut.io.K.poke(2.U)
      dut.io.N.poke(2.U)
      dut.io.useBias.poke(false.B)

      // Provide dummy data
      for (i <- 0 until 2) {
        dut.io.aData(i).poke(1.U)
        dut.io.bData(i).poke(1.U)
        dut.io.biasData(i).poke(0.U)
      }

      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B)

      // Run until done
      var cycles = 0
      while (!dut.io.done.peekBoolean() && cycles < 50) {
        dut.clock.step(1)
        cycles += 1
      }
      assert(cycles < 50, s"GEMMUnit did not finish within 50 cycles (ran $cycles)")
    }
  }
}
