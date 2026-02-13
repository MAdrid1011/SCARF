package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.compute._

/**
 * Test that each compute unit can be elaborated by Chisel.
 * This validates that all modules are structurally correct.
 */
class VerilogEmitTest extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "Chisel Elaboration"

  it should "elaborate PE" in {
    test(new PE) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate SystolicArray (size=2)" in {
    test(new SystolicArray(size = 2)) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate ConvEngine (size=4)" in {
    test(new ConvEngine(arraySize = 4)) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate OSPE" in {
    test(new OSPE) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate OutputStationaryArray (size=2)" in {
    test(new OutputStationaryArray(size = 2)) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate GEMMUnit (size=2)" in {
    test(new GEMMUnit(arraySize = 2)) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate BilinearSampler" in {
    test(new BilinearSampler) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate BilinearUnit (4 samplers)" in {
    test(new BilinearUnit(numSamplers = 4)) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate ActivationUnit" in {
    test(new ActivationUnit) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate NormUnit" in {
    test(new NormUnit) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate VectorALU (width=8)" in {
    test(new VectorALU(width = 8)) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate SoftmaxUnit" in {
    test(new SoftmaxUnit) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate PoolingUnit" in {
    test(new PoolingUnit) { dut =>
      dut.clock.step(1)
    }
  }

  it should "elaborate PadUnit" in {
    test(new PadUnit) { dut =>
      dut.clock.step(1)
    }
  }
}
