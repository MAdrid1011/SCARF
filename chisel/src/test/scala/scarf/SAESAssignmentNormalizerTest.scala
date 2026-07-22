package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.compute.SAESAssignmentNormalizer

/**
 * Exact Q0.16 normalization tests for SAES bilateral-kernel scores.
 *
 * These synthetic scores are independent of model features, depth, target RGB,
 * quality metrics, and routing. They lock the deterministic residual rule used
 * by the Python reference before the module is connected to a tile pipeline.
 */
class SAESAssignmentNormalizerTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SAESAssignmentNormalizer"

  private def idle(dut: SAESAssignmentNormalizer): Unit = {
    dut.io.start.poke(false.B)
    dut.io.anchorCount.poke(0.U)
    for (index <- 0 until 8) {
      dut.io.scores(index).poke(0.U)
    }
  }

  it should "emit an exactly normalized Q0.8 distribution with a stable residual anchor" in {
    test(new SAESAssignmentNormalizer(scoreWidth = 16, fractionBits = 8)) { dut =>
      idle(dut)
      dut.io.anchorCount.poke(4.U)
      dut.io.scores(0).poke(2.U)
      dut.io.scores(1).poke(3.U)
      dut.io.scores(2).poke(5.U)
      dut.io.scores(3).poke(0.U)
      dut.io.start.poke(true.B)
      dut.io.startAccepted.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.io.done.expect(true.B)
      dut.io.weights(0).expect(51.U)
      dut.io.weights(1).expect(76.U)
      dut.io.weights(2).expect(129.U)
      dut.io.weights(3).expect(0.U)
      dut.io.weightSum.expect(256.U)
      dut.io.residualAnchor.expect(2.U)
    }
  }

  it should "break maximum-score ties by the lowest active anchor and reject empty scores" in {
    test(new SAESAssignmentNormalizer(scoreWidth = 16, fractionBits = 8)) { dut =>
      idle(dut)
      dut.io.anchorCount.poke(3.U)
      dut.io.scores(0).poke(1.U)
      dut.io.scores(1).poke(1.U)
      dut.io.scores(2).poke(1.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.io.done.expect(true.B)
      dut.io.weights(0).expect(86.U)
      dut.io.weights(1).expect(85.U)
      dut.io.weights(2).expect(85.U)
      dut.io.residualAnchor.expect(0.U)

      dut.clock.step()
      dut.io.anchorCount.poke(2.U)
      dut.io.scores(0).poke(0.U)
      dut.io.scores(1).poke(0.U)
      dut.io.start.poke(true.B)
      dut.io.startAccepted.expect(false.B)
      dut.io.inputError.expect(true.B)
    }
  }
}
