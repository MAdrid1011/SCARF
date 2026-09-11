package scarf

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorBackendAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import scarf.SAESLevel
import scarf.control.SAESRepresentativeTileFlow

/**
  * Producer-to-buffer-to-consumer measurement for the representative T=4
  * retained-output path. Both instances use degree-2 (six-beat) records, one
  * producer beat per cycle, and the same sequential synchronous-read timing;
  * only selection differs.
  */
class SAESRepresentativeTileFlowTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SAESRepresentativeTileFlow"

  private case class FlowResult(
      cycles: Int,
      writeBeats: Int,
      committed: Int,
      readBeats: Int,
      records: Seq[(Int, Int, BigInt)],
  )

  private def sourceWord(pixel: Int, beat: Int): BigInt =
    (BigInt(pixel) << 112) | (BigInt(beat) << 64) | BigInt((pixel << 4) | beat)

  private def runFlow(
      dut: SAESRepresentativeTileFlow,
      expectedDescriptors: Int,
  ): FlowResult = {
    // Use distinct native records so the observed stream proves that the
    // selected source data, not an internally fabricated tag, reached the
    // synchronous consumer.
    for (pixel <- 0 until 16; beat <- 0 until 12) {
      dut.io.descriptorSource(pixel)(beat).poke(sourceWord(pixel, beat).U)
    }
    dut.io.start.poke(false.B)
    dut.clock.step()
    dut.io.start.poke(true.B)
    dut.io.startAccepted.expect(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)

    var cycles = 1
    var records = Vector.empty[(Int, Int, BigInt)]
    while (!dut.io.done.peekBoolean() && cycles < 400) {
      if (dut.io.readRecordValid.peekBoolean()) {
        records :+= (
          dut.io.readSourcePixel.peek().litValue.toInt,
          dut.io.readBeat.peek().litValue.toInt,
          dut.io.readData.peek().litValue,
        )
      }
      dut.clock.step()
      cycles += 1
    }
    assert(cycles < 400, s"representative flow did not finish (ran $cycles cycles)")
    dut.io.committedDescriptors.expect(expectedDescriptors.U)
    dut.io.acceptedWriteBeats.expect((expectedDescriptors * 6).U)
    dut.io.acceptedReadBeats.expect((expectedDescriptors * 6).U)
    FlowResult(
      cycles,
      dut.io.acceptedWriteBeats.peek().litValue.toInt,
      dut.io.committedDescriptors.peek().litValue.toInt,
      dut.io.acceptedReadBeats.peek().litValue.toInt,
      records,
    )
  }

  it should "execute only the four native L0 records through producer, buffer, and consumer" in {
    test(new SAESRepresentativeTileFlow(level = SAESLevel.sL0))
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
      val sparse = runFlow(dut, expectedDescriptors = 4)
      val l0Pixels = Seq(0, 3, 12, 15)
      val expectedSparseRecords = l0Pixels.flatMap { pixel =>
        (0 until 6).map(beat => (pixel, beat, sourceWord(pixel, beat)))
      }
      assert(sparse.records == expectedSparseRecords)
      assert(sparse.writeBeats == 24 && sparse.readBeats == 24 && sparse.committed == 4)
      assert(sparse.cycles == 73, s"expected 73 cycles, observed ${sparse.cycles}")
    }
  }

  it should "require the full dense record stream under the identical protocol" in {
    test(new SAESRepresentativeTileFlow(level = SAESLevel.sFull))
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
      val dense = runFlow(dut, expectedDescriptors = 16)
      val expectedDenseRecords = (0 until 16).flatMap { pixel =>
        (0 until 6).map(beat => (pixel, beat, sourceWord(pixel, beat)))
      }
      assert(dense.records == expectedDenseRecords)
      assert(dense.writeBeats == 96 && dense.readBeats == 96 && dense.committed == 16)
      assert(dense.cycles == 289, s"expected 289 cycles, observed ${dense.cycles}")
      // Each omitted degree-2 record removes six producer cycles and six
      // synchronous reads, each of which has a request and response cycle.
      assert(dense.cycles - 73 == 216)
    }
  }

  it should "execute the twelve native L1 anchors in the fixed scheduler order" in {
    test(new SAESRepresentativeTileFlow(level = SAESLevel.sL1))
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
      val l1 = runFlow(dut, expectedDescriptors = 12)
      val l1Pixels = Seq(0, 3, 12, 15, 1, 2, 4, 7, 8, 11, 13, 14)
      val expectedL1Records = l1Pixels.flatMap { pixel =>
        (0 until 6).map(beat => (pixel, beat, sourceWord(pixel, beat)))
      }
      assert(l1.records == expectedL1Records)
      assert(l1.writeBeats == 72 && l1.readBeats == 72 && l1.committed == 12)
      assert(l1.cycles == 217, s"expected 217 cycles, observed ${l1.cycles}")
      assert(289 - l1.cycles == 72)
    }
  }
}
