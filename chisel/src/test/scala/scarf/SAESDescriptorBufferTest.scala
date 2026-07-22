package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.memory.SAESDescriptorBuffer

/**
 * Contract tests for the staged SAES retained-descriptor buffer.
 *
 * The buffer is intentionally tested independently of the unimplemented
 * assignment/moment datapath.  These cases prove degree-dependent packed
 * lengths, atomic descriptor visibility, overwrite invalidation, and rejection
 * of malformed write streams; they do not establish an SAES timing claim.
 */
class SAESDescriptorBufferTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SAESDescriptorBuffer"

  private def idle(dut: SAESDescriptorBuffer): Unit = {
    dut.io.shDegree.poke(2.U)
    dut.io.writeEn.poke(false.B)
    dut.io.writeIndex.poke(0.U)
    dut.io.writeBeat.poke(0.U)
    dut.io.writeData.poke(0.U)
    dut.io.writeLast.poke(false.B)
    dut.io.readEn.poke(false.B)
    dut.io.readIndex.poke(0.U)
    dut.io.readBeat.poke(0.U)
  }

  private def writeBeat(
      dut: SAESDescriptorBuffer,
      index: Int,
      beat: Int,
      data: BigInt,
      last: Boolean,
  ): Unit = {
    dut.io.writeEn.poke(true.B)
    dut.io.writeIndex.poke(index.U)
    dut.io.writeBeat.poke(beat.U)
    dut.io.writeData.poke(data.U)
    dut.io.writeLast.poke(last.B)
    dut.io.writeAccepted.expect(true.B)
    dut.io.writeError.expect(false.B)
    dut.clock.step()
    dut.io.writeEn.poke(false.B)
  }

  it should "pack degree-2 descriptors into six beats and expose them only after commit" in {
    test(new SAESDescriptorBuffer(slots = 3)) { dut =>
      idle(dut)
      dut.io.activeBeats.expect(6.U)
      dut.io.maxBeats.expect(12.U)

      for (beat <- 0 until 5) {
        writeBeat(dut, index = 0, beat = beat, data = 0x100 + beat, last = false)
        dut.io.readIndex.poke(0.U)
        dut.io.descriptorValid.expect(false.B)
      }
      writeBeat(dut, index = 0, beat = 5, data = 0x105, last = true)
      dut.io.readIndex.poke(0.U)
      dut.io.descriptorValid.expect(true.B)
      dut.io.storedBeats.expect(6.U)

      // Sync-read one completed descriptor beat. A committed descriptor is the
      // only source that may become visible to a downstream GGU hand-off.
      dut.io.readIndex.poke(0.U)
      dut.io.readBeat.poke(3.U)
      dut.io.readEn.poke(true.B)
      dut.clock.step()
      dut.io.readEn.poke(false.B)
      dut.io.readValid.expect(true.B)
      dut.io.readData.expect(0x103.U)
    }
  }

  it should "invalidate an overwritten descriptor and reject malformed streams" in {
    test(new SAESDescriptorBuffer(slots = 3)) { dut =>
      idle(dut)

      // A terminal marker on the first degree-2 beat is not a valid commit.
      dut.io.writeEn.poke(true.B)
      dut.io.writeIndex.poke(0.U)
      dut.io.writeBeat.poke(0.U)
      dut.io.writeData.poke(0x1.U)
      dut.io.writeLast.poke(true.B)
      dut.io.writeAccepted.expect(false.B)
      dut.io.writeError.expect(true.B)
      dut.clock.step()
      dut.io.writeEn.poke(false.B)

      for (beat <- 0 until 6) {
        writeBeat(dut, index = 1, beat = beat, data = 0x200 + beat, last = beat == 5)
      }
      dut.io.readIndex.poke(1.U)
      dut.io.descriptorValid.expect(true.B)

      // The first beat of a replacement invalidates the old completed record.
      writeBeat(dut, index = 1, beat = 0, data = 0x300, last = false)
      dut.io.readIndex.poke(1.U)
      dut.io.descriptorValid.expect(false.B)

      // A skipped beat cannot be accepted; the original in-flight order stays
      // intact so the writer can provide the missing beat next.
      dut.io.writeEn.poke(true.B)
      dut.io.writeIndex.poke(1.U)
      dut.io.writeBeat.poke(2.U)
      dut.io.writeData.poke(0x302.U)
      dut.io.writeLast.poke(false.B)
      dut.io.writeAccepted.expect(false.B)
      dut.io.writeError.expect(true.B)
      dut.clock.step()
      dut.io.writeEn.poke(false.B)

      // Slot index three is outside this three-slot buffer.
      dut.io.writeEn.poke(true.B)
      dut.io.writeIndex.poke(3.U)
      dut.io.writeBeat.poke(0.U)
      dut.io.writeData.poke(0.U)
      dut.io.writeLast.poke(false.B)
      dut.io.writeAccepted.expect(false.B)
      dut.io.writeError.expect(true.B)
    }
  }

  it should "derive the degree-4 layout from the same 128-bit contract" in {
    test(new SAESDescriptorBuffer(slots = 3)) { dut =>
      idle(dut)
      dut.io.shDegree.poke(4.U)
      dut.io.activeBeats.expect(12.U)
    }
  }
}
