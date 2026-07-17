package scarf

import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import scarf.compute.{MMCU, SAESAssignmentNormalizer, SAESScalarMomentAccumulator}
import scarf.control.SAESController
import scarf.fsdr.FSDRCache
import scarf.ggu.GGUPE
import scarf.memory.{SAESDescriptorBuffer, TileBuffer}

class VerilogEmitTest extends AnyFlatSpec {

  behavior of "Current Chisel modules"

  it should "emit representative compute, reuse, control, GGU, and memory RTL" in {
    val saesRTL = ChiselStage.emitSystemVerilog(new SAESController)
    val outputs = Seq(
      ChiselStage.emitSystemVerilog(new MMCU(arraySize = 2)),
      ChiselStage.emitSystemVerilog(new FSDRCache(numEntries = 4, sigWidth = 4)),
      saesRTL,
      ChiselStage.emitSystemVerilog(new SAESDescriptorBuffer),
      ChiselStage.emitSystemVerilog(new SAESAssignmentNormalizer),
      ChiselStage.emitSystemVerilog(new SAESScalarMomentAccumulator),
      ChiselStage.emitSystemVerilog(new GGUPE),
      ChiselStage.emitSystemVerilog(new TileBuffer(sizeBytes = 1024)),
    )
    outputs.foreach { rtl =>
      assert(rtl.contains("module"))
      assert(!rtl.contains("FSGR"))
    }
    assert(saesRTL.contains("decisionCycles"))
  }
}
