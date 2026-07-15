package scarf

import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import scarf.compute.MMCU
import scarf.control.SAESController
import scarf.fsdr.FSDRCache
import scarf.ggu.GGUPE
import scarf.memory.TileBuffer

class VerilogEmitTest extends AnyFlatSpec {

  behavior of "Current Chisel modules"

  it should "emit representative compute, reuse, control, GGU, and memory RTL" in {
    val outputs = Seq(
      ChiselStage.emitSystemVerilog(new MMCU(arraySize = 2)),
      ChiselStage.emitSystemVerilog(new FSDRCache(numEntries = 4, sigWidth = 4)),
      ChiselStage.emitSystemVerilog(new SAESController),
      ChiselStage.emitSystemVerilog(new GGUPE),
      ChiselStage.emitSystemVerilog(new TileBuffer(sizeBytes = 1024)),
    )
    outputs.foreach { rtl =>
      assert(rtl.contains("module"))
      assert(!rtl.contains("FSGR"))
    }
  }
}
