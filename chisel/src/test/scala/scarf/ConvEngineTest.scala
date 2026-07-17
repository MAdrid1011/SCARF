package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.compute._
import scarf.control.SAESController
import scarf.fsdr.FSDRCache

class CurrentModuleBehaviorTest extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "VectorALU"

  it should "add vectors and assert valid after one cycle" in {
    test(new VectorALU(width = 8)) { dut =>
      for (i <- 0 until 8) {
        dut.io.a(i).poke((i + 1).U)
        dut.io.b(i).poke(2.U)
        dut.io.c(i).poke(0.U)
      }
      dut.io.op.poke(VectorOp.ADD)
      dut.io.enable.poke(true.B)
      dut.io.smStart.poke(false.B)
      dut.io.smLogitIn.poke(0.U)
      dut.io.smCandidateIn.poke(0.U)
      dut.io.smInValid.poke(false.B)
      dut.io.smNumElements.poke(1.U)
      dut.clock.step()
      dut.io.valid.expect(true.B)
      for (i <- 0 until 8) {
        dut.io.result(i).expect((i + 3).U)
      }
    }
  }

  behavior of "FSDRCache"

  it should "return an inserted exact-signature entry" in {
    test(new FSDRCache(numEntries = 4, sigWidth = 8))
      .withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.io.lookupEn.poke(false.B)
      dut.io.lookupSig.poke(0.U)
      dut.io.hammingThresh.poke(0.U)
      dut.io.insertEn.poke(true.B)
      dut.io.insertSig.poke("h5a".U)
      dut.io.insertDepth.poke("h1234".U)
      dut.io.insertPixelX.poke(3.U)
      dut.io.insertPixelY.poke(7.U)
      dut.clock.step()
      dut.io.insertEn.poke(false.B)
      dut.io.occupancy.expect(1.U)

      dut.io.lookupSig.poke("h5a".U)
      dut.io.lookupEn.poke(true.B)
      dut.clock.step()
      dut.io.lookupEn.poke(false.B)
      dut.io.hit.expect(true.B)
      dut.io.hitDepth.expect("h1234".U)
      dut.io.hitPixelX.expect(3.U)
      dut.io.hitPixelY.expect(7.U)
    }
  }

  behavior of "SAESController"

  it should "select L0 when feature variance is below threshold" in {
    test(new SAESController) { dut =>
      dut.io.config.numDepthCandidates.poke(64.U)
      dut.io.config.featureDim.poke(128.U)
      dut.io.config.imageH.poke(16.U)
      dut.io.config.imageW.poke(16.U)
      dut.io.config.tileSize.poke(4.U)
      dut.io.config.cnnLayers.poke(1.U)
      dut.io.config.transformerLayers.poke(1.U)
      dut.io.config.normGroups.poke(1.U)
      dut.io.config.shDegree.poke(2.U)
      dut.io.config.hasDINOv2.poke(false.B)
      dut.io.config.saesFeatureVarThresh.poke(10.U)
      dut.io.config.saesCrossCheckThresh.poke(10.U)
      dut.io.config.saesDepthStdThresh.poke(10.U)
      dut.io.config.saesEnabled.poke(true.B)
      dut.io.config.fsdrEnabled.poke(true.B)
      dut.io.config.fsdrCacheSize.poke(4.U)
      dut.io.config.fsdrHammingThresh.poke(1.U)
      dut.io.config.fsdrDepthValidThresh.poke(102.U)
      dut.io.probeFeatureVar.poke(5.U)
      dut.io.probeDepthStd.poke(20.U)
      dut.io.crossCheckError.poke(20.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.clock.step()
      dut.io.done.expect(true.B)
      dut.io.level.expect(SAESLevel.sL0)
      dut.io.decisionCycles.expect(2.U)
    }
  }

  it should "select L1 from depth without a Gaussian cross-check gate" in {
    test(new SAESController) { dut =>
      dut.io.config.numDepthCandidates.poke(64.U)
      dut.io.config.featureDim.poke(128.U)
      dut.io.config.imageH.poke(16.U)
      dut.io.config.imageW.poke(16.U)
      dut.io.config.tileSize.poke(4.U)
      dut.io.config.cnnLayers.poke(1.U)
      dut.io.config.transformerLayers.poke(1.U)
      dut.io.config.normGroups.poke(1.U)
      dut.io.config.shDegree.poke(2.U)
      dut.io.config.hasDINOv2.poke(false.B)
      dut.io.config.saesFeatureVarThresh.poke(10.U)
      dut.io.config.saesCrossCheckThresh.poke(0.U)
      dut.io.config.saesDepthStdThresh.poke(10.U)
      dut.io.config.saesEnabled.poke(true.B)
      dut.io.config.fsdrEnabled.poke(true.B)
      dut.io.config.fsdrCacheSize.poke(4.U)
      dut.io.config.fsdrHammingThresh.poke(1.U)
      dut.io.config.fsdrDepthValidThresh.poke(102.U)
      dut.io.probeFeatureVar.poke(20.U)
      dut.io.probeDepthStd.poke(5.U)
      dut.io.crossCheckError.poke(65535.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.clock.step(2)
      dut.io.done.expect(true.B)
      dut.io.level.expect(SAESLevel.sL1)
      dut.io.decisionCycles.expect(3.U)
    }
  }

  it should "report the same three-cycle first-hit latency for Full" in {
    test(new SAESController) { dut =>
      dut.io.config.numDepthCandidates.poke(64.U)
      dut.io.config.featureDim.poke(128.U)
      dut.io.config.imageH.poke(16.U)
      dut.io.config.imageW.poke(16.U)
      dut.io.config.tileSize.poke(4.U)
      dut.io.config.cnnLayers.poke(1.U)
      dut.io.config.transformerLayers.poke(1.U)
      dut.io.config.normGroups.poke(1.U)
      dut.io.config.shDegree.poke(2.U)
      dut.io.config.hasDINOv2.poke(false.B)
      dut.io.config.saesFeatureVarThresh.poke(10.U)
      dut.io.config.saesCrossCheckThresh.poke(0.U)
      dut.io.config.saesDepthStdThresh.poke(10.U)
      dut.io.config.saesEnabled.poke(true.B)
      dut.io.config.fsdrEnabled.poke(true.B)
      dut.io.config.fsdrCacheSize.poke(4.U)
      dut.io.config.fsdrHammingThresh.poke(1.U)
      dut.io.config.fsdrDepthValidThresh.poke(102.U)
      dut.io.probeFeatureVar.poke(20.U)
      dut.io.probeDepthStd.poke(20.U)
      dut.io.crossCheckError.poke(0.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.clock.step(2)
      dut.io.done.expect(true.B)
      dut.io.level.expect(SAESLevel.sFull)
      dut.io.decisionCycles.expect(3.U)
    }
  }
}
