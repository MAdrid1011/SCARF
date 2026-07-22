package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.control.PipelineController

class PipelineControllerTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "PipelineController"

  private def driveConfig(dut: PipelineController): Unit = {
    dut.io.config.numDepthCandidates.poke(4.U)
    dut.io.config.featureDim.poke(128.U)
    dut.io.config.imageH.poke(4.U)
    dut.io.config.imageW.poke(4.U)
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
    dut.io.config.fsdrEnabled.poke(false.B)
    dut.io.config.fsdrCacheSize.poke(32.U)
    dut.io.config.fsdrHammingThresh.poke(3.U)
    dut.io.config.fsdrDepthValidThresh.poke(102.U)
    dut.io.configValid.poke(true.B)

    dut.io.mmcuDone.poke(true.B)
    dut.io.bilinearDone.poke(true.B)
    dut.io.gguDone.poke(true.B)
    dut.io.saesClassifyDone.poke(false.B)
    dut.io.saesLevel.poke(SAESLevel.sFull)
    dut.io.fsdrDone.poke(true.B)
    dut.io.fsdrUseNarrow.poke(false.B)
    dut.io.fsdrNarrowCands.poke(1.U)
  }

  private def reachSAESDecision(dut: PipelineController): Unit = {
    dut.io.start.poke(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)
    dut.clock.step(4)
    dut.io.state.expect(PipeState.sS2S3_SAESClassify)
  }

  for (level <- Seq(SAESLevel.sL0, SAESLevel.sL1)) {
    it should s"keep $level tiles on the complete S2/S3 path until SAES merge RTL exists" in {
      test(new PipelineController) { dut =>
        driveConfig(dut)
        reachSAESDecision(dut)

        dut.io.saesLevel.poke(level)
        dut.io.saesClassifyDone.poke(true.B)
        dut.clock.step()
        dut.io.saesClassifyDone.poke(false.B)
        dut.io.state.expect(PipeState.sS2_CostVol)

        dut.clock.step()
        dut.io.state.expect(PipeState.sS2_UNet)
        dut.clock.step()
        dut.io.state.expect(PipeState.sS2_DepthHead)
        dut.clock.step()
        dut.io.state.expect(PipeState.sS2_Regression)
        dut.clock.step()
        dut.io.state.expect(PipeState.sS3_Refine)
        dut.clock.step()
        dut.io.state.expect(PipeState.sS3_GaussHead)
        dut.clock.step()
        dut.io.state.expect(PipeState.sGGU)
      }
    }
  }
}
