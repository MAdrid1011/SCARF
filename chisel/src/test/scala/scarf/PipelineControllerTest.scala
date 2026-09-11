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
    dut.io.config.dinov2Layers.poke(0.U)
    dut.io.config.saesFeatureVarThresh.poke(10.U)
    dut.io.config.saesCrossCheckThresh.poke(0.U)
    dut.io.config.saesDepthStdThresh.poke(10.U)
    dut.io.config.saesEnabled.poke(true.B)
    dut.io.config.fsdrEnabled.poke(false.B)
    dut.io.config.fsdrCacheSize.poke(32.U)
    dut.io.config.fsdrHammingThresh.poke(3.U)
    dut.io.config.fsdrDepthValidThresh.poke(102.U)
    dut.io.config.payloadBase.poke(0.U)
    dut.io.config.payloadBytes.poke(0.U)
    dut.io.config.payloadTensorCount.poke(0.U)
    dut.io.config.numGaussians.poke(0.U)
    dut.io.config.payloadFeatureOffset.poke(0.U)
    dut.io.config.payloadDepthOffset.poke(0.U)
    dut.io.config.payloadCandidateOffset.poke(0.U)
    dut.io.config.payloadProbabilityOffset.poke(0.U)
    dut.io.config.payloadFeatureBytes.poke(0.U)
    dut.io.config.payloadDepthBytes.poke(0.U)
    dut.io.config.payloadCandidateBytes.poke(0.U)
    dut.io.config.payloadProbabilityBytes.poke(0.U)
    dut.io.config.payloadSAESRouteOffset.poke(0.U)
    dut.io.config.payloadSAESRouteBytes.poke(0.U)
    dut.io.config.payloadValid.poke(false.B)
    dut.io.configValid.poke(true.B)

    dut.io.mmcuDone.poke(true.B)
    dut.io.bilinearDone.poke(true.B)
    dut.io.gguDone.poke(true.B)
    dut.io.tilePayloadReady.poke(true.B)
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
    // Each MMCU state now explicitly launches and then observes its own
    // completion, so S1 needs three cycles per configured command.
    dut.clock.step(8)
    dut.io.state.expect(PipeState.sS2S3_TileLoad)
    // A TileLoad entry cannot consume the predecessor's retained ready level.
    dut.clock.step()
    dut.io.state.expect(PipeState.sS2S3_SAESClassify)
  }

  for (level <- Seq(SAESLevel.sL0, SAESLevel.sL1)) {
    it should s"traverse the ordered S2/S3 state sequence for $level tiles" in {
      test(new PipelineController) { dut =>
        driveConfig(dut)
        reachSAESDecision(dut)

        dut.io.saesLevel.poke(level)
        dut.io.saesClassifyDone.poke(true.B)
        dut.clock.step()
        dut.io.saesClassifyDone.poke(false.B)
        dut.io.state.expect(PipeState.sS2_CostVol)

        dut.clock.step(3)
        dut.io.state.expect(PipeState.sS2_UNet)
        dut.clock.step(3)
        dut.io.state.expect(PipeState.sS2_DepthHead)
        dut.clock.step(3)
        dut.io.state.expect(PipeState.sS2_Regression)
        dut.clock.step(3)
        dut.io.state.expect(PipeState.sS3_Refine)
        dut.clock.step(3)
        dut.io.state.expect(PipeState.sS3_GaussHead)
        dut.clock.step(3)
        dut.io.state.expect(PipeState.sGGU)
      }
    }
  }

  it should "expose the accepted representative descriptor count" in {
    test(new PipelineController) { dut =>
      driveConfig(dut)
      reachSAESDecision(dut)
      dut.io.saesLevel.poke(SAESLevel.sL0)
      dut.io.saesClassifyDone.poke(true.B)
      dut.clock.step()
      dut.io.saesRetainedDescriptors.expect(4.U)
    }
  }

  it should "omit one terminal CostVol group for each regular SAES route" in {
    test(new PipelineController) { dut =>
      driveConfig(dut)
      dut.io.config.numDepthCandidates.poke(128.U)
      reachSAESDecision(dut)
      dut.io.saesLevel.poke(SAESLevel.sL0)
      dut.io.saesClassifyDone.poke(true.B)
      dut.clock.step()
      dut.io.costVolCandidates.expect(96.U)

      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      driveConfig(dut)
      dut.io.config.numDepthCandidates.poke(128.U)
      reachSAESDecision(dut)
      dut.io.saesLevel.poke(SAESLevel.sL1)
      dut.io.saesClassifyDone.poke(true.B)
      dut.clock.step()
      dut.io.costVolCandidates.expect(96.U)
    }
  }

  it should "leave combined L1 candidate work under FSDR ownership" in {
    test(new PipelineController) { dut =>
      driveConfig(dut)
      dut.io.config.numDepthCandidates.poke(128.U)
      dut.io.config.fsdrEnabled.poke(true.B)
      reachSAESDecision(dut)
      dut.io.saesLevel.poke(SAESLevel.sL1)
      dut.io.saesClassifyDone.poke(true.B)
      dut.clock.step()
      dut.io.state.expect(PipeState.sS2_FSDRLookup)
      dut.io.fsdrUseNarrow.poke(false.B)
      dut.io.fsdrDone.poke(true.B)
      dut.clock.step()
      dut.io.costVolCandidates.expect(128.U)
    }
  }

  it should "retain independent CostVol completion pulses" in {
    test(new PipelineController) { dut =>
      driveConfig(dut)
      reachSAESDecision(dut)

      dut.io.mmcuDone.poke(false.B)
      dut.io.bilinearDone.poke(false.B)
      dut.io.saesClassifyDone.poke(true.B)
      dut.clock.step()
      dut.io.saesClassifyDone.poke(false.B)
      dut.io.state.expect(PipeState.sS2_CostVol)

      // CostVol first issues both commands. A completion on its state-entry
      // cycle is not attributable to that newly launched pass.
      dut.clock.step()
      // Bilinear completes first and its pulse is withdrawn before MMCU.
      dut.io.bilinearDone.poke(true.B)
      dut.clock.step()
      dut.io.bilinearDone.poke(false.B)
      dut.io.state.expect(PipeState.sS2_CostVol)

      dut.io.mmcuDone.poke(true.B)
      dut.clock.step(3)
      dut.io.mmcuDone.poke(false.B)
      dut.io.state.expect(PipeState.sS2_UNet)
    }
  }
}
