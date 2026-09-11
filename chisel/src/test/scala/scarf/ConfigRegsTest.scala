package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.control.ConfigRegs

class ConfigRegsTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "ConfigRegs"

  it should "retain DMA role offsets, byte counts, SAES routes, and DINOv2 layer count" in {
    test(new ConfigRegs) { dut =>
      dut.io.writeEn.poke(false.B)
      dut.io.writeAddr.poke(0.U)
      dut.io.writeData.poke(0.U)
      dut.io.readAddr.poke(0.U)
      dut.clock.step()

      def write(address: Int, value: BigInt): Unit = {
        dut.io.writeAddr.poke(address.U)
        dut.io.writeData.poke(value.U)
        dut.io.writeEn.poke(true.B)
        dut.clock.step()
      }

      write(0x64, 0x1000)
      write(0x68, 0x2000)
      write(0x70, 0x3000)
      write(0x6c, 0x4000)
      write(0x74, 0x100)
      write(0x78, 0x10)
      write(0x7c, 0x10)
      write(0x80, 0x10)
      write(0x84, 12)
      write(0x88, 0x5000)
      write(0x8c, 0x200)
      dut.io.writeEn.poke(false.B)

      dut.io.config.payloadFeatureOffset.expect(0x1000.U)
      dut.io.config.payloadDepthOffset.expect(0x2000.U)
      dut.io.config.payloadCandidateOffset.expect(0x3000.U)
      dut.io.config.payloadProbabilityOffset.expect(0x4000.U)
      dut.io.config.payloadFeatureBytes.expect(0x100.U)
      dut.io.config.payloadDepthBytes.expect(0x10.U)
      dut.io.config.payloadCandidateBytes.expect(0x10.U)
      dut.io.config.payloadProbabilityBytes.expect(0x10.U)
      dut.io.config.dinov2Layers.expect(12.U)
      dut.io.config.payloadSAESRouteOffset.expect(0x5000.U)
      dut.io.config.payloadSAESRouteBytes.expect(0x200.U)

      dut.io.readAddr.poke(0x84.U)
      dut.clock.step()
      dut.io.readData.expect(12.U)
      dut.io.readAddr.poke(0x88.U)
      dut.clock.step()
      dut.io.readData.expect(0x5000.U)
    }
  }
}
