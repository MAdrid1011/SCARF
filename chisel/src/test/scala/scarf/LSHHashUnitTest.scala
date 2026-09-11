package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.fsdr.LSHHashUnit

class LSHHashComparison extends Module {
  val io = IO(new Bundle {
    val featureIn = Input(Vec(128, UInt(16.W)))
    val start = Input(Bool())
    val serializedDone = Output(Bool())
    val serializedSignature = Output(UInt(16.W))
    val parallelDone = Output(Bool())
    val parallelSignature = Output(UInt(16.W))
  })

  val serialized = Module(new LSHHashUnit(lanes = 1))
  val parallel = Module(new LSHHashUnit(lanes = 4))
  serialized.io.start := io.start
  parallel.io.start := io.start
  serialized.io.featureIn := io.featureIn
  parallel.io.featureIn := io.featureIn
  io.serializedDone := serialized.io.done
  io.serializedSignature := serialized.io.signature
  io.parallelDone := parallel.io.done
  io.parallelSignature := parallel.io.signature
}

class LSHHashUnitTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "LSHHashUnit"

  private def hash(dut: LSHHashUnit, values: Seq[Int]): (BigInt, Int) = {
    values.zipWithIndex.foreach { case (value, index) =>
      dut.io.featureIn(index).poke(value.U)
    }
    dut.io.start.poke(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)
    var cycles = 1
    while (!dut.io.done.peek().litToBoolean) {
      dut.clock.step()
      cycles += 1
    }
    val signature = dut.io.signature.peek().litValue
    dut.clock.step()
    (signature, cycles)
  }

  it should "match seed-42 software signatures for exact FP16 basis vectors" in {
    test(new LSHHashUnit()) { dut =>
      val zero = Seq.fill(128)(0x0000)
      val positiveE0 = zero.updated(0, 0x3c00)
      val negativeE1 = zero.updated(1, 0xbc00)

      assert(hash(dut, zero)._1 == BigInt("ffff", 16))
      assert(hash(dut, positiveE0)._1 == BigInt("e84b", 16))
      assert(hash(dut, negativeE1)._1 == BigInt("3ee8", 16))
    }
  }

  it should "preserve every signature bit while reducing projection latency" in {
    val values = Seq.tabulate(128)(index => if (index % 3 == 0) 0x3c00 else 0xbc00)
    test(new LSHHashComparison) { dut =>
      values.zipWithIndex.foreach { case (value, index) =>
        dut.io.featureIn(index).poke(value.U)
      }
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var cycles = 1
      var parallelDoneCycle = -1
      var serializedDoneCycle = -1
      var parallelSignature = BigInt(0)
      var serializedSignature = BigInt(0)
      while (serializedDoneCycle < 0 && cycles < 140) {
        if (dut.io.parallelDone.peekBoolean()) {
          parallelDoneCycle = cycles
          parallelSignature = dut.io.parallelSignature.peek().litValue
        }
        if (dut.io.serializedDone.peekBoolean()) {
          serializedDoneCycle = cycles
          serializedSignature = dut.io.serializedSignature.peek().litValue
        }
        dut.clock.step()
        cycles += 1
      }
      assert(serializedDoneCycle == 130)
      assert(parallelDoneCycle == 34)
      assert(parallelSignature == serializedSignature)
    }
  }
}
