package scarf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scarf.fsdr.LSHHashUnit

class LSHHashUnitTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "LSHHashUnit"

  private def hash(dut: LSHHashUnit, values: Seq[Int]): BigInt = {
    values.zipWithIndex.foreach { case (value, index) =>
      dut.io.featureIn(index).poke(value.U)
    }
    dut.io.start.poke(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)
    while (!dut.io.done.peek().litToBoolean) {
      dut.clock.step()
    }
    val signature = dut.io.signature.peek().litValue
    dut.clock.step()
    signature
  }

  it should "match seed-42 software signatures for exact FP16 basis vectors" in {
    test(new LSHHashUnit()) { dut =>
      val zero = Seq.fill(128)(0x0000)
      val positiveE0 = zero.updated(0, 0x3c00)
      val negativeE1 = zero.updated(1, 0xbc00)

      assert(hash(dut, zero) == BigInt("ffff", 16))
      assert(hash(dut, positiveE0) == BigInt("e84b", 16))
      assert(hash(dut, negativeE1) == BigInt("3ee8", 16))
    }
  }
}
