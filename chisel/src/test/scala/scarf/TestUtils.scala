package scarf

import chisel3._
import chiseltest._

/** Shared test utilities for SCARF Chisel tests. */
object TestUtils {

  /** Default tolerance for floating-point comparisons (relative). */
  val DefaultRelTol: Double = 0.02 // 2% relative tolerance

  /** Default tolerance for floating-point comparisons (absolute). */
  val DefaultAbsTol: Double = 1e-4

  /**
   * Check if two values are approximately equal within relative tolerance.
   *
   * @param actual   the computed value
   * @param expected the golden reference value
   * @param relTol   relative tolerance (default 2%)
   * @param absTol   absolute tolerance for near-zero values
   * @return true if values match within tolerance
   */
  def approxEqual(
    actual: Double,
    expected: Double,
    relTol: Double = DefaultRelTol,
    absTol: Double = DefaultAbsTol,
  ): Boolean = {
    val diff = math.abs(actual - expected)
    if (math.abs(expected) < absTol) {
      diff < absTol
    } else {
      diff / math.abs(expected) < relTol
    }
  }

  /**
   * Assert approximate equality with a descriptive error message.
   */
  def assertApprox(
    actual: Double,
    expected: Double,
    label: String = "",
    relTol: Double = DefaultRelTol,
    absTol: Double = DefaultAbsTol,
  ): Unit = {
    assert(
      approxEqual(actual, expected, relTol, absTol),
      s"${if (label.nonEmpty) s"[$label] " else ""}Expected $expected, got $actual " +
        s"(diff=${math.abs(actual - expected)}, relTol=$relTol, absTol=$absTol)",
    )
  }

  /**
   * Convert a Float to its UInt bit representation (IEEE 754 FP32).
   */
  def floatToUInt(f: Float): BigInt = {
    BigInt(java.lang.Float.floatToRawIntBits(f)) & 0xFFFFFFFFL
  }

  /**
   * Convert a UInt bit representation back to Float.
   */
  def uintToFloat(u: BigInt): Float = {
    java.lang.Float.intBitsToFloat(u.toInt)
  }

  /**
   * Convert a Double to FP16 half-precision representation (16-bit UInt).
   * Simplified: truncates mantissa, does not handle denormals.
   */
  def doubleToFP16(d: Double): Int = {
    val f = d.toFloat
    val bits = java.lang.Float.floatToRawIntBits(f)
    val sign = (bits >>> 31) & 0x1
    val exp  = (bits >>> 23) & 0xFF
    val man  = bits & 0x7FFFFF

    if (exp == 0) {
      // Zero or denormal → FP16 zero
      (sign << 15).toInt
    } else if (exp == 0xFF) {
      // Inf or NaN
      ((sign << 15) | 0x7C00 | (if (man != 0) 0x0200 else 0)).toInt
    } else {
      val newExp = exp - 127 + 15
      if (newExp >= 31) {
        // Overflow → Inf
        ((sign << 15) | 0x7C00).toInt
      } else if (newExp <= 0) {
        // Underflow → zero
        (sign << 15).toInt
      } else {
        val newMan = man >>> 13 // Truncate from 23-bit to 10-bit
        ((sign << 15) | (newExp << 10) | newMan).toInt
      }
    }
  }

  /**
   * Convert FP16 (16-bit Int) back to Double.
   */
  def fp16ToDouble(h: Int): Double = {
    val sign = (h >>> 15) & 0x1
    val exp  = (h >>> 10) & 0x1F
    val man  = h & 0x3FF

    val value: Double = if (exp == 0) {
      // Zero or denormal
      math.pow(2, -14) * (man.toDouble / 1024.0)
    } else if (exp == 31) {
      // Inf or NaN
      if (man == 0) Double.PositiveInfinity else Double.NaN
    } else {
      math.pow(2, exp - 15) * (1.0 + man.toDouble / 1024.0)
    }

    if (sign == 1) -value else value
  }

  /** SCARF hardware parameters — must match encoder/types.py */
  object HWParams {
    val PEArraySize: Int     = 48   // ConvEngine & GEMM systolic array
    val BilinearChannels: Int = 32  // BilinearUnit parallel channels
    val VectorALUWidth: Int  = 64   // VectorALU SIMD width
    val GGUPECount: Int      = 32   // GGU processing elements
    val WeightBufferKB: Int  = 128  // ConvEngine weight buffer
    val FeatureBufferKB: Int = 256  // Feature SRAM
    val TileSPMKB: Int       = 64   // Tile scratchpad
    val GEMMBufferKB: Int    = 64   // GEMM A/B buffer
    val TileSize: Int        = 4    // SAES tile size
    val ActivationLUTSize: Int = 256 // Activation LUT entries
  }
}
