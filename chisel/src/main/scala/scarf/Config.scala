package scarf

import chisel3._
import chisel3.util._

object ScarfConfig {
  // ---- MMCU (Multi-Mode Compute Unit) ----
  val PEArraySize: Int = 48      // 48×48 output-stationary array = 2304 MACs/cycle

  // ---- Data Widths ----
  val DataWidth: Int   = 16      // FP16 data path
  val AccWidth: Int    = 32      // FP32 accumulation
  val AddrWidth: Int   = 32
  val CoordFracBits: Int = 8     // BilinearUnit fixed-point precision

  // ---- Compute Unit Sizes ----
  val BilinearChannels: Int = 32   // Parallel bilinear samplers
  val VectorALUWidth: Int   = 64   // Total SIMD width
  val VectorALULanes: Int   = 8    // Number of SIMD lanes (8 elements/lane)
  val GGUPECount: Int       = 32   // GGU processing elements
  val ActivationLUTSize: Int = 256

  // ---- FSDR Cache ----
  val FSDRCacheEntries: Int = 32
  val LSHDim: Int           = 16   // Hash signature bits
  val LSHFeatureDim: Int    = 128  // Input feature dimension for LSH

  // ---- Memory Sizes (bytes) ----
  val WeightBufferBytes: Int    = 128 * 1024  // 128 KB
  val FeatureBufferBytes: Int   = 256 * 1024  // 256 KB
  val TileBufferBytes: Int      = 64 * 1024   // 64 KB

  // ---- Pipeline Parameters ----
  val TileSize: Int = 4
  val MaxDepthCandidates: Int = 128
  val MaxFeatureDim: Int = 256
  val MaxImageDim: Int = 512

  // ---- Supported Kernel Sizes ----
  val SupportedKernels: Seq[Int] = Seq(1, 3, 5, 7, 9, 14)
  val MaxKernelSize: Int = 14

  // ---- AXI4 Interface ----
  val AXIDataWidth: Int = 128
  val AXIIDWidth: Int   = 4
}

class ModelConfig extends Bundle {
  val numDepthCandidates = UInt(8.W)
  val featureDim         = UInt(9.W)     // Up to 256
  val imageH             = UInt(10.W)
  val imageW             = UInt(10.W)
  val tileSize           = UInt(4.W)
  val cnnLayers          = UInt(8.W)
  val transformerLayers  = UInt(4.W)
  val normGroups         = UInt(4.W)
  val shDegree           = UInt(3.W)
  val hasDINOv2          = Bool()

  val saesFeatureVarThresh = UInt(16.W)
  val saesCrossCheckThresh = UInt(16.W)
  val saesDepthStdThresh   = UInt(16.W)
  val saesEnabled          = Bool()

  val fsdrEnabled       = Bool()
  val fsdrCacheSize     = UInt(10.W)
  val fsdrHammingThresh = UInt(4.W)
  val fsdrDepthValidThresh = UInt(10.W) // Q0.10 relative depth tolerance
}

object ModelPresets {
  def transplat: Map[String, BigInt] = Map(
    "numDepthCandidates" -> 64,
    "featureDim"         -> 256,
    "imageH"             -> 256,
    "imageW"             -> 256,
    "tileSize"           -> 4,
    "cnnLayers"          -> 6,
    "transformerLayers"  -> 6,
    "normGroups"         -> 8,
    "shDegree"           -> 4,
    "hasDINOv2"          -> 0,
    "saesEnabled"        -> 1,
    "fsdrEnabled"        -> 1,
    "fsdrCacheSize"      -> 32,
    "fsdrHammingThresh"  -> 3,
    "fsdrDepthValidThresh" -> 102,
  )

  def mvsplat: Map[String, BigInt] = Map(
    "numDepthCandidates" -> 64,
    "featureDim"         -> 256,
    "imageH"             -> 256,
    "imageW"             -> 256,
    "tileSize"           -> 4,
    "cnnLayers"          -> 6,
    "transformerLayers"  -> 6,
    "normGroups"         -> 8,
    "shDegree"           -> 4,
    "hasDINOv2"          -> 0,
    "saesEnabled"        -> 1,
    "fsdrEnabled"        -> 1,
    "fsdrCacheSize"      -> 32,
    "fsdrHammingThresh"  -> 3,
    "fsdrDepthValidThresh" -> 102,
  )

  def depthsplat: Map[String, BigInt] = Map(
    "numDepthCandidates" -> 96,
    "featureDim"         -> 256,
    "imageH"             -> 384,
    "imageW"             -> 512,
    "tileSize"           -> 4,
    "cnnLayers"          -> 6,
    "transformerLayers"  -> 8,
    "normGroups"         -> 4,
    "shDegree"           -> 2,
    "hasDINOv2"          -> 1,
    "saesEnabled"        -> 1,
    "fsdrEnabled"        -> 1,
    "fsdrCacheSize"      -> 32,
    "fsdrHammingThresh"  -> 3,
    "fsdrDepthValidThresh" -> 102,
  )
}
