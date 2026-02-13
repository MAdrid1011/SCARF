package scarf

import chisel3._
import chisel3.util._

/**
 * SCARF global hardware configuration.
 *
 * These parameters match encoder/types.py (ConvConfig, GEMMConfig, BilinearConfig)
 * and scripts/demo.py (SCARFConfig).
 */
object ScarfConfig {
  // ---- Systolic Array Dimensions ----
  val PEArraySize: Int = 48     // ConvEngine & GEMM: 48×48 = 2304 MACs/cycle

  // ---- Data Widths ----
  val DataWidth: Int   = 16     // FP16 for data path
  val AccWidth: Int    = 32     // FP32 for accumulation
  val AddrWidth: Int   = 32     // Memory address width
  val CoordFracBits: Int = 8    // BilinearUnit coordinate precision

  // ---- Compute Unit Sizes ----
  val BilinearChannels: Int = 32  // Parallel bilinear samplers
  val VectorALUWidth: Int   = 64  // SIMD width
  val GGUPECount: Int       = 32  // GGU processing elements
  val ActivationLUTSize: Int = 256 // LUT entries for activation functions

  // ---- Memory Sizes (bytes) ----
  val WeightBufferBytes: Int  = 128 * 1024  // 128 KB
  val FeatureBufferBytes: Int = 256 * 1024  // 256 KB
  val TileSPMBytes: Int       = 64 * 1024   // 64 KB
  val GEMMBufferBytes: Int    = 64 * 1024   // 64 KB

  // ---- Pipeline Parameters ----
  val TileSize: Int = 4   // SAES tile size (4×4 = 16 pixels)
  val MaxDepthCandidates: Int = 128
  val MaxFeatureDim: Int = 128
  val MaxImageDim: Int = 512  // Maximum image width/height

  // ---- Supported Kernel Sizes ----
  val SupportedKernels: Seq[Int] = Seq(1, 3, 5, 7, 9, 14)
  val MaxKernelSize: Int = 14

  // ---- AXI4 Interface ----
  val AXIDataWidth: Int = 128  // 128-bit AXI data bus
  val AXIIDWidth: Int   = 4
}

/**
 * Model configuration bundle — written via MMIO ConfigRegs.
 *
 * This is the ONLY mechanism for model differentiation.
 * No if-else model branches exist in the data path.
 */
class ModelConfig extends Bundle {
  val numDepthCandidates = UInt(8.W)    // 32 / 64 / 128
  val featureDim         = UInt(8.W)    // 128
  val imageH             = UInt(10.W)   // 256
  val imageW             = UInt(10.W)   // 256
  val tileSize           = UInt(4.W)    // 4
  val cnnLayers          = UInt(8.W)    // Number of CNN layers in backbone
  val transformerLayers  = UInt(4.W)    // Number of transformer layers
  val normGroups         = UInt(4.W)    // GroupNorm groups (4 or 8)
  val shDegree           = UInt(3.W)    // SH degree (2 or 4)
  val hasDINOv2          = Bool()       // DepthSplat only

  // SAES thresholds (FP16 encoded)
  val saesFeatureVarThresh = UInt(16.W)
  val saesCrossCheckThresh = UInt(16.W)
  val saesDepthStdThresh   = UInt(16.W)
  val saesEnabled          = Bool()

  // FSGR parameters
  val fsgrEnabled       = Bool()
  val fsgrCacheSize     = UInt(10.W)   // Up to 1024 entries
  val fsgrHammingThresh = UInt(4.W)    // Hamming distance threshold
}

/**
 * Predefined model configurations.
 * These would be loaded into ConfigRegs at startup.
 */
object ModelPresets {
  def transplat: Map[String, BigInt] = Map(
    "numDepthCandidates" -> 128,
    "featureDim"         -> 128,
    "imageH"             -> 256,
    "imageW"             -> 256,
    "tileSize"           -> 4,
    "cnnLayers"          -> 6,
    "transformerLayers"  -> 6,
    "normGroups"         -> 8,
    "shDegree"           -> 4,
    "hasDINOv2"          -> 0,
    "saesEnabled"        -> 1,
    "fsgrEnabled"        -> 1,
    "fsgrCacheSize"      -> 512,
    "fsgrHammingThresh"  -> 4,
  )

  def mvsplat: Map[String, BigInt] = Map(
    "numDepthCandidates" -> 32,
    "featureDim"         -> 128,
    "imageH"             -> 256,
    "imageW"             -> 256,
    "tileSize"           -> 4,
    "cnnLayers"          -> 6,
    "transformerLayers"  -> 6,
    "normGroups"         -> 8,
    "shDegree"           -> 4,
    "hasDINOv2"          -> 0,
    "saesEnabled"        -> 1,
    "fsgrEnabled"        -> 1,
    "fsgrCacheSize"      -> 512,
    "fsgrHammingThresh"  -> 4,
  )

  def depthsplat: Map[String, BigInt] = Map(
    "numDepthCandidates" -> 128,
    "featureDim"         -> 128,
    "imageH"             -> 256,
    "imageW"             -> 256,
    "tileSize"           -> 4,
    "cnnLayers"          -> 6,
    "transformerLayers"  -> 6,
    "normGroups"         -> 4,  // DepthSplat uses GroupNorm(4)
    "shDegree"           -> 2,
    "hasDINOv2"          -> 1,  // DepthSplat has DINOv2
    "saesEnabled"        -> 1,
    "fsgrEnabled"        -> 1,
    "fsgrCacheSize"      -> 512,
    "fsgrHammingThresh"  -> 4,
  )
}
