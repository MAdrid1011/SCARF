package scarf

import chisel3._
import chisel3.util._

// ============================================================
// Pipeline FSM States
// ============================================================

/** Top-level pipeline states. */
object PipeState extends ChiselEnum {
  val sIdle, sLoadConfig,
      sS1_CNN, sS1_Transformer, sS1_DINOv2,
      sS2S3_TileLoad, sS2S3_SAESClassify,
      sS2_CostVol, sS2_UNet, sS2_DepthHead, sS2_Regression,
      sS3_Refine, sS3_GaussHead,
      sS2S3_ProbeOnly,  // Lightweight probe path for SAES-skipped tiles
      sGGU,
      sS2S3_NextTile,
      sDone = Value
}

/** SAES tile classification result. */
object SAESLevel extends ChiselEnum {
  val sFull,   // No early-stop: run full S2+S3
      sL0,     // Feature-uniform: probe only (skip S2+S3)
      sL1,     // Depth-uniform: probe only (skip S2+S3)
      sL2      // Gaussian cross-check: skip S3 only
      = Value
}

// ============================================================
// Compute Unit Command & Status
// ============================================================

/** Unified command interface for all compute units. */
class ComputeCmd extends Bundle {
  val opcode    = UInt(4.W)    // Operation type (unit-specific)
  val inAddr    = UInt(ScarfConfig.AddrWidth.W)  // Input data address in SRAM
  val outAddr   = UInt(ScarfConfig.AddrWidth.W)  // Output data address in SRAM
  val weightAddr = UInt(ScarfConfig.AddrWidth.W) // Weight address (conv/gemm)
  val param0    = UInt(16.W)   // Generic parameter 0 (e.g., channels, kernel_size)
  val param1    = UInt(16.W)   // Generic parameter 1 (e.g., height, width)
  val param2    = UInt(16.W)   // Generic parameter 2 (e.g., stride, groups)
}

/** Compute unit status output. */
class UnitStatus extends Bundle {
  val busy   = Bool()
  val done   = Bool()
  val cycles = UInt(32.W)  // Cycle count for current operation
}

// ============================================================
// Data Packets
// ============================================================

/** Generic data packet for inter-unit transfers. */
class DataPacket(val dataWidth: Int = ScarfConfig.DataWidth) extends Bundle {
  val data  = UInt(dataWidth.W)
  val valid = Bool()
  val last  = Bool()         // Last element in current transfer
  val tag   = UInt(4.W)      // Source/destination tag for routing
}

/** Feature map metadata (accompanies feature data in FeatureBuffer). */
class FeatureMapMeta extends Bundle {
  val height   = UInt(10.W)
  val fmWidth  = UInt(10.W)   // "width" is reserved in Chisel Bundle
  val channels = UInt(8.W)
  val baseAddr = UInt(ScarfConfig.AddrWidth.W)
}

// ============================================================
// ConvEngine-Specific Types
// ============================================================

/** ConvEngine operation parameters. */
class ConvParams extends Bundle {
  val inChannels  = UInt(10.W)
  val outChannels = UInt(10.W)
  val kernelSize  = UInt(4.W)    // 1, 3, 5, 7, 9, or 14
  val stride      = UInt(3.W)    // 1 or 2
  val padding     = UInt(4.W)
  val dilation    = UInt(3.W)    // 1 or 2
  val groups      = UInt(10.W)   // Grouped convolution
  val useBias     = Bool()
  val fuseReLU    = Bool()       // Conv-BN-ReLU fusion
  val fuseBN      = Bool()       // Fused batch normalization
}

// ============================================================
// GEMM-Specific Types
// ============================================================

/** GEMM operation parameters. */
class GEMMParams extends Bundle {
  val M = UInt(16.W)   // Rows of A / output
  val K = UInt(16.W)   // Columns of A / rows of B
  val N = UInt(16.W)   // Columns of B / output
  val useBias = Bool()
  val accumulate = Bool()  // Accumulate into existing output
}

// ============================================================
// BilinearUnit-Specific Types
// ============================================================

/** Bilinear sampling coordinate (fixed-point). */
class SampleCoord extends Bundle {
  val x = SInt((8 + ScarfConfig.CoordFracBits).W)  // Integer + fractional
  val y = SInt((8 + ScarfConfig.CoordFracBits).W)
}

/** Bilinear operation parameters. */
class BilinearParams extends Bundle {
  val inH       = UInt(10.W)
  val inW       = UInt(10.W)
  val outH      = UInt(10.W)
  val outW      = UInt(10.W)
  val channels  = UInt(10.W)
  val mode      = UInt(2.W)   // 0=nearest, 1=bilinear, 2=bicubic
}

// ============================================================
// GGU-Specific Types
// ============================================================

/** Raw Gaussian parameters (output from S3 neural network). */
class RawGaussian extends Bundle {
  val depth    = UInt(ScarfConfig.DataWidth.W)  // FP16 depth value
  val scales   = Vec(3, UInt(ScarfConfig.DataWidth.W))  // 3D scale (FP16)
  val rotation = Vec(4, UInt(ScarfConfig.DataWidth.W))  // Quaternion (FP16)
  val sh       = Vec(75, UInt(ScarfConfig.DataWidth.W)) // SH coefficients: 25 × 3ch (FP16)
  val opacity  = UInt(ScarfConfig.DataWidth.W)  // Raw opacity (FP16, pre-sigmoid)
}

/** Complete 3D Gaussian output. */
class GaussianOutput extends Bundle {
  val position   = Vec(3, UInt(ScarfConfig.AccWidth.W))   // World position (FP32)
  val covariance = Vec(6, UInt(ScarfConfig.AccWidth.W))   // Upper triangle of 3×3 cov (FP32)
  val harmonics  = Vec(75, UInt(ScarfConfig.DataWidth.W)) // Rotated SH (FP16)
  val opacity    = UInt(ScarfConfig.DataWidth.W)           // Sigmoid-activated (FP16)
}

/** Camera intrinsics and extrinsics for GGU position calculation. */
class CameraParams extends Bundle {
  val intrinsics = Vec(4, UInt(ScarfConfig.AccWidth.W))  // fx, fy, cx, cy (FP32)
  val extrinsics = Vec(12, UInt(ScarfConfig.AccWidth.W)) // 3×4 [R|t] matrix (FP32)
}

// ============================================================
// SAES Types
// ============================================================

/** SAES tile classification input (4 corner probe features). */
class SAESTileInput extends Bundle {
  val probeFeatures = Vec(4, Vec(ScarfConfig.MaxFeatureDim, UInt(ScarfConfig.DataWidth.W)))
  val probeDepths   = Vec(4, UInt(ScarfConfig.DataWidth.W))  // Only after S2 probe
  val tileRow       = UInt(8.W)
  val tileCol       = UInt(8.W)
}

/** SAES classification output. */
class SAESResult extends Bundle {
  val level         = SAESLevel()
  val crossCheckErr = UInt(ScarfConfig.DataWidth.W)  // Max leave-one-out error (FP16)
}

// ============================================================
// AXI4-Lite Interface (Simplified for ConfigRegs)
// ============================================================

class AXI4LiteWriteAddr extends Bundle {
  val addr = UInt(ScarfConfig.AddrWidth.W)
  val prot = UInt(3.W)
}

class AXI4LiteWriteData extends Bundle {
  val data = UInt(32.W)
  val strb = UInt(4.W)
}

class AXI4LiteReadAddr extends Bundle {
  val addr = UInt(ScarfConfig.AddrWidth.W)
  val prot = UInt(3.W)
}

class AXI4LiteReadData extends Bundle {
  val data = UInt(32.W)
  val resp = UInt(2.W)
}
