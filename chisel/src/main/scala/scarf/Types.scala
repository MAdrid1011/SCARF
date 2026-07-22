package scarf

import chisel3._
import chisel3.util._

// ============================================================
// Pipeline FSM States
// ============================================================

object PipeState extends ChiselEnum {
  val sIdle, sLoadConfig,
      sS1_CNN, sS1_Transformer, sS1_DINOv2,
      sS2S3_TileLoad, sS2S3_SAESClassify,
      sS2_FSDRLookup,
      sS2_CostVol, sS2_UNet, sS2_DepthHead, sS2_Regression,
      sS3_Refine, sS3_GaussHead,
      sGGU,
      sS2S3_NextTile,
      sDone = Value
}

object SAESLevel extends ChiselEnum {
  val sFull, sL0, sL1, sL2 = Value
}

// ============================================================
// MMCU Modes
// ============================================================

object MMCUMode extends ChiselEnum {
  val mConv, mGEMM, mAttn = Value
}

// ============================================================
// Compute Unit Command & Status
// ============================================================

class ComputeCmd extends Bundle {
  val opcode    = UInt(4.W)
  val inAddr    = UInt(ScarfConfig.AddrWidth.W)
  val outAddr   = UInt(ScarfConfig.AddrWidth.W)
  val weightAddr = UInt(ScarfConfig.AddrWidth.W)
  val param0    = UInt(16.W)
  val param1    = UInt(16.W)
  val param2    = UInt(16.W)
}

class UnitStatus extends Bundle {
  val busy   = Bool()
  val done   = Bool()
  val cycles = UInt(32.W)
}

// ============================================================
// Data Packets
// ============================================================

class DataPacket(val dataWidth: Int = ScarfConfig.DataWidth) extends Bundle {
  val data  = UInt(dataWidth.W)
  val valid = Bool()
  val last  = Bool()
  val tag   = UInt(4.W)
}

class FeatureMapMeta extends Bundle {
  val height   = UInt(10.W)
  val fmWidth  = UInt(10.W)
  val channels = UInt(9.W)
  val baseAddr = UInt(ScarfConfig.AddrWidth.W)
}

// ============================================================
// MMCU Parameters
// ============================================================

class MMCUParams extends Bundle {
  val mode        = MMCUMode()
  val M           = UInt(16.W)
  val K           = UInt(16.W)
  val N           = UInt(16.W)
  val kernelSize  = UInt(4.W)
  val stride      = UInt(3.W)
  val padding     = UInt(4.W)
  val inChannels  = UInt(10.W)
  val outChannels = UInt(10.W)
  val useBias     = Bool()
}

// ============================================================
// BilinearUnit Types
// ============================================================

class SampleCoord extends Bundle {
  val x = SInt((8 + ScarfConfig.CoordFracBits).W)
  val y = SInt((8 + ScarfConfig.CoordFracBits).W)
}

class BilinearParams extends Bundle {
  val inH       = UInt(10.W)
  val inW       = UInt(10.W)
  val outH      = UInt(10.W)
  val outW      = UInt(10.W)
  val channels  = UInt(10.W)
  val mode      = UInt(2.W)
}

// ============================================================
// GGU Types
// ============================================================

class RawGaussian extends Bundle {
  val depth    = UInt(ScarfConfig.DataWidth.W)
  val scales   = Vec(3, UInt(ScarfConfig.DataWidth.W))
  val rotation = Vec(4, UInt(ScarfConfig.DataWidth.W))
  val sh       = Vec(75, UInt(ScarfConfig.DataWidth.W))
  val opacity  = UInt(ScarfConfig.DataWidth.W)
}

class GaussianOutput extends Bundle {
  val position   = Vec(3, UInt(ScarfConfig.AccWidth.W))
  val covariance = Vec(6, UInt(ScarfConfig.AccWidth.W))
  val harmonics  = Vec(75, UInt(ScarfConfig.DataWidth.W))
  val opacity    = UInt(ScarfConfig.DataWidth.W)
}

class CameraParams extends Bundle {
  val intrinsics = Vec(4, UInt(ScarfConfig.AccWidth.W))
  val extrinsics = Vec(12, UInt(ScarfConfig.AccWidth.W))
}

// ============================================================
// SAES Types
// ============================================================

class SAESTileInput extends Bundle {
  val probeFeatures = Vec(4, Vec(ScarfConfig.MaxFeatureDim, UInt(ScarfConfig.DataWidth.W)))
  val probeDepths   = Vec(4, UInt(ScarfConfig.DataWidth.W))
  val tileRow       = UInt(8.W)
  val tileCol       = UInt(8.W)
}

class SAESResult extends Bundle {
  val level         = SAESLevel()
  val crossCheckErr = UInt(ScarfConfig.DataWidth.W)
}

// ============================================================
// AXI4-Lite Interface
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
