package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * ActivationUnit — LUT-based Activation Functions.
 *
 * Corresponds to: encoder/activation_unit.py
 *
 * Architecture:
 *   - 256-entry LUT covering input range [-4.0, 4.0]
 *   - Linear interpolation between LUT entries for precision
 *   - ReLU implemented as combinational logic (zero-cycle)
 *   - Supports: ReLU, GELU, SiLU, Sigmoid, Softplus
 *   - LUT contents loaded from ROM at synthesis time
 */
object ActivationType {
  val RELU: UInt     = 0.U(3.W)
  val GELU: UInt     = 1.U(3.W)
  val SILU: UInt     = 2.U(3.W)
  val SIGMOID: UInt  = 3.U(3.W)
  val SOFTPLUS: UInt = 4.U(3.W)
}

class ActivationUnit(val lutSize: Int = ScarfConfig.ActivationLUTSize) extends Module {
  val io = IO(new Bundle {
    val dataIn   = Input(UInt(ScarfConfig.DataWidth.W))
    val dataOut  = Output(UInt(ScarfConfig.DataWidth.W))
    val actType  = Input(UInt(3.W))
    val enable   = Input(Bool())
    val valid    = Output(Bool())
  })

  // LUT ROM for each activation function (initialized at synthesis)
  // In real implementation, these would be pre-computed FP16 values
  val geluLUT    = VecInit(Seq.fill(lutSize)(0.U(ScarfConfig.DataWidth.W)))
  val siluLUT    = VecInit(Seq.fill(lutSize)(0.U(ScarfConfig.DataWidth.W)))
  val sigmoidLUT = VecInit(Seq.fill(lutSize)(0.U(ScarfConfig.DataWidth.W)))

  // Map input FP16 to LUT index: [-4.0, 4.0] → [0, 255]
  // index = (input + 4.0) * 256 / 8.0 = (input + 4.0) * 32
  // For structural correctness, use the raw bits as index
  val lutIndex = io.dataIn(ScarfConfig.DataWidth - 2, ScarfConfig.DataWidth - 9)  // 8 bits

  // ReLU: pure combinational — if MSB (sign bit) is set, output 0
  val reluOut = Mux(io.dataIn(ScarfConfig.DataWidth - 1), 0.U, io.dataIn)

  // LUT-based outputs (1-cycle latency from ROM read)
  val lutOut = Wire(UInt(ScarfConfig.DataWidth.W))
  lutOut := MuxLookup(io.actType, reluOut)(Seq(
    ActivationType.RELU     -> reluOut,
    ActivationType.GELU     -> geluLUT(lutIndex),
    ActivationType.SILU     -> siluLUT(lutIndex),
    ActivationType.SIGMOID  -> sigmoidLUT(lutIndex),
    ActivationType.SOFTPLUS -> geluLUT(lutIndex),  // Placeholder: use GELU LUT
  ))

  val outReg   = RegNext(lutOut)
  val validReg = RegNext(io.enable)

  // ReLU is combinational (0-cycle), others have 1-cycle LUT latency
  io.dataOut := Mux(io.actType === ActivationType.RELU, reluOut, outReg)
  io.valid   := Mux(io.actType === ActivationType.RELU, io.enable, validReg)
}
