package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * PoolingUnit — Max/Average Pooling.
 *
 * Corresponds to: encoder/pooling_unit.py
 *
 * Supports 2×2 and 3×3 pooling with stride 1 or 2.
 * Operates on streaming input with windowed accumulation.
 */
object PoolType {
  val MAX: UInt = 0.U(1.W)
  val AVG: UInt = 1.U(1.W)
}

class PoolingUnit extends Module {
  val io = IO(new Bundle {
    val dataIn   = Input(UInt(ScarfConfig.DataWidth.W))
    val dataOut  = Output(UInt(ScarfConfig.DataWidth.W))
    val poolType = Input(UInt(1.W))       // 0=max, 1=avg
    val poolSize = Input(UInt(2.W))       // 2 or 3
    val enable   = Input(Bool())
    val valid    = Output(Bool())
  })

  // Window buffer (max 3×3 = 9 elements)
  val windowBuf = RegInit(VecInit(Seq.fill(9)(0.U(ScarfConfig.DataWidth.W))))
  val count     = RegInit(0.U(4.W))
  val totalElems = io.poolSize * io.poolSize

  when(io.enable) {
    windowBuf(count) := io.dataIn
    count := Mux(count === totalElems - 1.U, 0.U, count + 1.U)
  }

  // Max pooling: tree comparison
  val maxVal = windowBuf.reduce((a, b) => Mux(a > b, a, b))

  // Avg pooling: sum / count (structural: shift-based division for pool_size=2)
  val sumVal = windowBuf.take(9).reduce(_ + _)
  val avgVal = Mux(io.poolSize === 2.U,
    sumVal >> 2,  // Divide by 4
    sumVal / 9.U  // Divide by 9
  )

  io.dataOut := Mux(io.poolType === PoolType.MAX, maxVal, avgVal)(ScarfConfig.DataWidth - 1, 0)
  io.valid   := io.enable && (count === totalElems - 1.U)
}
