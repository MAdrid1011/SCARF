package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * GEMMUnit — General Matrix Multiply with Output-Stationary Dataflow.
 *
 * Corresponds to: encoder/gemm_unit.py
 *
 * Architecture:
 *   - PEArraySize × PEArraySize output-stationary array (48×48 = 2304 MACs/cycle)
 *   - Each PE accumulates one element of the output tile C[i,j]
 *   - A matrix rows and B matrix columns stream through alternately
 *   - Tiled execution for matrices larger than array size
 *   - Dual 32 KB buffers for A and B matrix tiles
 *
 * Used in: S1 (Transformer QKV projections), S2 (regression), GGU (covariance)
 * Single instance, time-multiplexed via PipelineController.
 */

// ============================================================
// Output-Stationary PE
// ============================================================

/** Output-stationary PE: accumulates C[i,j] = sum_k A[i,k] * B[k,j]. */
class OSPE extends Module {
  val io = IO(new Bundle {
    val aIn     = Input(UInt(ScarfConfig.DataWidth.W))   // A element (broadcast per row)
    val bIn     = Input(UInt(ScarfConfig.DataWidth.W))   // B element (broadcast per col)
    val accOut  = Output(UInt(ScarfConfig.AccWidth.W))    // Accumulated result
    val enable  = Input(Bool())
    val clear   = Input(Bool())
  })

  val accReg = RegInit(0.U(ScarfConfig.AccWidth.W))

  when(io.clear) {
    accReg := 0.U
  }.elsewhen(io.enable) {
    val product = (io.aIn * io.bIn)(ScarfConfig.AccWidth - 1, 0)
    accReg := accReg + product
  }

  io.accOut := accReg
}

// ============================================================
// Output-Stationary Array
// ============================================================

/**
 * Output-stationary array for GEMM.
 *
 * Each PE[i,j] accumulates C[i,j]. During each K-step:
 *   - Row i receives A[i,k] (broadcast to all PEs in row i)
 *   - Col j receives B[k,j] (broadcast to all PEs in col j)
 *   - All PEs compute product and accumulate simultaneously
 */
class OutputStationaryArray(val size: Int = ScarfConfig.PEArraySize) extends Module {
  val io = IO(new Bundle {
    val aRow    = Input(Vec(size, UInt(ScarfConfig.DataWidth.W)))  // A[0..size-1, k]
    val bCol    = Input(Vec(size, UInt(ScarfConfig.DataWidth.W)))  // B[k, 0..size-1]
    val results = Output(Vec(size, Vec(size, UInt(ScarfConfig.AccWidth.W))))
    val enable  = Input(Bool())
    val clear   = Input(Bool())
  })

  val pes = Seq.tabulate(size, size) { (i, j) =>
    Module(new OSPE)
  }

  for (i <- 0 until size) {
    for (j <- 0 until size) {
      pes(i)(j).io.aIn    := io.aRow(i)    // Broadcast A[i,k] to row i
      pes(i)(j).io.bIn    := io.bCol(j)    // Broadcast B[k,j] to col j
      pes(i)(j).io.enable := io.enable
      pes(i)(j).io.clear  := io.clear
      io.results(i)(j)    := pes(i)(j).io.accOut
    }
  }
}

// ============================================================
// GEMM Unit Top-Level
// ============================================================

object GEMMState extends ChiselEnum {
  val sIdle, sCompute, sWriteBack, sDone = Value
}

/**
 * GEMMUnit: C[M,N] = A[M,K] × B[K,N] + bias[N] (optional).
 *
 * Tiled execution: iterates over (M/size, K, N/size) tiles.
 * For each output tile C_tile[size,size]:
 *   1. Clear accumulators
 *   2. For k in 0..ceil(K/size): load A_tile column, B_tile row, accumulate
 *   3. Write back C_tile (+ optional bias)
 */
class GEMMUnit(val arraySize: Int = ScarfConfig.PEArraySize) extends Module {
  val io = IO(new Bundle {
    val start    = Input(Bool())
    val done     = Output(Bool())
    val busy     = Output(Bool())

    // GEMM dimensions
    val M        = Input(UInt(16.W))
    val K        = Input(UInt(16.W))
    val N        = Input(UInt(16.W))
    val useBias  = Input(Bool())

    // Memory interface
    val aAddr    = Output(UInt(ScarfConfig.AddrWidth.W))
    val aData    = Input(Vec(arraySize, UInt(ScarfConfig.DataWidth.W)))
    val bAddr    = Output(UInt(ScarfConfig.AddrWidth.W))
    val bData    = Input(Vec(arraySize, UInt(ScarfConfig.DataWidth.W)))
    val cAddr    = Output(UInt(ScarfConfig.AddrWidth.W))
    val cData    = Output(Vec(arraySize, UInt(ScarfConfig.AccWidth.W)))
    val cWr      = Output(Bool())
    val biasAddr = Output(UInt(ScarfConfig.AddrWidth.W))
    val biasData = Input(Vec(arraySize, UInt(ScarfConfig.DataWidth.W)))
  })

  val array = Module(new OutputStationaryArray(arraySize))

  // FSM
  val state = RegInit(GEMMState.sIdle)

  // Tiling counters
  val mTile = RegInit(0.U(10.W))  // Current M tile index
  val nTile = RegInit(0.U(10.W))  // Current N tile index
  val kStep = RegInit(0.U(16.W))  // Current K step

  val totalMTiles = Wire(UInt(10.W))
  val totalNTiles = Wire(UInt(10.W))
  val totalKSteps = Wire(UInt(16.W))

  totalMTiles := (io.M + (arraySize - 1).U) / arraySize.U
  totalNTiles := (io.N + (arraySize - 1).U) / arraySize.U
  totalKSteps := (io.K + (arraySize - 1).U) / arraySize.U

  // Row counter for write-back
  val wbRow = RegInit(0.U(log2Ceil(arraySize + 1).W))

  // Default outputs
  val doneReg = RegInit(false.B)
  val busyReg = RegInit(false.B)
  io.done := doneReg
  io.busy := busyReg

  io.aAddr    := (mTile * io.K + kStep * arraySize.U)
  io.bAddr    := (kStep * arraySize.U * io.N + nTile * arraySize.U)
  io.cAddr    := (mTile * arraySize.U * io.N + nTile * arraySize.U + wbRow * io.N)
  io.biasAddr := nTile * arraySize.U

  // Write-back: output one row at a time from the 2D result array
  io.cData := VecInit(Seq.fill(arraySize)(0.U(ScarfConfig.AccWidth.W)))
  io.cWr   := false.B

  // Array connections
  array.io.aRow   := io.aData
  array.io.bCol   := io.bData
  array.io.enable := false.B
  array.io.clear  := false.B

  switch(state) {
    is(GEMMState.sIdle) {
      doneReg := false.B
      busyReg := false.B
      when(io.start) {
        state   := GEMMState.sCompute
        busyReg := true.B
        mTile   := 0.U
        nTile   := 0.U
        kStep   := 0.U
        array.io.clear := true.B
      }
    }

    is(GEMMState.sCompute) {
      // Each cycle: broadcast A column and B row, accumulate in PEs
      array.io.enable := true.B

      kStep := kStep + 1.U
      when(kStep === totalKSteps - 1.U) {
        state := GEMMState.sWriteBack
        wbRow := 0.U
      }
    }

    is(GEMMState.sWriteBack) {
      // Output results row by row
      io.cWr := true.B
      for (j <- 0 until arraySize) {
        val result = array.io.results(wbRow)(j)
        io.cData(j) := Mux(io.useBias,
          result + io.biasData(j),  // Add bias
          result)
      }

      wbRow := wbRow + 1.U
      when(wbRow === (arraySize - 1).U) {
        // Move to next tile
        nTile := nTile + 1.U
        when(nTile === totalNTiles - 1.U) {
          nTile := 0.U
          mTile := mTile + 1.U
          when(mTile === totalMTiles - 1.U) {
            state := GEMMState.sDone
          }.otherwise {
            state := GEMMState.sCompute
            kStep := 0.U
            array.io.clear := true.B
          }
        }.otherwise {
          state := GEMMState.sCompute
          kStep := 0.U
          array.io.clear := true.B
        }
      }
    }

    is(GEMMState.sDone) {
      doneReg := true.B
      busyReg := false.B
      state   := GEMMState.sIdle
    }
  }
}
