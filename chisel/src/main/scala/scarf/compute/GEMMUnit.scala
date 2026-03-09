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
 * Write-back design: snapshot + background DMA
 *   After all K-steps for a tile complete, the PE results are snapshotted into
 *   outBufReg in a single cycle (sSnapshot).  A background DMA sub-FSM then
 *   serialises outBufReg to external memory one row per cycle (arraySize cycles)
 *   while the systolic array immediately begins computing the next tile.
 *   If the DMA is still running when the next snapshot is ready, the main FSM
 *   stalls for exactly the remaining DMA rows (dmaStall).  For large K this
 *   overlap completely hides the write-back latency.
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
  val sIdle, sCompute, sSnapshot, sDone = Value
}

/**
 * GEMMUnit: C[M,N] = A[M,K] × B[K,N] + bias[N] (optional).
 *
 * Tiled execution: iterates over (M/size, K, N/size) tiles.
 * For each output tile C_tile[size,size]:
 *   1. Clear accumulators
 *   2. For k in 0..ceil(K/size): load A_tile column, B_tile row, accumulate
 *   3. Snapshot: capture all PE results into outBufReg in 1 cycle
 *   4. DMA sub-FSM writes outBufReg to memory (arraySize cycles, overlapped
 *      with next tile's computation)
 *
 * Overlap policy: the main FSM transitions from sSnapshot directly to sCompute
 * for the next tile.  If dmaActive is still set when sSnapshot fires again
 * (i.e. K is very small and DMA hasn't finished), the FSM waits (dmaStall)
 * until dmaActive de-asserts before overwriting outBufReg.
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

  // ---- Main FSM ----
  val state = RegInit(GEMMState.sIdle)

  // Tiling counters
  val mTile = RegInit(0.U(10.W))
  val nTile = RegInit(0.U(10.W))
  val kStep = RegInit(0.U(16.W))

  val totalMTiles = Wire(UInt(10.W))
  val totalNTiles = Wire(UInt(10.W))
  val totalKSteps = Wire(UInt(16.W))

  totalMTiles := (io.M + (arraySize - 1).U) / arraySize.U
  totalNTiles := (io.N + (arraySize - 1).U) / arraySize.U
  totalKSteps := (io.K + (arraySize - 1).U) / arraySize.U

  // ---- Snapshot / output buffer ----
  // Holds the completed tile while DMA serialises it to memory.
  // Sized arraySize × arraySize (e.g. 48 × 48 FP32 words).
  // No reset initialisation needed: sSnapshot always writes every element
  // before the DMA sub-FSM reads them.
  val outBufReg = Reg(Vec(arraySize, Vec(arraySize, UInt(ScarfConfig.AccWidth.W))))

  // Tile address at the time of snapshot (needed for DMA address generation)
  val snapMTile = RegInit(0.U(10.W))
  val snapNTile = RegInit(0.U(10.W))

  // ---- DMA sub-FSM ----
  val dmaActive = RegInit(false.B)
  val dmaRow    = RegInit(0.U(log2Ceil(arraySize + 1).W))

  // ---- Status registers ----
  val doneReg = RegInit(false.B)
  val busyReg = RegInit(false.B)
  io.done := doneReg
  io.busy := busyReg

  // ---- Default memory outputs ----
  io.aAddr    := (mTile * io.K + kStep * arraySize.U)
  io.bAddr    := (kStep * arraySize.U * io.N + nTile * arraySize.U)
  io.biasAddr := snapNTile * arraySize.U

  // Use Wires so the DMA sub-FSM can override without multiple-driver conflicts.
  val cAddrWire = Wire(UInt(ScarfConfig.AddrWidth.W))
  val cDataWire = Wire(Vec(arraySize, UInt(ScarfConfig.AccWidth.W)))
  val cWrWire   = Wire(Bool())

  // Default: outputs are inactive; DMA sub-FSM overrides cAddrWire/cDataWire/cWrWire
  // when dmaActive.  cData is don't-care when cWr is false, so 0 is fine.
  cAddrWire := 0.U
  for (j <- 0 until arraySize) { cDataWire(j) := 0.U }
  cWrWire := false.B

  io.cAddr := cAddrWire
  io.cData := cDataWire
  io.cWr   := cWrWire

  // Array connections
  array.io.aRow   := io.aData
  array.io.bCol   := io.bData
  array.io.enable := false.B
  array.io.clear  := false.B

  // ---- DMA sub-FSM (runs every cycle independently of main FSM) ----
  // When active, serialises outBufReg → external memory one row per cycle.
  when(dmaActive) {
    cWrWire   := true.B
    cAddrWire := (snapMTile * arraySize.U * io.N + snapNTile * arraySize.U +
                  dmaRow * io.N)
    for (j <- 0 until arraySize) {
      cDataWire(j) := Mux(io.useBias,
        outBufReg(dmaRow)(j) + io.biasData(j),
        outBufReg(dmaRow)(j))
    }

    dmaRow := dmaRow + 1.U
    when(dmaRow === (arraySize - 1).U) {
      dmaActive := false.B
      dmaRow    := 0.U
    }
  }

  // ---- Main FSM ----
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
        state := GEMMState.sSnapshot
      }
    }

    is(GEMMState.sSnapshot) {
      // Wait if the previous DMA is still draining outBufReg.
      // This stall is rare: only when K < arraySize (very small matrices).
      when(!dmaActive) {
        // 1-cycle parallel snapshot of all PE results into outBufReg
        for (i <- 0 until arraySize) {
          for (j <- 0 until arraySize) {
            outBufReg(i)(j) := array.io.results(i)(j)
          }
        }
        // Record the tile address for DMA address generation
        snapMTile := mTile
        snapNTile := nTile

        // Kick off background DMA
        dmaActive := true.B
        dmaRow    := 0.U

        // Clear PE array for the next tile
        array.io.clear := true.B

        // Advance tile counters and decide next state
        nTile := nTile + 1.U
        when(nTile === totalNTiles - 1.U) {
          nTile := 0.U
          mTile := mTile + 1.U
          when(mTile === totalMTiles - 1.U) {
            state := GEMMState.sDone
          }.otherwise {
            state := GEMMState.sCompute
            kStep := 0.U
          }
        }.otherwise {
          state := GEMMState.sCompute
          kStep := 0.U
        }
      }
      // else: stay in sSnapshot (dmaStall) until DMA finishes
    }

    is(GEMMState.sDone) {
      // Wait for the background DMA to finish draining the last tile's outBufReg
      // before asserting done.  Without this, the downstream PipelineController
      // would see done=true while the final tile's data is still being written.
      when(!dmaActive) {
        doneReg := true.B
        busyReg := false.B
        state   := GEMMState.sIdle
      }
      // busyReg stays true until DMA flushes (unit is still writing output)
    }
  }
}
