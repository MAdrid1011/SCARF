package scarf.compute

import chisel3._
import chisel3.util._
import scarf.{ScarfConfig, MMCUMode}

/**
 * MMCU — Multi-Mode Compute Unit.
 *
 * Unified 48×48 output-stationary systolic array supporting:
 *   - Conv mode:  im2col + GEMM (weight preload, input streaming)
 *   - GEMM mode:  direct matrix multiply C = A × B + bias
 *   - Attn mode:  QKV projections + scaled dot-product (sequence of GEMMs)
 *
 * Single instance, time-multiplexed across S1/S2/S3 by PipelineController.
 */

class OSPE extends Module {
  val io = IO(new Bundle {
    val aIn     = Input(UInt(ScarfConfig.DataWidth.W))
    val bIn     = Input(UInt(ScarfConfig.DataWidth.W))
    val accOut  = Output(UInt(ScarfConfig.AccWidth.W))
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

class OutputStationaryArray(val size: Int = ScarfConfig.PEArraySize) extends Module {
  val io = IO(new Bundle {
    val aRow    = Input(Vec(size, UInt(ScarfConfig.DataWidth.W)))
    val bCol    = Input(Vec(size, UInt(ScarfConfig.DataWidth.W)))
    val results = Output(Vec(size, Vec(size, UInt(ScarfConfig.AccWidth.W))))
    val enable  = Input(Bool())
    val clear   = Input(Bool())
  })

  val pes = Seq.tabulate(size, size) { (i, j) =>
    Module(new OSPE)
  }

  for (i <- 0 until size; j <- 0 until size) {
    pes(i)(j).io.aIn    := io.aRow(i)
    pes(i)(j).io.bIn    := io.bCol(j)
    pes(i)(j).io.enable := io.enable
    pes(i)(j).io.clear  := io.clear
    io.results(i)(j)    := pes(i)(j).io.accOut
  }
}

object MMCUState extends ChiselEnum {
  val sIdle, sCompute, sSnapshot, sDone = Value
}

class MMCU(val arraySize: Int = ScarfConfig.PEArraySize) extends Module {
  val io = IO(new Bundle {
    val start    = Input(Bool())
    val done     = Output(Bool())
    val busy     = Output(Bool())

    val mode     = Input(MMCUMode())

    // GEMM dimensions (also used for im2col-transformed Conv)
    val M        = Input(UInt(16.W))
    val K        = Input(UInt(16.W))
    val N        = Input(UInt(16.W))
    val useBias  = Input(Bool())

    // Conv-specific (mode == mConv)
    val kernelSize  = Input(UInt(4.W))
    val stride      = Input(UInt(3.W))
    val inChannels  = Input(UInt(10.W))
    val outChannels = Input(UInt(10.W))

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

    // Weight address output for WeightBuffer reads
    val weightAddr = Output(UInt(ScarfConfig.AddrWidth.W))
  })

  val array = Module(new OutputStationaryArray(arraySize))

  val state = RegInit(MMCUState.sIdle)

  // Tiling counters
  val mTile = RegInit(0.U(10.W))
  val nTile = RegInit(0.U(10.W))
  val kStep = RegInit(0.U(16.W))

  // Effective dimensions: in Conv mode, im2col transforms (C_in*K*K, C_out) → GEMM(M', K', N')
  val effM = Wire(UInt(16.W))
  val effK = Wire(UInt(16.W))
  val effN = Wire(UInt(16.W))
  effM := Mux(io.mode === MMCUMode.mConv,
    io.outChannels.pad(16),
    io.M)
  effK := Mux(io.mode === MMCUMode.mConv,
    (io.inChannels * io.kernelSize * io.kernelSize).pad(16),
    io.K)
  effN := Mux(io.mode === MMCUMode.mConv,
    io.outChannels.pad(16),
    io.N)

  val totalMTiles = (effM + (arraySize - 1).U) / arraySize.U
  val totalNTiles = (effN + (arraySize - 1).U) / arraySize.U
  val totalKSteps = (effK + (arraySize - 1).U) / arraySize.U

  // Snapshot/output buffer
  val outBufReg = Reg(Vec(arraySize, Vec(arraySize, UInt(ScarfConfig.AccWidth.W))))
  val snapMTile = RegInit(0.U(10.W))
  val snapNTile = RegInit(0.U(10.W))

  // DMA sub-FSM
  val dmaActive = RegInit(false.B)
  val dmaRow    = RegInit(0.U(log2Ceil(arraySize + 1).W))

  val doneReg = RegInit(false.B)
  val busyReg = RegInit(false.B)
  io.done := doneReg
  io.busy := busyReg

  // Address generation
  io.aAddr := mTile * effK + kStep * arraySize.U
  io.bAddr := kStep * arraySize.U * effN + nTile * arraySize.U
  io.biasAddr := snapNTile * arraySize.U
  io.weightAddr := Mux(io.mode === MMCUMode.mConv,
    (mTile * totalKSteps + kStep) * arraySize.U,
    io.bAddr)

  val cAddrWire = Wire(UInt(ScarfConfig.AddrWidth.W))
  val cDataWire = Wire(Vec(arraySize, UInt(ScarfConfig.AccWidth.W)))
  val cWrWire   = Wire(Bool())
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

  // DMA sub-FSM: serialise outBufReg → external memory
  when(dmaActive) {
    val dmaRowIdx = dmaRow(log2Ceil(arraySize) - 1, 0)
    cWrWire   := true.B
    cAddrWire := snapMTile * arraySize.U * effN + snapNTile * arraySize.U +
                 dmaRow * effN
    for (j <- 0 until arraySize) {
      cDataWire(j) := Mux(io.useBias,
        outBufReg(dmaRowIdx)(j) + io.biasData(j),
        outBufReg(dmaRowIdx)(j))
    }
    dmaRow := dmaRow + 1.U
    when(dmaRow === (arraySize - 1).U) {
      dmaActive := false.B
      dmaRow    := 0.U
    }
  }

  switch(state) {
    is(MMCUState.sIdle) {
      doneReg := false.B
      busyReg := false.B
      when(io.start) {
        state   := MMCUState.sCompute
        busyReg := true.B
        mTile   := 0.U
        nTile   := 0.U
        kStep   := 0.U
        array.io.clear := true.B
      }
    }

    is(MMCUState.sCompute) {
      array.io.enable := true.B
      kStep := kStep + 1.U
      when(kStep === totalKSteps - 1.U) {
        state := MMCUState.sSnapshot
      }
    }

    is(MMCUState.sSnapshot) {
      when(!dmaActive) {
        for (i <- 0 until arraySize; j <- 0 until arraySize) {
          outBufReg(i)(j) := array.io.results(i)(j)
        }
        snapMTile := mTile
        snapNTile := nTile
        dmaActive := true.B
        dmaRow    := 0.U
        array.io.clear := true.B

        nTile := nTile + 1.U
        when(nTile === totalNTiles - 1.U) {
          nTile := 0.U
          mTile := mTile + 1.U
          when(mTile === totalMTiles - 1.U) {
            state := MMCUState.sDone
          }.otherwise {
            state := MMCUState.sCompute
            kStep := 0.U
          }
        }.otherwise {
          state := MMCUState.sCompute
          kStep := 0.U
        }
      }
    }

    is(MMCUState.sDone) {
      when(!dmaActive) {
        doneReg := true.B
        busyReg := false.B
        state   := MMCUState.sIdle
      }
    }
  }
}
