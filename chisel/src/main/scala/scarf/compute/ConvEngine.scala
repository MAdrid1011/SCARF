package scarf.compute

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * ConvEngine — 2D Convolution Engine with Weight-Stationary Systolic Array.
 *
 * Corresponds to: encoder/conv_engine.py
 *
 * Architecture:
 *   - PEArraySize × PEArraySize systolic array (48×48 = 2304 MACs/cycle)
 *   - Weight-stationary dataflow: weights preloaded, inputs stream left-to-right
 *   - Partial sums flow top-to-bottom
 *   - Im2col FSM converts conv operations to matrix multiplies
 *   - Supports kernel sizes: 1, 3, 5, 7, 9, 14
 *   - Conv-BN-ReLU fusion in output stage
 *
 * This is the single ConvEngine instance shared across S1 (CNN), S2 (U-Net,
 * DepthHead), and S3 (Refine, GaussHead) via FSM time-multiplexing.
 */

// ============================================================
// Processing Element (PE)
// ============================================================

/** Single MAC processing element in the systolic array. */
class PE extends Module {
  val io = IO(new Bundle {
    // Weight loading
    val weightIn  = Input(UInt(ScarfConfig.DataWidth.W))
    val weightLoad = Input(Bool())

    // Data flow (left-to-right for inputs, top-to-bottom for partial sums)
    val dataIn    = Input(UInt(ScarfConfig.DataWidth.W))    // From left neighbor
    val dataOut   = Output(UInt(ScarfConfig.DataWidth.W))   // To right neighbor
    val psumIn    = Input(UInt(ScarfConfig.AccWidth.W))     // From top neighbor
    val psumOut   = Output(UInt(ScarfConfig.AccWidth.W))    // To bottom neighbor

    // Control
    val enable    = Input(Bool())
    val clear     = Input(Bool())
  })

  // Weight register (stationary — loaded once, reused for all inputs)
  val weightReg = RegInit(0.U(ScarfConfig.DataWidth.W))
  when(io.weightLoad) {
    weightReg := io.weightIn
  }

  // Pipeline registers for systolic data flow
  val dataReg = RegInit(0.U(ScarfConfig.DataWidth.W))
  val psumReg = RegInit(0.U(ScarfConfig.AccWidth.W))

  when(io.clear) {
    dataReg := 0.U
    psumReg := 0.U
  }.elsewhen(io.enable) {
    // Pass input data to right neighbor (1-cycle delay = systolic timing)
    dataReg := io.dataIn

    // MAC: psum_out = psum_in + weight * data_in
    // In real HW this would be FP16 mul + FP32 accumulate;
    // here we model as integer arithmetic for structural correctness.
    val product = (io.dataIn * weightReg)(ScarfConfig.AccWidth - 1, 0)
    psumReg := io.psumIn + product
  }

  io.dataOut := dataReg
  io.psumOut := psumReg
}

// ============================================================
// Systolic Array
// ============================================================

/**
 * Weight-stationary systolic array of PEArraySize × PEArraySize PEs.
 *
 * Data flow:
 *   - Inputs stream from left edge, propagate right (1 cycle/PE)
 *   - Partial sums flow from top edge (initialized to 0), accumulate downward
 *   - Bottom edge outputs final accumulated results
 *   - Weights are preloaded column-by-column before computation starts
 */
class SystolicArray(val size: Int = ScarfConfig.PEArraySize) extends Module {
  val io = IO(new Bundle {
    // Weight loading interface
    val weightData  = Input(Vec(size, UInt(ScarfConfig.DataWidth.W)))
    val weightCol   = Input(UInt(log2Ceil(size + 1).W))
    val weightLoad  = Input(Bool())

    // Data input (left edge) — one value per row per cycle
    val dataIn      = Input(Vec(size, UInt(ScarfConfig.DataWidth.W)))

    // Partial sum output (bottom edge)
    val psumOut     = Output(Vec(size, UInt(ScarfConfig.AccWidth.W)))

    // Control
    val enable      = Input(Bool())
    val clear       = Input(Bool())
  })

  // Instantiate size × size PE grid
  val pes = Seq.tabulate(size, size) { (r, c) =>
    Module(new PE)
  }

  // Wire up the systolic array
  for (r <- 0 until size) {
    for (c <- 0 until size) {
      val pe = pes(r)(c)

      // Control signals
      pe.io.enable := io.enable
      pe.io.clear  := io.clear

      // Weight loading: load column `weightCol` with data from weightData
      pe.io.weightLoad := io.weightLoad && (io.weightCol === c.U)
      pe.io.weightIn   := io.weightData(r)

      // Horizontal data flow: left-to-right
      if (c == 0) {
        pe.io.dataIn := io.dataIn(r)      // Left edge: external input
      } else {
        pe.io.dataIn := pes(r)(c - 1).io.dataOut  // From left neighbor
      }

      // Vertical partial sum flow: top-to-bottom
      if (r == 0) {
        pe.io.psumIn := 0.U                // Top edge: initialize to 0
      } else {
        pe.io.psumIn := pes(r - 1)(c).io.psumOut  // From top neighbor
      }
    }
  }

  // Bottom edge outputs
  for (c <- 0 until size) {
    io.psumOut(c) := pes(size - 1)(c).io.psumOut
  }
}

// ============================================================
// ConvEngine Top-Level (Array + Control FSM)
// ============================================================

/** ConvEngine FSM states. */
object ConvState extends ChiselEnum {
  val sIdle, sLoadWeights, sCompute, sWriteBack, sDone = Value
}

/**
 * ConvEngine: complete convolution engine with Im2col control.
 *
 * The FSM orchestrates:
 *   1. Weight loading: read weight tiles from WeightBuffer
 *   2. Compute: stream im2col-transformed input through systolic array
 *   3. Write-back: write accumulated output tile to output buffer
 *   4. Repeat for all output channel tiles
 *
 * Parameters are set externally via the `params` input (from ConfigRegs).
 */
class ConvEngine(val arraySize: Int = ScarfConfig.PEArraySize) extends Module {
  val io = IO(new Bundle {
    // Command interface
    val start     = Input(Bool())
    val done      = Output(Bool())
    val busy      = Output(Bool())

    // Convolution parameters (from ConfigRegs / PipelineController)
    val inChannels  = Input(UInt(10.W))
    val outChannels = Input(UInt(10.W))
    val kernelSize  = Input(UInt(4.W))
    val stride      = Input(UInt(3.W))
    val padding     = Input(UInt(4.W))
    val fuseReLU    = Input(Bool())

    // Memory interface (simplified: address + data)
    val weightAddr  = Output(UInt(ScarfConfig.AddrWidth.W))
    val weightData  = Input(Vec(arraySize, UInt(ScarfConfig.DataWidth.W)))
    val inputAddr   = Output(UInt(ScarfConfig.AddrWidth.W))
    val inputData   = Input(Vec(arraySize, UInt(ScarfConfig.DataWidth.W)))
    val outputAddr  = Output(UInt(ScarfConfig.AddrWidth.W))
    val outputData  = Output(Vec(arraySize, UInt(ScarfConfig.AccWidth.W)))
    val outputWr    = Output(Bool())
  })

  // Systolic array
  val array = Module(new SystolicArray(arraySize))

  // FSM
  val state = RegInit(ConvState.sIdle)
  val done  = RegInit(false.B)
  val busy  = RegInit(false.B)

  // Tiling counters
  val outTile    = RegInit(0.U(10.W))  // Current output channel tile
  val kStep      = RegInit(0.U(16.W))  // Current step within im2col unrolling
  val totalOutTiles = Wire(UInt(10.W))
  val totalKSteps   = Wire(UInt(16.W))

  // Compute tile counts: ceil(outChannels / arraySize), ceil(inChannels * K * K / arraySize)
  totalOutTiles := (io.outChannels + (arraySize - 1).U) / arraySize.U
  totalKSteps   := (io.inChannels * io.kernelSize * io.kernelSize + (arraySize - 1).U) / arraySize.U

  // Address generation
  val weightBaseAddr = RegInit(0.U(ScarfConfig.AddrWidth.W))
  val inputBaseAddr  = RegInit(0.U(ScarfConfig.AddrWidth.W))
  val outputBaseAddr = RegInit(0.U(ScarfConfig.AddrWidth.W))

  // Default outputs
  io.done := done
  io.busy := busy
  io.weightAddr := weightBaseAddr + (outTile * totalKSteps + kStep) * arraySize.U
  io.inputAddr  := inputBaseAddr + kStep * arraySize.U
  io.outputAddr := outputBaseAddr + outTile * arraySize.U
  io.outputData := array.io.psumOut
  io.outputWr   := false.B

  // Array defaults
  array.io.weightData := io.weightData
  array.io.weightCol  := kStep(log2Ceil(arraySize + 1) - 1, 0)
  array.io.weightLoad := false.B
  array.io.dataIn     := io.inputData
  array.io.enable     := false.B
  array.io.clear      := false.B

  // Weight loading counter
  val weightLoadCol = RegInit(0.U(log2Ceil(arraySize + 1).W))

  switch(state) {
    is(ConvState.sIdle) {
      done := false.B
      busy := false.B
      when(io.start) {
        state    := ConvState.sLoadWeights
        busy     := true.B
        outTile  := 0.U
        kStep    := 0.U
        weightLoadCol := 0.U
        array.io.clear := true.B
      }
    }

    is(ConvState.sLoadWeights) {
      // Load one column of weights per cycle into the systolic array
      array.io.weightLoad := true.B
      array.io.weightCol  := weightLoadCol

      weightLoadCol := weightLoadCol + 1.U
      when(weightLoadCol === (arraySize - 1).U) {
        state := ConvState.sCompute
        kStep := 0.U
        weightLoadCol := 0.U
      }
    }

    is(ConvState.sCompute) {
      // Stream input data through the array
      array.io.enable := true.B

      kStep := kStep + 1.U
      when(kStep === totalKSteps - 1.U) {
        state := ConvState.sWriteBack
      }
    }

    is(ConvState.sWriteBack) {
      // Write accumulated results to output buffer
      io.outputWr := true.B

      // Optional ReLU fusion: clamp negative outputs to 0
      when(io.fuseReLU) {
        for (i <- 0 until arraySize) {
          when(array.io.psumOut(i)(ScarfConfig.AccWidth - 1)) {
            // MSB set = negative in 2's complement → clamp to 0
            io.outputData(i) := 0.U
          }
        }
      }

      // Move to next output tile or finish
      outTile := outTile + 1.U
      when(outTile === totalOutTiles - 1.U) {
        state := ConvState.sDone
      }.otherwise {
        state := ConvState.sLoadWeights
        array.io.clear := true.B
        weightLoadCol := 0.U
        kStep := 0.U
      }
    }

    is(ConvState.sDone) {
      done := true.B
      busy := false.B
      state := ConvState.sIdle
    }
  }
}
