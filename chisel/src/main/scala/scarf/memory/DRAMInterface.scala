package scarf.memory

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * DRAMInterface — Simplified AXI4 DRAM controller interface.
 *
 * Provides burst read/write access to external DRAM for:
 * - Loading input images and model weights
 * - Writing final Gaussian outputs
 * - Weight prefetch into WeightBuffer
 *
 * Simplified model: Decoupled read/write channels with burst support.
 * Real implementation would use full AXI4 protocol with outstanding transactions.
 */
class DRAMInterface extends Module {
  val io = IO(new Bundle {
    // Internal interface (to pipeline controller)
    val readReq   = Flipped(Decoupled(new DRAMReadReq))
    val readResp  = Decoupled(new DRAMReadResp)
    val writeReq  = Flipped(Decoupled(new DRAMWriteReq))
    val writeDone = Output(Bool())

    // External AXI4 interface (to DRAM PHY)
    val axiArAddr = Output(UInt(ScarfConfig.AddrWidth.W))
    val axiArLen  = Output(UInt(8.W))
    val axiArValid = Output(Bool())
    val axiArReady = Input(Bool())
    val axiRData  = Input(UInt(ScarfConfig.AXIDataWidth.W))
    val axiRValid = Input(Bool())
    val axiRReady = Output(Bool())
    val axiRLast  = Input(Bool())

    val axiAwAddr  = Output(UInt(ScarfConfig.AddrWidth.W))
    val axiAwLen   = Output(UInt(8.W))
    val axiAwValid = Output(Bool())
    val axiAwReady = Input(Bool())
    val axiWData   = Output(UInt(ScarfConfig.AXIDataWidth.W))
    val axiWValid  = Output(Bool())
    val axiWReady  = Input(Bool())
    val axiWLast   = Output(Bool())
  })

  // Read channel FSM
  val sRdIdle :: sRdAddr :: sRdData :: Nil = Enum(3)
  val rdState = RegInit(sRdIdle)
  val rdAddr  = RegInit(0.U(ScarfConfig.AddrWidth.W))
  val rdLen   = RegInit(0.U(8.W))
  val rdCount = RegInit(0.U(8.W))

  io.readReq.ready  := rdState === sRdIdle
  io.readResp.valid := rdState === sRdData && io.axiRValid
  io.readResp.bits  := DontCare

  io.axiArAddr  := rdAddr
  io.axiArLen   := rdLen
  io.axiArValid := rdState === sRdAddr
  io.axiRReady  := rdState === sRdData

  switch(rdState) {
    is(sRdIdle) {
      when(io.readReq.valid) {
        rdAddr  := io.readReq.bits.addr
        rdLen   := io.readReq.bits.burstLen
        rdCount := 0.U
        rdState := sRdAddr
      }
    }
    is(sRdAddr) {
      when(io.axiArReady) { rdState := sRdData }
    }
    is(sRdData) {
      when(io.axiRValid && io.readResp.ready) {
        io.readResp.bits.data := io.axiRData
        io.readResp.bits.last := io.axiRLast
        when(io.axiRLast) { rdState := sRdIdle }
      }
    }
  }

  // Write channel FSM
  val sWrIdle :: sWrAddr :: sWrData :: Nil = Enum(3)
  val wrState = RegInit(sWrIdle)
  val wrAddr  = RegInit(0.U(ScarfConfig.AddrWidth.W))
  val wrLen   = RegInit(0.U(8.W))
  val wrCount = RegInit(0.U(8.W))

  io.writeReq.ready := wrState === sWrIdle
  io.writeDone      := false.B

  io.axiAwAddr  := wrAddr
  io.axiAwLen   := wrLen
  io.axiAwValid := wrState === sWrAddr
  io.axiWData   := io.writeReq.bits.data
  io.axiWValid  := wrState === sWrData && io.writeReq.valid
  io.axiWLast   := wrCount === wrLen

  switch(wrState) {
    is(sWrIdle) {
      when(io.writeReq.valid) {
        wrAddr  := io.writeReq.bits.addr
        wrLen   := io.writeReq.bits.burstLen
        wrCount := 0.U
        wrState := sWrAddr
      }
    }
    is(sWrAddr) {
      when(io.axiAwReady) { wrState := sWrData }
    }
    is(sWrData) {
      when(io.axiWReady && io.axiWValid) {
        wrCount := wrCount + 1.U
        when(wrCount === wrLen) {
          wrState     := sWrIdle
          io.writeDone := true.B
        }
      }
    }
  }
}

class DRAMReadReq extends Bundle {
  val addr     = UInt(ScarfConfig.AddrWidth.W)
  val burstLen = UInt(8.W)
}

class DRAMReadResp extends Bundle {
  val data = UInt(ScarfConfig.AXIDataWidth.W)
  val last = Bool()
}

class DRAMWriteReq extends Bundle {
  val addr     = UInt(ScarfConfig.AddrWidth.W)
  val data     = UInt(ScarfConfig.AXIDataWidth.W)
  val burstLen = UInt(8.W)
}
