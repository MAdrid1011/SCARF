package scarf.control

import chisel3._
import chisel3.util._
import scarf.SAESLevel

/** Schedules the native retained S2/S3 outputs after an accepted SAES route.
  *
  * The submitted hardware configuration fixes tiles at 4x4. L0 requests the
  * four primary probes and L1 requests the same prefix followed by four
  * deterministic lightweight anchors. Each request advances only after the
  * upstream S2/S3 producer confirms a real descriptor is available. This
  * scheduler does not classify tiles, create descriptors, or authorize a
  * bypass; it is intentionally left outside ScarfTop until the producer,
  * moment path, descriptor packing, and S4 hand-off are connected.
  */
class SAESRetainedOutputScheduler extends Module {
  private val maxRetained = 8
  private val ordinalWidth = log2Ceil(maxRetained)

  val io = IO(new Bundle {
    val start = Input(Bool())
    val level = Input(SAESLevel())
    val startAccepted = Output(Bool())
    val inputError = Output(Bool())

    val requestValid = Output(Bool())
    val requestOrdinal = Output(UInt(ordinalWidth.W))
    val requestPixelIndex = Output(UInt(4.W))
    val expectedDescriptorCount = Output(UInt(4.W))
    val upstreamDescriptorValid = Input(Bool())
    val descriptorAccepted = Output(Bool())

    val busy = Output(Bool())
    val done = Output(Bool())
  })

  val sIdle :: sRequest :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val activeLevel = RegInit(SAESLevel.sFull)
  val ordinal = RegInit(0.U(ordinalWidth.W))

  val primaryPositions = VecInit(Seq(0.U(4.W), 3.U(4.W), 12.U(4.W), 15.U(4.W)))
  val lightweightPositions = VecInit(
    Seq(0.U(4.W), 3.U(4.W), 12.U(4.W), 15.U(4.W), 5.U(4.W), 10.U(4.W), 1.U(4.W), 2.U(4.W)),
  )
  val activeIsSparse = activeLevel === SAESLevel.sL0 || activeLevel === SAESLevel.sL1
  val levelIsSparse = io.level === SAESLevel.sL0 || io.level === SAESLevel.sL1
  val activeCount = Mux(activeLevel === SAESLevel.sL1, 8.U(4.W), 4.U(4.W))

  io.startAccepted := state === sIdle && io.start && levelIsSparse
  io.inputError := (state === sIdle && io.start && !levelIsSparse) ||
    (state =/= sIdle && io.start)
  io.requestValid := state === sRequest && activeIsSparse
  io.requestOrdinal := Mux(io.requestValid, ordinal, 0.U)
  io.requestPixelIndex := Mux(
    activeLevel === SAESLevel.sL1,
    lightweightPositions(ordinal),
    primaryPositions(ordinal(1, 0)),
  )
  io.expectedDescriptorCount := Mux(activeIsSparse, activeCount, 0.U)
  io.descriptorAccepted := io.requestValid && io.upstreamDescriptorValid
  io.busy := state === sRequest
  io.done := state === sDone

  switch(state) {
    is(sIdle) {
      when(io.startAccepted) {
        activeLevel := io.level
        ordinal := 0.U
        state := sRequest
      }
    }
    is(sRequest) {
      when(io.descriptorAccepted) {
        when(ordinal === activeCount - 1.U) {
          state := sDone
        }.otherwise {
          ordinal := ordinal + 1.U
        }
      }
    }
    is(sDone) {
      state := sIdle
    }
  }
}
