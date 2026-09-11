package scarf.control

import chisel3._
import chisel3.util._
import scarf.SAESLevel
import scarf.memory.SAESDescriptorBuffer

/**
  * Directed T=4 retained-descriptor flow used to measure the implemented SAES
  * L0 hand-off against the same dense descriptor protocol.
  *
  * This is deliberately a bounded integration module, not a ScarfTop bypass:
  * its producer copies source-bound native descriptor records only after the
  * retained scheduler requests them, the buffer commits each record, and the
  * consumer reads every committed beat. The dense instance uses the identical
  * source, buffer, and consumer timing with all sixteen tile pixels selected.
  */
class SAESRepresentativeTileFlow(
    val level: SAESLevel.Type,
    val shDegree: Int = 2,
) extends Module {
  require(shDegree >= 0 && shDegree <= SAESDescriptorBuffer.MaxSupportedSHDegree)
  require(level == SAESLevel.sL0 || level == SAESLevel.sL1 || level == SAESLevel.sFull)

  private val tilePixels = 16
  private val sparse = level == SAESLevel.sL0 || level == SAESLevel.sL1
  private val retainedDescriptors = if (level == SAESLevel.sL0) 4 else if (level == SAESLevel.sL1) 12 else tilePixels
  private val countWidth = 8
  private val maxDescriptorBeats = SAESDescriptorBuffer.descriptorBeats(
    SAESDescriptorBuffer.MaxSupportedSHDegree,
  )

  val io = IO(new Bundle {
    val start = Input(Bool())
    val startAccepted = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())

    // Native S3 records arrive before the retained-output hand-off.  The
    // flow stores and returns these actual words; it does not synthesize an
    // opaque value for a selected anchor.
    val descriptorSource = Input(Vec(tilePixels, Vec(maxDescriptorBeats, UInt(128.W))))

    val acceptedWriteBeats = Output(UInt(countWidth.W))
    val committedDescriptors = Output(UInt(5.W))
    val acceptedReadBeats = Output(UInt(countWidth.W))
    val readRecordValid = Output(Bool())
    val readSourcePixel = Output(UInt(4.W))
    val readBeat = Output(UInt(4.W))
    val readData = Output(UInt(128.W))
  })

  val scheduler = Module(new SAESRetainedOutputScheduler)
  val descriptorBuffer = Module(new SAESDescriptorBuffer(slots = tilePixels))

  val sIdle :: sProduce :: sReadIssue :: sReadWait :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)
  val producerOrdinal = RegInit(0.U(4.W))
  val producerBeat = RegInit(0.U(4.W))
  val readSlot = RegInit(0.U(4.W))
  val readBeat = RegInit(0.U(4.W))
  val writeBeats = RegInit(0.U(countWidth.W))
  val committed = RegInit(0.U(5.W))
  val readBeats = RegInit(0.U(countWidth.W))

  val startAccepted = state === sIdle && io.start
  io.startAccepted := startAccepted
  io.busy := state =/= sIdle && state =/= sDone
  io.done := state === sDone
  io.acceptedWriteBeats := writeBeats
  io.committedDescriptors := committed
  io.acceptedReadBeats := readBeats

  scheduler.io.start := startAccepted && sparse.B
  scheduler.io.level := level

  val sparseProducerReady = scheduler.io.requestValid
  val producerReady = if (sparse) sparseProducerReady else true.B
  val sourcePixel = if (sparse) scheduler.io.requestPixelIndex else producerOrdinal
  val slotIndex = if (sparse) scheduler.io.requestOrdinal else producerOrdinal
  val writeLast = producerBeat === (descriptorBuffer.io.activeBeats - 1.U)
  val writeEn = state === sProduce && producerReady

  descriptorBuffer.io.shDegree := shDegree.U
  descriptorBuffer.io.writeEn := writeEn
  descriptorBuffer.io.writeIndex := slotIndex
  descriptorBuffer.io.writeBeat := producerBeat
  descriptorBuffer.io.writeData := io.descriptorSource(sourcePixel)(producerBeat)
  descriptorBuffer.io.writeLast := writeLast

  scheduler.io.upstreamDescriptorValid := descriptorBuffer.io.writeAccepted && writeLast

  descriptorBuffer.io.readEn := state === sReadIssue
  descriptorBuffer.io.readIndex := readSlot
  descriptorBuffer.io.readBeat := readBeat
  io.readRecordValid := descriptorBuffer.io.readValid
  io.readSourcePixel := descriptorBuffer.io.readData(7, 4)
  io.readBeat := descriptorBuffer.io.readData(3, 0)
  io.readData := descriptorBuffer.io.readData

  assert(!descriptorBuffer.io.writeEn || descriptorBuffer.io.writeAccepted)
  assert(!descriptorBuffer.io.readEn || descriptorBuffer.io.descriptorValid)

  switch(state) {
    is(sIdle) {
      when(startAccepted) {
        producerOrdinal := 0.U
        producerBeat := 0.U
        readSlot := 0.U
        readBeat := 0.U
        writeBeats := 0.U
        committed := 0.U
        readBeats := 0.U
        state := sProduce
      }
    }
    is(sProduce) {
      when(descriptorBuffer.io.writeAccepted) {
        writeBeats := writeBeats + 1.U
        when(writeLast) {
          committed := committed + 1.U
          producerBeat := 0.U
          val finalDescriptor = if (sparse) {
            scheduler.io.requestOrdinal === (retainedDescriptors - 1).U
          } else {
            producerOrdinal === (retainedDescriptors - 1).U
          }
          when(finalDescriptor) {
            state := sReadIssue
          }.otherwise {
            producerOrdinal := producerOrdinal + 1.U
          }
        }.otherwise {
          producerBeat := producerBeat + 1.U
        }
      }
    }
    is(sReadIssue) {
      state := sReadWait
    }
    is(sReadWait) {
      when(descriptorBuffer.io.readValid) {
        readBeats := readBeats + 1.U
        when(readBeat === (descriptorBuffer.io.activeBeats - 1.U)) {
          when(readSlot === (retainedDescriptors - 1).U) {
            state := sDone
          }.otherwise {
            readSlot := readSlot + 1.U
            readBeat := 0.U
            state := sReadIssue
          }
        }.otherwise {
          readBeat := readBeat + 1.U
          state := sReadIssue
        }
      }
    }
    is(sDone) {
      state := sIdle
    }
  }
}
