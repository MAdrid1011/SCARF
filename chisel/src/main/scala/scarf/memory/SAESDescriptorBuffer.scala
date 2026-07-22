package scarf.memory

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/** Packed retained-descriptor storage for the staged SAES RTL data path.
  *
  * This module deliberately stores opaque 128-bit descriptor beats. Numeric
  * assignment and moment matching are a later, separately verified stage. A
  * descriptor is readable only after an ordered, complete packed write; this
  * prevents a sparse tile from accidentally reusing a stale or partial output.
  */
object SAESDescriptorBuffer {
  val MaxSupportedSHDegree: Int = 4

  def descriptorBytes(shDegree: Int): Int = {
    require(shDegree >= 0 && shDegree <= MaxSupportedSHDegree)
    val meanBytes = 3 * 4
    val covarianceBytes = 6 * 4
    val shBytes = 3 * (shDegree + 1) * (shDegree + 1) * 2
    val opacityBytes = 2
    meanBytes + covarianceBytes + shBytes + opacityBytes
  }

  def descriptorBeats(
      shDegree: Int,
      beatBytes: Int = ScarfConfig.AXIDataWidth / 8,
  ): Int = {
    require(beatBytes > 0)
    (descriptorBytes(shDegree) + beatBytes - 1) / beatBytes
  }
}

class SAESDescriptorBuffer(
    val slots: Int = ScarfConfig.GGUPECount,
    val wordWidth: Int = ScarfConfig.AXIDataWidth,
) extends Module {
  require(slots > 0)
  require(wordWidth > 0 && wordWidth % 8 == 0)

  private val beatBytes = wordWidth / 8
  private val maxBeats = SAESDescriptorBuffer.descriptorBeats(
    SAESDescriptorBuffer.MaxSupportedSHDegree,
    beatBytes,
  )
  private val indexWidth = math.max(1, log2Ceil(slots))
  private val beatWidth = math.max(1, log2Ceil(maxBeats + 1))
  private val addressWidth = math.max(1, log2Ceil(slots * maxBeats))

  val io = IO(new Bundle {
    val shDegree = Input(UInt(3.W))
    val activeBeats = Output(UInt(beatWidth.W))
    val maxBeats = Output(UInt(beatWidth.W))

    val writeEn = Input(Bool())
    val writeIndex = Input(UInt(indexWidth.W))
    val writeBeat = Input(UInt(beatWidth.W))
    val writeData = Input(UInt(wordWidth.W))
    val writeLast = Input(Bool())
    val writeAccepted = Output(Bool())
    val writeError = Output(Bool())

    val readEn = Input(Bool())
    val readIndex = Input(UInt(indexWidth.W))
    val readBeat = Input(UInt(beatWidth.W))
    val readData = Output(UInt(wordWidth.W))
    val readValid = Output(Bool())
    val descriptorValid = Output(Bool())
    val storedBeats = Output(UInt(beatWidth.W))
  })

  private def beatsForDegree(degree: UInt): UInt = {
    MuxLookup(degree, 0.U(beatWidth.W))(
      (0 to SAESDescriptorBuffer.MaxSupportedSHDegree).map { shDegree =>
        shDegree.U -> SAESDescriptorBuffer.descriptorBeats(shDegree, beatBytes).U(beatWidth.W)
      },
    )
  }

  val activeBeats = beatsForDegree(io.shDegree)
  val degreeValid = activeBeats =/= 0.U
  io.activeBeats := activeBeats
  io.maxBeats := maxBeats.U(beatWidth.W)

  val descriptorMemory = SyncReadMem(slots * maxBeats, UInt(wordWidth.W))
  val completed = RegInit(VecInit(Seq.fill(slots)(false.B)))
  val nextBeat = RegInit(VecInit(Seq.fill(slots)(0.U(beatWidth.W))))
  val inFlightBeats = RegInit(VecInit(Seq.fill(slots)(0.U(beatWidth.W))))
  val completedBeats = RegInit(VecInit(Seq.fill(slots)(0.U(beatWidth.W))))

  val writeIndexValid = io.writeIndex < slots.U
  val writeNextBeat = WireDefault(0.U(beatWidth.W))
  val writeInFlightBeats = WireDefault(0.U(beatWidth.W))
  for (slot <- 0 until slots) {
    when(io.writeIndex === slot.U) {
      writeNextBeat := nextBeat(slot)
      writeInFlightBeats := inFlightBeats(slot)
    }
  }
  val writeExpectedBeats = Mux(
    writeNextBeat === 0.U,
    activeBeats,
    writeInFlightBeats,
  )
  val writeLastExpected = io.writeBeat === (writeExpectedBeats - 1.U)
  val writeAccepted = io.writeEn && degreeValid && writeIndexValid &&
    (io.writeBeat === writeNextBeat) && (io.writeBeat < writeExpectedBeats) &&
    (io.writeLast === writeLastExpected)
  io.writeAccepted := writeAccepted
  io.writeError := io.writeEn && !writeAccepted

  val writeAddress = (
    io.writeIndex * maxBeats.U + io.writeBeat
  )(addressWidth - 1, 0)
  when(writeAccepted) {
    descriptorMemory.write(writeAddress, io.writeData)
    for (slot <- 0 until slots) {
      when(io.writeIndex === slot.U) {
        when(io.writeBeat === 0.U) {
          completed(slot) := false.B
          completedBeats(slot) := 0.U
        }
        when(writeLastExpected) {
          completed(slot) := true.B
          completedBeats(slot) := writeExpectedBeats
          nextBeat(slot) := 0.U
          inFlightBeats(slot) := 0.U
        }.otherwise {
          nextBeat(slot) := io.writeBeat + 1.U
          inFlightBeats(slot) := writeExpectedBeats
        }
      }
    }
  }

  val readIndexValid = io.readIndex < slots.U
  val selectedComplete = WireDefault(false.B)
  val selectedBeats = WireDefault(0.U(beatWidth.W))
  for (slot <- 0 until slots) {
    when(io.readIndex === slot.U) {
      selectedComplete := completed(slot)
      selectedBeats := completedBeats(slot)
    }
  }
  io.descriptorValid := readIndexValid && selectedComplete
  io.storedBeats := Mux(readIndexValid, selectedBeats, 0.U)

  val readAccepted = io.readEn && readIndexValid && selectedComplete &&
    (io.readBeat < selectedBeats)
  val readAddress = (
    io.readIndex * maxBeats.U + io.readBeat
  )(addressWidth - 1, 0)
  io.readData := descriptorMemory.read(readAddress, readAccepted)
  io.readValid := RegNext(readAccepted, false.B)
}
