package scarf.fsdr

import chisel3._
import chisel3.util._
import scarf.ScarfConfig

/**
 * FSDRCache — Feature-Similarity Depth Reuse Cache (512-entry CAM).
 *
 * Semantic-indexed cache with parallel Hamming distance lookup.
 * Each entry: (valid[1], signature[16], depth[16], pixX[10], pixY[10], lru[8]) = 61 bits.
 * Parallel XOR-popcount for all entries + min-selector tree + threshold comparator.
 * LRU replacement for eviction.
 */

class FSDRCacheEntry(val sigWidth: Int = ScarfConfig.LSHDim) extends Bundle {
  val valid     = Bool()
  val signature = UInt(sigWidth.W)
  val depth     = UInt(ScarfConfig.DataWidth.W)
  val pixelX    = UInt(10.W)
  val pixelY    = UInt(10.W)
  val lruCount  = UInt(8.W)
}

class FSDRCache(
  val numEntries: Int = ScarfConfig.FSDRCacheEntries,
  val sigWidth: Int = ScarfConfig.LSHDim,
) extends Module {
  val io = IO(new Bundle {
    val lookupSig     = Input(UInt(sigWidth.W))
    val lookupEn      = Input(Bool())
    val hammingThresh = Input(UInt(4.W))

    val hit           = Output(Bool())
    val hitDepth      = Output(UInt(ScarfConfig.DataWidth.W))
    val hitHamming    = Output(UInt(5.W))
    val hitPixelX     = Output(UInt(10.W))
    val hitPixelY     = Output(UInt(10.W))

    val insertEn      = Input(Bool())
    val insertSig     = Input(UInt(sigWidth.W))
    val insertDepth   = Input(UInt(ScarfConfig.DataWidth.W))
    val insertPixelX  = Input(UInt(10.W))
    val insertPixelY  = Input(UInt(10.W))

    val occupancy     = Output(UInt(log2Ceil(numEntries + 1).W))
  })

  val entries = RegInit(VecInit(Seq.fill(numEntries)(
    0.U.asTypeOf(new FSDRCacheEntry(sigWidth))
  )))

  val lruTimer = RegInit(0.U(8.W))

  // ── Parallel Hamming distance (combinational) ──
  val hammingDists = Wire(Vec(numEntries, UInt(5.W)))
  val validMask    = Wire(Vec(numEntries, Bool()))
  for (i <- 0 until numEntries) {
    hammingDists(i) := HammingDistance(io.lookupSig, entries(i).signature, sigWidth)
    validMask(i)    := entries(i).valid
  }

  // ── Min-selector tree ──
  val indexedDists = (0 until numEntries).map { i =>
    (hammingDists(i), validMask(i), i.U(log2Ceil(numEntries).W))
  }
  val (bestDist, bestValid, bestIdx) = indexedDists.reduce { (a, b) =>
    val (distA, validA, idxA) = a
    val (distB, validB, idxB) = b
    val aWins = validA && (!validB || distA <= distB)
    (Mux(aWins, distA, distB), validA || validB, Mux(aWins, idxA, idxB))
  }

  // ── Lookup result (registered) ──
  val hitReg   = RegInit(false.B)
  val depthReg = RegInit(0.U(ScarfConfig.DataWidth.W))
  val hamReg   = RegInit(0.U(5.W))
  val pixXReg  = RegInit(0.U(10.W))
  val pixYReg  = RegInit(0.U(10.W))

  when(io.lookupEn) {
    val isHit = bestValid && (bestDist <= io.hammingThresh)
    hitReg := isHit
    hamReg := bestDist
    when(isHit) {
      depthReg := entries(bestIdx).depth
      pixXReg  := entries(bestIdx).pixelX
      pixYReg  := entries(bestIdx).pixelY
      entries(bestIdx).lruCount := lruTimer
      lruTimer := lruTimer + 1.U
    }
  }

  io.hit        := hitReg
  io.hitDepth   := depthReg
  io.hitHamming := hamReg
  io.hitPixelX  := pixXReg
  io.hitPixelY  := pixYReg

  // ── Insert logic (LRU eviction) ──
  when(io.insertEn) {
    val emptySlot = Wire(Valid(UInt(log2Ceil(numEntries).W)))
    emptySlot.valid := false.B
    emptySlot.bits  := 0.U
    for (i <- numEntries - 1 to 0 by -1) {
      when(!entries(i).valid) {
        emptySlot.valid := true.B
        emptySlot.bits  := i.U
      }
    }

    val lruCandidates = (0 until numEntries).map { i =>
      (entries(i).lruCount, entries(i).valid, i.U(log2Ceil(numEntries).W))
    }
    val (_, _, lruIdx) = lruCandidates.reduce { (a, b) =>
      val (cntA, validA, idxA) = a
      val (cntB, validB, idxB) = b
      val aOlder = validA && (!validB || cntA < cntB)
      (Mux(aOlder, cntA, cntB), validA || validB, Mux(aOlder, idxA, idxB))
    }

    val insertIdx = Mux(emptySlot.valid, emptySlot.bits, lruIdx)
    entries(insertIdx).valid     := true.B
    entries(insertIdx).signature := io.insertSig
    entries(insertIdx).depth     := io.insertDepth
    entries(insertIdx).pixelX    := io.insertPixelX
    entries(insertIdx).pixelY    := io.insertPixelY
    entries(insertIdx).lruCount  := lruTimer
    lruTimer := lruTimer + 1.U
  }

  io.occupancy := PopCount(entries.map(_.valid))
}
