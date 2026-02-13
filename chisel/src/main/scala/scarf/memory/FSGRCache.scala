package scarf.memory

import chisel3._
import chisel3.util._
import scarf.ScarfConfig
import scarf.compute.HammingDistance

/**
 * FSGRCache — Semantic-Indexed Cache Table with Hamming Lookup.
 *
 * Corresponds to: fsgr/cache_table.py (CacheTable)
 *
 * Architecture:
 *   - N entries, each storing: {valid, signature[K], depth[16], lru_count[8]}
 *   - Parallel Hamming distance calculation for ALL entries simultaneously
 *   - Min-selector tree to find best match
 *   - Threshold comparator to determine hit/miss
 *   - LRU replacement policy for eviction
 *
 * Hardware cost:
 *   - N × (K XOR gates + popcount tree) for parallel lookup
 *   - log2(N)-level min-selector tree
 *   - N × (K + 16 + 8 + 1) bits = N × 41 bits SRAM (for K=16)
 *   - Total: ~300 LUTs + 5 KB SRAM for N=512
 *
 * Latency: 1 cycle lookup, 1 cycle insert
 *
 * Key design: This is a specialized CAM (Content-Addressable Memory).
 * It is NOT a general cache — the lookup key is a Hamming distance
 * comparison, not an exact address match.
 */

/** Single cache entry. */
class FSGRCacheEntry(val sigWidth: Int = 16) extends Bundle {
  val valid     = Bool()
  val signature = UInt(sigWidth.W)
  val depth     = UInt(ScarfConfig.DataWidth.W)  // Cached depth value (FP16)
  val pixelX    = UInt(10.W)                     // Source pixel X
  val pixelY    = UInt(10.W)                     // Source pixel Y
  val lruCount  = UInt(8.W)                      // LRU counter (lower = older)
}

class FSGRCache(
  val numEntries: Int = 512,
  val sigWidth: Int = 16,
) extends Module {
  val io = IO(new Bundle {
    // Lookup interface
    val lookupSig     = Input(UInt(sigWidth.W))
    val lookupEn      = Input(Bool())
    val hammingThresh = Input(UInt(4.W))         // From ConfigRegs

    // Lookup result (available 1 cycle after lookupEn)
    val hit           = Output(Bool())
    val hitDepth      = Output(UInt(ScarfConfig.DataWidth.W))
    val hitHamming    = Output(UInt(5.W))         // Hamming distance of best match
    val hitPixelX     = Output(UInt(10.W))
    val hitPixelY     = Output(UInt(10.W))

    // Insert interface
    val insertEn      = Input(Bool())
    val insertSig     = Input(UInt(sigWidth.W))
    val insertDepth   = Input(UInt(ScarfConfig.DataWidth.W))
    val insertPixelX  = Input(UInt(10.W))
    val insertPixelY  = Input(UInt(10.W))

    // Status
    val occupancy     = Output(UInt(log2Ceil(numEntries + 1).W))
  })

  // ── Cache storage ──
  val entries = RegInit(VecInit(Seq.fill(numEntries)(
    0.U.asTypeOf(new FSGRCacheEntry(sigWidth))
  )))

  // Global LRU timestamp
  val lruTimer = RegInit(0.U(8.W))

  // ── Parallel Hamming distance calculation (combinational) ──
  val hammingDists = Wire(Vec(numEntries, UInt(5.W)))
  val validMask    = Wire(Vec(numEntries, Bool()))

  for (i <- 0 until numEntries) {
    hammingDists(i) := HammingDistance(io.lookupSig, entries(i).signature, sigWidth)
    validMask(i)    := entries(i).valid
  }

  // ── Min-selector: find entry with minimum Hamming distance ──
  // Using tree reduction for O(log N) depth
  val indexedDists = (0 until numEntries).map { i =>
    (hammingDists(i), validMask(i), i.U(log2Ceil(numEntries).W))
  }

  // Tree reduce: find (minDist, isValid, index)
  val (bestDist, bestValid, bestIdx) = indexedDists.reduce { (a, b) =>
    val (distA, validA, idxA) = a
    val (distB, validB, idxB) = b
    // Prefer valid entries; among valid, prefer lower Hamming distance
    val aWins = validA && (!validB || distA <= distB)
    (Mux(aWins, distA, distB),
     validA || validB,
     Mux(aWins, idxA, idxB))
  }

  // ── Lookup result (registered for timing) ──
  val hitReg     = RegInit(false.B)
  val depthReg   = RegInit(0.U(ScarfConfig.DataWidth.W))
  val hamReg     = RegInit(0.U(5.W))
  val pixXReg    = RegInit(0.U(10.W))
  val pixYReg    = RegInit(0.U(10.W))

  when(io.lookupEn) {
    val isHit = bestValid && (bestDist <= io.hammingThresh)
    hitReg   := isHit
    hamReg   := bestDist
    when(isHit) {
      depthReg := entries(bestIdx).depth
      pixXReg  := entries(bestIdx).pixelX
      pixYReg  := entries(bestIdx).pixelY
      // Update LRU
      entries(bestIdx).lruCount := lruTimer
      lruTimer := lruTimer + 1.U
    }
  }

  io.hit        := hitReg
  io.hitDepth   := depthReg
  io.hitHamming := hamReg
  io.hitPixelX  := pixXReg
  io.hitPixelY  := pixYReg

  // ── Insert logic ──
  when(io.insertEn) {
    // Find insertion slot: first empty, or LRU eviction
    val emptySlot = Wire(Valid(UInt(log2Ceil(numEntries).W)))
    emptySlot.valid := false.B
    emptySlot.bits  := 0.U

    // Priority: first empty slot
    for (i <- numEntries - 1 to 0 by -1) {
      when(!entries(i).valid) {
        emptySlot.valid := true.B
        emptySlot.bits  := i.U
      }
    }

    // LRU: find entry with smallest lruCount (oldest access)
    // Use tree reduction to avoid combinational loop
    val lruCandidates = (0 until numEntries).map { i =>
      (entries(i).lruCount, entries(i).valid, i.U(log2Ceil(numEntries).W))
    }
    val (_, _, lruIdx) = lruCandidates.reduce { (a, b) =>
      val (cntA, validA, idxA) = a
      val (cntB, validB, idxB) = b
      // Among valid entries, pick the one with smaller lruCount (older)
      val aOlder = validA && (!validB || cntA < cntB)
      (Mux(aOlder, cntA, cntB), validA || validB, Mux(aOlder, idxA, idxB))
    }

    val insertIdx = Mux(emptySlot.valid, emptySlot.bits, lruIdx)

    // Write new entry
    entries(insertIdx).valid     := true.B
    entries(insertIdx).signature := io.insertSig
    entries(insertIdx).depth     := io.insertDepth
    entries(insertIdx).pixelX    := io.insertPixelX
    entries(insertIdx).pixelY    := io.insertPixelY
    entries(insertIdx).lruCount  := lruTimer
    lruTimer := lruTimer + 1.U
  }

  // ── Occupancy counter ──
  io.occupancy := PopCount(entries.map(_.valid))
}
