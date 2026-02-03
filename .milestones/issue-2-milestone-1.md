# Milestone 1 for Issue #2

**Branch:** issue-2  
**Created:** 2026-02-03  
**LOC Implemented:** ~2906 lines  
**Test Status:** 0/0 tests (tests not yet created - Phase 1 documentation only)

---

## Work Completed

### Phase 1: Documentation (Complete)

**Step 1: FSDR Architecture Documentation** ✓ (797 LOC)
- `docs/fsdr-architecture.md` - Complete FSDR design documentation
  - Cache entry structure (70-bit layout)
  - LSH signature generation algorithm
  - Three-level correction strategy
  - Hardware mapping with Verilog examples
  - Integration with DSU

**Step 2: DSU Architecture Documentation** ✓ (592 LOC)
- `docs/dsu-architecture.md` - Complete DSU design documentation
  - Depth sampling and projection
  - Cost volume computation
  - Softmax aggregation
  - Hardware mapping

**Step 3: GGU Architecture Documentation** ✓ (523 LOC)
- `docs/ggu-architecture.md` - Complete GGU design documentation
  - Position calculation
  - Covariance building
  - SH rotation
  - Hardware mapping

**Step 4: Hardware Resource Summary** ✓ (309 LOC)
- `docs/hardware-resource-summary.md` - Complete resource analysis
  - Component breakdown (FSDR, DSU, GGU, SAES)
  - Total: 11,750 LUTs, 272 DSPs, 6KB SRAM
  - FPGA/ASIC feasibility analysis
  - Performance comparison vs baseline

**Step 5: Multi-Model Integration Guide** ✓ (685 LOC)
- `docs/multi-model-integration.md` - Adapter pattern documentation
  - BaseAdapter interface
  - TransplatAdapter, MVSplatAdapter, DepthSplatAdapter
  - Integration examples
  - Testing strategy

---

## Work Remaining

### Phase 2: Test Cases (~3312 LOC estimated)

- Step 6: FSDR Test Infrastructure & Tests (~1310 LOC)
  - `tests/fsdr/conftest.py` - Test fixtures
  - `tests/fsdr/test_lsh_hasher.py` - LSH hasher tests
  - `tests/fsdr/test_cache_table.py` - Cache table tests
  - `tests/fsdr/test_depth_corrector.py` - Depth corrector tests
  - `tests/fsdr/test_light_verifier.py` - Light verifier tests
  - `tests/fsdr/test_fsdr_processor.py` - Main processor tests
  - `tests/fsdr/test_integration.py` - Integration tests

- Step 7: DSU Tests (~651 LOC)
  - `tests/dsu/conftest.py`
  - `tests/dsu/test_cost_volume.py`
  - `tests/dsu/test_depth_sampler.py`
  - `tests/dsu/test_dsu_processor.py`

- Step 8: GGU Tests (~431 LOC)
  - `tests/ggu/conftest.py`
  - `tests/ggu/test_covariance_builder.py`
  - `tests/ggu/test_ggu_processor.py`

- Step 9-11: Adapter and Integration Tests (~601 LOC)

### Phase 3: Implementation (~3455 LOC estimated)

- Steps 12-19: FSDR implementation
- Steps 20-25: DSU implementation
- Steps 26-31: GGU implementation
- Steps 32-38: Adapters and integration

---

## Next File Changes (Estimated for Next Milestone)

**Phase 2 Test Suite (targeting ~1600 LOC for Milestone 2)**

- `tests/fsdr/__init__.py` (~1 LOC)
- `tests/fsdr/conftest.py` (~180 LOC)
- `tests/fsdr/test_lsh_hasher.py` (~200 LOC)
- `tests/fsdr/test_cache_table.py` (~250 LOC)
- `tests/fsdr/test_depth_corrector.py` (~220 LOC)
- `tests/fsdr/test_light_verifier.py` (~180 LOC)
- `tests/fsdr/test_fsdr_processor.py` (~280 LOC)

**Total estimated for next milestone:** ~1310 LOC

---

## Test Status

**No tests yet** - This milestone covers Phase 1 (Documentation only).

Tests will be created in Milestone 2 following Design-first TDD:
- Documentation ✓ (this milestone)
- Tests → Next milestone
- Implementation → Future milestones

---

## Technical Notes

1. **FSDR Design Decisions:**
   - 128 cache entries chosen for best cost/benefit (75% hit rate vs 1.1KB SRAM)
   - 16-bit LSH signature balances precision vs hardware cost
   - Three-level correction provides graceful degradation

2. **DSU Design Decisions:**
   - 4× parallel DSU units for balanced throughput/area
   - Bilinear sampling with boundary clamping
   - Softmax with temperature control for model compatibility

3. **Multi-Model Support:**
   - Adapter pattern isolates model differences
   - All adapters share projection logic
   - Model-specific: softmax direction, depth candidates, feature dim

---

## Progress Summary

| Phase | Status | LOC |
|-------|--------|-----|
| Phase 1: Documentation | ✓ Complete | 2906 |
| Phase 2: Test Cases | Pending | ~3312 (est) |
| Phase 3: Implementation | Pending | ~3455 (est) |
| **Total** | **In Progress** | **~9673** |

---

**Next Action:** Create FSDR test suite (Step 6-7)
