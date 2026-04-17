# Direction: Post-Audit Action Plan (2026-02-24)

Derived from a 6-agent comprehensive audit of the entire codebase (81K lines,
1137 tests, 114 commits). Each item has a concrete scope, affected files, and
estimated effort. Items are grouped by priority tier.

**Guiding principle**: fix correctness issues first, then improve observability,
then clean up infrastructure. Don't refactor for aesthetics -- refactor where
confusion or bugs are likely.

**Status as of 2026-02-24 end-of-day**: All Tier 1-5 items complete. Tier 3
data-driven constants extended further with E/W direction ranges and
TileParams DMA channel counts. 1155 tests passing. Next focus: Chess
compiler compatibility.

---

## Tier 1: Silent Failures -- ALL DONE

| Item | Status | Commit |
|------|--------|--------|
| 1.1 Log unknown CDO opcodes with opcode name | DONE | 999f81d |
| 1.2 Handle or reject NPU opcodes 5 and 6 | DONE | 999f81d |
| 1.3 Validate DDR patch payload bounds | DONE | 999f81d |
| 1.4 CDO DmaWrite 64-bit address handling | DONE | 999f81d |

---

## Tier 2: Observability & Trace Consistency -- ALL DONE

| Item | Status | Commit |
|------|--------|--------|
| 2.1 Add port events to internal EventLog (Perfetto) | DONE | 518af8b, 051e50c |
| 2.2 CDO checksum: use log::warn instead of eprintln | DONE | 999f81d |

---

## Tier 3: Hardcoded Architecture Knowledge -- ALL DONE

| Item | Status | Commit |
|------|--------|--------|
| 3.1 Extract stream switch port constants | DONE | 999f81d (DMA/TRACE), then E/W/S ranges added later |
| 3.2 Data-drive DMA channel-to-port mapping | DONE | 999f81d (named constants replace inline arithmetic) |
| 3.3 Data-drive shim DMA register offsets | DONE | 999f81d (shim_bd_base/shim_channel_base from regdb) |
| 3.4 Expand shim mux arrays beyond 2 channels | DONE | 999f81d (Vec from DMA constants), then TileParams.dma_s2mm/mm2s_channels |

---

## Tier 4: Interpreter Completeness -- MOSTLY DONE

| Item | Status | Commit |
|------|--------|--------|
| 4.1 Implement Rotl/Rotr/Bswap in semantic dispatch | DONE | 999f81d |
| 4.2 Document intentionally-delegated SemanticOps | DONE | 999f81d |
| 4.3 Resolve ScalarAlu / semantic dispatch duplication | PARTIAL | ScalarAlu cleaned up; full dedup deferred |
| 4.4 Validate SRS rounding against hardware | DONE | e7841b0 (golden reference tests) |

### 4.3 details

ScalarAlu has been cleaned up but both paths still exist. The semantic
dispatch chain runs first; ScalarAlu catches anything it misses. The risk
of divergence is low since semantic handles all arithmetic/bitwise/comparison
ops. Full dedup would require verifying every ScalarAlu handler is redundant
with its semantic equivalent, which is safe to defer.

---

## Tier 5: Dead Infrastructure Cleanup -- ALL DONE

| Item | Status | Commit |
|------|--------|--------|
| 5.1 Mark timing submodules as future work | DONE | 999f81d |
| 5.2 Document trace unit auto-start workaround | DONE | 999f81d (FIXME(broadcast-events) added) |
| 5.3 Move diagnostic test to examples | DONE | 999f81d (282-line test removed) |
| 5.4 Decide on build_progress.rs | DONE | Integrated into npu-test binary (613f9c3) |

---

## Tier 6: Test Infrastructure Improvements

| Item | Status |
|------|--------|
| 6.1 Refactor run_mlir_aie_tests.rs main loop | DONE (runner unification, 613f9c3) |
| 6.2 Add intermediate assertions to large tests | DONE (already present) |
| 6.3 Add unit tests for under-covered subsystems | ONGOING |
| 6.4 Unify or deprecate duplicate test runners | DONE (npu-test binary, 613f9c3) |

### 6.3 priority order (unchanged)
1. Stream packet routing (end-to-end, not just local)
2. Multi-tile lock synchronization
3. MemTile DMA
4. CDO application correctness (not just parsing)

---

## Tier 7: Minor Cleanups (do opportunistically)

| Item | File | Status |
|------|------|--------|
| `mem_port_running_hw_id()` dead | trace/mod.rs | DONE |
| ChannelStats never updated | dma/engine.rs | RESOLVED (audit was wrong) |
| HostMemory statistics unused | host_memory.rs | DONE |
| StreamPort.route_to write-only | stream_switch.rs | DEFERRED (useful for GUI) |
| tlast_flags parallel Vec | stream_switch.rs | DEFERRED (low ROI) |
| ControlPacketState Vec for 4 items | tile.rs | DONE |
| LockResult::WouldUnderflow naming | tile.rs | DONE |
| Deprecated coordinator aliases | coordinator.rs | DONE |
| FFI error API stubbed | ffi/mod.rs | DEFERRED |
| ELF lifetime transmute | parser/elf.rs | DEFERRED |
| Lock convenience methods unused | tile.rs | DONE |
| native_hw.rs orphaned | testing/native_hw.rs | RESOLVED (audit was wrong) |

---

## Not In Scope (intentional deferrals)

These were noted by the audit but are not actionable now:

- **Vector matmul pipeline** (~70% now, was ~5%): Dense matmul implemented
  for multiple dtypes. MAC permutation modes (~50%) and sparse format
  metadata remain. Tracked in docs/plan-100-percent-compatibility.md.
- **Vector permute MAC modes**: Framework ready, ~50% of modes implemented.
- **Broadcast event system**: Would fix trace auto-start (5.2) properly.
  Significant effort. Not needed until trace start/stop conditions matter.
- **Full hazard/memory timing**: The disabled hazard checking in
  cycle_accurate.rs is intentional. Re-enable only when correctness is
  perfect and we're chasing cycle-accuracy.
- **AIE2P support**: All Tier 3 items prepare for this, but actual AIE2P
  implementation is future work.

---

## Next Priority: Chess Compiler Compatibility

With the audit complete, the highest-impact remaining work is ensuring
Chess-compiled binaries run correctly on the emulator. Chess exercises
different instruction encodings, NOP patterns, and loop constructs
(JNZD) than Peano. Known issues:

- ~6 Chess tests hit unknown opcode 0x0000 (scalar NOP encoding?)
- Chess uses JNZD as loop counter (fixed 2026-02-19, f4a9fbf)
- Chess section layout differs from Peano (orphan sections)
- Chess intrinsic usage may exercise untested vector paths
