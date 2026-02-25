# Direction: Post-Audit Action Plan (2026-02-24)

Derived from a 6-agent comprehensive audit of the entire codebase (81K lines,
1137 tests, 114 commits). Each item has a concrete scope, affected files, and
estimated effort. Items are grouped by priority tier.

**Guiding principle**: fix correctness issues first, then improve observability,
then clean up infrastructure. Don't refactor for aesthetics -- refactor where
confusion or bugs are likely.

---

## Tier 1: Silent Failures (fix these first)

These can cause wrong emulation results without any error message.

### 1.1 Log unknown CDO opcodes with opcode name
**Files**: `src/device/state.rs:180-182`
**Problem**: Unknown CDO opcodes increment a counter but don't log *which*
opcode was dropped. ~30 CDO opcodes fall through silently. Most are irrelevant
to NPU emulation (PM commands, SSIT sync, etc.), but if a new xclbin uses
one that matters, we'll never know.
**Fix**: Change the catch-all to `log::warn!("CDO opcode {:#x} not implemented,
skipping", op)`. Add a summary at end of CDO application: "N commands skipped
(opcodes: ...)".
**Effort**: Small (30 min)

### 1.2 Handle or reject NPU opcodes 5 and 6
**Files**: `src/npu/parser.rs:320-328`, `src/npu/mod.rs`
**Problem**: `ConfigShimDmaBd` (5) and `ConfigShimDmaDmaBufBd` (6) are defined
but never matched in the parser. They silently discard 8 bytes. If any xclbin
uses them, shim DMA won't be configured.
**Fix**: First, check whether mlir-aie ever emits these opcodes (grep
mlir-aie source for the opcode numbers). If never emitted, add explicit
`log::warn!` stubs. If emitted, implement them -- they're just BD register
writes like the existing DdrPatch handler.
**Effort**: Small-Medium (1-2 hours, depending on whether implementation needed)

### 1.3 Validate DDR patch payload bounds
**Files**: `src/npu/parser.rs:235-250`
**Problem**: DdrPatch parsing reads hardcoded offsets (16..20, 24..28, 32..36)
without verifying the payload is long enough. Malformed input silently becomes
zeros.
**Fix**: Return `Err` if `payload.len() < expected_size` before any reads.
**Effort**: Small (15 min)

### 1.4 CDO DmaWrite 64-bit address handling
**Files**: `src/parser/cdo.rs:512-531`
**Problem**: DmaWrite warns about non-zero addr_hi but uses only low 32 bits.
DDR above 4GB would silently go to wrong address.
**Fix**: Either combine addr_hi:addr_lo into u64 and pass through, or error
if addr_hi != 0 (since our emulated address space is 32-bit anyway).
**Effort**: Small (30 min)

---

## Tier 2: Observability & Trace Consistency

The emulator produces correct results but some output paths are incomplete.

### 2.1 Add port events to internal EventLog (Perfetto)
**Files**: `src/interpreter/engine/coordinator.rs:494-573`
**Problem**: PORT_IDLE/RUNNING/STALLED/TLAST events go to hardware trace units
but NOT to the internal EventLog used for Perfetto JSON export. This means
Perfetto traces are missing all port activity.
**Fix**: Add `EventType` variants for port events (PortIdle, PortRunning,
PortStalled, PortTlast -- each with port index). Record them in the trace_log
alongside the hardware trace unit notification.
**Effort**: Medium (1-2 hours)

### 2.2 CDO checksum: use log::warn instead of eprintln
**Files**: `src/parser/cdo.rs:316-325`
**Problem**: Checksum mismatch uses `eprintln!` instead of the logging system.
**Fix**: Replace with `log::warn!`.
**Effort**: Tiny (5 min)

---

## Tier 3: Hardcoded Architecture Knowledge

These work perfectly for NPU1/AIE2 but will break for AIE2P and are
maintenance liabilities.

### 3.1 Extract stream switch port constants
**Files**: `src/device/array.rs` (throughout), `src/device/aie2_spec.rs`
**Problem**: Port indices (8, 12-17, 22, 23, 24) are scattered as magic numbers
throughout array.rs routing functions. Comments reference AM025 but values
aren't derived from any data source.
**Fix**: Create a `PortLayout` struct in `aie2_spec.rs` that maps bundle names
to port index ranges (e.g., `north_slaves: 8..14`, `trace_slave: 23`,
`mem_trace_slave: 24`). Build it from the device model JSON if possible,
otherwise from named constants. Replace all magic numbers with lookups.
**Effort**: Medium-Large (half day)

### 3.2 Data-drive DMA channel-to-port mapping
**Files**: `src/device/array.rs:818-864, 911-927`
**Problem**: Channel-to-port formulas are hardcoded per tile type with inline
arithmetic (e.g., `if channel >= 6 { channel - 6 } else { channel }`).
**Fix**: Add a mapping table to `PortLayout` or derive from aie-rt. Replace
inline formulas with table lookups.
**Effort**: Medium (2-3 hours, shares work with 3.1)

### 3.3 Data-drive shim DMA register offsets
**Files**: `src/npu/executor.rs:460-463`
**Problem**: Queue register addresses (0x1D204 etc.) and BD base (0x1D000)
hardcoded.
**Fix**: Derive from regdb (already loaded at startup). The register names
are known; look them up by name instead of hardcoding offsets.
**Effort**: Small-Medium (1-2 hours)

### 3.4 Expand shim mux arrays beyond 2 channels
**Files**: `src/device/tile.rs:625-633`
**Problem**: `[Option<usize>; 2]` limits shim to 2 DMA channels. NPU1 has 2,
but the code should handle the device model's actual channel count.
**Fix**: Use `Vec<Option<usize>>` or `SmallVec` sized from device model.
**Effort**: Small (1 hour)

---

## Tier 4: Interpreter Completeness

Filling gaps in instruction execution coverage.

### 4.1 Implement Rotl/Rotr/Bswap in semantic dispatch
**Files**: `src/interpreter/execute/semantic.rs`
**Problem**: Three SemanticOp variants return false (fall through). These are
simple bitwise operations (~5 lines each).
**Fix**: Implement directly in the match arms.
**Effort**: Tiny (15 min)

### 4.2 Document intentionally-delegated SemanticOps
**Files**: `src/interpreter/execute/semantic.rs`
**Problem**: Call, Ret, Done, LockAcquire, LockRelease, Intrinsic, Load, Store,
Br, BrCond all return false. Some are correct (Load/Store/Br need tile access),
others are ambiguous.
**Fix**: Add comments explaining which are intentionally delegated vs genuinely
unimplemented. Group the match arms with section comments.
**Effort**: Small (30 min)

### 4.3 Resolve ScalarAlu / semantic dispatch duplication
**Files**: `src/interpreter/execute/scalar.rs`, `semantic.rs`, `cycle_accurate.rs`
**Problem**: Both ScalarAlu and semantic dispatch handle the same operations.
The fallback chain (try semantic first, then ScalarAlu) works but creates risk
of divergence.
**Fix**: Audit which operations are fully covered by semantic dispatch. For
those, remove the ScalarAlu fallback handler (or mark it unreachable). Leave
ScalarAlu for operations not yet migrated.
**Effort**: Medium (2-3 hours, needs careful verification)

### 4.4 Validate SRS rounding against hardware
**Files**: `src/interpreter/execute/vector_srs.rs`
**Problem**: 10 rounding modes implemented but Conv-Even/Conv-Odd may have
edge case bugs. No hardware reference output to validate against.
**Fix**: Build a small test that runs SRS operations on real NPU (via XRT)
and compares emulator output. Use the existing hardware comparison
infrastructure.
**Effort**: Medium (half day, requires hardware test setup)

---

## Tier 5: Dead Infrastructure Cleanup

Code that exists but doesn't run. Not harmful, but confusing.

### 5.1 Mark timing submodules as future work
**Files**: `src/interpreter/timing/` (deadlock.rs, barrier.rs, sync.rs,
arbitration.rs, slots.rs)
**Problem**: ~2000 lines of timing infrastructure built but never instantiated
or called. Deadlock detection, barrier tracking, arbitration modeling -- all
reasonable future directions, but currently dead.
**Fix**: Add a module-level doc comment to each: "// Future work: not yet
wired into the coordinator pipeline." Don't delete -- these are
well-designed and will be needed for true cycle-accuracy.
**Effort**: Small (30 min)

### 5.2 Document trace unit auto-start workaround
**Files**: `src/device/trace_unit.rs:174-181`
**Problem**: Real hardware starts tracing on a broadcast event. We auto-start
on configuration because we don't model the broadcast event system. Tests can
pass that would fail on real hardware.
**Fix**: Add a more prominent comment explaining the deviation and its
implications. Add a `// FIXME(broadcast-events)` marker so it's searchable.
Consider logging a warning when auto-starting.
**Effort**: Small (15 min)

### 5.3 Move diagnostic test to examples
**Files**: `src/interpreter/test_runner.rs:914-1195`
**Problem**: `test_add_one_diagnostic_trace()` is a 282-line debugging artifact.
It runs in the test suite but isn't a real regression test.
**Fix**: Move to `examples/diagnostic_trace.rs` where it can be run on demand
without cluttering the test suite.
**Effort**: Small (30 min)

### 5.4 Decide on build_progress.rs
**Files**: `src/build_progress.rs` (1058 lines)
**Problem**: Complete parallel build progress UI, never called by anything.
**Fix**: Either integrate into lit_runner (the natural consumer) or move to
a standalone tool. If neither is imminent, add a module doc comment explaining
its purpose and intended consumer.
**Effort**: Small (comment) or Medium (integrate)

---

## Tier 6: Test Infrastructure Improvements

Not blocking correctness, but important for long-term velocity.

### 6.1 Refactor run_mlir_aie_tests.rs main loop
**Files**: `examples/run_mlir_aie_tests.rs`
**Problem**: 906-line monolithic main() with tangled control flow.
**Fix**: Extract `run_single_test_matrix()`, `run_hw_validation()`, and
`discover_and_filter()` as helper functions. Keep the example file as the
orchestrator but move logic into `src/testing/` modules.
**Effort**: Medium-Large (half day)

### 6.2 Add intermediate assertions to large tests
**Files**: `src/interpreter/test_runner.rs` (6 tests, 100-200 lines each)
**Problem**: Large tests mix setup/execution/validation without phase breaks.
Failures are hard to diagnose.
**Fix**: Add "phase complete" assertions after setup, after execution, and
before final validation. Example: assert DMA has data before checking values.
**Effort**: Medium (2-3 hours across 6 tests)

### 6.3 Add unit tests for under-covered subsystems
**Priority order**:
1. Stream packet routing (end-to-end, not just local)
2. Multi-tile lock synchronization
3. MemTile DMA
4. CDO application correctness (not just parsing)
**Effort**: Large (ongoing, one test suite at a time)

### 6.4 Unify or deprecate duplicate test runners
**Files**: `examples/run_mlir_aie_tests.rs`, `src/bin/lit_runner.rs`
**Problem**: Two runners with divergent features and discovery paths.
**Fix**: Decide which is primary. If lit_runner, migrate features from
run_mlir_aie_tests. If run_mlir_aie_tests, integrate lit discovery.
**Effort**: Large (full day)

---

## Tier 7: Minor Cleanups (do opportunistically)

These are not worth a dedicated session, but fix them when touching nearby code.

| Item | File | Fix |
|------|------|-----|
| `mem_port_running_hw_id()` dead | trace/mod.rs:606 | Remove or #[allow(dead_code)] with comment |
| ChannelStats never updated | dma/engine.rs:178 | Either populate or remove the struct |
| HostMemory statistics unused | host_memory.rs:150 | Wire into GUI or remove |
| StreamPort.route_to write-only | stream_switch.rs:136 | Use for debugging display or remove |
| tlast_flags parallel Vec | stream_switch.rs:113 | Refactor to Vec<(u32, bool)> |
| ControlPacketState Vec for 4 items | tile.rs:537 | Change to [u32; 4] |
| LockResult::WouldUnderflow naming | tile.rs:66 | Rename to PreconditionNotMet |
| Deprecated coordinator aliases | coordinator.rs:171 | Remove in next breaking change |
| FFI error API stubbed | ffi/mod.rs:553 | Implement with thread_local! when FFI is priority |
| ELF lifetime transmute | parser/elf.rs:170 | Refactor to owned data when parser is revisited |
| Lock convenience methods unused | tile.rs:129-142 | Verify and remove if truly dead |
| native_hw.rs orphaned | testing/native_hw.rs | Decide: integrate or remove |

---

## Not In Scope (intentional deferrals)

These were noted by the audit but are not actionable now:

- **Vector matmul pipeline** (~5% complete): Requires mulmac.py study, config
  word decoding, accumulator pipeline. Large effort, blocked on understanding
  the hardware semantics. Tracked in docs/plan-100-percent-compatibility.md.
- **Vector permute MAC modes**: Deferred until matmul pipeline exists.
- **Broadcast event system**: Would fix trace auto-start (5.2) properly.
  Significant effort -- model event broadcast network, wire into all tile
  modules. Not needed until trace start/stop conditions matter.
- **Full hazard/memory timing**: The disabled hazard checking in
  cycle_accurate.rs is intentional. Re-enable only when correctness is
  perfect and we're chasing cycle-accuracy.
- **AIE2P support**: All Tier 3 items prepare for this, but actual AIE2P
  implementation is future work.

---

## Suggested Work Order

For a single session:
1. **Tier 1 complete** (1.1 through 1.4) -- ~2 hours, high impact
2. **Tier 2.1** (port events in EventLog) -- ~1-2 hours
3. **Tier 4.1 + 4.2** (quick SemanticOp wins) -- ~45 min
4. **Tier 5.1 + 5.2 + 5.3** (documentation/comments) -- ~1 hour
5. **Tier 2.2** (eprintln fix) -- 5 min

That covers all the "silent failure" and "confusion" issues in one pass.

For subsequent sessions:
- Tier 3 (data-drive port mapping) -- standalone refactoring session
- Tier 6 (test infrastructure) -- standalone cleanup session
- Tier 4.3 + 4.4 (scalar dispatch + SRS validation) -- needs care
