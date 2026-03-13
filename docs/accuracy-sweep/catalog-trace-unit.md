# Trace Unit -- Divergence Catalog

Audit of `src/device/trace_unit.rs` and `src/trace/mod.rs` against aie-rt.

## [TRACE-1] Single1/Single2 event slot bit position was wrong

- **Severity**: CRITICAL
- **Our behavior**: Single1 encoded slot in bits [6:4] of byte0 (`slot << 4`),
  Single2 did the same. This produced correct decoding only for slot 0
  (by coincidence). Slots 1-7 decoded to wrong event numbers.
- **aie-rt behavior**: N/A (aie-rt does not encode trace bytes; the hardware
  does). Reference is mlir-aie `python/utils/trace/utils.py` decode logic
  which expects slot in bits [4:2] of byte0 for both formats.
- **Impact**: Every trace packet containing a Single1 or Single2 event for
  slots 1-7 would produce incorrect event IDs when decoded by mlir-aie's
  parse.py. Affects all bridge tests with trace sweep that use multi-slot
  configurations.
- **Suggested fix**: Change `slot << 4` to `slot << 2` in both encoders.
- **Fixed in-place**: YES. Encoding corrected, 4 new tests added (including
  exhaustive round-trip for all 8 slots x 12 deltas = 96 encode/decode pairs).

## [TRACE-2] No read_register / Trace_Status not readable

- **Severity**: LOW
- **Our behavior**: TraceUnit has `write_register()` but no `read_register()`.
  Trace_Status (offset +0x08) is not readable. The running state, mode, and
  configuration are tracked internally but cannot be read back via register
  interface.
- **aie-rt behavior**: `XAie_TraceGetState()` reads Trace_Status at
  `StatusRegOff` (xaie_trace.c:432). `XAie_TraceGetMode()` reads
  `StatusRegOff` for mode bits (xaie_trace.c:498). Both are register reads.
  (xaiemlgbl_params.h: STATE at bits [9:8], MODE at bits [2:0])
- **Impact**: No current test relies on reading trace status. CDO flows only
  write trace configuration, never read it back. Only impacts hypothetical
  future MMIO-read paths.
- **Suggested fix**: Add `read_register()` to TraceUnit returning the status
  register value synthesized from `self.state` and `self.mode`.
- **Fixed in-place**: no (LOW priority, no test path exercises it)

## [TRACE-3] Memory module trace has no mode field but we parse one

- **Severity**: LOW (cosmetic)
- **Our behavior**: `write_register(0x00, value)` always extracts mode from
  bits [1:0], even for memory module where the register mask is 0x7F7F0000
  (no mode bits).
- **aie-rt behavior**: Memory module TraceMod has
  `ModeConfig = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE}`.
  The `XAie_TraceModeConfig()` function rejects mode writes to memory
  modules. `XAie_TraceControlConfig()` sets `Mode = 0` when ModeConfig
  is unavailable (xaie_trace.c:578).
- **Impact**: None. Memory module Trace_Control0 bits [1:0] are reserved
  and always 0. We read 0 and store EventTime (mode=0), which is correct.
- **Suggested fix**: Could add a `has_mode_field` flag to distinguish
  core vs memory modules, but not functionally necessary.
- **Fixed in-place**: no (no functional impact)

## [TRACE-4] 8-bit event extraction vs 7-bit register fields

- **Severity**: LOW (cosmetic)
- **Our behavior**: Event fields extracted with `& 0xFF` (8 bits) for all
  modules uniformly.
- **aie-rt behavior**: Core module and memory module use 7-bit fields
  (mask=0x7F). MemTile uses 8-bit fields (mask=0xFF, per
  XAIEMLGBL_MEM_TILE_MODULE_TRACE_CONTROL0_TRACE_START_EVENT_WIDTH=8).
- **Impact**: None. The extra bit (bit 7) is reserved and always 0 in
  core/memory modules. For MemTile, 8-bit extraction is actually required
  since events go up to 160 (USER_EVENT_1).
- **Suggested fix**: Could mask per-module, but unnecessary.
- **Fixed in-place**: no (actually correct for MemTile, benign for others)

## [TRACE-5] Performance counter events not generated

- **Severity**: MEDIUM
- **Our behavior**: Perf counter event IDs (PERF_CNT_0=5 through
  PERF_CNT_3=8 for core, PERF_CNT_0=5 and PERF_CNT_1=6 for memory) can
  be configured in trace event slots, but the emulator never fires these
  events because performance counters are not implemented.
- **aie-rt behavior**: aie-rt configures perf counters via a separate API
  (`XAie_PerfCounterControlSet`, `XAie_PerfCounterEventValueSet`). When a
  counter reaches its configured value, it fires a PERF_CNT_N event that
  can trigger trace start/stop or appear in trace event slots.
- **Impact**: Trace configurations that use perf counter events as
  start/stop triggers will not work. Some advanced tracing scenarios use
  perf counters for windowed trace capture.
- **Suggested fix**: Implement basic perf counter infrastructure: counter
  registers, increment-on-event logic, and event generation when threshold
  reached.
- **Fixed in-place**: no (requires new subsystem, not a trace unit fix)

## [TRACE-6] Combo events not generated

- **Severity**: MEDIUM
- **Our behavior**: Combo event IDs (COMBO_EVENT_0=9 through
  COMBO_EVENT_3=12) can be configured in trace event slots but are never
  generated. The combo event configuration registers are not implemented.
- **aie-rt behavior**: Combo events combine multiple input events using
  logical operations (AND, OR, etc.) configured via
  `XAie_EventComboConfig()`. The hardware generates derived events.
- **Impact**: Trace configurations that use combo events as start/stop
  triggers or in event slots will not produce expected output. Some
  mlir-aie test flows use combo events.
- **Suggested fix**: Implement combo event configuration registers and
  evaluation logic (4 combo events per module, each combining 2 input
  events with a configurable operation).
- **Fixed in-place**: no (separate subsystem)

## [TRACE-7] Multiple0/Multiple1 encoding not implemented

- **Severity**: LOW
- **Our behavior**: Only Single0/Single1/Single2 and Start encodings are
  implemented. Multiple0 and Multiple1 formats (which encode multiple
  simultaneous events in one compact encoding) are not generated.
- **aie-rt behavior**: N/A (encoding is hardware's job). mlir-aie's
  decode logic handles Multiple0 (0b1100XXXX, 2 bytes) and Multiple1
  (0b110100XX, 3 bytes) formats.
- **Impact**: When multiple traced events fire in the same cycle or in
  nearby cycles, the hardware may use Multiple encoding for efficiency.
  Our emulator always uses Single encoding, which is functionally correct
  but produces slightly larger trace output. Decoded events are identical.
- **Suggested fix**: Could implement Multiple encoding as an optimization
  when multiple slots fire in the same cycle. Not required for correctness.
- **Fixed in-place**: no (optimization, not correctness)

## [TRACE-8] EventPC and Execution trace modes not implemented

- **Severity**: LOW
- **Our behavior**: Mode values are parsed and stored, but only EventTime
  mode (mode=0) actually produces correct output. EventPC (mode=1) should
  include the program counter in trace packets instead of cycle deltas.
  Execution (mode=2) should produce per-instruction execution traces.
- **aie-rt behavior**: `XAie_TraceModeConfig()` sets the mode in
  Trace_Control0 bits [1:0]. Hardware behavior differs by mode.
- **Impact**: None for current workloads. All bridge tests and trace-inject
  use EventTime mode exclusively.
- **Suggested fix**: Implement EventPC mode (record PC values instead of
  cycle deltas). Execution mode is a lower priority.
- **Fixed in-place**: no (no test exercises non-EventTime modes)
