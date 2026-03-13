# Trace Unit -- Divergence Catalog

Audit of `src/device/trace_unit.rs` against aie-rt and mlir-aie.
Audited: 2026-03-12 (re-run)

## [TRACE-1] Single1/Single2 event slot bit position was wrong (FIXED previously)

- **Severity**: CRITICAL
- **Status**: FIXED (prior session)
- **Details**: Single1/Single2 encoded slot in bits [6:4] of byte0 (`slot << 4`).
  Correct position is bits [4:2] (`slot << 2`), per mlir-aie decode logic.

## [TRACE-2] Packet header pkt_type shifted to wrong bit position

- **Severity**: CRITICAL
- **Status**: FIXED (this audit)
- **Our behavior (before fix)**: `packet_type << 11` placed pkt_type at bits [12:11].
- **mlir-aie behavior**: `extract_tile()` reads pkt_type from `(data >> 12) & 0x3`
  (bits [13:12]). Additionally, `parse_pkt_hdr_in_stream()` checks that bits [11:5]
  are all zero -- if pkt_type was at bit 11, any non-zero pkt_type would cause the
  header to be rejected as invalid.
- **Impact**: Headers with pkt_type > 0 (MEM=1, SHIMTILE=2, MEMTILE=3) would be:
  (a) decoded to wrong pkt_type by extract_tile, AND (b) rejected as invalid by
  parse_pkt_hdr_in_stream. Only pkt_type=0 (Core) worked by coincidence.
- **Fix**: Changed `<< 11` to `<< 12`. Updated test assertions. Added new test
  `test_packet_header_matches_mlir_aie_decoder` that validates all 4 pkt_types
  against the mlir-aie validity checks and field extraction.

## [TRACE-3] Memory module trace has no mode field but we parse one

- **Severity**: LOW (cosmetic)
- **Our behavior**: `write_register(0x00, value)` always extracts mode from
  bits [1:0], even for memory module where the register mask is 0x7F7F0000
  (no mode bits).
- **aie-rt behavior**: Memory module TraceMod has
  `ModeConfig = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE}`.
  `XAie_TraceControlConfig()` sets `Mode = 0` when ModeConfig is unavailable.
- **Impact**: None. Memory module Trace_Control0 bits [1:0] are reserved
  and always 0. We read 0 and store EventTime (mode=0), which is correct.
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
- **Fixed in-place**: no (actually correct for MemTile, benign for others)

## [TRACE-5] TraceState::Stopped vs XAIE_TRACE_OVERRUN naming

- **Severity**: LOW (semantic)
- **Our behavior**: Stop event transitions to `TraceState::Stopped` (value=2).
- **aie-rt behavior**: State 2 is `XAIE_TRACE_OVERRUN` (buffer overflow).
  On real hardware, when the stop event fires, the trace unit likely returns
  to Idle (state 0). State 2 is reserved for overflow conditions.
- **Impact**: Purely cosmetic for emulator purposes -- we never overflow
  since packets are emitted immediately. No tools read the status register
  during normal operation. The state value (2) is written to Trace_Status
  bits [9:8] via read_register(0x08).
- **Fixed in-place**: no (would need hardware validation to confirm
  post-stop-event state transition)

## [TRACE-6] Performance counter events not generated

- **Severity**: MEDIUM
- **Our behavior**: Perf counter event IDs (PERF_CNT_0=5 through
  PERF_CNT_3=8 for core) can be configured in trace event slots, but the
  emulator never fires these events because performance counters are not
  implemented.
- **aie-rt behavior**: aie-rt configures perf counters via
  `XAie_PerfCounterControlSet`. When a counter reaches its threshold, it
  fires a PERF_CNT_N event that can trigger trace start/stop.
- **Impact**: Trace configurations using perf counter events as start/stop
  triggers will not work. Some advanced tracing scenarios use perf counters
  for windowed trace capture.
- **Fixed in-place**: no (requires new subsystem)

## [TRACE-7] Combo events not generated

- **Severity**: MEDIUM
- **Our behavior**: Combo event IDs (COMBO_EVENT_0=9 through
  COMBO_EVENT_3=12) can be configured in trace event slots but are never
  generated.
- **aie-rt behavior**: Combo events combine multiple input events using
  logical operations configured via `XAie_EventComboConfig()`.
- **Impact**: Trace configurations using combo events as start/stop triggers
  or in event slots will not produce expected output.
- **Fixed in-place**: no (separate subsystem)

## [TRACE-8] Multiple0/Multiple1/Multiple2 encoding not implemented

- **Severity**: LOW
- **Our behavior**: Only Single0/Single1/Single2 and Start encodings are
  generated. Multiple0/1/2 formats (which encode multiple simultaneous
  events) are not emitted.
- **mlir-aie behavior**: Decoder handles Multiple0 (0b1100XXXX, 2 bytes),
  Multiple1 (0b110100XX, 3 bytes), and Multiple2 (0b110101XX, 4 bytes).
- **Impact**: When multiple traced events fire in the same cycle, the
  hardware may use Multiple encoding for efficiency. Our emulator always
  uses Single encoding, which is functionally correct but produces slightly
  larger trace output. Decoded events are identical.
- **Fixed in-place**: no (optimization, not correctness)

## [TRACE-9] EventPC and Execution trace modes not implemented

- **Severity**: LOW
- **Our behavior**: Mode values are parsed and stored, but only EventTime
  mode (mode=0) actually produces correct output. EventPC (mode=1) should
  include the program counter instead of cycle deltas. Execution (mode=2)
  should produce per-instruction execution traces.
- **Impact**: None for current workloads. All bridge tests and trace-inject
  use EventTime mode exclusively.
- **Fixed in-place**: no (no test exercises non-EventTime modes)

## [TRACE-10] Repeat0/Repeat1 encoding not implemented

- **Severity**: LOW
- **Our behavior**: Repeat encodings (0b1110RRRR for Repeat0,
  0b110110RR + byte for Repeat1) are never generated.
- **mlir-aie behavior**: Decoder handles Repeat0 and Repeat1 formats that
  repeat the previous event/delta pattern.
- **Impact**: The hardware uses repeat encoding to compress repeated events.
  Our emulator encodes each event independently, which is functionally
  correct but less space-efficient.
- **Fixed in-place**: no (compression optimization)

## [TRACE-11] Event_Sync (0xFF) encoding not implemented

- **Severity**: LOW
- **Our behavior**: The 0xFF byte (Event_Sync marker) is never generated.
- **mlir-aie behavior**: Decoder handles 0xFF as Event_Sync, adding 0x3FFFF
  to the running timer (used for long idle gaps).
- **Impact**: Long idle periods between events may produce slightly different
  trace output compared to hardware (we use Single2 with large deltas
  instead of Event_Sync). Functionally equivalent for the decoder.
- **Fixed in-place**: no (compression optimization)
