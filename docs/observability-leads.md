# Observability Leads -- Untapped Capabilities in aie-rt and xdna-driver

Survey notes from a 2026-04-24 sweep of `../aie-rt/` and `../xdna-driver/`.
Captures debug, trace, and observability infrastructure we are not yet using
in xdna-emu's trace pipeline. Preserved here so the leads survive across
sessions and so the next round of sequence-skeleton / drift-correction work
can pick them up without re-doing the survey.

The driving question was: what would let us anchor the trace window to
deterministic events, partition the variance budget, or get an independent
ground truth for "the kernel body took N cycles"?

## Current status (2026-06-28)

**DONE:**
- **Event-bounded trace prototype (lead #1):** experiment complete. The
  comparative table inline in Section 1 gives the conclusion: mlir-aie's
  default broadcast configuration (`BROADCAST_15`/`BROADCAST_14`) is
  near-optimal for this workload. Span SD ~220cy vs call-return ~200cy; the
  residual is broadcast-latency variance that on-tile re-anchoring cannot
  reduce. Action closed.
- **In-tree trace decoder (leads #1/#3):** shipped at `tools/trace_decoder/`
  (MIT), covering modes 0 (EVENT_TIME), 1 (EVENT_PC), and 2 (INST_EXEC).
  `parse-trace.py --decoder=ours` is the default. Cross-validation via
  `--decoder=mlir-aie` remains available.

**STILL OPEN:**
- Leads #2, #4, #5: perfcnt-span sidecar, ftrace capture, QUERY_HW_CONTEXTS
  per-batch sample. See Action priority section below.
- Modes 1/2/3 decoder: `trace_decoder/` covers the format; semantic
  validation against real hardware output is pending.

Sorted by relevance to the trace-sweep / sequence-skeleton work.

---

## Drift correction & sequence-skeleton

### 1. Event-bounded trace windows

`aie-rt/driver/src/trace/xaie_trace.h`

```c
XAie_TraceStartEvent(DevInst, Loc, Module, Event)
XAie_TraceStopEvent(DevInst, Loc, Module, Event)
XAie_TraceControlConfig(DevInst, Loc, Module, StartEvent, StopEvent, Mode)
XAie_TraceModeConfig(DevInst, Loc, Module, Mode)
  // modes: XAIE_TRACE_EVENT_TIME, XAIE_TRACE_EVENT_PC, XAIE_TRACE_INST_EXEC
```

The trace window is bounded by start/stop events. Trace_Control0
register (offset 0x340D0 on core, 0x140D0 on memmod, 0x340D0 on shim,
0x940D0 on memtile) carries the start_event (bits [16:22], 7 bits on
core/mem/shim, 8 bits on memtile) and stop_event (bits [24:30/31]).

**Discovery (2026-04-24):** mlir-aie's compiled output is *not*
free-running -- it sets `start_event=BROADCAST_15` (122 on core) and
`stop_event=BROADCAST_14` (121). The host fires those broadcasts via
shim control packets to bracket the kernel run.

`tools/trace-patch-events.py` (extended in this session) supports
`--start-event` / `--stop-event` / `--mode` flags to override these
fields in an existing insts.bin. No mlir-aie patch needed.

**Empirical comparison** (`build/experiments/cdo-preamble-test/test_event_bounded.py`,
add_one_objFifo, N=20 same-batch iters under hwctx reuse + CDO preamble):

| Mode | start | stop | mean cycles | sd | range | n_events sd |
|------|-------|------|-------------|----|----|------|
| broadcast (default) | 122 | 121 | 3160 | **221** | 1049 | **0.0** |
| call-return | INSTR_CALL=35 | INSTR_RETURN=36 | 3109 | **199** | 772 | 1.2 |
| call-disabled | INSTR_CALL=35 | CORE_DISABLED=29 | 9911 | 2664 | 8944 | 1.6 |

Broadcast is essentially as stable as call-return on span (220 vs 200
cycles sd) but gives **perfect event-count stability** (sd=0.0,
exactly 146 events every iteration). Call-disabled is broken under
hwctx-reuse: CORE_DISABLED doesn't fire predictably so the trace runs
3x longer and varies 12x more.

**Take-away:** mlir-aie's broadcast configuration is near-optimal for
this workload. The remaining ~7% span jitter is residual broadcast
latency variance, not something we can reduce by re-anchoring on
on-tile events. Future direction: try anchoring on PERF_CNT events
(perfcnt threshold reached -> fire one-shot event) to get a more
precise on-tile bound, or use this as evidence that the broadcast bound
is the right default and shift focus to calibrating the residual.

PC-trace mode (`XAIE_TRACE_EVENT_PC`, mode=1) remains unexplored.

**Trace mode survey (2026-04-24).** Patched add_one_objFifo to each
mode value (0..3) via the patcher and captured under bridge-runner.
Per-run trimmed-trace sizes from fresh hwctx (no reuse, so the trace
buffer is clean each run):

| mode | name | bytes | discriminator at off 7 |
|------|------|-------|------------------------|
| 0 | EVENT_TIME | 320 | `f0` |
| 1 | EVENT_PC | 896 | `f1` |
| 2 | INST_EXEC | 192 | `f0` (1st pkt), `f2` (2nd pkt) |
| 3 | reserved (per AM020) | 896 | `f2` |

Modes 1 and 3 are NOT empty -- both produce 896-byte payloads, larger
than mode 0. Mode 3 has the same discriminator byte as mode 2's
secondary packet, suggesting it's a related variant rather than truly
unimplemented.

**Authoritative decoder situation (2026-04-25 update).** We now ship an
in-tree decoder at `tools/trace_decoder/` (MIT) covering modes 0
(EVENT_TIME), 1 (EVENT_PC), and 2 (INST_EXEC).  `parse-trace.py
--decoder=ours` is the default and authoritative for the xdna-emu
cycle-diff pipeline; mlir-aie's `parse_trace` (mode 0 only) remains
selectable via `--decoder=mlir-aie` for cross-validation.  Not a
permanent fork: if/when mlir-aie's `parse_trace` covers all three
modes upstream, we swap back -- trace decoding is post-mortem so the
swap-back is a one-line default change with no hot-path cost.

**Historical context** (the situation that drove us to write our own):
mlir-aie's `parse_trace` (`mlir-aie/python/utils/trace/parse.py`)
decodes mode 0 only -- no mention of `atom`, `INST_EXEC`, or PC mode
anywhere in mlir-aie's trace utilities. aie-rt configures the trace
mode register but ships no host decoder for any mode (the trace data
lives off-chip in the trace buffer; aie-rt's job ends at the register
write). AM020 §2 documents mode 2 as emitting "conditional and
unconditional direct branches, all indirect branches, and ZOL LC"
-- a branch-trace
record format -- but does not specify byte layout.

We have an experimental decoder at
`tools/trace-mode-tests/decode_trace_experiment.py` that interprets
mode-2 bytes as `E`/`N` atoms (instruction-slot execute bits). **This
interpretation is unverified** -- AM020's "branches and ZOL LC"
description is incompatible with per-instruction-slot atom bits, so
the decoder is probably wrong about the semantics even if its byte
walk happens to be self-consistent.

**Stability under fresh hwctx (n=3 per mode):** all modes produce
constant-size traces but their byte content differs run-to-run --
SHA-256s of trimmed bytes are unique each run. So *some* of the per-
run variance is timestamp/delta-encoded into the bytes. No mode
gives us bit-identical traces of the same kernel.

**Earlier "atom-identical across 5 iterations" claim (RETRACTED):**
that result came from runs under hwctx-reuse where the trace buffer
**accumulates across iterations** -- bridge-trace-runner does not zero
the trace BO between batches. Iter 0's data starts at offset 0; each
subsequent iter appends. The decoder always read offset 0 and got
iter 0's bytes back every time. data_end grew 192 -> 448 -> 640 -> 896
-> 1088 across the 5 runs. This is also a runner bug worth fixing
(or at least documenting): for reuse-mode batches, the runner should
either zero the trace BO or report a per-batch "new data starts at
offset N."

PC-trace mode (1) and reserved mode (3) remain unexamined beyond
"yes, they produce data."

**Status:** patcher works for any mode value; modes 1, 2, 3 produce
captures but we have no trustworthy decoder for them. Reverse-
engineering needed (or finding a decoder buried in aietools); until
then mode 0 is the only mode whose contents we can interpret.

#### Aietools hunt result (2026-04-24)

Searched the local aietools install for an authoritative on-tile
trace decoder. Found a partial trail; the `.cpp` implementation is
**not shipped** in this install, but a Synopsys-copyrighted header
discloses the **semantic** structure of the trace stream
(read-only reference per project policy; do not copy):

`tps/lnx64/target_aie_ml/chessdir/checkersdir/include/checkers_trace_decoder.h`

The header defines a decoder-callback API whose virtual methods
enumerate exactly what the on-tile trace stream encodes:

- cycle markers
- instruction-address (PC) values at indirect/taken branches
- conditional-branch outcomes (taken / not-taken bits)
- memory request and response (address + value)
- function-call actions (jsr, rts, entry, ji/rti, tailcall, delay-slot)
- hardware stalls
- IO/stream changes
- trace control (start/stop) and overflow markers
- trace-stage transitions

This **matches AM020 §2's "branches and ZOL LC" description** and
proves our experimental decoder's per-instruction-slot E/N atom
interpretation is wrong. The stream is event-driven (records
deltas on branches and accesses) not slot-driven (no E/N bit per
issue slot).

**For Event-PC mode (mode 1)**: the stream is essentially mode 0's
event firings interleaved with `process_instruction_address(pc)`
records -- each event captured along with the PC at which it fired,
giving us PC-anchored events instead of cycle-anchored.

**For INST_EXEC mode (mode 2)**: pure execution-flow trace --
sequence of `process_cjump(taken)` and `process_instruction_address(pc)`
calls reconstructible into a branch-by-branch path through the ELF.

**Byte format remains opaque** -- the header shows the API surface,
not the wire format. Other aietools artefacts checked and dismissed:

- `data/DynamicEventTraceSchema.json`, `EventTraceConfigSchema.json`:
  configuration-side schemas, not decode formats
- `bin/x86sim_decode_event_trace.py`: x86sim's internal event log,
  not on-tile trace
- `data/eventanalyze/event_type_table.txt`: high-level analyzer event
  types (CORE_WAIT, PC_CHANGE, etc.), not byte format
- `bin/eventanalyze`, `bin/hwanalyze`: bash wrappers for missing
  loader binaries; the actual analyzers aren't in this install
- `data/pl_fileio/libpl_trace_decoder.so`: programmable-logic trace
  (FPGA fabric), not AIE
- `tps/lnx64/target_aie_ml/chessdir/ychessdir/iss_hw_tracing.tcl`:
  ISS-debugger UI plugin; references load_trace but the actual
  binary plumbing isn't here either

**Where to go:** either try a fuller aietools install (the AMD
unified installer ships more components -- the `.cpp` decoder may
be in a sibling package), or reverse-engineer the byte format using
the semantic structure above as a guide. The latter is now far more
tractable: we know we're looking for a stream that contains *PCs,
branch outcomes, cycle markers, and memory accesses* -- not E/N
atoms. A diff between mode-0 and mode-1 bytes for the same kernel
should reveal where PC values are inserted.

#### Aietools follow-up (2026-04-24): full decoder library found

Earlier search missed the binaries because `find` without `-L` did
not follow the `aietools/` symlink (since removed). Searching the
canonical path `amd-unified-software/aietools/` reveals the real
install includes a complete trace-decoder shared library (read-only
reference -- do not copy):

`amd-unified-software/aietools/lib/lnx64.o/libxv_trace_decoder_opt.so`

Symbol-table inspection (read-only knowledge of hardware behavior)
exposes the full frame taxonomy in `cardano::Trace`:

| Class | What it represents |
|-------|--------------------|
| `Execution_Start` | trace start marker |
| `Execution_Stop` | trace stop marker |
| `Execution_Sync` | sync / resync (carries the sync_pc anchor) |
| `Execution_E_atom` | one cycle in which an instruction executed |
| `Execution_N_atom` | one cycle in which the core did not execute (stall) |
| `Execution_New_PC` | taken-branch destination PC |
| `Execution_New_PC_AIE4` | New_PC variant for AIE4 |
| `Execution_LC` | zero-overhead-loop loop counter snapshot |
| `Execution_Filler0/1` | padding frames |
| `Execution_Repeat0/1` | run-length compression frames |

So our experimental decoder's "EEENN..." atom string interpretation
was right in spirit -- the frames really are per-cycle execute /
no-execute markers, not slot-bit packing. Each atom = one cycle.
The `Execution_New_PC` frames interleaved with atoms give us the
actual taken-branch destinations. Combined: a fully reconstructible
cycle-by-cycle execution path through the kernel ELF.

For Event-PC mode (1), diagnostic strings reveal the parallel set:
`Event PC frame`, `Event PC Sync frame`, `Event PC Filler frame`,
`Event PC Start frame Timer value:`, `Event PC Repeat 0/1`. So
mode 1 is structured similarly to mode 0 but with PC values
attached to event records instead of (or alongside) timer deltas.

Other library symbols hint at the decode internals:
- `TraceDecoder::decode_packet(uint*, ..., name, offset)` -- main
  byte-stream decoder
- `TraceDecoder::decodeExecutionTrace(...)` -- mode 2 entry point
- `TraceDecoder::record_single_event`, `record_multiple_events`,
  `record_event_repeat` -- mode 0 event-record handlers
- `EVENT_TIME_FRAME_TYPE` enum (mode 0 frame discriminator)
- All frame `decode()` methods take `(uint, uint, int, bitset<32>&,
  ofstream&)` -- the `bitset<32>` strongly implies 32-bit-word-
  aligned frame headers

**Path forward:** write our own decoder informed by the frame
taxonomy and the adf::Trace API contract (below) and validate
against the open-source mode-0 oracle (mlir-aie's parse_trace).
We will *not* link aietools at runtime -- aietools' licensing
posture is too ambiguous for a runtime dependency. Read for
understanding, ship original code.

#### adf::Trace::TraceDecoder API contract (read-only reference)

A second decoder library `lib/lnx64.o/libevent_trace_decoder.so`
(distinct from the Synopsys cardano `libxv_trace_decoder_opt.so`,
this one is in the AMD `adf::Trace::` namespace) directly maps the
mode-0/mode-1 surface we care about. Symbol-table inspection only;
no code or strings copied:

| Method (signature, demangled)                                                | What it tells us |
|------------------------------------------------------------------------------|------------------|
| `TraceDecoder::decodePacket(uint8_t (&buf)[8], uint pkt_index)`              | The byte-stream reader works on 8-byte windows -- consistent with mode-0 frame opcodes that range 1..8 bytes (Single*, Multiple*, Start, Repeat*) |
| `TraceDecoder::processStart(col, row, module_type, Module*, bool, trace_mode, uint64 ts)` | Start frame carries a 64-bit timer value AND the trace_mode -- so the same frame layout serves all modes; the mode dictates which event-handlers fire after |
| `TraceDecoder::processStop(..., uint ts)`                                    | Stop is timestamped |
| `TraceDecoder::processSync(...)`                                             | Sync resyncs the decoder; carries no payload (inferred from no extra arg) |
| `TraceDecoder::processRepeat(..., uint count)`                               | Run-length compression frame |
| `TraceDecoder::processAssertedEvents(..., bitset<8>, uint cycles_or_ts)`     | **Mode 0 (EVENT_TIME)**: 8-bit event mask + cycle delta |
| `TraceDecoder::processEventPC(..., bitset<8>, uint pc)`                      | **Mode 1 (EVENT_PC)**: same 8-bit event mask but with PC value instead of cycles |
| `TraceDecoder::processEventPCStop(...)`                                      | Mode-1 specific stop variant -- mode-1 has its own end-of-trace marker |
| `FileOutputTraceDecoder::streamOutDecodedTimeEvent(col, row, mod_type, ?, ?, event_status, uint64)` | Mode-0 emit: 7 fields incl. event_status |
| `FileOutputTraceDecoder::streamOutDecodedPCEvent(col, row, mod_type, ?, ?, uint)` | Mode-1 emit: 6 fields, no event_status, but a uint PC field |

**Key inference**: modes 0 and 1 share the same encoder framework
(Start, Stop, Sync, Repeat all common; bitset<8> event-mask format
common). The single difference is the secondary value attached to
each event-firing record: cycles in mode 0, PC in mode 1. Mode 1's
own stop marker (`processEventPCStop`) suggests the per-mode header
discriminator (`f0`/`f1`) tells the decoder which terminal opcode
to expect.

**For implementation purposes**: build mode-0 fully (validated
against mlir-aie's parse_trace), then mode-1 reuses Start/Stop/
Sync/Repeat verbatim and replaces the cycle-encoded Single*/Multiple*
opcodes with PC-encoded variants. Empirical bytes diff (mode-0 vs
mode-1 of the same kernel) will localise the new opcodes.

**Mode 2 (INST_EXEC)** is a separate beast handled by
`libxv_trace_decoder_opt.so` (cardano::Trace, Synopsys-copyrighted)
-- the Execution_E_atom / Execution_New_PC frame taxonomy described
above. Cardano is the reference for mode 2 only; libevent doesn't
cover it.

**Quick win available:** even without parsing every frame type, we
can bucket-count frame headers in a captured trace to verify our
experimental decoder is roughly hitting the right structural
classes, and probe modes 1/3 byte content with the same approach.

### 2. Performance counters with start/stop/reset events

`aie-rt/driver/src/perfcnt/xaie_perfcnt.h`

```c
XAie_PerfCounterControlSet(DevInst, Loc, Module, Counter,
                           StartEvent, StopEvent)
XAie_PerfCounterResetControlSet(DevInst, Loc, Module, Counter, ResetEvent)
XAie_PerfCounterEventValueSet(...)   // threshold trigger
XAie_PerfCounterGet(...)             // host-side register read
```

Four 32-bit counters per tile (core: 0x31500 control block, 0x31520 value,
0x31580 threshold). Each counts cycles while in the "armed" state
(armed by start_event, disarmed by stop_event). When the count hits the
threshold, fires `PERF_CNT_N` event.

**Hard constraint discovered (2026-04-24):** the NPU instruction set
exposed by XRT (`Write32`, `MaskWrite`, `MaskPoll`) has no `Read32` op,
so we cannot read counter values back to host through the runtime
sequence. The only way to surface a perfcnt measurement in our XRT
path is via the `PERF_CNT_N` event into the trace.

Trace timestamps and perfcnt both run off the same 64-bit free-running
timer, so perfcnt-via-trace is **not** an independent cycle source --
it gives us the same timestamps the trace already captures.

What perfcnt-via-trace *can* still buy us:

1. **Deterministic trace stop**: route `PERF_CNT_0` to Trace_Control0's
   stop_event slot, with perfcnt configured to fire at threshold N
   cycles after BROADCAST_15. This replaces the host-fired BROADCAST_14
   (variable latency) with a tile-internal "exactly N cycles after
   start" stop. Useful for fixed-window captures.
2. **Periodic timing markers**: small threshold (e.g., 100), with
   counter not auto-resetting -- fires `PERF_CNT_0` every 100 cycles,
   gives us cycle-stride markers in the trace.
3. **Threshold-as-bound check**: set threshold to expected cycle count;
   trace shows PERF_CNT firing iff actual count met threshold. Useful
   for asserting "the body took ≥N cycles."

**Implementation hurdle:** existing insts.bin doesn't write to perfcnt
registers, and CDO preamble doesn't either (verified by scanning init
and enable blobs). To enable perfcnt we'd need to *insert* new Write32
ops into insts.bin (header has `num_ops` and `total_size` fields that
must be updated), or compile perfcnt config into the source MLIR.

**MDM (Microcode DMA) performance counters** are a separate set
(`XAie_MdmPerfCounterStart/Stop/Sample/Get`) that may be readable
differently -- not yet investigated.

**Status:** scoped down. Worth implementing the insert-ops capability
in the patcher when we need a fixed-window trace, but no longer the
"independent ground truth" the lead originally promised.

### 3. Event broadcast + directional blocking

`aie-rt/driver/src/events/xaie_events.h`

```c
XAie_EventBroadcast(DevInst, Loc, Module, BroadcastId, Event)
XAie_EventBroadcastBlockDir(DevInst, Loc, Module, ..., Dir)
  // Dir: North/South/East/West
XAie_EventComboConfig(...)            // AND/OR/AND_NOT, up to 8 events
XAie_EventGroupControl(...)           // group events bitmap
```

Cross-tile anchor capability we don't use today. If sequence-skeleton
extends to multi-tile, broadcasting a single "go" event into multiple
trace control registers gives synchronous start across the array.
Directional blocking lets us shape the broadcast region (e.g., one column
only).

Combo events let one trace slot fire on (A AND B) or (A OR B), which could
reduce the number of slots needed for compound conditions.

**Edge detection** events (rising/falling edges of monitored signals) are
new in AIE2P -- not available on Phoenix/NPU1.

**Status:** useful when we go multi-tile. Not blocking current work.

---

## Orthogonal observability

### 4. Kernel ftrace tracepoints

`xdna-driver/src/driver/amdxdna/amdxdna_trace.h`

```
xdna_job             (sched_job, name, string, seq, op)
mbox_set_tail        (channel_id, opcode, msg_id)
mbox_set_head        (channel_id, opcode, msg_id)
mbox_irq_handle      (irq, msix_index)
uc_irq_handle        (irq, msix_index)
mbox_rx_worker       (...)
mbox_poll_handle     (...)
amdxdna_debug_point  (name, number, string)
```

Enable via `/sys/kernel/debug/tracing/events/amdxdna_trace/`. Captures the
host-device handshake at the kernel level: job submit, mailbox push/pull,
IRQ arrival, worker wake-up.

Lets us partition variance across the stack: "did this 50us jitter come
from kernel scheduling, mailbox latency, or on-tile execution?" Free to
enable -- no instrumentation in our code.

**Status:** add to runner once user's debugfs is back. Cheap and additive.

### 5. `QUERY_HW_CONTEXTS` ioctl

`xdna-driver/include/uapi/drm/amdxdna_accel.h`

`DRM_IOCTL_AMDXDNA_GET_INFO` with type `QUERY_HW_CONTEXTS` returns an array
of `amdxdna_drm_hwctx_entry`:

```c
struct amdxdna_drm_hwctx_entry {
  u64 context_id;
  struct { start_col, num_col } partition;
  u32 pid;
  u64 command_submissions;
  u64 command_completions;
  u64 migrations;
  u64 preemptions;
  u64 errors;
};
```

Per-batch sanity check independent of the trace buffer. Useful for
distinguishing "trace was empty because nothing ran" from "trace was empty
because event config was wrong." Also surfaces preemption/migration events
that would otherwise be invisible in the trace.

Other useful `GET_INFO` queries:
- `QUERY_RESOURCE_INFO` -- npu_clk_max, npu_tops_curr, npu_task_curr
- `QUERY_CLOCK_METADATA` -- mp_npu_clock and h_clock with current freq_mhz
- `QUERY_SENSORS` -- power (mW), column utilization
- `QUERY_AIE_STATUS` -- per-column live bitmap
- `GET_POWER_MODE` -- DEFAULT / LOW / MEDIUM / HIGH / TURBO

**Status:** doable today via XRT shim or raw ioctl.

### 6. Firmware log/trace ring buffers

`xdna-driver/src/driver/amdxdna/aie2_dpt.c`

Two 8KB ring buffers maintained by firmware (PMC):

| Buffer | Debugfs entry | Contents |
|--------|---------------|----------|
| FW log   | `dump_fw_log_buffer`   | Errors, warnings, info, debug |
| FW trace | `dump_fw_trace_buffer` | Detailed firmware execution events |

Frame format (both buffers):

```
header: magic 0xCA, data_word_len, seq_num, reserved
data:   timestamp (u64), format_flag, level, app_id, argc, line, module
footer: reserved, seq_num, data_word_len, magic 0xBA
```

Sequence numbers detect wraparound. Levels: ERR / WRN / INF / DBG.

Probably contains the kernel-FW handshake timing we currently can't see.
Could correlate firmware ops with our on-tile trace events.

**Status:** read on demand via debugfs once user's debugfs is back.

---

## Auxiliary capabilities (lower priority but preserve)

### Debug halt / interrupt backtracking

`aie-rt/driver/src/core/xaie_core.h` and
`aie-rt/driver/src/interrupt/xaie_interrupt_backtrack.c`:

```c
XAie_CoreDebugHalt / XAie_CoreDebugUnhalt
XAie_CoreGetDebugHaltStatus
XAie_CoreConfigDebugControl1        // event-triggered halt
XAie_CoreConfigureErrorHaltEvent
XAie_BacktrackErrorInterrupts       // walks error log, finds source tile
```

Synchronization primitive (freeze cores on error) plus a post-mortem
backtracking API. Useful if we ever need to pause emulation on a
trace-detected anomaly.

### Power / clock / DPM

xdna-driver:
- Debugfs `dpm_level` -- read and write DPM level (frequency pair
  `[npuclk, hclk]` MHz). NPU1 has a discrete table.
- `aie2_smu.c` exposes `AIE_SMU_SET_HARD_DPMLEVEL` and `SOFT_DPMLEVEL`
- `QUERY_CLOCK_METADATA` ioctl reads current frequencies

aie-rt:
- `XAie_PmSetColumnClk` -- column-wide clock gate. Could "pause" a batch
  for inspection at power level.

Relevant to the future "6 MHz downclock so DMA delay becomes irrelevant"
direction. The DPM table is discrete, so we're constrained to the levels
firmware exposes -- not arbitrary frequencies. Worth checking what the
lowest level actually is.

### TDR (Timeout Detection & Recovery)

`xdna-driver/src/driver/amdxdna/aie2_tdr.c`

Periodic watchdog (default 2s, module param `timeout_in_sec`). Detects
stuck contexts via submission/completion counter stagnation, then either
dumps state (`tdr_dump_ctx=true`) or resets the context.

`docs/driver-diagnostics.md` covers the existing `aie2_diag` debugfs
surface (diag_stats, diag_tdr_history, diag_cert_state) -- this is the
userspace view of TDR events.

### FAL (Full Abstraction Layer)

`aie-rt/fal/src/`

C++ resource manager + profiling helpers. Pre-built widgets:

- `XAieActiveCycles` (counter on ACTIVE vs DISABLED)
- `XAieStallCycles` (counter on GROUP_CORE_STALL vs GROUP_CORE_PROGRAM_FLOW)
- `XAieStallOccurrences` (event count, not duration)
- Trace-resource class with slot reservation, cross-module event support

Reference for managing shared resources (events, counters, broadcast
channels, trace slots) when we have many concurrent measurements. Not
needed for current single-trace setup.

### Telemetry types (firmware-side aggregates)

`xdna-driver/src/driver/amdxdna/aie2_msg_priv.h`

5 telemetry buffer types, queryable via debugfs `telemetry_*` files or
`QUERY_TELEMETRY` ioctl:

- DISABLED, HEALTH, ERROR_INFO, **PROFILING**, DEBUG

Each is an 8KB DMA buffer fetched from FW on read. PROFILING is the
interesting one for fidelity validation -- per-tile / aggregate
performance metrics that we could compare against emulated values.

NPU4 collapses this to a single PERF_COUNTER type with different layout.

---

## Action priority for trace-sweep work

1. ~~**Event-bounded trace prototype (lead #1).**~~ **DONE (2026-04-24).**
   Experiment ran; conclusion: broadcast (`BROADCAST_15`/`14`) is near-optimal.
   Residual ~220cy SD is broadcast-latency variance; on-tile re-anchoring does
   not reduce it further. See the comparative table in Section 1.

2. ~~**In-tree decoder (leads #1/#3).**~~ **DONE (2026-04-25).** Decoder
   shipped at `tools/trace_decoder/` (MIT). `parse-trace.py --decoder=ours`
   is the default. Modes 0/1/2 covered; semantic HW-validation of modes 1/2
   remains pending.

3. **Perfcnt-span sidecar (lead #2).** One perf counter started on first
   INSTR_CALL, stopped on last INSTR_RETURN, read out post-batch.
   Independent ground truth for "did the kernel body take a constant
   number of cycles?" (Requires inserting Write32 ops into insts.bin.)

4. **Ftrace capture in the runner (lead #4).** Subscribe to `xdna_job` and
   `mbox_*` tracepoints during a sweep. Partitions kernel-side variance from
   on-tile variance. Cheap and additive; doable today via debugfs.

5. **`QUERY_HW_CONTEXTS` per-batch sample (lead #5).** Cheap meta-anchor
   for batch validity. Append to runner JSON status output.

Leads 6 (firmware log/trace buffers) and the FAL profilers are useful adjuncts
but not in the critical path.

---

## Source paths cheat sheet

```
aie-rt/driver/src/trace/xaie_trace.h         -- trace API
aie-rt/driver/src/perfcnt/xaie_perfcnt.h     -- perfcnt API
aie-rt/driver/src/events/xaie_events.h       -- event broadcast / combo / group
aie-rt/driver/src/core/xaie_core.h           -- debug halt
aie-rt/driver/src/interrupt/                 -- L1/L2, backtracking
aie-rt/driver/src/pm/xaie_clock.h            -- column clock gating
aie-rt/fal/src/rsc/xaiefal-trace.hpp         -- FAL trace resource
aie-rt/fal/src/rsc/xaiefal-perf.hpp          -- FAL perfcnt resource
aie-rt/fal/src/profile/xaiefal-profile.hpp   -- pre-built profilers

xdna-driver/include/uapi/drm/amdxdna_accel.h            -- ioctl + query types
xdna-driver/src/driver/amdxdna/amdxdna_trace.h          -- ftrace tracepoints
xdna-driver/src/driver/amdxdna/aie2_debugfs.c           -- NPU1 debugfs surface
xdna-driver/src/driver/amdxdna/aie2_dpt.c               -- FW log/trace parsing
xdna-driver/src/driver/amdxdna/aie2_tdr.c               -- TDR watchdog
xdna-driver/src/driver/amdxdna/aie2_msg_priv.h          -- telemetry types
xdna-driver/src/driver/amdxdna/aie2_smu.c               -- DPM commands
xdna-driver/src/driver/amdxdna/amdxdna_error.h          -- error codes
```
