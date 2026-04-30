# A.2b: Mode-2 (Execution) Trace — Encoder + Comparator + ZOL Instrumentation

**Status:** Design approved 2026-04-29; ready for implementation plan.
**Predecessor:** A.2 PC-anchored validation (mode-0/1 paths shipped).
**Findings doc:** `docs/superpowers/findings/2026-04-28-a2b-mode2-decoder-deferred.md`

## 1. Problem and goals

A.2 shipped mode-0 (event-time) and mode-1 (event-PC) trace comparison. Mode-2
(execution) was deferred because:

- Cycle-by-cycle alignment was meaningless until the cycle accumulator stabilized.
- The LC frame's bit-28 flag is undocumented across the entire toolchain
  (verified across aietools, mlir-aie, llvm-aie, aie-rt as of 2026-04-29).
- HW baselines existed but no EMU encoder did.

A.5 cycle-budget work is now in place. Mode-2 baselines are captured by
`trace-sweep.py --with-mode2-baseline` and listed (without comparison) by
`compare_sweep_dir_with_opts`.

A.2b's deliverable is end-to-end mode-2 support:

1. EMU emits mode-2 streams that decode cleanly via the same `mode2.py` decoder
   that consumes HW captures.
2. A comparator that pairs HW and EMU mode-2 streams and reports correctness
   divergences (PC sequence, LC counts) separately from timing deltas (atom
   counts within PC segments).
3. Interpreter ZOL instrumentation surfaces loop-boundary events to the trace
   unit, including the LC frame's flag bit (semantics pinned in Phase 0).
4. Bridge tests exercise the full pipeline against real HW captures on
   selected kernels.

## 2. Scope

**In scope:** mode-2 encoder for all four frame types we generate
(E_atom, N_atom, New_PC, LC), Start/Stop framing, RLE compression
(Repeat0/Repeat1), Filler0 word-alignment padding, ZOL instrumentation in
the interpreter, comparator with PC-sequence and LC-count layers,
informational atom-window timing breakdown, bridge-test integration.

**Out of scope (explicit):**

- **Sync and Filler1 emission.** EMU never drops bytes, so decoder-resync
  markers are not emitted. Stubbed with comments; trivial to add later if
  fuzzing or scrubbing scenarios call for them.
- **Mode 3 (Reserved).** Confirmed unused via `xaie_trace.h`'s
  `XAie_TraceMode` enum (only EVENT_TIME / EVENT_PC / INST_EXEC defined).
  No encoder dispatch beyond the existing match arm.
- **Differential fuzzing of the encoder.** Roadmap item, not A.2b.
- **JNZD-driven loop instrumentation** unless Phase 0 captures show HW emits
  LC frames for them. Hookpoint is symmetric to ZLS; trivial to add if needed.

## 3. Design rationale: PC vs cycle alignment

**Mode 2's value is correctness, not cycle-accuracy.** This shapes the
entire comparator:

- **PC anchors** (New_PC, LC) are deterministic functions of the input
  program. EMU and HW must produce identical New_PC and LC sequences for
  any deterministic kernel, regardless of cycle-skew.
- **Atoms** (E_atom, N_atom) are cycle-anchored. EMU and HW will produce
  *different* atom counts within the same PC segment because EMU's stall
  accounting is intentionally not cycle-perfect (DMA stalls in particular
  are nondeterministic on real silicon and not worth chasing).

Therefore:

- **Layer 1 (PC sequence) and Layer 2 (LC counts) gate test pass/fail.**
  These are correctness signals, immune to cycle skew.
- **Layer 3 (atom windows) is informational only.** It measures stall-
  accuracy delta between PC anchors. Useful for tuning the cycle
  accumulator; never gates a test.

Future readers should *not* harden Layer 3 into a regression check —
DMA-stall variance alone makes that path noise.

## 4. Phase 0 — Bit-28 flag reverse engineering

Phase 0 sits before any code work. Deliverable: a findings doc
(`docs/superpowers/findings/2026-04-29-mode2-lc-flag-semantics.md`) that
pins both unknowns about the LC frame:

1. **Bit-28 flag semantics** — what the flag actually marks
   (first-iter, last-iter, nest level, branch-taken, or other).
2. **Trigger frequency** — does HW emit LC frames every iteration, only
   on loop exit, only at outer-loop boundaries in nested cases, or some
   other rule?

### Method

Hand-write 4–6 small AIE2 kernels with controlled ZOL structure, run on
real NPU with mode-2 trace enabled, decode via existing `mode2.py`,
observe the flag and frame timing patterns.

Test scenarios:

1. **Single non-nested loop, fixed iteration count** — flag pattern across
   iterations, frame frequency.
2. **Single loop with early exit** — does flag distinguish natural
   termination from break?
3. **Nested 2-deep loops** — does flag distinguish outer vs inner; does
   trigger frequency differ between the two levels?
4. **Unit-iteration loop (count=1)** — does the flag/frame still fire?
   Tests whether trigger is "boundary" or "decrement".
5. **Tight inner loop (count > 100)** — observe trigger density to
   confirm whether HW LC frames are per-iteration or per-loop.
6. **JNZD-driven software loop** — do JNZD boundaries also emit LC
   frames, or are LC frames ZLS-exclusive?

### Outputs

- Findings doc with: pinned flag bit semantics, pinned trigger frequency,
  JNZD scope decision, captured fixture filenames stored under
  `tools/mode2_capture_fixtures/`.
- Encoder-side rule statement we'll implement directly.
- Comparator stance on the flag bit: strict if pinned cleanly, advisory
  with warning if residual ambiguity remains.

### Time estimate

1–2 days of capture + analysis. Failure mode: if the bit pattern is
ambiguous from observation alone, expand to additional scenarios before
falling back to "ship LC structurally with flag=0, comparator ignores
the bit." That fallback is acceptable but undesirable; we prefer to
land A.2b with the flag fully pinned.

## 5. Encoder structure

### 5.1 New fields on `TraceUnit` (`src/device/trace_unit/mod.rs`)

```rust
pending_word: u32,
pending_word_bits: u8,
pending_atoms_run: Option<AtomRun>,         // see 5.5
pending_mode2_frames: Vec<PendingMode2Frame>, // queued New_PC / LC for current cycle
```

`pending_word` / `pending_word_bits` form the bit accumulator.
`pending_atoms_run` carries the in-flight RLE state.
`pending_mode2_frames` queues non-atom frames (New_PC, LC) within the
current cycle so `commit_cycle` can flush them in the
Phase-0-pinned order alongside the atom. Mode-0/1 paths never touch any
of these.

### 5.2 New private helpers

```rust
fn push_bits(&mut self, value: u32, count: u8);
fn align_to_word_via_filler0(&mut self);
fn flush_word_if_full(&mut self);  // pushes 4 BE bytes to byte_buffer
fn emit_long_frame(&mut self, word: u32);  // align + push whole word
```

Output of the bit accumulator flushes to the existing `byte_buffer`
(4 big-endian bytes per word) so all downstream packetization
(28 bytes → 8 words → packet header) stays unchanged.

### 5.3 New mode-2 frame encoders

```rust
fn encode_atom(&mut self, executed: bool);            // 4-bit prefix
fn encode_new_pc(&mut self, pc: u16);                 // 2b prefix + 14b PC = 16b
fn encode_lc(&mut self, flag: u8, count: u32);        // 32b long frame
fn encode_repeat0(&mut self, n: u8);                  // 4b prefix + 4b count = 8b
fn encode_repeat1(&mut self, n: u16);                 // 6b prefix + 10b count = 16b
fn encode_mode2_start(&mut self, anchor_pc: u16);     // 32b long frame
fn encode_mode2_stop(&mut self);                      // 32b long frame
```

A private `drain_mode2_pending()` helper assembles per-cycle frames in
the Phase-0-pinned order (atom → New_PC → LC by default) and pushes
them through `push_bits` / `emit_long_frame`.

`Sync` and `Filler1` are stubbed with `// not emitted by EMU; see spec
section 2 (out of scope)` comments.

### 5.4 Mode dispatch

The existing match in `commit_pending_frame` (line 643 of trace_unit/mod.rs)
gains the `Execution` arm:

```rust
TraceMode::Execution => {
    self.drain_mode2_pending();  // flushes per-cycle accumulated frames
}
TraceMode::Reserved => {
    // mode 3 not defined per xaie_trace.h; ignore.
}
```

### 5.5 RLE strategy

Per-cycle atom emission feeds a run state:

```rust
pending_atoms_run: Option<{ exec: bool, count: u32 }>,
```

Each cycle's atom appends to the run. Flush triggers:

- Atom polarity flips (E ↔ N).
- A non-atom frame (New_PC, LC) is queued for this cycle.
- Run count would overflow Repeat1's 10-bit field (1023).

Flush emission rule:

- count = 1: emit single E_atom or N_atom.
- count ≤ 16: emit E_atom or N_atom + Repeat0(count - 1).
- count ≤ 1024: emit E_atom or N_atom + Repeat1(count - 1).
- count > 1024: chain Repeat1 frames until exhausted.

## 6. ZOL instrumentation

The interpreter already implements ZLS via `Context::check_hardware_loop`
(state/context.rs:425). Instrumentation is additive:

### 6.1 New event variant

```rust
// src/interpreter/state/event_trace.rs
EventType::LoopBoundary {
    lc_before: u32,
    lc_after: u32,
    le_pc: u32,
}
```

`le_pc` carried for cross-checks during divergence diagnosis. Cheap to
include, easy to drop later if it proves dead weight.

### 6.2 Refactor `check_hardware_loop`

Returns `Option<LoopBoundaryInfo>` instead of `()`. The 6 callsites in
`src/interpreter/core/interpreter.rs` (lines 194, 207, 220, 349, 365, 381)
capture the return and:

1. Record `EventType::LoopBoundary` to the per-tile EventLog (matches
   the existing pattern used for `BranchTaken` at
   `cycle_accurate.rs:585`).
2. Call `tile.core_trace.notify_loop_boundary(cycle, info.lc_before,
   info.lc_after)` to route to the trace unit.

### 6.3 New TraceUnit entry points

```rust
pub fn notify_core_active(&mut self, cycle: u64);   // queues E_atom
pub fn notify_core_stalled(&mut self, cycle: u64);  // queues N_atom
pub fn notify_branch_taken(&mut self, cycle: u64, retire_pc: u32);
pub fn notify_loop_boundary(&mut self, cycle: u64, lc_before: u32, lc_after: u32);
pub fn is_running(&self) -> bool;
```

All four notify methods are no-ops unless
`mode == Execution && state == Running`. Mode-0/1 paths are unaffected.

These are direct calls from the executor to the trace unit, not drained
from EventLog. Architectural justification (see section 8): real silicon
mode-2 trace observes dedicated lines from the core, not the event
broadcast network. Direct calls mirror that.

## 7. Coordinator wiring

### 7.1 Per-cycle atom emission (Phase 3f, additive)

Inside the existing Phase 3f loop in
`src/interpreter/engine/coordinator.rs`:

```rust
if matches!(tile.core_trace.mode(), TraceMode::Execution)
    && tile.core_trace.is_running()
{
    if cores[i].active_this_cycle {
        tile.core_trace.notify_core_active(cycle);
    } else {
        tile.core_trace.notify_core_stalled(cycle);
    }
}
tile.core_trace.commit_cycle(cycle);
```

Mem trace units don't get atoms — mode-2 is core-only per the regdb (only
`CORE_MODULE.Trace_Control0` has the Mode bitfield). This matches the
existing mode-1 guard on `mode_supports_pc()`.

### 7.2 Branch wiring (Phase 2, executor)

At `cycle_accurate.rs:585` (the existing BranchTaken record_event call),
add:

```rust
tile.core_trace.notify_branch_taken(cycle, target);
```

No-op outside mode-2; existing event-time / event-PC paths untouched.

### 7.3 Loop-boundary wiring (Phase 2, executor)

The 6 `check_hardware_loop` callsites in `interpreter.rs` (lines 194, 207,
220, 349, 365, 381) capture the new `Option<LoopBoundaryInfo>`, record
`EventType::LoopBoundary` to EventLog, and notify the trace unit.

### 7.4 Frame ordering within a cycle

`commit_cycle()` flushes per-cycle pending frames in this order:

1. Atom (E or N).
2. New_PC frames (if a branch retired).
3. LC frames (if loop boundary fired).

Order is Phase-0-pinned. If captures show HW differs, the order is a
single Vec drain — mechanical change.

### 7.5 Start/Stop frame emission

Mode-2 Start fires at Idle → Running transition (start_event observed),
carrying the current PC as anchor. Mode-2 Stop fires at Running →
Stopped transition. `encode_start` already mode-dispatches via prefix
byte; mode-2 routes through `encode_mode2_start(anchor_pc)`.

## 8. Comparator

### 8.1 Module structure

New file `src/trace/compare_mode2.rs`. Top-level entry point:

```rust
pub fn compare_mode2_for_tile(
    hw_stream: &[u8],
    emu_stream: &[u8],
    tile: TileKey,
) -> Mode2CompareResult;
```

Aggregator hook in `compare_sweep_dir_with_opts` (`src/trace/compare.rs`)
finds matching `mode2-baseline/<test>/<batch>` files, runs the per-tile
comparator, includes results in the report.

### 8.2 Rust-side mode-2 decoder

New file `src/trace/mode2_decode.rs`. Ports the frame tree from
`tools/trace_decoder/modes/mode2.py` to Rust. Original implementation;
the frame tree was already recovered from the decoder library symbols
once.

Rationale: per-cycle traces can be huge; Python decode is slow at scale;
the comparator wants to be self-contained for unit testing; the
encoder's round-trip tests need a Rust decoder anyway.

### 8.3 Three comparison layers

**Layer 1 — PC sequence diff (correctness).** Extract New_PC sequence
from each stream. Pair by index. First divergence reported with
context. This is the primary bug-finding layer.

**Layer 2 — LC count diff (correctness).** Pair LC frames by index.
Report any count mismatch. Flag-bit handling depends on Phase 0:
- If pinned: comparator checks both flag and count strictly.
- If residual ambiguity: count is strict, flag mismatch is a warning.

**Layer 3 — Atom windows (informational).** For each pair of consecutive
New_PC anchors, report HW vs EMU atom counts in that segment. Format:
`PC 0x300 → 0x340: HW 47 cycles, EMU 52 cycles (Δ +5)`.
Never gates pass/fail — DMA-stall nondeterminism makes Layer 3 noise
in the aggregate.

### 8.4 Frame count mismatch

If HW and EMU disagree on total New_PC or LC frame count, that's itself
a divergence (control-flow took a different path). Reported as Layer 1
or Layer 2 failure depending on which frame type diverged.

### 8.5 Pass/fail decision

Test passes iff Layer 1 has zero divergences AND Layer 2 has zero count
divergences (flag-bit handling per Phase 0 stance).

## 9. Testing

### 9.1 Encoder unit tests

Extend `src/device/trace_unit/tests.rs`:

- Round-trip per frame type via the Rust decoder.
- Bit-accumulator: MSB-first packing, Filler0 alignment, BE word flush.
- RLE: 1, 16, 17, 1024, 1025 atoms — verify Repeat0 / Repeat1 / chained
  Repeat1.
- Mixed-frame ordering: atom + New_PC + LC interleaved.

### 9.2 ZOL instrumentation tests

Extend `src/interpreter/state/context.rs::tests` and add a test in
`cycle_accurate.rs`:

- `check_hardware_loop` returns `Some(_)` on boundary, `None` otherwise.
- LoopBoundary events recorded with correct lc_before/lc_after/le_pc
  across single and nested loops.
- `notify_loop_boundary` called with correct cycle.

### 9.3 Bridge-test integration

Extend `scripts/emu-bridge-test.sh` and `tools/parse-trace.py`:

- 2–3 existing bridge tests with known ZOL structure (e.g.,
  `add_one_using_dma` plus one with explicit nested loops).
- HW capture via existing `--with-mode2-baseline` (already supported).
- EMU side runs, decoder consumes both streams, comparator runs.
- Pass criteria: Layer 1 + Layer 2 match. Layer 3 printed but not gated.

### 9.4 Phase 0 RE captures

Stored under `tools/mode2_capture_fixtures/`. Small kernels, captured
once, decoded for analysis. Not part of CI / regression — they are
scientific instruments, not tests.

### 9.5 Cost

- Encoder unit tests: sub-second.
- ZOL tests: sub-second.
- Bridge integration: alongside existing bridge sweep, no new HW-time cost.
- Round-trip decoder tests: Rust-only fixtures, no HW required.

## 10. Phase 7 — Upstream contribution

After Phase 0 lands the bit-28 finding and the encoder ships, package
the result for upstream submission to mlir-aie:

- Captures + decoded fixtures demonstrating the flag semantics.
- Rust mode-2 decoder ported back to Python, contributed as a
  `mlir-aie/python/utils/trace/parse_mode2.py` enhancement.
- Documentation note explaining the bit-28 flag (the project that owns
  the trace tooling deserves to know what their own format means).

This is a publishable artifact — the project's MIT licensing and
open-source posture make contributing back natural.

## 11. Phase ordering and ownership

| Phase | Owner | Description | Deliverable |
|-------|-------|-------------|-------------|
| 0 | Critical path | Reverse-engineer bit-28 flag + LC trigger frequency | Findings doc + fixtures |
| 1 | Encoder | Bit accumulator + frame encoders + mode dispatch | trace_unit/mod.rs changes + tests |
| 2 | ZOL instr | Refactor `check_hardware_loop` + LoopBoundary event | context.rs + interpreter.rs + tests |
| 3 | Coordinator wiring | Atom emission, branch/loop notify | coordinator.rs + cycle_accurate.rs |
| 4 | Rust decoder | Port mode2.py to Rust | trace/mode2_decode.rs |
| 5 | Comparator | Three-layer compare + aggregator hook | trace/compare_mode2.rs + compare.rs |
| 6 | Bridge integration | Test wiring + sweep updates | scripts + tools/parse-trace.py |
| 7 | Upstream | Findings + decoder contributed to mlir-aie | PR to mlir-aie |

Phases 1–3 can proceed in parallel after Phase 0 lands. Phase 4 unblocks
both the encoder's round-trip tests and the comparator. Phase 5 needs
Phase 4. Phase 6 needs everything. Phase 7 follows shipment.

## 12. Risks and open questions

- **Phase 0 ambiguity.** If HW captures don't pin the flag cleanly,
  fall back to advisory comparator handling. Documented in section 4.
- **Frame ordering within a cycle.** Pinned by Phase 0 captures.
  Implementation is a single Vec drain — adjusting order is mechanical.
- **JNZD scope.** Pinned by Phase 0. If HW emits LC for JNZD too, the
  hookpoint is symmetric to ZLS — small additional scope.
- **Atom-window size for Layer 3.** No default to set — Layer 3 is
  informational, not gated, so window size is a display preference
  rather than a correctness parameter.

## 13. References

- A.2 plan: `docs/superpowers/plans/2026-04-25-a2-pc-anchored-validation.md`
- A.2 design: `docs/superpowers/specs/2026-04-25-a2-pc-anchored-validation-design.md`
- A.2b deferral notes: `docs/superpowers/findings/2026-04-28-a2b-mode2-decoder-deferred.md`
- HW frame tree: `tools/trace_decoder/modes/mode2.py`
- Mode 3 confirmed dead: `aietools/include/drivers/aiengine/xaiengine/xaie_trace.h` (`XAie_TraceMode` enum)
- ZLS register IDs: `aietools/data/aie_ml/lib/isg/me_regid.txt` (LC=284, LE=285, LS=287)
- Existing ZLS impl: `src/interpreter/state/context.rs:425` (`check_hardware_loop`)
- Existing mode-0/1 encoder: `src/device/trace_unit/mod.rs`
- Existing comparator: `src/trace/compare.rs`
- mlir-aie trace attr declaration: `include/aie/Dialect/AIE/IR/AIETraceAttrs.td`
