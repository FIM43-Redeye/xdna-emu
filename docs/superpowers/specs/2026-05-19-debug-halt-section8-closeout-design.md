# debug_halt §8 close-out — design

**Date:** 2026-05-19
**Parent spec:** `docs/superpowers/specs/2026-05-18-debug-halt-design.md` (§8 Forward-commitment)
**Status:** approved (Maya, 2026-05-19) — ready for writing-plans

This spec resolves the desk-actionable subset of the parent spec's §8
forward-commitments. It does **not** reopen Phase A/B; Phase B is
conclusively closed (`debug_halt` coverage `Modeled { Full }`).

---

## 0. Scope

Five §8 forward-commitments exist. They split into three desk-closeable
and two genuinely hardware-budget-gated:

| §8 item | Disposition here |
|---------|------------------|
| G1 halt-timing (bounded escalation) | **RESOLVED (bookkeeping).** The contingency never fired — G1 was DERIVED and SHIPPED (findings `2026-05-18-...` conclusion (a)). The §8 bullet describes a fallback that did not happen; flip it to RESOLVED. |
| `OUTBUF_ADDR` probe-artifact robustness | **RESOLVED (Item 2).** Unified build-time derivation. |
| `Core_Status` RESET-bit EMU/HW divergence | **RESOLVED (Item 3).** Emulator fidelity fix. |
| Count-step silicon-fidelity | **STAYS OPEN.** Only `N=4` was observable (`LANDED:0`); finer cadence / larger-N / `0x11`-on-silicon need register-poke tooling or a dedicated hardware-observation budget. Untouched; stays surfaced in the `debug_halt` coverage narrative. |
| Resume hardware-verification | **STAYS OPEN.** A runtime sequence cannot deassert a breakpoint mid-run and re-observe a second core pass; needs a dedicated silicon probe pass. Untouched; stays surfaced. |

Decisions locked with Maya (2026-05-19, do not re-litigate):

- **Probe scope:** de-fragilize *both* `OUTBUF_ADDR` and the sibling
  `TRAP_PC` (same fragility class), via one unified derivation mechanism.
- **Core_Status approach:** **B** — route the runtime enable path through
  `write_control()`; scoped to the enable path only (`disable_core`
  unchanged).
- **Derivation approach:** **1** — build-time derivation owned by our prep
  glue, golden-record-preserving (committed probe bytes stay the record;
  the build re-derives and *guards* them).

The output of this spec is itself this document plus pointer updates in
the parent spec's §8. Implementation is handed to `writing-plans`.

---

## 1. Item 3 — `Core_Status` RESET-bit reconciliation (Approach B)

### 1.1 The divergence (restated from §8)

Debug-halted at the trap, hardware reports `Core_Status = 0x10001`
(`DEBUG_HALT | ENABLE`); the emulator reports `0x10003`
(`DEBUG_HALT | RESET | ENABLE`) — bit 1 (`RESET`) left set.

Root cause: `CoreDebugState.reset` defaults `true`
(`src/device/core_debug/mod.rs:314`, "initial state matches hardware
reset"). The runtime enable path
`Coordinator::enable_core()` (`src/interpreter/engine/coordinator.rs:383`)
mirrors to core-debug via `set_enabled(true)`
(`src/device/core_debug/mod.rs:602`), which is a bare
`self.enabled = enabled;` — it never touches `reset`. By contrast a CDO
`Core_Control=0x1` write routes through `write_control()`
(`src/device/core_debug/mod.rs:447`), which sets *both* bits from the
value and therefore clears `reset`. The two enable paths diverge; the
observed run was enabled via the runtime path, so `reset` stayed `true`.

### 1.2 Change

1. Add a method to `CoreDebugState` (`src/device/core_debug/mod.rs`):

   ```rust
   /// Enable the core via the same register semantics as a CDO
   /// `Core_Control = 0x1` write: sets `enabled`, clears `reset`.
   /// Used by the runtime enable path so it cannot diverge from the
   /// CDO write path (Core_Status RESET-bit fidelity, §8 close-out).
   pub fn enable(&mut self) {
       self.write_control(CTRL_ENABLE_MASK);
   }
   ```

   This encapsulates the module-private `CTRL_ENABLE_MASK`
   (`src/device/core_debug/mod.rs:122`) so the coordinator never sees
   the raw mask.

2. In `Coordinator::enable_core()`
   (`src/interpreter/engine/coordinator.rs:383`), replace
   `tile.core_debug.set_enabled(true)` with `tile.core_debug.enable()`.
   The interpreter-side `core.enabled = true` line is unchanged.

3. `disable_core()` is **unchanged** — it stays on `set_enabled(false)`.
   Disabling a core does not re-assert `reset` on hardware (Core_Control
   bits 0 and 1 are independent), and the disable path is not the
   indicted path.

`set_enabled()` is retained (still used by `disable_core` and any
false-path); it is simply no longer the enable path, so the divergence
cannot recur through enable.

### 1.3 Why this is correct, not a patch

After the change the runtime enable path is byte-identical to a CDO
`Core_Control=0x1` write: `write_control(CTRL_ENABLE_MASK)` →
`enabled=true, reset=false`. Because `reset` is `false`, the
reset-clears-runtime-state block in `write_control` is correctly
skipped — `done`, `pc`, halt-causes, etc. are *not* clobbered (the core
is being enabled, not reset). `Core_Status` while halted at the trap
becomes `0x10001`, matching silicon. The fix unifies the two enable
semantics so the bug cannot reappear via a future `set_enabled` caller —
this is the "derive correctly / design for flexibility" resolution, not
a symptom patch at one call site.

### 1.4 Fallout

Tests that enable a core and then inspect `is_reset()`,
`read_control()`, or `Core_Status` may encode the old (incorrect)
RESET-set expectation. No test hardcodes the literal `0x10003` /
`65539` (verified), so the blast radius is small. The implementer
audits `enable_core` / `core_debug` test sites during TDD and corrects
stale expectations — these corrections are the *intended* behavior
change (§8 states the EMU was wrong here).

---

## 2. Item 2 — Unified `OUTBUF_ADDR` + `TRAP_PC` derivation (Approach 1)

### 2.1 The constants (restated)

The Phase A probe (`../mlir-aie/test/npu-xrt/debug_halt_probe/`) carries
two hand-derived magic constants of the same fragility class:

- `OUTBUF_ADDR` — `static constexpr uint32_t OUTBUF_ADDR = 0x0400;`
  (`test.cpp:54`). The tile-local address of `output_buffer[0]`, used to
  build OP_READ control-packet headers (host-side only).
- `TRAP_PC` — the PC of the VLIW bundle that stores the `0xBB` marker to
  `output_buffer[1]`, encoded into the `@seq` `PC_Event0` write32 as
  `PC_Event0 = 0x80000000 | (TRAP_PC & 0x3FFF)` (`aie.mlir`). Arms the
  breakpoint.

Both were hand-derived from `llvm-objdump-aie` of the compiled core ELF
(`p0 = 0x70400`; `OUTBUF_ADDR = p0 & 0xFFFF = 0x0400`; trap store bundle
at `0x184`). They are self-documenting (re-derive warnings) and
partially self-checking (the EMU no-trap marker readback), but a kernel
or allocation change can silently desync them.

### 2.2 Governing principle: golden-record-preserving

The probe is a **permanent Phase A artifact whose exact committed bytes
produced the G1 and G2 verdicts**. Those bytes must remain in git as the
record. Therefore derivation does not blindly rewrite the committed
files — it runs every build and **guards** them: re-derive, compare to
the committed golden constants, and fail loudly on drift.

This satisfies "derived, not hardcoded" (the value is computed and
verified on every build) while preserving the artifact bytes and Phase A
integrity. It is safe because: `OUTBUF_ADDR` is host-only (no effect on
the AIE compile), and `TRAP_PC` changes only a `@seq` runtime-sequence
`write32` immediate (not compute-core codegen) — so re-deriving and
regenerating is schedule-neutral for the core the probe measured.

### 2.3 Components

**Templates** (committed alongside the golden concrete files):

- `../mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp.in` — `test.cpp`
  with the literal `0x0400` replaced by the token `@OUTBUF_ADDR@`.
- `../mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir.in` — `aie.mlir`
  with the `PC_Event0` immediate replaced by `@PC_EVENT0_VALUE@` and any
  in-comment `TRAP_PC`/`OUTBUF_ADDR` derivation notes left intact
  (comments are documentation, not substituted).

The committed `test.cpp` and `aie.mlir` remain in git **unchanged** as
the golden record.

**Derivation tool:** `xdna-emu/tools/debug-halt-probe-derive.py` (lives
in our tree per the working-directory convention, not in the mlir-aie
tree). Given the probe project directory after a first aiecc compile:

- `OUTBUF_ADDR`: run `llvm-nm-aie` (fallback `nm`) on
  `*_core_*.elf`; read the absolute value of the `output_buffer`
  symbol; `OUTBUF_ADDR = value & 0xFFFF`. (Verified: `nm` yields
  `00070400 A output_buffer` → `0x0400`. This is the clean "from a
  symbol" path; no disasm scrape for this constant.)
- `TRAP_PC`: run `llvm-objdump-aie -d` on the core ELF; locate the
  bundle that stores the `0xBB` immediate to `[p0, #4]` where `p0` was
  materialized as the `output_buffer` base address; emit that bundle's
  PC. Derive `PC14 = TRAP_PC & 0x3FFF` and
  `PC_EVENT0_VALUE = 0x80000000 | PC14`. (No symbol exists for an
  arbitrary instruction; a structured disasm scrape is unavoidable for
  this constant, but it is deterministic given the kernel.)
- **Error handling:** if the symbol is absent, or the trap store bundle
  cannot be located unambiguously, the tool **exits non-zero with a
  diagnostic** — it never emits a default or guessed value.

**Single-pass guarded build inside `run.lit`** (revised 2026-05-19
during execution — see §2.5 for why this replaced the original two-pass
design):

1. Compile the probe once: the existing `aiecc` invocation on
   `aie_arch.mlir` (the committed golden `aie.mlir` with `NPUDEVICE`
   substituted), producing `aie_arch.mlir.prj/*core*.elf`.
2. **Derive + guard** (a `run.lit` step after `aiecc`, before the host
   `clang`/run): run `debug-halt-probe-derive.py` against the produced
   core ELF → derived `OUTBUF_ADDR`, `PC_EVENT0_VALUE`; regenerate
   `aie.mlir` / `test.cpp` from the `.in` templates; diff regenerated vs
   committed golden.
   - **Equal:** the bytes just compiled *are* the committed golden
     (Phase A integrity intact); the step exits 0 and the host
     `clang`/run lines proceed.
   - **Different:** the tool exits non-zero, the `run.lit` step **fails
     the test loudly** (the host compile/run lines do not execute),
     printing committed-vs-derived and the instruction: *if the
     kernel/allocation change was intentional, commit the regenerated
     files as the new golden record and re-run.*

Why one compile suffices: the committed golden `aie.mlir` is the only
kernel source, and the tool emits golden bytes verbatim on a match — so
a separate scratch first pass is redundant. The guard runs *after* the
single compile rather than before; on a match this is identical (golden
was what compiled), and on drift the build still fails loudly and never
ships drifted artifacts (worst case: one doomed compile is wasted before
the guard aborts the test). The constants are still computed and
verified on every build; silent rot is still impossible.

### 2.5 Why single-pass (revised from two-pass during execution)

The originally-approved design used a *two-pass* `run.lit`: a scratch
`cp %S/aie.mlir scratch_arch.mlir` + first `aiecc`, derive, then a
second `aiecc` of the verified golden. Implementation surfaced that this
is incompatible with `emu-bridge-test.sh` — the probe's **primary**
validation path (the XRT bridge is the real validation target;
`debug_halt_probe` has committed bridge results). The bridge harness
**skips all `cp` RUN lines** (`emu-bridge-test.sh:719`) and instead
**synthesizes a single `aie_arch.mlir`** from canonical `aie.mlir`
itself (≈ lines 1522-1535); it never creates `scratch_arch.mlir`, so the
two-pass Pass-1 (`sed`/`aiecc scratch_arch.mlir`) fails under the bridge.
The single-pass design needs no scratch `cp` and a single `aiecc` — the
exact shape the bridge harness already supports — so it is compatible
with **both** upstream `llvm-lit` and the bridge harness, while
remaining functionally equivalent on the match path and equally loud on
drift. (Maya, 2026-05-19: chose single-pass over extending the bridge
harness.)

### 2.4 What this is *not*

It is not approach 2 (restructuring the probe so `TRAP_PC` is a symbol):
that risks changing the compiled schedule G1 measured, which is an
unacceptable risk to the committed artifact. It is not the
"TRAP_PC self-check only" middle ground (explicitly rejected when the
unified-derivation scope was chosen).

---

## 3. Item 1 — G1 §8 bookkeeping

The §8 "G1 halt-timing (bounded escalation)" bullet describes a
contingency ("if the HW core does not halt … ship the after-commit model
as an explicit assumption") that **never fired**: findings
`docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`
conclusion (a) records *"G1 — synchronous-trap halt boundary:
DERIVED and SHIPPED."* This is not open debt; it is a tracker the data
already retired.

Edits (all bookkeeping, no logic):

- **Parent spec `2026-05-18-debug-halt-design.md` §8:**
  - "G1 halt-timing (bounded escalation)" bullet → prepend
    **`RESOLVED (2026-05-19): contingency never fired — G1 DERIVED and
    SHIPPED, see findings conclusion (a).`**
  - "`OUTBUF_ADDR`" bullet → prepend
    **`RESOLVED (2026-05-19): unified build-time derivation — see
    docs/superpowers/specs/2026-05-19-debug-halt-section8-closeout-design.md.`**
  - "`Core_Status` RESET-bit EMU/HW divergence" bullet → prepend
    **`RESOLVED (2026-05-19): enable path routed through write_control —
    see 2026-05-19-debug-halt-section8-closeout-design.md.`**
  - Count-step silicon-fidelity and Resume hardware-verification bullets
    are **left unchanged** (stay OPEN).
- **Findings doc:** the `Core_Status` divergence note gets a one-line
  RESOLVED addendum (EMU now reports `0x10001`).
- **`crates/xdna-archspec/src/coverage/units.rs`:** if the `debug_halt`
  narrative cites any of the three resolved items as open, update it to
  read "3 §8 commitments resolved 2026-05-19; count-step silicon-fidelity
  and resume HW-verification remain OPEN (HW-budget-gated)." Then
  regenerate coverage artifacts via
  `cargo run -p xdna-archspec --example gen_coverage_artifacts` —
  expect zero drift except the narrative line. `debug_halt` stays
  `Modeled { Full }`.

---

## 4. Testing

- **Item 3 (TDD):**
  - `src/device/core_debug/tests.rs`: failing test first — after
    `enable()`, `is_reset()` is `false` and `read_control()` has the
    RESET bit clear; with the core halted, `Core_Status` has bit 1
    clear (`0x10001` semantics, not `0x10003`).
  - A coordinator-level test: `enable_core(col,row)` then the tile's
    `core_debug.is_reset()` is `false`.
  - Audit and correct any pre-existing test that enabled a core and
    asserted the old RESET-set state.
- **Item 2:**
  - The build-time guard *is* the regression — exercised whenever the
    probe builds under lit or `emu-bridge-test.sh`.
  - Unit test for the derive tool's parsers: feed captured `nm` and
    `llvm-objdump-aie -d` fixture snippets (stored under
    `xdna-emu/tools/`, never `/tmp`); assert `OUTBUF_ADDR == 0x0400`
    and `TRAP_PC == 0x184` / `PC_EVENT0_VALUE == 0x80000184`.
  - A negative test: malformed/empty disasm → tool exits non-zero, no
    value emitted.
- **Coherence:** full `cargo test --lib` green; coverage artifact regen
  zero-drift except the narrative line; `debug_halt` remains
  `Modeled { Full }`.

---

## 5. Component boundaries and file structure

**Modify:**

- `src/device/core_debug/mod.rs` — add `enable()`.
- `src/interpreter/engine/coordinator.rs` (~line 383) — `enable_core`
  uses `enable()`.
- `src/device/core_debug/tests.rs` — Item 3 tests + stale-expectation
  corrections.
- `../mlir-aie/test/npu-xrt/debug_halt_probe/run.lit` — single-pass
  build with a post-`aiecc` derive + guard step (§2.5).
- `docs/superpowers/specs/2026-05-18-debug-halt-design.md` — §8 pointer
  updates (Item 1).
- `docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`
  — Core_Status RESOLVED addendum.
- `crates/xdna-archspec/src/coverage/units.rs` — narrative; regenerate
  artifacts.

**Create:**

- `../mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp.in`
- `../mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir.in`
- `xdna-emu/tools/debug-halt-probe-derive.py` + parser unit test +
  `nm`/`objdump` fixture snippets under `xdna-emu/tools/`.

The committed `test.cpp` / `aie.mlir` remain unchanged as the golden
record.

**Execution:** subagent-driven development. Order: Item 3 (Core_Status,
tightest, fidelity bug) first; then Item 2 (derivation); Item 1
bookkeeping folded into whichever commit closes §8 last. Spec/plan/
findings coherence committed before implementation per the established
rhythm.

---

## 6. Out of scope (stays OPEN, by design)

- **Count-step silicon-fidelity.** Decrement cadence, larger-N, exact
  instruction boundary, `0x11`-on-silicon — only `N=4` observable
  (`LANDED:0`). HW-budget-gated; stays a tracked goal in §8 and the
  coverage narrative.
- **Resume hardware-verification.** Needs a dedicated silicon probe
  pass. HW-budget-gated; stays a tracked goal.

Neither is touched by this work; both remain discoverable in §8 and the
`debug_halt` coverage narrative.
