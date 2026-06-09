# Consumer-Side Bypass Matrix — Vector Register Result Visibility

**Date:** 2026-06-09
**Status:** Design approved (Approach A), spec under review
**Supersedes:** the producer-side `def_bypass` model and the consumer-keyed
"store-lag" experiment currently in the working tree (Task #108).

## 1. Problem

AIE2 vector-register results do not become visible to consumers at a fixed
issue+N. Visibility depends on a forwarding (bypass) network whose latency is a
function of *both* the producer and the consumer:

- the producer's result-availability cycle (`l_def`) and result bypass class
  (`def_bypass`), and
- the consumer's per-operand read cycle (`use_cycle`) and per-operand bypass
  class (`use_bypass`).

The decisive evidence is that latency is **consumer-keyed**, not producer-keyed.
The same producer `VMOV x, bml` feeds:

- a compute consumer (`VEXTRACT`, reads its vector operand at use_cycle 2) at
  **issue+1** (the `matrix_multiplication_using_cascade.cascade` prologue), but
- a store consumer (`VST`, reads stored data at use_cycle 1) at **issue+2**
  (the `vec_mac_bf16` epilogue).

A producer-side-only model (`def_bypass` attached to the write) cannot
distinguish these two consumers and corrupts one or the other. The interim
"store-lag" hack approximated the consumer side with a binary
(`use == Mov ? -1 : 0`); it is empirically correct on every *currently testable*
kernel but collapses two independent quantities (`use_cycle` and bypass `match`)
into one flag and cannot express cases we don't yet exercise (per-operand
`use_cycle` variation within one instruction, same-domain `match` forwarding,
and `l_def` variation). Per the project fidelity goal — get the model right now,
while Phoenix hardware is available to validate against — we build the full
matrix.

## 2. The Forwarding Model

For a producer P writing a vector register and a consumer C reading it as source
operand *k*:

```
match      = (def_bypass(P) == use_bypass(C, k)) && def_bypass(P) != NoBypass
eff        = max(1, l_def(P) - use_cycle(C, k) + 1 - (match ? 1 : 0))
visible_at = issue_bundle(P) + eff
```

- `l_def(P)` = itinerary `operand_cycles[0]` of the producer (result
  availability cycle). Already extracted.
- `def_bypass(P)` = itinerary `Forwardings[FirstOperandCycle]` of the producer.
  Already extracted.
- `use_cycle(C, k)` = itinerary `operand_cycles[num_defs + k]` of the consumer.
  **New extraction.**
- `use_bypass(C, k)` = itinerary `Forwardings[FirstOperandCycle + num_defs + k]`
  of the consumer. **New extraction.**

The `+1` is the bundle-clock convention constant (visibility expressed as
issue+N over the issued-bundle clock, not LLVM's relative-cycle convention);
it is calibrated against the two anchors below and held fixed.

**Anchor derivation** (`VMOV x, bml`: `l_def = 2`, both consumers `match = 0`):

| Consumer  | use_cycle | eff = 2 - use_cycle + 1 - 0 | visible_at | Observed |
|-----------|-----------|-----------------------------|------------|----------|
| `VST`     | 1         | 2                           | issue+2    | issue+2  |
| `VEXTRACT`| 2         | 1                           | issue+1    | issue+1  |

Both fall out of the formula with no special-casing. The clamp `max(1, …)`
enforces pure-VLIW read-old within the producer's own bundle (the existing
per-bundle snapshot already guarantees issue+0 reads the old value; the clamp
keeps the pending-overlay consistent with it).

The exact per-opcode constants (`l_def`, `def_bypass`, `use_cycle`,
`use_bypass`) are all derived from the LLVM itinerary — never hardcoded. Only the
`+1` bundle-clock offset is a calibrated convention constant, documented as such.

### 2.1 Scope boundary: vector register file only

This model governs the **W/X vector register file** (`MOV_Bypass` / `NoBypass`
results). The **accumulator / CM-domain file** (`VEC_Bypass`, MAC/MUL results)
is deferred: those writes currently flow through
`ExecutionContext::queue_matmul_accum_write` with its own pipeline-latency model
(commit 2958ba6). That path keeps its existing behavior and carries a prominent
`FIXME(bypass-model)` noting that when the accumulator file joins this model,
(a) the `Bypass::Vec` case must be distinguished in `LatencyTable::def_bypass`
and `use_bypass`, and (b) the MAC accumulator timing may need to move here. We
do **not** change accumulator timing in this work.

## 3. Architecture (Approach A: attach at decode time)

Forwarding metadata is resolved once at decode time, where the TableGen operand
ordering is unambiguous, and stored on the `SlotOp` aligned with its `sources`.
This mirrors the producer side, which already attaches `result_latency` /
`result_bypass` per slot via `ExecutionContext`.

Data flow:

```
llvm-aie itinerary
   │  (FFI, once at init)
   ▼
RawInstrInfo / InstrInfo          ← per-opcode arrays: operand_cycles[], operand_bypass[], num_defs
   │
   ▼
LatencyTable                      ← use_cycle(opcode, k), use_bypass(opcode, k) accessors
   │  (decode time, in build_slot_op)
   ▼
SlotOp.source_forward[k] = (use_cycle, use_bypass)   ← aligned with SlotOp.sources
   │  (execute time, at every vector read)
   ▼
VectorRegisterFile::resolve(reg, use_cycle, use_bypass)
   │
   ▼
visible_at(pending_write, use_cycle, use_bypass)     ← the formula in §2
```

### 3.1 FFI extraction (`aie2_decoder.{h,cpp}`, `decoder_ffi.rs`)

Extend `Aie2InstrInfo` / `aie2_get_instr_info` to emit, per opcode:

- `operand_cycle[i]` for `i` in `0..min(num_operand_cycles, AIE2_MAX_OPERANDS)`
  via `g_iid.getOperandCycle(sched_class, i)`,
- `operand_bypass[i]` via `g_iid.Forwardings[FirstOperandCycle + i]` (guarded by
  `FirstOperandCycle + i < LastOperandCycle`),
- `num_operand_cycles` = `LastOperandCycle - FirstOperandCycle` (so Rust knows
  the valid range).

`num_defs` is already emitted. The existing `latency` (= `operand_cycle[0]`) and
`def_bypass` (= `operand_bypass[0]`) fields are retained — they remain the
producer-side source of truth and now also fall out of the arrays, so they can
be derived from the arrays to avoid duplication, or kept as-is. Keep as-is to
minimize churn; assert consistency in a unit test.

`RawInstrInfo` / `InstrInfo` gain fixed-size arrays (`[i16; AIE2_MAX_OPERANDS]`
cycles, `[u16; AIE2_MAX_OPERANDS]` bypass) plus `num_operand_cycles: u8`.

### 3.2 LatencyTable accessors (`timing/latency.rs`)

Replace the scalar `llvm_def_bypass: Vec<u16>` with per-opcode arrays:

- `llvm_operand_cycles: Vec<[i16; N]>`
- `llvm_operand_bypass: Vec<[u16; N]>`
- `llvm_num_defs: Vec<u8>` (or reuse from InstrInfo)

New public methods:

- `use_cycle(opcode, source_idx) -> u8` → `operand_cycles[num_defs + source_idx]`
  clamped to the valid range, defaulting to **1** (read-at-issue) when out of
  range or unavailable. Default 1 is the conservative choice: it makes an
  unmodeled operand behave like a store-data read (latest visibility), avoiding
  spurious early forwarding.
- `use_bypass(opcode, source_idx) -> Bypass` →
  `operand_bypass[num_defs + source_idx]` mapped 0→No, nonzero→Mov (the same
  vector-register-only mapping as `def_bypass`, with the same `Vec` FIXME).

`def_bypass(opcode)` and the producer-side `latency` lookups are unchanged in
behavior (now backed by `operand_cycles[0]` / `operand_bypass[0]`).

### 3.3 SlotOp source_forward (`bundle.rs`, `slot_builder.rs`)

`SlotOp` gains:

```rust
/// Per-source forwarding metadata, aligned 1:1 with `sources`.
/// `source_forward[k] = (use_cycle, use_bypass)` for the consumer read of
/// `sources[k]`, derived from the itinerary at decode time. Empty when the
/// opcode has no itinerary data (legacy decode path) — reads then fall back
/// to the conservative default (use_cycle 1, NoBypass).
pub source_forward: SmallVec<[(u8, Bypass); N]>,
```

In `build_slot_op`, after sources are set, populate `source_forward` by walking
`sources` in order and calling `LatencyTable::use_cycle/use_bypass(opcode, k)`.
The `extra_sources` appended after the TableGen `input_order` operands (ag_*,
synthesized memory/pointer operands) do **not** correspond to itinerary operand
slots; they receive the default `(1, No)` and are never vector-register reads
in practice, so they never affect visibility.

### 3.4 Read-time resolve (`registers.rs`, `vector_helpers.rs`, `memory/mod.rs`)

`VectorRegisterFile`:

- `resolve(reg, use_cycle, use_bypass)` and
  `visible_at(w, use_cycle, use_bypass)` implement §2 (replacing the
  experiment's `resolve(reg, use_bypass)` / store-lag `visible_at`).
- The primary entry point becomes `read_with(reg, use_cycle, use_bypass)`, which
  the execute path always calls with explicit operand context from
  `source_forward`.
- `read(reg)` / `read_store(reg)` are retained only as convenience shims for
  callers without operand context (unit tests, legacy helpers), each documented
  with a fixed default pair (`read` → compute-like, `read_store` → `(1, No)`).
  No execute-path read relies on these defaults once threading is complete.

`vector_helpers.rs`:

- `read_vector_operand` / `get_vector_source` / `get_wide_vec_source` gain the
  `source_idx` (already available at most call sites) and look up
  `op.source_forward[source_idx]` to pass `(use_cycle, use_bypass)` into
  `resolve`. Where the helper currently scans by type, the `source_idx` is the
  position within `sources`, so the lookup aligns with `source_forward`.

`memory/mod.rs`:

- `read_store_data_wide` / `read_store_operand` for a `VectorReg` source look up
  the store instruction's `source_forward` for that operand (store-data is a
  source of the ST) and pass `(use_cycle, use_bypass)` — typically `(1, No)`,
  reproducing issue+2 from the formula rather than via a special `read_store`.

### 3.5 The operand-index mapping cross-check

The correctness of the whole model rests on `source_forward[k]` lining up with
the itinerary operand at MI index `num_defs + k`. This holds because `sources`
is built from TableGen `input_order` (operand_extraction.rs:730). Risks:

- **Tied operands** (e.g. VMAC accumulator is both def and use): the tied use
  may occupy a def slot in MI order. VMAC results go through the accumulator
  path (out of scope here), so vector-register reads are unaffected — but the
  cross-check must confirm no vector-register read lands on a tied/def slot.
- **extra_sources**: handled by the default (§3.3).
- **Immediates**: occupy itinerary slots but are never vector-register reads;
  default is harmless.

A debug-only decode-time cross-check (behind a cfg/env flag) validates, for the
target kernels, that each vector-register source's resolved `use_cycle` matches
the itinerary slot at `num_defs + k`, logging any mismatch. This is a
verification scaffold, not a runtime cost.

## 4. Testing & Validation

**TDD unit tests** (`registers.rs`), rewritten for the full formula:

- compute read (use_cycle 2, No def) → issue+1
- store read (use_cycle 1, No def) → issue+2
- matched forwarding (use_cycle 2, Mov def == Mov use) → issue+1 via `match`
  (eff = l_def-2+1-1, clamped)
- per-operand variation: one instruction reading two vector sources at
  different use_cycles resolves each independently
- `l_def` variation: a producer with `l_def = 3` shifts visibility accordingly
- clamp: eff ≤ 0 resolves to issue+1 (read-old in own bundle)
- latest-issued-write-wins, wide-write split, zero-latency immediate write
  (retained from the current set)

The current `test_bypass_nobypass_def_compute_read_visible_issue_plus_2` (which
encodes the *producer-keyed* expectation) is replaced — under the matrix a
NoBypass-def compute read resolves by `use_cycle`, not by producer class.

**Per-kernel gate** (`./scripts/emu-bridge-test.sh --no-hw --chess-only <name>`):
`vec_mac_bf16`, `two_col`, and all three
`matrix_multiplication_using_cascade` variants must PASS.

**Regression gate:** `cargo test --lib` clean, then a full `--no-hw` chess
sweep with **0 regressions** against the pre-change baseline.

**Hardware validation (Phoenix, gated):** capture the three target kernels on
the real NPU and confirm EMU matches. This is the reason to build the matrix now
— record the result as `Verified{evidence}` for the bf16 matmul class. Per
project rules, HW captures run outside the Claude sandbox.

## 5. Migration From Current Working Tree

The working tree holds the producer-side build plus the store-lag experiment
hack. Transition:

1. Keep: the `Bypass` enum, `PendingVecWrite` (producer fields `l_def`,
   `def_bypass`), `advance_bundle`, `queue_write`/`queue_write_wide`, the
   per-bundle snapshot, the `VECTOR_MAC_F = 6` float-MAC latency.
2. Extend: FFI arrays, `LatencyTable` accessors, `SlotOp.source_forward`,
   `build_slot_op` population, read-site threading.
3. Replace: `visible_at` / `resolve` signatures (the store-lag hack →
   the §2 formula), and the one producer-keyed unit test.
4. Remove: the `let _ = w.def_bypass;` experiment marker and its comment.

## 6. Commit Plan

1. **Float MAC latency** (`VECTOR_MAC_F = 6`) — already staged content; commit
   standalone with its bf16-MAC rationale.
2. **Consumer-side bypass matrix** — the model in this spec, as one cohesive
   commit (FFI + table + SlotOp + decode + resolve + tests), gated on §4.

Docs to update on landing: `tests/vector-verify/README.md` (final model),
`docs/known-fidelity-gaps.md` (accumulator/VEC_Bypass deferral row), and the
existing plan doc at
`docs/superpowers/plans/2026-06-09-vector-write-result-latency.md`.
