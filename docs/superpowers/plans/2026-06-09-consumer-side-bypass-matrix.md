# Consumer-Side Bypass Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Model AIE2 vector-register result visibility with the full per-operand
forwarding matrix — every vector read supplies `(use_cycle, use_bypass)` from the
consumer itinerary, resolved against each pending write's `(l_def, def_bypass)`.

**Architecture:** Extend the FFI to extract full per-operand cycle+bypass arrays
into `InstrInfo`. At decode time, `build_slot_op` reads those arrays from the
decoder's `instr_info` table and attaches `(use_cycle, use_bypass)` per source
onto a new `SlotOp.source_forward`. At execute time, every vector read passes its
operand's pair into `VectorRegisterFile::resolve`, which applies the forwarding
formula. The accumulator/`VEC_Bypass` file stays on its existing path with a
FIXME.

**Tech Stack:** Rust, C++ FFI (llvm-aie MCDisassembler/InstrItineraryData),
`smallvec`, the bridge-test harness for kernel validation.

**Spec:** `docs/superpowers/specs/2026-06-09-consumer-side-bypass-matrix-design.md`

**The forwarding formula (the heart):**
```
match      = (def_bypass(P) == use_bypass(C,k)) && def_bypass(P) != NoBypass
eff        = max(1, l_def(P) - use_cycle(C,k) + 1 - (match ? 1 : 0))
visible_at = issue_bundle(P) + eff
```

**Build order rationale:** Task 1 builds the pure, fully-unit-testable formula
core. Task 2 adds the FFI extraction. Task 3 **retires the load-bearing risk**
(the `num_defs + k` operand-index mapping) before any execute wiring depends on
it. Tasks 4–5 wire decode-time attach and read-time threading. Task 6 runs the
gates. Task 7 cleans up and splits commits.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/interpreter/state/registers.rs` | Forwarding formula + pending-write resolution | Modify: `visible_at`/`resolve`/`read_with`/shims + tests |
| `crates/xdna-archspec/decoder_ffi/aie2_decoder.h` | C struct for instr metadata | Modify: add per-operand arrays |
| `crates/xdna-archspec/decoder_ffi/aie2_decoder.cpp` | Itinerary extraction | Modify: fill per-operand arrays |
| `crates/xdna-archspec/src/aie2/isa/decoder_ffi.rs` | `RawInstrInfo`/`InstrInfo` + accessors | Modify: arrays + `use_cycle`/`use_bypass_raw` |
| `src/interpreter/bundle/slot.rs` | `SlotOp` definition | Modify: add `source_forward` field |
| `src/interpreter/decode/slot_builder.rs` | `build_slot_op` | Modify: populate `source_forward` |
| `src/interpreter/execute/vector_helpers.rs` | Vector source reads | Modify: thread `source_forward` into reads |
| `src/interpreter/execute/memory/mod.rs` | Store-data reads | Modify: thread `source_forward` into store reads |
| `tests/vector-verify/README.md` | Vector-verify docs | Modify (Task 7) |
| `docs/known-fidelity-gaps.md` | Fidelity gaps | Modify (Task 7): accumulator deferral row |

---

## Task 0: Land the float-MAC latency as a standalone commit

The working tree mixes three uncommitted strands: (a) the float-MAC bump
(`VECTOR_MAC_F = 6`), (b) the producer-side model (the `Bypass` enum,
`PendingVecWrite`, `llvm_def_bypass`, `result_bypass` plumbing — kept per spec
§5), and (c) the store-lag experiment hack (removed in Task 1). Split (a) out
first so it has its own reviewable commit, before the matrix work.

**Files:**
- `crates/xdna-archspec/src/aie2/mod.rs` (`VECTOR_MAC_F` ~149)
- `src/interpreter/timing/latency.rs` (`LATENCY_VECTOR_MAC_F` ~160-165)

- [ ] **Step 1: Confirm what the float-MAC change comprises**

Run: `git diff crates/xdna-archspec/src/aie2/mod.rs src/interpreter/timing/latency.rs`
Confirm the float-MAC lines (`VECTOR_MAC_F`, `LATENCY_VECTOR_MAC_F`, and any
`vector_matmul`/dispatch line that selects MAC-F for float) are separable from
the producer-side `llvm_def_bypass` additions in `latency.rs`.

- [ ] **Step 2: Stage only the float-MAC hunks**

Use `git add -p` to stage *only* the `VECTOR_MAC_F`/`LATENCY_VECTOR_MAC_F` hunks
(and the float-MAC routing line in `src/interpreter/execute/vector_matmul/mod.rs`
if present), leaving the `llvm_def_bypass` and `def_bypass()` hunks unstaged.

Run: `git add -p crates/xdna-archspec/src/aie2/mod.rs src/interpreter/timing/latency.rs src/interpreter/execute/vector_matmul/mod.rs`

- [ ] **Step 3: Verify the staged set is float-MAC-only**

Run: `git diff --cached`
Expected: only the float-MAC constant + its consumer. No `llvm_def_bypass`, no
`source_forward`, no `Bypass` enum.

- [ ] **Step 4: Commit**

```bash
git commit -m "interpreter: float (bf16/fp32) vector MAC/MUL latency = 6

II_VMACf / II_VMULf operand_cycles[0] = 6, one cycle longer than the integer
MAC for the float normalization stage. Fixes the bf16 compiled-matmul tile
phase shift.

Generated using Claude Code."
```

- [ ] **Step 5: Confirm the remaining tree still builds**

Run: `TMPDIR=/tmp/claude-1000 cargo build 2>&1 | tail -5`
Expected: builds clean (the producer-side blob + experiment hack remain
uncommitted, compiling as before).

---

## Task 1: Forwarding formula core in `registers.rs`

Rewrite the pending-write resolution to take per-operand `(use_cycle,
use_bypass)` and apply the §2 formula. This is pure logic, fully unit-testable
with hand-built writes — no FFI. Removes the store-lag experiment hack.

**Files:**
- Modify: `src/interpreter/state/registers.rs` (`visible_at` ~428, `resolve` ~441, `read`/`read_store`/`read_lane` ~498-524)
- Test: `src/interpreter/state/registers.rs` (tests module, the `test_bypass_*` group)

- [ ] **Step 1: Write the failing tests**

Replace the existing `test_bypass_*` group (including the producer-keyed
`test_bypass_nobypass_def_compute_read_visible_issue_plus_2`) with the matrix
tests. Add to the `tests` module:

```rust
// ── Forwarding matrix (consumer-keyed) ──────────────────────────────
// Producer VMOV-class: l_def = 2. Visibility per the itinerary formula
//   eff = max(1, l_def - use_cycle + 1 - match), match = def==use != No.

/// Helper: file with one pending write at issue bundle 0, advanced to `at`.
fn pending_at(l_def: u8, def: Bypass, at: u64) -> VectorRegisterFile {
    let mut f = VectorRegisterFile::new();
    f.queue_write(4, [0xAAAA_AAAA; 8], l_def, def);
    f.advance_bundle(at); // lands into regs only when issue+l_def <= at
    f.cur_bundle = at;    // ensure resolve sees `at` as current bundle
    f
}

#[test]
fn test_matrix_compute_read_l2_visible_issue_plus_1() {
    // use_cycle 2 (compute), No-def: eff = 2-2+1-0 = 1 -> visible at bundle 1.
    let f = pending_at(2, Bypass::No, 0);
    assert_eq!(f.read_with(4, 2, Bypass::No), [0u32; 8], "issue+0 reads old");
    let f1 = pending_at(2, Bypass::No, 1);
    assert_eq!(f1.read_with(4, 2, Bypass::No), [0xAAAA_AAAA; 8], "issue+1 visible");
}

#[test]
fn test_matrix_store_read_l2_visible_issue_plus_2() {
    // use_cycle 1 (store), No-def: eff = 2-1+1-0 = 2 -> visible at bundle 2.
    let f1 = pending_at(2, Bypass::No, 1);
    assert_eq!(f1.read_with(4, 1, Bypass::No), [0u32; 8], "issue+1 still old for store");
    let f2 = pending_at(2, Bypass::No, 2);
    assert_eq!(f2.read_with(4, 1, Bypass::No), [0xAAAA_AAAA; 8], "issue+2 visible");
}

#[test]
fn test_matrix_matched_forwarding_shaves_a_cycle() {
    // use_cycle 2, Mov-def == Mov-use: eff = 2-2+1-1 = 0 -> clamp 1 -> issue+1.
    let f1 = pending_at(2, Bypass::Mov, 1);
    assert_eq!(f1.read_with(4, 2, Bypass::Mov), [0xAAAA_AAAA; 8], "matched: issue+1");
}

#[test]
fn test_matrix_l_def_variation_shifts_visibility() {
    // l_def = 3, use_cycle 1 (store), No: eff = 3-1+1 = 3 -> issue+3.
    let f2 = pending_at(3, Bypass::No, 2);
    assert_eq!(f2.read_with(4, 1, Bypass::No), [0u32; 8], "l_def 3 store: still old at +2");
    let f3 = pending_at(3, Bypass::No, 3);
    assert_eq!(f3.read_with(4, 1, Bypass::No), [0xAAAA_AAAA; 8], "l_def 3 store: visible at +3");
}

#[test]
fn test_matrix_clamp_never_resolves_in_own_bundle() {
    // Even maximally-forwarded, eff clamps to >= 1: issue+0 always reads old.
    let f0 = pending_at(1, Bypass::Mov, 0);
    assert_eq!(f0.read_with(4, 2, Bypass::Mov), [0u32; 8], "own bundle reads old");
}

#[test]
fn test_matrix_per_operand_independence() {
    // Two reads of the same reg at the SAME bundle with different use_cycles
    // resolve independently (compute sees it, store does not).
    let f1 = pending_at(2, Bypass::No, 1);
    assert_eq!(f1.read_with(4, 2, Bypass::No), [0xAAAA_AAAA; 8], "compute (uc2) visible at +1");
    assert_eq!(f1.read_with(4, 1, Bypass::No), [0u32; 8], "store (uc1) not visible at +1");
}
```

Keep the existing `test_latest_issued_write_wins`, `test_wide_write_splits_and_resolves`,
and `test_zero_latency_writes_immediately` tests, updating any `resolve`/`read`
call that used the old `(reg, use_bypass)` signature to `read_with(reg, use_cycle, use_bypass)`
or the `read`/`read_store` shims as appropriate.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib state::registers::tests::test_matrix`
Expected: FAIL — `read_with` does not exist yet, store-lag `visible_at` gives wrong bundles.

- [ ] **Step 3: Implement the formula**

Replace `visible_at` (currently the experiment hack at ~428) with:

```rust
    /// Effective visibility bundle of an in-flight write to a consumer reading
    /// it at `use_cycle` with `use_bypass`. AIE2 forwarding-network latency:
    ///   eff = max(1, l_def - use_cycle + 1 - (match ? 1 : 0))
    /// where match = producer/consumer bypass classes equal and not NoBypass.
    /// The `+1` is the bundle-clock convention constant (calibrated against the
    /// VMOV->VST issue+2 / VMOV->VEXTRACT issue+1 anchors; see the design spec).
    /// Clamped to >= 1: a write never resolves within its own issue bundle.
    #[inline]
    fn visible_at(w: &PendingVecWrite, use_cycle: u8, use_bypass: Bypass) -> u64 {
        let matched = w.def_bypass == use_bypass && w.def_bypass != Bypass::No;
        let eff = (w.l_def as i64) - (use_cycle as i64) + 1 - (matched as i64);
        let eff = eff.max(1) as u64;
        w.issue_bundle + eff
    }
```

Replace `resolve` (~441) with the three-arg form:

```rust
    /// Resolve register `reg` for a consumer reading at `(use_cycle, use_bypass)`:
    /// the committed base value, overridden by the latest-issued in-flight write
    /// visible to this consumer at the current bundle.
    #[inline]
    fn resolve(&self, reg: u8, use_cycle: u8, use_bypass: Bypass) -> [u32; 8] {
        let idx = (reg & 0x1F) as usize;
        let mut val = self.active()[idx];
        for w in &self.pending {
            if (w.reg & 0x1F) as usize == idx
                && Self::visible_at(w, use_cycle, use_bypass) <= self.cur_bundle
            {
                val = w.value;
            }
        }
        val
    }

    /// Primary vector-read entry point. The execute path supplies the consuming
    /// operand's `(use_cycle, use_bypass)` from `SlotOp::source_forward`.
    #[inline]
    pub fn read_with(&self, reg: u8, use_cycle: u8, use_bypass: Bypass) -> [u32; 8] {
        self.resolve(reg, use_cycle, use_bypass)
    }
```

Update the shims (`read`, `read_store`, `read_lane`) to the new signature.
`read` is the compute-like default for callers without operand context (tests,
legacy helpers); `read_store` is the store default `(1, No)`:

```rust
    pub fn read(&self, reg: u8) -> [u32; 8] {
        // Compute-consumer default (use_cycle 2, MOV_Bypass). Only for callers
        // without operand context; the execute path uses read_with.
        self.resolve(reg, 2, Bypass::Mov)
    }
    // ...
    pub fn read_store(&self, reg: u8) -> [u32; 8] {
        self.resolve(reg, 1, Bypass::No)
    }
    // ...
    pub fn read_lane(&self, reg: u8, lane: u8) -> u32 {
        self.resolve(reg, 2, Bypass::Mov)[(lane & 0x07) as usize]
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib state::registers::tests`
Expected: PASS (all matrix + retained tests).

- [ ] **Step 5: Run the full lib suite for regressions**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: PASS. (The three target kernels' unit-adjacent behavior is unchanged
for the realistic `l_def in {1,2}` range — `read`/`read_store` defaults reproduce
the prior compute/store split.)

- [ ] **Step 6: Commit**

```bash
git add src/interpreter/state/registers.rs
git commit -m "interpreter: forwarding-matrix vector visibility (consumer use_cycle+bypass)

Generated using Claude Code."
```

---

## Task 2: FFI per-operand cycle+bypass extraction

Extend the FFI to extract the full per-operand `operand_cycle[]` /
`operand_bypass[]` arrays (not just operand 0) and expose `use_cycle` /
`use_bypass_raw` on `InstrInfo`.

**Files:**
- Modify: `crates/xdna-archspec/decoder_ffi/aie2_decoder.h` (struct ~77-85)
- Modify: `crates/xdna-archspec/decoder_ffi/aie2_decoder.cpp` (`aie2_get_instr_info` ~311-353)
- Modify: `crates/xdna-archspec/src/aie2/isa/decoder_ffi.rs` (`RawInstrInfo` ~60-78, `InstrInfo` ~276-328, `query_all_instr_info` ~335-389)
- Test: `crates/xdna-archspec/src/aie2/isa/decoder_ffi.rs` (tests module)

- [ ] **Step 1: Write the failing test**

Add to the `decoder_ffi.rs` tests module:

```rust
    #[test]
    fn test_per_operand_arrays_consistent_with_op0() {
        // The new per-operand arrays must agree with the existing operand-0
        // fields for every opcode that has itinerary data.
        let infos = query_all_instr_info();
        assert!(!infos.is_empty(), "FFI returned no instr info");
        let mut checked = 0;
        for info in &infos {
            if info.num_operand_cycles == 0 {
                continue;
            }
            // operand_cycle[0] == latency (both are operand_cycles[0]).
            if let Some(lat) = info.latency {
                assert_eq!(
                    info.operand_cycle[0], lat as i16,
                    "operand_cycle[0] disagrees with latency for sched_class {}",
                    info.sched_class
                );
            }
            // operand_bypass[0] == def_bypass (both are Forwardings[First]).
            assert_eq!(
                info.operand_bypass[0], info.def_bypass,
                "operand_bypass[0] disagrees with def_bypass for sched_class {}",
                info.sched_class
            );
            checked += 1;
        }
        assert!(checked > 0, "no opcodes had operand-cycle data");
    }

    #[test]
    fn test_use_cycle_defaults_when_out_of_range() {
        // A source index beyond the itinerary range yields the conservative
        // default of 1.
        let info = InstrInfo {
            flags: 0, num_defs: 1, latency: Some(2), stage_latency: None,
            sched_class: 0, def_bypass: 0,
            operand_cycle: [-1; AIE2_MAX_OPERANDS],
            operand_bypass: [0; AIE2_MAX_OPERANDS],
            num_operand_cycles: 0,
        };
        assert_eq!(info.use_cycle(0), 1, "out-of-range source defaults to use_cycle 1");
    }
```

- [ ] **Step 2: Run the test to verify it fails to compile**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec --lib decoder_ffi::tests::test_use_cycle 2>&1 | head -20`
Expected: FAIL — `operand_cycle`, `num_operand_cycles`, `use_cycle`, `AIE2_MAX_OPERANDS` not defined.

- [ ] **Step 3a: Extend the C struct**

In `aie2_decoder.h`, add to `struct Aie2InstrInfo` (after `def_bypass`):

```c
    uint16_t def_bypass;     // Forwarding-network id of result operand 0 (0 = NoBypass)
    // Per-operand itinerary data, indexed by MI operand position (defs then
    // uses). operand_cycle[i] is the cycle operand i is read/produced;
    // operand_bypass[i] its forwarding id. Valid for i < num_operand_cycles.
    int16_t operand_cycle[AIE2_MAX_OPERANDS];
    uint16_t operand_bypass[AIE2_MAX_OPERANDS];
    uint8_t num_operand_cycles;
```

- [ ] **Step 3b: Fill the arrays in the cpp**

In `aie2_decoder.cpp`, inside `aie2_get_instr_info`, replace the `def_bypass`
extraction block (the `if (g_iid.Itineraries && g_iid.Forwardings)` block) with:

```cpp
        // Per-operand itinerary cycles + forwarding ids. OperandCycles and
        // Forwardings are parallel arrays indexed by FirstOperandCycle + i,
        // matching MI operand order (defs then uses). def_bypass/latency stay
        // as the operand-0 shorthands the producer side already uses.
        if (g_iid.Itineraries) {
            const InstrItinerary &itin = g_iid.Itineraries[sched_class];
            unsigned n = (itin.LastOperandCycle > itin.FirstOperandCycle)
                             ? (itin.LastOperandCycle - itin.FirstOperandCycle)
                             : 0;
            if (n > AIE2_MAX_OPERANDS)
                n = AIE2_MAX_OPERANDS;
            out->num_operand_cycles = (uint8_t)n;
            for (unsigned i = 0; i < n; i++) {
                if (g_iid.OperandCycles)
                    out->operand_cycle[i] =
                        (int16_t)g_iid.OperandCycles[itin.FirstOperandCycle + i];
                if (g_iid.Forwardings)
                    out->operand_bypass[i] =
                        (uint16_t)g_iid.Forwardings[itin.FirstOperandCycle + i];
            }
            if (n > 0 && g_iid.Forwardings)
                out->def_bypass = out->operand_bypass[0];
        }
```

(The `memset(out, 0, sizeof(*out))` at the top already zeroes the arrays and
`num_operand_cycles`, so untouched slots are 0 and bounded by `num_operand_cycles`.)

- [ ] **Step 3c: Mirror the fields in `decoder_ffi.rs`**

Add `pub const AIE2_MAX_OPERANDS: usize = 16;` near the top of `decoder_ffi.rs`
(matching the C `#define`). Extend `RawInstrInfo` (the `#[repr(C)]` struct) with,
appended in the same order as the C struct:

```rust
    pub operand_cycle: [i16; AIE2_MAX_OPERANDS],
    pub operand_bypass: [u16; AIE2_MAX_OPERANDS],
    pub num_operand_cycles: u8,
```

Extend `InstrInfo` with the same three fields, and add the accessors:

```rust
    /// use_cycle for source operand `source_idx` (0-based among sources). Maps
    /// to itinerary operand index `num_defs + source_idx`. Returns the
    /// conservative default 1 (read-at-issue) when out of range — an unmodeled
    /// operand then behaves like a store-data read (latest visibility).
    pub fn use_cycle(&self, source_idx: usize) -> u8 {
        let mi = self.num_defs as usize + source_idx;
        if mi < self.num_operand_cycles as usize {
            let c = self.operand_cycle[mi];
            if c >= 0 {
                return c as u8;
            }
        }
        1
    }

    /// Raw forwarding id for source operand `source_idx` (0 = NoBypass).
    /// Caller maps to a named `Bypass` per the destination register file (the
    /// nonzero->MOV_Bypass mapping is only unambiguous for vector-register reads
    /// — see the VEC_Bypass FIXME at queue_matmul_accum_write).
    pub fn use_bypass_raw(&self, source_idx: usize) -> u16 {
        let mi = self.num_defs as usize + source_idx;
        if mi < self.num_operand_cycles as usize {
            return self.operand_bypass[mi];
        }
        0
    }
```

In `query_all_instr_info`, initialize the new `RawInstrInfo` fields in the
zero-init literal (`operand_cycle: [0; AIE2_MAX_OPERANDS]`, `operand_bypass:
[0; AIE2_MAX_OPERANDS]`, `num_operand_cycles: 0`), copy them into both the
ok-path and the failure-path `InstrInfo` construction (failure path uses
`operand_cycle: [-1; AIE2_MAX_OPERANDS]`, `operand_bypass: [0; ...]`,
`num_operand_cycles: 0`), and on the ok path copy `raw.operand_cycle` etc.
through.

- [ ] **Step 4: Rebuild the FFI and run the test**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec --lib decoder_ffi::tests`
Expected: PASS (the C++ is rebuilt by the build script; both new tests pass).

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/decoder_ffi/aie2_decoder.h \
        crates/xdna-archspec/decoder_ffi/aie2_decoder.cpp \
        crates/xdna-archspec/src/aie2/isa/decoder_ffi.rs
git commit -m "archspec: extract full per-operand itinerary cycle+bypass arrays

Generated using Claude Code."
```

---

## Task 3: Retire the operand-index mapping risk

Before any execute wiring depends on it, prove that `num_defs + k` maps a vector
source operand to the right itinerary slot, on the real target kernels. This is a
diagnostic test, not production code — it decodes the kernels' key instructions
and asserts the mapping is sane (vector sources land on use slots with plausible
use_cycles, never on a def/tied slot).

**Files:**
- Test: `src/interpreter/decode/operand_extraction.rs` (tests module) — a focused decode-mapping test.

- [ ] **Step 1: Write the diagnostic test**

The `VMOV x, bml` / `VST` / `VEXTRACT` opcodes are the anchors. Decode each from a
known encoding (reuse an existing decode-test helper in this module — search for
`fn decode_slot` / existing `decode` test helpers) and assert:

```rust
    #[test]
    fn test_vector_source_operand_index_mapping() {
        // The forwarding matrix relies on source k mapping to itinerary slot
        // num_defs + k. Validate on the anchor instructions: a vector source's
        // resolved use_cycle must be in the plausible range [1, 4] and the
        // instruction must report num_operand_cycles > num_defs (i.e. uses are
        // modeled). Guards against tied/def-slot collisions.
        let decoder = InstructionDecoder::load_cached();
        // Anchor encodings (slot, bits) — reuse the constants/helpers the other
        // decode tests in this module already use; if none exist for these,
        // decode via decoder.decode() over a captured bundle from the cascade
        // and bf16 build dirs.
        for (name, info) in anchor_instr_infos(&decoder) {
            assert!(
                info.num_operand_cycles as usize > info.num_defs as usize,
                "{name}: no modeled use operands (num_operand_cycles {} <= num_defs {})",
                info.num_operand_cycles, info.num_defs
            );
            for k in 0..(info.num_operand_cycles as usize - info.num_defs as usize) {
                let uc = info.use_cycle(k);
                assert!((1..=4).contains(&uc), "{name}: source {k} use_cycle {uc} out of plausible range");
            }
        }
    }
```

Implement `anchor_instr_infos` to return `(&str, &InstrInfo)` for the VMOV-x,
VST, and VEXTRACT opcodes by name via `decoder.get_instr_info(opcode)`, resolving
opcodes through the decoder's opcode-name table (search this module / `decoder.rs`
for an existing name→opcode helper; if absent, decode a real captured bundle and
read `slot_op.llvm_opcode`).

- [ ] **Step 2: Run it (this is the risk gate)**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib decode::operand_extraction::tests::test_vector_source_operand_index_mapping -- --nocapture`
Expected: PASS. **If it fails**, STOP and investigate — the `num_defs + k`
assumption is broken for one of the anchors (tied operand, unexpected operand
order). Do not proceed to Task 4 until the mapping is understood; the spec's §3.5
notes VMAC's tied accumulator as the prime suspect, but VMAC is out of scope so
the vector-register anchors should be clean.

- [ ] **Step 3: Commit**

```bash
git add src/interpreter/decode/operand_extraction.rs
git commit -m "test: prove vector source operand-index mapping on anchor instrs

Generated using Claude Code."
```

---

## Task 4: `SlotOp.source_forward` + decode-time population

Add the per-source forwarding field and populate it in `build_slot_op` from the
decoder's `instr_info` table.

**Files:**
- Modify: `src/interpreter/bundle/slot.rs` (`SlotOp` struct ~233-290, constructors `nop` ~481, `from_semantic` ~560)
- Modify: `src/interpreter/decode/slot_builder.rs` (`build_slot_op` ~118-126)
- Test: `src/interpreter/decode/slot_builder.rs` or `operand_extraction.rs` tests

- [ ] **Step 1: Write the failing test**

Add a decode test asserting `source_forward` aligns 1:1 with `sources` and
carries the itinerary `use_cycle` for a known instruction (use the same anchor
decode helper from Task 3):

```rust
    #[test]
    fn test_source_forward_aligned_with_sources() {
        let decoder = InstructionDecoder::load_cached();
        let slot_op = decode_anchor_vst(&decoder); // helper: decodes a VST bundle
        assert_eq!(
            slot_op.source_forward.len(),
            slot_op.sources.len(),
            "source_forward must align 1:1 with sources"
        );
        // The stored-vector-data source of a VST reads at use_cycle 1 (store).
        let opcode = slot_op.llvm_opcode.expect("anchor has opcode");
        let info = decoder.get_instr_info(opcode).expect("anchor has instr info");
        for (k, (uc, _ub)) in slot_op.source_forward.iter().enumerate() {
            assert_eq!(*uc, info.use_cycle(k), "source_forward[{k}].use_cycle mismatch");
        }
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib test_source_forward_aligned_with_sources 2>&1 | head`
Expected: FAIL — `source_forward` field does not exist.

- [ ] **Step 3: Add the field**

In `slot.rs`, add the import `use xdna_archspec::aie2::Bypass;` (if not present)
and add to `struct SlotOp`:

```rust
    /// Per-source forwarding metadata, aligned 1:1 with `sources`.
    /// `source_forward[k] = (use_cycle, use_bypass)` for the consumer read of
    /// `sources[k]`, derived from the itinerary at decode time. Empty when the
    /// opcode has no itinerary data — reads then fall back to read()/read_store()
    /// defaults.
    pub source_forward: SmallVec<[(u8, Bypass); 4]>,
```

Initialize it to `SmallVec::new()` in every `SlotOp { ... }` literal
(`nop` ~481 and `from_semantic` ~560 — grep `SlotOp {` in this file to find all).

- [ ] **Step 4: Populate it in `build_slot_op`**

In `slot_builder.rs`, after the `for src in sources { slot_op = slot_op.with_source(src); }`
loop, add:

```rust
        // Attach per-source forwarding metadata from the itinerary (consumer
        // side of the bypass network). source k maps to itinerary operand
        // num_defs + k. Vector-register reads consume this in resolve(); other
        // operand kinds carry it harmlessly. nonzero forwarding id -> MOV_Bypass
        // (vector-register reads only carry MOV/NoBypass; VEC_Bypass accumulator
        // reads go through the accumulator path -- see queue_matmul_accum_write
        // FIXME).
        if let Some(opcode) = slot_op.llvm_opcode {
            if let Some(info) = self.get_instr_info(opcode) {
                for k in 0..slot_op.sources.len() {
                    let uc = info.use_cycle(k);
                    let ub = match info.use_bypass_raw(k) {
                        0 => Bypass::No,
                        _ => Bypass::Mov,
                    };
                    slot_op.source_forward.push((uc, ub));
                }
            }
        }
```

Add `use xdna_archspec::aie2::Bypass;` to `slot_builder.rs` imports. Confirm
`slot_op.llvm_opcode` is set before `build_slot_op` returns — it is set at
`decoder.rs:391` *after* `build_slot_op` is called, so **move** the
`slot_op.llvm_opcode = ffi_opcode;` assignment to before `build_slot_op`, or pass
the opcode into `build_slot_op` as a parameter. Prefer passing it as a parameter
(`build_slot_op(..., llvm_opcode: Option<u32>)`) and setting it inside, so the
population sees it. Update the single call site at `decoder.rs:390`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib test_source_forward_aligned_with_sources`
Expected: PASS.

- [ ] **Step 6: Run the full lib suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: PASS (field added, nothing reads it yet — behavior unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/interpreter/bundle/slot.rs src/interpreter/decode/slot_builder.rs src/interpreter/decode/decoder.rs
git commit -m "interpreter: attach per-source forwarding metadata at decode time

Generated using Claude Code."
```

---

## Task 5: Thread `source_forward` into the read path

Replace `read`/`read_store` calls in the vector and store read paths with
`read_with`, supplying each operand's `source_forward` pair.

**Files:**
- Modify: `src/interpreter/execute/vector_helpers.rs` (`read_vector_operand` ~38, `get_vector_source` ~28, `get_wide_vec_source` ~437)
- Modify: `src/interpreter/execute/memory/mod.rs` (`read_store_data_wide`, `read_store_operand` VectorReg arms)
- Test: rely on the kernel gates in Task 6 (integration); add one focused unit test below.

- [ ] **Step 1: Write the failing test**

The cleanest direct assertion is a small helper test that a `SlotOp` with a
populated `source_forward` routes the pair into the read. Add to
`vector_helpers.rs` tests:

```rust
    #[test]
    fn test_get_vector_source_uses_source_forward() {
        let mut ctx = ExecutionContext::new();
        // Pending compute write to reg 3, l_def 2, No-def, at issue bundle 0.
        ctx.vector.queue_write(3, [0x1234_5678; 8], 2, crate::interpreter::state::Bypass::No);
        ctx.vector.advance_bundle(1);
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.sources = smallvec![Operand::VectorReg(3)];
        // Store consumer (use_cycle 1, No): not visible at issue+1.
        op.source_forward = smallvec![(1u8, crate::interpreter::state::Bypass::No)];
        assert_eq!(VectorAlu::get_vector_source(&op, &ctx, 0), [0u32; 8], "store read: old at +1");
        // Compute consumer (use_cycle 2, No): visible at issue+1.
        op.source_forward = smallvec![(2u8, crate::interpreter::state::Bypass::No)];
        assert_eq!(VectorAlu::get_vector_source(&op, &ctx, 0), [0x1234_5678; 8], "compute read: visible at +1");
    }
```

(Export `Bypass` from `crate::interpreter::state` if not already — it's re-exported
in `registers.rs`.)

- [ ] **Step 2: Run it to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib test_get_vector_source_uses_source_forward 2>&1 | head`
Expected: FAIL — `get_vector_source` ignores `source_forward`, returns the compute
default for both.

- [ ] **Step 3: Thread the pair through the read helpers**

Add a small helper on `VectorAlu` that resolves a source's `(use_cycle,
use_bypass)`, defaulting to compute `(2, Mov)` when `source_forward` is absent:

```rust
    /// Forwarding pair for source operand `idx`, from `source_forward`.
    /// Falls back to the compute default when the op carries no itinerary data.
    #[inline]
    fn source_fwd(op: &SlotOp, idx: usize) -> (u8, crate::interpreter::state::Bypass) {
        op.source_forward
            .get(idx)
            .copied()
            .unwrap_or((2, crate::interpreter::state::Bypass::Mov))
    }
```

Change `read_vector_operand` to take the resolved pair (or the `op` + `idx`).
Simplest: have `get_vector_source` resolve the pair and call a new
`read_vector_operand_fwd(operand, ctx, use_cycle, use_bypass)`:

```rust
    pub(super) fn get_vector_source(op: &SlotOp, ctx: &ExecutionContext, idx: usize) -> [u32; 8] {
        let (uc, ub) = Self::source_fwd(op, idx);
        op.sources.get(idx).map_or([0; 8], |src| Self::read_vector_operand_fwd(src, ctx, uc, ub))
    }
```

`read_vector_operand_fwd` mirrors `read_vector_operand` but the `VectorReg(r)` arm
calls `ctx.vector.read_with(*r, use_cycle, use_bypass)`. Keep the old
`read_vector_operand` as a thin wrapper passing the compute default, for the few
callers that lack op/idx context. Apply the same change to `get_wide_vec_source`
(call `ctx.vector.read_wide_with(*r, uc, ub)` — add a `read_wide_with` that
resolves each 256-bit half via `read_with`).

- [ ] **Step 4: Thread store reads in `memory/mod.rs`**

In `read_store_data_wide` and `read_store_operand`, the `VectorReg` arms currently
call `ctx.vector.read_store(*r)`. Replace with a lookup of the store
instruction's `source_forward` for that operand, falling back to the store default
`(1, No)`:

```rust
            Operand::VectorReg(r) => {
                // Store-data is a source operand of the ST; use its itinerary
                // forwarding pair (typically use_cycle 1, NoBypass -> issue+2).
                let (uc, ub) = op.source_forward.get(idx).copied()
                    .unwrap_or((1, xdna_archspec::aie2::Bypass::No));
                ctx.vector.read_with(*r, uc, ub)
            }
```

(Confirm the store read sites have the operand index `idx` and the `op`; if they
read a single fixed store-data operand, find its index in `op.sources` first.)

- [ ] **Step 5: Run the focused test + full suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: PASS (focused test green; no lib regressions).

- [ ] **Step 6: Commit**

```bash
git add src/interpreter/execute/vector_helpers.rs src/interpreter/execute/memory/mod.rs
git commit -m "interpreter: read vector operands through per-source forwarding pair

Generated using Claude Code."
```

---

## Task 6: Integration gates (kernels + sweep)

Rebuild the FFI `.so` and validate against the real kernels. No code changes
unless a gate fails.

**Files:** none (validation only), unless a failure requires a fix.

- [ ] **Step 1: Rebuild the plugin `.so`**

Run: `TMPDIR=/tmp/claude-1000 cargo build -p xdna-emu-ffi`
Expected: builds clean. (Bridge tests load `target/debug/libxdna_emu.so`.)

- [ ] **Step 2: Per-kernel bridge gates (chess, no HW)**

Run each and confirm PASS:
```bash
./scripts/emu-bridge-test.sh --no-hw --chess-only vec_mac_bf16
./scripts/emu-bridge-test.sh --no-hw --chess-only two_col
./scripts/emu-bridge-test.sh --no-hw --chess-only matrix_multiplication_using_cascade
```
Expected: `vec_mac_bf16` PASS, `two_col` PASS, all three
`matrix_multiplication_using_cascade` variants (plain/buffer/cascade) PASS.

If any fails: this is the real itinerary disagreeing with the prior store-lag
approximation. Use the memory-watch / VECDBG path from CLAUDE.md to find the
specific instruction whose `use_cycle` shifted, and verify against the chess
`.lst` disassembly. Do not patch the formula to force a pass — confirm the
itinerary value is what HW does.

- [ ] **Step 3: Full lib suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: PASS.

- [ ] **Step 4: Full `--no-hw` chess sweep (regression gate)**

Run: `./scripts/emu-bridge-test.sh --no-hw --chess-only` (background it; redirect
to a log under `build/`, do not pipe through tail).
Expected: **0 regressions** vs the pre-change baseline. Compare against the
known-good set; any newly-failing kernel is a regression to root-cause before
proceeding.

- [ ] **Step 5: Commit (validation note only, if anything was adjusted)**

If Step 2–4 required a code fix, commit it with a message naming the kernel and
the itinerary fact that drove it. Otherwise no commit.

---

## Task 7: Cleanup, docs, and final commit split

Remove dead experiment scaffolding, update docs, and confirm the commit history
matches the intended split.

**Files:**
- Modify: `tests/vector-verify/README.md`
- Modify: `docs/known-fidelity-gaps.md`
- Modify: `docs/superpowers/plans/2026-06-09-vector-write-result-latency.md` (status note)

- [ ] **Step 1: Verify no experiment markers remain**

Run: `grep -rn "EXPERIMENT\|store-lag\|let _ = w.def_bypass" src/ crates/`
Expected: no matches (the Task 1 rewrite removed them). Fix any stragglers.

- [ ] **Step 2: Verify the FIXME(bypass-model) is intact**

Run: `grep -rn "FIXME(bypass-model)" src/`
Expected: the block at `ExecutionContext::queue_matmul_accum_write` is present and
references the `Bypass::Vec` / accumulator-timing deferral. Confirm
`LatencyTable::def_bypass` and the `use_bypass_raw->Bypass` mapping both note the
nonzero->Mov assumption is vector-register-only.

- [ ] **Step 3: Update `tests/vector-verify/README.md`**

Replace any store-lag / producer-keyed description with the matrix model: each
vector read supplies `(use_cycle, use_bypass)` from the consumer itinerary,
resolved against pending writes via `eff = max(1, l_def - use_cycle + 1 - match)`.
Note the accumulator/`VEC_Bypass` deferral.

- [ ] **Step 4: Add a known-fidelity-gaps row**

In `docs/known-fidelity-gaps.md`, add a row: accumulator/CM-domain
(`VEC_Bypass`) result visibility is NOT yet modeled by the forwarding matrix — it
uses the separate MAC-pipeline-latency path (`queue_matmul_accum_write`).
Validated-against-HW status pending the Phoenix capture.

- [ ] **Step 5: Confirm commit split**

Run: `git log --oneline -8`
Expected, in order (oldest first within this work):
1. `spec: consumer-side bypass matrix ...` (Task 0, already committed)
2. The float-MAC `VECTOR_MAC_F = 6` commit — **verify it exists as its own
   commit**. The working tree at plan start had it staged-but-uncommitted inside
   the broader diff; if it is not yet a standalone commit, create it now from the
   `crates/xdna-archspec/src/aie2/mod.rs` + `latency.rs` MAC-F lines BEFORE the
   matrix commits, or confirm it already landed.
3. Tasks 1–5 matrix commits.
4. This docs commit.

- [ ] **Step 6: Commit docs**

```bash
git add tests/vector-verify/README.md docs/known-fidelity-gaps.md \
        docs/superpowers/plans/2026-06-09-vector-write-result-latency.md
git commit -m "docs: consumer-side bypass matrix model + accumulator deferral gap

Generated using Claude Code."
```

---

## Task 8 (Phoenix-gated, separate session): HW validation

**Not run in the Claude sandbox.** Capture `vec_mac_bf16`, `two_col`, and the
cascade matmul on the real NPU and confirm EMU matches. On success, record
`Verified{evidence}` for the bf16 matmul class in the vector-verify records and
flip the known-fidelity-gaps row's HW-status. This is the reason to build the
matrix now, while Phoenix is available.

---

## Self-Review Notes

- **Spec coverage:** §2 formula → Task 1. §3.1 FFI → Task 2. §3.2 accessors →
  Task 2 (on `InstrInfo`, since the decoder already holds `instr_info`; the
  spec's mention of `LatencyTable` accessors is satisfied equivalently —
  producer-side `LatencyTable` is untouched). §3.3 `source_forward` → Task 4.
  §3.4 read threading → Task 5. §3.5 mapping cross-check → Task 3. §2.1
  accumulator deferral → Task 7 (FIXME verified) + existing code. §4 testing →
  Tasks 1,3,4,5,6. §5 migration → Tasks 1 (replace hack) + 7 (cleanup). §6
  commit split → Task 7.
- **Known deviation from spec §3.2:** consumer-side `use_cycle`/`use_bypass` live
  on `InstrInfo` (decoder-reachable) rather than `LatencyTable`, because the
  decoder owns `instr_info` and the executor's `LatencyTable` is not reachable at
  decode time. This is strictly better (computed once at decode, not per
  execution) and keeps the producer-side `LatencyTable` untouched.
- **Risk placement:** Task 3 runs the mapping gate after the table exists (Task 2)
  and before any execute path depends on it (Tasks 4–5), per Maya's "know where
  the risk is" requirement.
