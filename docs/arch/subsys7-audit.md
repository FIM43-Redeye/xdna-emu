# Subsystem 7 -- ISA Execute Audit

**Subsystem:** 7 of 8 (Phase 1b of the device-family refactor)
**Spec:** [../superpowers/specs/2026-04-21-subsys7-isa-execute-design.md](../superpowers/specs/2026-04-21-subsys7-isa-execute-design.md)
**Plan:** [../superpowers/plans/2026-04-21-subsys7-isa-execute.md](../superpowers/plans/2026-04-21-subsys7-isa-execute.md)

## Baseline (pre-subsystem, at phase1-subsys-stream-switch tag / HEAD)

- `cargo test --lib`: 2684 passed; 0 failed; 5 ignored
- `cargo test -p xdna-archspec --lib`: 297 passed; 0 failed; 2 ignored
- `cargo build --release`: clean
- Bridge smoke (`--no-hw -v add_one_cpp_aiecc`): green

Known pre-existing failures (carry through):
- `bd_chain_repeat_on_memtile` EMU deadlock (bridge suite; see NEXT-STEPS.md).

## Audit methodology

Per the spec, this audit is a per-file deep dive (Question 3 option
B) over the 20 files in `src/interpreter/execute/` plus the 9-file
`interpreter/timing/` submodule, grouped by functional area rather
than alphabetically.

Per-file subsection template:

- **Size + responsibility.** One sentence.
- **AIE2 hardcode count.** Grep count of literal `"AIE2"`,
  `AIE_ML_*`, `aie2`/`Aie2` identifiers, and arch-branded constants.
- **Divergence risks vs AIE1/AIE2P.** Evidence from file comments,
  llvm-aie TableGen, aie-rt per-arch headers.
- **Prescribed migration verb.** `move-to-archspec` /
  `read-archspec-via-accessor` / `wrap-in-trait` / `leave-alone`.
- **Estimated LOC impact.** Lines changing xdna-emu-side + lines
  added archspec-side.

Two files get ~2 pages each: `vmac_routing.rs` and `memory/mod.rs`.

---

## 1. Dispatcher / orchestration

Files: `execute/mod.rs`, `semantic.rs`, `cycle_accurate.rs`,
`vector_dispatch.rs`.

(Filled in by Task 1 Step 3.)

## 2. Scalar / control / stream / cascade

Files: `control.rs`, `stream.rs`, `cascade.rs`.

(Filled in by Task 1 Step 4.)

## 3. Memory

Files: `memory/mod.rs`, `memory/neighbor.rs`. (Deep dive for `mod.rs`.)

(Filled in by Task 1 Step 5.)

## 4. Vector ALU

Files: `vector_arith.rs`, `vector_compare.rs`, `vector_misc.rs`,
`vector_pack.rs`, `vector_ups.rs`, `vector_srs.rs`, `vector_helpers.rs`,
`vector_semantic.rs`, `vector_permute.rs`, `vector_float.rs`,
`vector_config.rs`, `vector_convert.rs`, `vector_validate.rs`.

(Filled in by Task 1 Step 6.)

## 5. VMAC / matmul

Files: `vmac_routing.rs` (deep dive), `vmac_hw.rs`, `vector_matmul/`.

(Filled in by Task 1 Step 7.)

## 6. Timing

Files: `interpreter/timing/{arbitration, barrier, deadlock, hazards,
latency, memory, mod, slots, sync}.rs`, plus `execute/cycle_accurate.rs`
latency tables.

(Filled in by Task 1 Step 8.)

---

## Closing summary

(Filled in by Task 1 Step 9.)

### Tentative trait method list

### Data migration list

### AIE1 projection

---

## Completion

(Filled in at the end of Subsystem 7, in the Part B final task.)
