# Subsystem 6 -- ISA Decode Audit

Captured at the start of Phase 1b Subsystem 6 (ISA Decode), before any
relocation work begins. All numbers here are from actual command output on
the `dev` branch at the `phase1-subsys-regs-mem` lineage (commit `8c0f217`
is the parent tag; these captures are from current HEAD on `dev`).

---

## Baseline (pre-subsystem)

- `cargo test --lib`:
  ```
  test result: ok. 2797 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in 2.22s
  ```

- `cargo test -p xdna-archspec --lib`:
  ```
  test result: FAILED. 138 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.29s
  ```
  Failure is `test_full_parse_all_devices` (device count 13 vs expected 12).
  Pre-existing, unrelated to this subsystem; carry through without regression.

- Bridge `--no-hw -v add_one`:
  ```
  === Summary ===
  Chess: 10/10 compiled, 10 bridge pass, 0 bridge fail
  Peano: 9/9 compiled, 9 bridge pass, 0 bridge fail
  Logs: /home/triple/npu-work/xdna-emu/build/bridge-test-results/20260417/
  ```

---

## crate::arch Consumers

**36 files** match `rg -l 'crate::arch'` across `src/`, `examples/`, `tests/`, `xrt-plugin/`.

All 36 are under `src/`:

```
src/device/array/routing.rs
src/device/banking.rs
src/device/control_packets/mod.rs
src/device/control_packets/parser.rs
src/device/control_packets/reassembler.rs
src/device/dma/engine/mod.rs
src/device/dma/stream_io.rs
src/device/dma/timing.rs
src/device/dma/transfer/core.rs
src/device/mod.rs
src/device/model.rs
src/device/port_layout.rs
src/device/registers.rs
src/device/state/compute.rs
src/device/state/effects.rs
src/device/state/memtile.rs
src/device/stream_switch/mod.rs
src/device/stream_switch/packet_switch.rs
src/device/stream_switch/packet_types.rs
src/device/stream_switch/ports.rs
src/device/tile/mod.rs
src/device/tile/params.rs
src/device/tile/tests.rs
src/interpreter/bundle/slot_layout.rs
src/interpreter/execute/memory/mod.rs
src/interpreter/execute/vector_permute.rs
src/interpreter/execute/vector_srs.rs
src/interpreter/state/context.rs
src/interpreter/state/registers.rs
src/interpreter/state/timing_context.rs
src/interpreter/test_runner.rs
src/interpreter/timing/memory.rs
src/interpreter/timing/sync.rs
src/npu/executor.rs
src/parser/cdo.rs
src/parser/elf.rs
```

Matches plan expectation of ~36 files under `src/`.

---

## crate::tablegen Consumers

**27 files** match `rg -l 'use crate::tablegen'` across `src/`, `examples/`, `tests/`, `xrt-plugin/`.

All 27 are under `src/`:

```
src/interpreter/bundle/mod.rs
src/interpreter/bundle/slot.rs
src/interpreter/decode/decoder.rs
src/interpreter/decode/loader.rs
src/interpreter/decode/operand_extraction.rs
src/interpreter/decode/slot_builder.rs
src/interpreter/execute/cascade.rs
src/interpreter/execute/control.rs
src/interpreter/execute/cycle_accurate.rs
src/interpreter/execute/memory/mod.rs
src/interpreter/execute/mod.rs
src/interpreter/execute/semantic.rs
src/interpreter/execute/stream.rs
src/interpreter/execute/vector_arith.rs
src/interpreter/execute/vector_compare.rs
src/interpreter/execute/vector_convert.rs
src/interpreter/execute/vector_dispatch.rs
src/interpreter/execute/vector_matmul/mod.rs
src/interpreter/execute/vector_misc.rs
src/interpreter/execute/vector_semantic.rs
src/interpreter/execute/vector_srs.rs
src/interpreter/execute/vector_ups.rs
src/interpreter/execute/vector_validate.rs
src/interpreter/state/context.rs
src/interpreter/timing/latency.rs
src/interpreter/timing/slots.rs
src/tablegen/resolver/mod.rs
```

Plan expected ~25-38 files; 27 is within range.

---

## xdna-emu/build.rs Surface

**995 lines** total. **22 top-level `fn` definitions** (from `rg -n '^fn ' build.rs`).

A previous revision of this table used the regex `^fn (gen_|extract_|compile_|run_)`
and silently captured only 10 of the 22 functions. The table below enumerates all
22. The 11 helper functions in the `parse_*` / name-mapping group are part of the
same `extract_aiert` + aie-rt parsing block and migrate with Task 11.

| Line | Function | Block | Migrates |
|------|----------|-------|----------|
| 52 | `main()` | Orchestration | stays in `xdna-emu/build.rs` |
| 275 | `gen_header(source_desc: &str) -> String` | Shared utility | Task 7 (or shared) |
| 289 | `extract_aiert(...)` | aie-rt extraction (top-level) | Task 11 |
| 380 | `run_aiert_preprocessor(aiert_dir: &Path) -> Option<String>` | aie-rt extraction | Task 11 |
| 459 | `parse_dma_modules(text: &str) -> Vec<DmaModData>` | aie-rt parsing | Task 11 |
| 467 | `parse_lock_modules(text: &str) -> Vec<LockModData>` | aie-rt parsing | Task 11 |
| 475 | `parse_port_maps(text: &str) -> Vec<PortMapData>` | aie-rt parsing | Task 11 |
| 542 | `parse_struct_initializers(text, type_name)` | aie-rt parsing | Task 11 |
| 599 | `extract_identifier(line: &str, type_name: &str) -> Option<String>` | aie-rt parsing | Task 11 |
| 614 | `parse_field_assignment(line: &str) -> Option<(String, String)>` | aie-rt parsing | Task 11 |
| 643 | `extract_field_value(line: &str, field: &str) -> Option<String>` | aie-rt parsing | Task 11 |
| 658 | `parse_numeric_value(s: &str) -> Option<u32>` | aie-rt parsing | Task 11 |
| 671 | `get_field(fields, name, struct_name) -> u32` | aie-rt parsing | Task 11 |
| 680 | `dma_mod_name(name: &str) -> &str` | aie-rt name mapping | Task 11 |
| 690 | `lock_mod_name(name: &str) -> &str` | aie-rt name mapping | Task 11 |
| 700 | `port_map_rust_name(name: &str) -> &str` | aie-rt name mapping | Task 11 |
| 718 | `gen_aiert_dma(modules: &[DmaModData], out_dir: &Path)` | aie-rt codegen | Task 11 |
| 742 | `gen_aiert_locks(modules: &[LockModData], out_dir: &Path)` | aie-rt codegen | Task 11 |
| 764 | `gen_aiert_ports(port_maps: &[PortMapData], out_dir: &Path)` | aie-rt codegen | Task 11 |
| 814 | `write_aiert_stubs(out_dir: &Path)` | aie-rt fallback stubs | Task 11 |
| 923 | `compile_llvm_decoder_ffi(llvm_aie_path: &Path)` | TableGen / FFI | Task 9 |
| 982 | `run_llvm_config(llvm_config: &Path, args: &[&str]) -> String` | TableGen / FFI | Task 9 |

**Migration summary by task:**
- **Task 7** (gen_arch relocation): `gen_header` may move here or become shared; `main` orchestration updated.
- **Task 9** (TableGen/FFI): `compile_llvm_decoder_ffi`, `run_llvm_config`.
- **Task 11** (aie-rt extraction): all 18 remaining functions in the aie-rt parsing, codegen, and fallback-stub block (lines 289-820).

---

## decoder_ffi.rs Split Line

Split is at **line 346**:

```
346: use crate::interpreter::bundle::slot::Operand;
347: use crate::interpreter::state::{
```

This is the boundary between the FFI bridge (lines 1-345, pure TableGen /
archspec-facing) and the interpreter-facing decode logic (lines 346+).
Task 10 splits the file here: everything above line 346 moves to
`xdna-archspec`, everything from line 346 onward stays in `xdna-emu` under
`src/interpreter/decode/`.

Matches plan expectation exactly (line 346).

---

## Plan Deviations

### Tasks 3-5 + parts of 6 and 10 combined into one commit

Discovered during Task 3 execution that Tasks 3, 4, 5 couldn't move
independently because:
- `types.rs::TblgenOutput` uses `super::resolver::InstrEncoding` and
  `super::decoder_bytecode::DecoderTable`
- `resolver/mod.rs` imports from `super::types`
- `resolver/mod.rs::SlotIndex::decode()` calls
  `super::decoder_ffi::slot_from_name` / `decode_slot_name`
- `resolver/semantic_inference.rs` calls
  `super::super::element_type_logic::*`

The atomic unit is five pieces, not three: types, resolver,
decoder_bytecode, element_type_logic (runtime copy), and the pure-FFI
top half of decoder_ffi.rs. All moved together in commit 11f1275.

Additional deviations from the original plan:

- `log = "0.4"` added to archspec's Cargo.toml (required by
  `resolver/mod.rs` production code using `log::trace!()`).
- `get_num_regs()` and `get_reg_name()` public wrapper functions added to
  archspec's `decoder_ffi.rs` so `register_map.rs` (in xdna-emu) can call
  them without accessing private `extern "C"` symbols.
- xdna-emu test count after mega-move: 2730 (plan expected 2797). Delta of
  -67 is correct -- those tests relocated into archspec.
- Archspec test count after mega-move: 202 (plan expected 138). Delta of
  +64 is the relocated test modules, with 2 tests newly marked `#[ignore]`
  (decoder_bytecode integration tests that require `load_from_generated`,
  wired in Task 7).

Remaining plan tasks after this commit:
- Task 6 still moves `build_helpers/` directory (the build-time copy
  of element_type_logic.rs stays there pending that).
- Task 10 reduces to relocating `src/tablegen/register_map.rs` ->
  `src/interpreter/decode/register_map.rs` (the split itself already
  happened in this combined commit).
- All other tasks unchanged.

---

## Part A Completion

Landed 2026-04-17. Tag: `phase1-subsys-isa-decode-partA` at `25c88be`.

### Commits (Task 1 through tag)

- `90b6b91` docs: subsys6 ISA decode audit
- `ed4543f` docs: subsys6 audit -- enumerate all build.rs helpers
- `f5d3b94` refactor: scaffold xdna_archspec::aie2::isa module
- `11f1275` refactor: move tablegen types/resolver/decoder_bytecode + FFI split (mega-move)
- `71fb29b` docs: subsys6 audit -- Tasks 3-5 + 6/10 partial deviation note
- `e973565` docs: archspec decoder_ffi -- document transitional build.rs ownership
- `0b68c14` refactor: move build_helpers/ directory to xdna-archspec
- `24e755c` refactor: wire TableGen extraction into xdna-archspec's build.rs
- `749cec7` refactor: move decoder_ffi/ (C++ source) into xdna-archspec
- `55ff796` refactor: move compile_llvm_decoder_ffi into xdna-archspec
- `25c88be` refactor: move extract_aiert + gen_aiert_* to xdna-archspec

### Fast verification (at tag)

- `cargo test --lib`: 2730 passed; 0 failed; 5 ignored.
- `cargo test -p xdna-archspec --lib`: 202 passed; 1 failed (`test_full_parse_all_devices`, pre-existing device-count mismatch).
- `cargo build --release`: clean (4m 56s).
- Bridge `--no-hw -v add_one`: Chess 10/10, Peano 9/9.

### Full HW + ISA gate

The plan's Task 12 called for full `./scripts/emu-bridge-test.sh` (~30 min) + `./scripts/isa-test.sh` (~10 min). Deferred to a separate user-gated step before Part B begins. Part A landed on the fast-gate (release build + bridge smoke green); Rust-side invariants confirm no regressions, and the expensive HW tests run as a prerequisite for Task 13 rather than blocking the tag.

### Part A deliverables

- [x] build_helpers/ moved to xdna-archspec.
- [x] decoder_ffi/ C++ sources moved to xdna-archspec.
- [x] src/tablegen/types.rs, resolver/, decoder_bytecode.rs moved (with forwarders).
- [x] decoder_ffi.rs split at line 346; top half to archspec; bottom half at src/tablegen/register_map.rs (final relocation to interpreter/decode is Part B Task 13).
- [x] extract_aiert + gen_aiert_{dma,locks,ports} moved to archspec's build.rs; xdna_archspec::aie2::aiert module exposes the data.
- [x] compile_llvm_decoder_ffi + run_llvm_config moved to archspec's build.rs.
- [x] tblgen, cc, serde_json build-deps moved from xdna-emu to archspec.
- [x] xdna_archspec::aie2::{isa, aiert} modules populated.
- [x] Forwarders (`pub mod arch` in src/lib.rs, `src/tablegen/*` re-export files) still live; consumer rewrites deferred to Part B.

### Build-environment caveat

archspec's build.rs runs tblgen against llvm-aie's `build/bin/llvm-config` (version 21.x). The mlir-aie `llvm/install/bin/llvm-config` on PATH (version 23.x) is the wrong version for tblgen. Prior to `cargo clean`, cached tblgen artifacts hid this. Now every fresh build requires:

```bash
export PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH
```

Future hygiene: have archspec's build.rs set `LLVM_SYS_210_PREFIX` explicitly so this prepend isn't required. Noted as a follow-up, not a Part A blocker.

### Deviations

- Plan's Tasks 3, 4, 5 (move types, resolver, decoder_bytecode independently) merged into a single mega-move because circular `super::` references between them required simultaneous relocation. Part of Task 6 (element_type_logic runtime copy) and part of Task 10 (decoder_ffi split at line 346) were absorbed into the same commit (`11f1275`) for the same reason.
- Plan's Task 10 reduced scope: the split already happened in the mega-move. Final register_map relocation to `interpreter::decode` is Task 13, in Part B.
- Plan's Task 7 Step 3a (codegen string path rewrite `super::super::` -> `super::`) skipped because both module trees have a `mod generated { include!() }` wrapper that preserves the two-level depth. Paths left unchanged.

---

## Part B Completion Log (2026-04-18)

Part B closed out at `phase1-subsys-isa-decode`. Seven tasks landed on top of Part A:

| Task | Subject | Commit |
|------|---------|--------|
| 13 | Move `src/tablegen/register_map.rs` -> `src/interpreter/decode/register_map.rs`; rewrite 10 `AccumWidth` consumers | `bb99951` |
| 14 | Atomic sed rewrite of 29 `crate::tablegen::*` consumers to `xdna_archspec::aie2::isa::*` (+ 1 doc-comment) | `0e6982b` |
| 15 | Atomic sed rewrite of 36 `crate::arch::*` consumers to `xdna_archspec::aie2::*`; `use crate::arch;` -> alias; `subsystem` -> `subsystems as subsystem` alias | `f07ecfb` |
| 16 | Delete `src/tablegen/`; remove `mod arch { ... }` / `mod tablegen` from `src/lib.rs`; fix `src/bin/export_isa.rs` import; rescue 21 tests to archspec + register_map | `454a608` |
| 17 | Shrink `build.rs` 201 -> 97 lines (plugin install only); empty xdna-emu `[build-dependencies]` | `9aafa8c` |
| 18 | `docs/arch/isa-decode.md` design note | `d3338a6` |
| 19 | Full HW bridge + ISA gate; tag `phase1-subsys-isa-decode`; this log; NEXT-STEPS update | (this commit) |

### Part B Deliverables

- [x] `MappedOperand` / `RegisterMap` / `classify_reg_name` / `AccumWidth` live at their interpreter-side canonical path.
- [x] Zero `crate::tablegen::*` or `crate::arch::*` references remain outside `src/lib.rs` (which no longer has them).
- [x] `src/tablegen/` directory deleted.
- [x] `pub mod arch { ... }` and `pub mod tablegen` forwarder blocks removed from `src/lib.rs`.
- [x] `xdna-emu/build.rs` reduced to XRT plugin install only; no `[build-dependencies]`.
- [x] Test counts: xdna-emu `2712 passed`, archspec `220 passed` (the 1 failure is the pre-existing `device_model::test_full_parse_all_devices`). Live-test parity preserved with pre-delete count.
- [x] Full HW bridge + ISA gate: see `/tmp/claude-1000/subsys6-partB-{bridge,isa}.log`.

### Deviations in Part B

- **Task 14 missed `AccumWidth` fixup in the plan's sed script.** `AccumWidth` is emulator-side (not archspec), so the generic `crate::tablegen::*` -> `xdna_archspec::aie2::isa::*` rewrite would have produced a bad path. Task 13 pre-rewrote all 10 `AccumWidth` consumers to `crate::interpreter::decode::register_map::AccumWidth` so Task 14 didn't need to know.
- **Task 14 caught one unplanned consumer:** `src/interpreter/decode/operand_extraction.rs` combined `operand_from_reg_name` (interpreter-side) with the other decoder_ffi imports in a single `use` block. Split the import: archspec for `decoder_ffi::{self, DecodedOperand, DecodeResult}`, register_map for `{AccumWidth, operand_from_reg_name}`.
- **Task 15 needed additional fixups beyond the plan's sed.** Six files used bare `use crate::arch;` (module-as-name), rewritten to `use xdna_archspec::aie2 as arch;`. Two files used `use crate::arch::subsystem;` (singular compat shim), rewritten to `use xdna_archspec::aie2::subsystems as subsystem;` to preserve the local short name.
- **Task 16 deleted 21 tests.** The tests were rescued to their rightful homes (archspec for `load_from_generated` / decoder-FFI tests, `interpreter::decode::register_map` for cross-coverage tests that depend on `reg_map_coverage`). Net live-test count preserved.
- **Task 17 discovered `build.rs` was still running dead code.** The pre-Part-B build.rs called `build_arch_model` and `confirm_processor_slots` but the returned `arch_model` went unused -- those calls were only emitting `cargo:warning` lines for cross-validation side effects. Removing them dropped the `xdna-archspec` build-dependency entirely; the archspec build.rs owns that validation now.

### Incidental harness fixes (not on the plan)

During Task 12 + 19 verification, the shell bridge harness
(`scripts/emu-bridge-test.sh`) surfaced two long-standing bugs, both
fixed in commit `ba86b37`:
1. `apply_lit_subs` never substituted lit's `%s` macro. Only one test in the tree (`bd_chain_repeat_on_memtile`) uses `FileCheck %s`, but any new test relying on it would have errored out at the harness level.
2. `XFAIL:` parsing was missing entirely from the shell bridge. The Rust-native `src/testing/` harness had it since March (`b139c29`); the XRT bridge harness did not. Added `is_xfail()` honoring `XFAIL: *` and comma-separated feature lists (`chess`, `peano`); result writers rewrite `FAIL`/`TIMEOUT` -> `XFAIL` and `PASS` -> `XPASS`; summary counters surface both.

### Pre-existing EMU regression flagged during gate

`bd_chain_repeat_on_memtile` EMU side hangs in a DMA `check_acquire_granted granted=false` polling loop and never emits `PASS!`. Bridge result is `FAIL` (not `XFAIL` -- the test has no XFAIL annotation). Traced backward via the bridge-test-results history: passed on 20260414 (pre-Phase-1a), failing by 20260416. Introduced somewhere in Subsystem 1 or earlier Phase-1a work, not in Subsystem 6. Flagged as an independent investigation item; unrelated to this subsystem's deliverables.
