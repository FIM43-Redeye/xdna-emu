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
