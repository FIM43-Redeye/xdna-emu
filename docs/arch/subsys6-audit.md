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

**995 lines** total.

Functions matching `^fn (gen_|extract_|compile_|run_)`:

| Line | Function |
|------|----------|
| 275 | `gen_header(source_desc: &str) -> String` |
| 289 | `extract_aiert(...)` |
| 380 | `run_aiert_preprocessor(aiert_dir: &Path) -> Option<String>` |
| 599 | `extract_identifier(line: &str, type_name: &str) -> Option<String>` |
| 643 | `extract_field_value(line: &str, field: &str) -> Option<String>` |
| 718 | `gen_aiert_dma(modules: &[DmaModData], out_dir: &Path)` |
| 742 | `gen_aiert_locks(modules: &[LockModData], out_dir: &Path)` |
| 764 | `gen_aiert_ports(port_maps: &[PortMapData], out_dir: &Path)` |
| 923 | `compile_llvm_decoder_ffi(llvm_aie_path: &Path)` |
| 982 | `run_llvm_config(llvm_config: &Path, args: &[&str]) -> String` |

Functions slated for migration to `xdna-archspec/build.rs` in Tasks 7, 9, 11:
- `compile_llvm_decoder_ffi` (Task 9)
- `extract_aiert`, `run_aiert_preprocessor`, `extract_identifier`,
  `extract_field_value`, `gen_aiert_dma`, `gen_aiert_locks`,
  `gen_aiert_ports` (Task 11)

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
