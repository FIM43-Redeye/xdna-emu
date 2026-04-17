# Subsystem 1 -- Registers & Memory Map Audit

## Baseline (pre-subsystem)

- `cargo test --lib`:

  ```
  test result: ok. 2798 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in 2.35s
  ```

- `cargo test -p xdna-archspec --lib`:

  ```
  test result: FAILED. 138 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.28s
  ```

- Bridge `--no-hw -v add_one`:

  ```
  === Summary ===
  Chess: 10/10 compiled, 10 bridge pass, 0 bridge fail
  Peano: 9/9 compiled, 9 bridge pass, 0 bridge fail
  ```

Failures to carry through: `test_full_parse_all_devices` (archspec, pre-existing,
device count 13 vs expected 12 -- unrelated).

---

## crate::arch Consumers

37 files total. All matches are under `src/` -- no consumers in `examples/`,
`tests/`, or `xrt-plugin/`.

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
src/device/registers_spec.rs
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

No hidden consumers outside `src/`.

---

## Codegen Include Sites

12 `include!(concat!(env!("OUT_DIR"), ...))` sites across 5 files:

| File | Line | Generated file |
|------|------|----------------|
| `src/lib.rs` | 61 | `gen_arch.rs` |
| `src/lib.rs` | 89 | `gen_subsystems.rs` |
| `src/lib.rs` | 93 | `gen_stream_ports.rs` |
| `src/lib.rs` | 97 | `gen_stream_ranges.rs` |
| `src/trace/mod.rs` | 20 | `trace_event_codes.rs` |
| `src/device/aiert_validation.rs` | 9 | `gen_aiert_dma.rs` |
| `src/device/aiert_validation.rs` | 95 | `gen_aiert_locks.rs` |
| `src/device/aiert_validation.rs` | 151 | `gen_aiert_ports.rs` |
| `src/tablegen/mod.rs` | 28 | `gen_tablegen.rs` |
| `src/device/registers_spec.rs` | 58 | `gen_memory_lock.rs` |
| `src/device/registers_spec.rs` | 80 | `gen_core_module.rs` |
| `src/device/registers_spec.rs` | 99 | `gen_memtile_lock.rs` |

---

## build.rs Codegen Functions

Codegen functions (`^fn gen_`):

| Line | Function |
|------|----------|
| 285 | `gen_header(source_desc: &str) -> String` |
| 297 | `gen_arch(model: &xdna_archspec::types::ArchModel, out_dir: &Path)` |
| 619 | `gen_subsystems(model: &xdna_archspec::types::ArchModel, out_dir: &Path)` |
| 710 | `gen_core_module(regdb: &RegisterDb, out_dir: &Path)` |
| 781 | `gen_lock_request(...)` |
| 946 | `gen_stream_ports(regdb: &RegisterDb, out_dir: &Path) -> PortArrayData` |
| 1127 | `gen_stream_ranges(...)` |
| 1382 | `gen_trace_events(bridge_path: &Path, out_dir: &Path)` |
| 1947 | `gen_aiert_dma(modules: &[DmaModData], out_dir: &Path)` |
| 1971 | `gen_aiert_locks(modules: &[LockModData], out_dir: &Path)` |
| 1993 | `gen_aiert_ports(port_maps: &[PortMapData], out_dir: &Path)` |

**Call-site to output-file mapping** (relevant for tracing `include!()` back to a generator):
`gen_lock_request` is parameterized and called twice; there is no `fn gen_memory_lock` or
`fn gen_memtile_lock` to search for.

| build.rs line | Call | Output file |
|---------------|------|-------------|
| 146 | `gen_core_module(&regdb, &out_dir)` | `gen_core_module.rs` |
| 147 | `gen_lock_request(&regdb, &out_dir, "memory", "gen_memory_lock.rs")` | `gen_memory_lock.rs` |
| 148 | `gen_lock_request(&regdb, &out_dir, "memory_tile", "gen_memtile_lock.rs")` | `gen_memtile_lock.rs` |

aie-rt extraction function:

| Line | Function |
|------|----------|
| 1525 | `extract_aiert(...)` |

LLVM decoder FFI compilation:

| Line | Symbol |
|------|--------|
| 197 | call site: `compile_llvm_decoder_ffi(llvm_aie_path)` |
| 2169 | definition: `compile_llvm_decoder_ffi(llvm_aie_path: &Path)` |

---

## sign_extend_7bit Call Sites

Two independent implementations exist -- one in `registers_spec.rs` (takes
`u32`, public, `const fn`) and one private copy in `dma/bd.rs` (takes `u8`).
They must be unified in Part B.

Call sites in `src/device/dma/bd.rs` (using the local `u8` version):

| Line | Usage |
|------|-------|
| 202 | `lock_rel_value: sign_extend_7bit(lay.lock_rel_value.extract(w5) as u8)` |
| 205 | `lock_acq_value: sign_extend_7bit(lay.lock_acq_value.extract(w5) as u8)` |
| 305 | `lock_rel_value: sign_extend_7bit(lay.lock_rel_value.extract(w7) as u8)` |
| 308 | `lock_acq_value: sign_extend_7bit(lay.lock_acq_value.extract(w7) as u8)` |
| 385 | `lock_rel_value: sign_extend_7bit(lay.lock_rel_value.extract(w7) as u8)` |
| 388 | `lock_acq_value: sign_extend_7bit(lay.lock_acq_value.extract(w7) as u8)` |
| 612 | definition (local, private, takes `u8`) |
| 626 | test (`test_sign_extend_7bit`) |

Definition and tests in `src/device/registers_spec.rs` (takes `u32`, public):

| Line | Usage |
|------|-------|
| 17 | module doc listing |
| 141 | definition (`pub const fn sign_extend_7bit(val: u32) -> i8`) |
| 159 | test (`test_sign_extend_7bit`) |

---

## registers_spec.rs Consumers

9 use sites across 6 files:

| File | Line | Item imported |
|------|------|---------------|
| `src/interpreter/test_runner.rs` | 172 | `AIE_DATA_MEMORY_BASE` |
| `src/interpreter/test_runner.rs` | 2178 | `AIE_DATA_MEMORY_BASE` |
| `src/device/tile/registers.rs` | 18 | `memory_module as mm`, `mem_tile_module as mt` |
| `src/device/tile/registers.rs` | 213 | `memory_module as mm`, `mem_tile_module as mt` |
| `src/device/state/dispatch.rs` | 277 | `PROGRAM_MEMORY_BASE` |
| `src/device/state/dispatch.rs` | 302 | `MEM_TILE_DATA_MEMORY_END` |
| `src/device/regdb/tests.rs` | 596 | `core_module as cm` |
| `src/device/state/compute.rs` | 434 | `core_module as cm` |
| `src/device/state/compute.rs` | 478 | `core_module as cm` |
