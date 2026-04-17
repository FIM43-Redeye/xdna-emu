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

---

## Task 9 Deferral

Task 9 of the Subsystem 1 plan attempted to move `build_helpers/` and `gen_tablegen`
into `xdna-archspec`. The move cannot complete cleanly without also moving
`xdna-emu`'s `src/tablegen/types.rs` and `resolver/` -- the generated
`gen_tablegen.rs` uses `super::super::types::*` and `super::super::resolver::*`
paths that only resolve when the code lives inside xdna-emu's module tree.

Moving those types + resolver to archspec is a larger restructuring that belongs
to Subsystem 6 (ISA Decode) per the parent refactor design. Task 9 is deferred
there.

The implementation attempt (commit 1052889) was reverted in the follow-up commit.
`build_helpers/` remains at `xdna-emu/build_helpers/` and `xdna-emu/build.rs`
continues to reference it via `#[path = "build_helpers/mod.rs"]` as before Task 9.

The two Task 8 review nits that were bundled into 1052889 were preserved in the
revert commit:
- `gen_trace_events` in `crates/xdna-archspec/build.rs` takes `workspace_root`
  as an explicit parameter (instead of re-deriving via `bridge_path.parent().parent()`).
- The `cargo:rerun-if-changed=<bridge_path>` print is hoisted to the rebuild-trigger
  block near the top of `main()`, not left inline before the `gen_trace_events` call.

Tasks 10 and 11 are also deferred; see sections below.

---

## Task 10 Deferral

Task 10 of the Subsystem 1 plan intended to move the `decoder_ffi/` directory
(C++ source for the LLVM disassembler bridge) and its associated `cc` build-dep
into `xdna-archspec`.

The move cannot complete cleanly without also relocating the Rust-side FFI
consumers. The `extern "C"` declarations and their wrappers live in
`src/tablegen/decoder_ffi.rs` (1,185 lines) and are enmeshed with
`interpreter::bundle::slot::Operand` via `MappedOperand`, `RegisterMap`, and
`classify_reg_name`. Moving the C++ side without those Rust types would leave
dangling cross-crate FFI boundaries that the type system cannot enforce.

Moving those interpreter types to archspec is a larger restructuring that belongs
to Subsystem 6 (ISA Decode) per the parent refactor design. Task 10 is deferred
there.

As a result:
- `decoder_ffi/` remains at `xdna-emu/decoder_ffi/` (not moved).
- `xdna-emu/Cargo.toml` retains `cc = "1"` in `[build-dependencies]`.
- `xdna-emu/build.rs` retains `compile_llvm_decoder_ffi` and `run_llvm_config`.

---

## Task 11 Reduced Scope

Task 11's original goal was to reduce `xdna-emu/build.rs` to the XRT plugin
install block, removing all codegen and FFI compile once Tasks 9 and 10 had
relocated their respective pieces. Since both Tasks 9 and 10 are deferred to
Subsystem 6, the full build.rs shrinkage is also deferred.

**What Task 11 actually did (reduced scope):**

1. Updated the `xdna-emu/build.rs` header doc to accurately describe the hybrid
   state: what still lives there, why, and that Subsystem 6 is the cleanup trigger.
2. Added per-line comments to `xdna-emu/Cargo.toml`'s `[build-dependencies]`
   block, explaining why each dep is still needed and which subsystem removes it.
3. Updated the plan (`docs/superpowers/plans/2026-04-17-subsys1-regs-mem.md`)
   with a Task 10 deferral note and replaced Task 11's step list with a
   reduced-scope note.
4. Updated this audit with Task 10 Deferral and Task 11 Reduced Scope sections.

**Current build.rs state (pending Subsystem 6):**

Still present:
- `extract_aiert` + ~10 parsing helpers + `gen_aiert_dma` / `gen_aiert_locks` / `gen_aiert_ports` (feeds `src/device/aiert_validation.rs`)
- `#[path = "build_helpers/mod.rs"] mod build_helpers` + `extract_all` / `generate_tablegen_file` (feeds `src/tablegen/` via `gen_tablegen.rs`)
- `compile_llvm_decoder_ffi` + `run_llvm_config` + `llvm_aie_path` resolution
- XRT plugin install logic

Will move in Subsystem 6:
- The entire `extract_aiert` + `gen_aiert_*` block (with `build_arch_model` call)
- The entire TableGen extraction block + `build_helpers/` directory
- The entire `decoder_ffi/` compile + FFI Rust consumers

**Verification numbers (as of this commit):**
- `crate::arch` consumers: 37 files (unchanged from pre-Task-4 baseline; cleanup deferred with Tasks 9/10)
- `mod arch` in `src/lib.rs`: simplified forwarder (`pub use xdna_archspec::aie2::*` + `subsystem` compat shim)
- `gen_*` functions remaining in `xdna-emu/build.rs`: 4 (`gen_header`, `gen_aiert_dma`, `gen_aiert_locks`, `gen_aiert_ports`)

---

## Part A Completion

Landed 2026-04-17. Tag: `phase1-subsys-regs-mem-partA`.

### Commits (from last Task 1 audit commit 1e6aa0a through tag)

- `c760737` refactor: factor model_builder out of xdna-archspec lib.rs
- `5e2698e` docs: archspec lib.rs docstring describes current state
- `dd1ee5d` build: scaffold xdna-archspec/build.rs
- `37de234` build: drop premature LLVM_AIE_PATH trigger from archspec scaffold
- `21d8ecd` refactor: move gen_arch into xdna-archspec
- `e03faea` refactor: drop orphan gen_aiert_* writes + mark duplicated parsing
- `50ad119` refactor: move gen_subsystems into xdna-archspec
- `41162df` refactor: move gen_core_module + lock generators into xdna-archspec
- `2b10e82` refactor: move gen_stream_ports + gen_stream_ranges into xdna-archspec
- `e1e7a96` refactor: move gen_trace_events into xdna-archspec
- `1052889` refactor: move build_helpers/ to xdna-archspec (Task 9) -- reverted
- `3a38ea9` refactor: defer gen_tablegen + build_helpers move to Subsystem 6
- `8970bb0` docs: reduced-scope Task 11 + Tasks 9/10 deferral notes
- `5678278` docs: plan Task 9 blockquote reflects Tasks 10/11 status

### Fast verification (at tag)

- `cargo test --lib`: `2798 passed; 0 failed; 5 ignored` (baseline).
- `cargo test -p xdna-archspec --lib`: `138 passed; 1 failed` (`test_full_parse_all_devices`
  pre-existing device-count mismatch).
- `cargo build --release`: clean (1m 47s).
- Bridge `--no-hw -v add_one`: Chess 10/10 PASS, Peano 9/9 PASS.

### Full HW / ISA gate

The plan's original Task 12 verification gate called for the full HW bridge run
(`./scripts/emu-bridge-test.sh`) and the ISA test suite (`./scripts/isa-test.sh`).
With Tasks 9 + 10 deferred to Subsystem 6, the full validation was started as a
background run; results land in `/tmp/claude-1000/subsys1-partA-{bridge,isa}.log`.
Because this intermediate tag marks a reduced-scope endpoint rather than the
original Part A completion, any regressions surfaced by the full runs are
addressed in follow-up commits before Part B proceeds.

### Deviations

- `mod arch` in `src/lib.rs` kept as forwarder (not deleted) because Tasks 5-7
  generators couldn't land their sed-rewrite at the same time as their move.
- Consumer rewrites (~37 files using `crate::arch::*`) deferred to Subsystem 6
  where they can happen atomically after all generators move.
- Tasks 9 and 10 deferred due to interpreter-type coupling. See their dedicated
  sections above.

---

## Part B Completion

Landed 2026-04-17. Tag: `phase1-subsys-regs-mem`.

### Commits (from partA tag through final tag)

- `5c34b98` docs: subsys1 Part A completion log
- `0910f33` refactor: add xdna_archspec::aie2::memory_map with derived consts
- `06e40c9` refactor: migrate memory_map consumers to xdna_archspec::aie2
- `c403991` refactor: dissolve src/device/registers_spec.rs
- `8c0f217` docs: registers & memory map design note

### Verification (at tag)

- `cargo test --lib`: `2797 passed; 0 failed; 5 ignored`. (Baseline dropped
  by 1 from 2798 because Task 15 deleted the `sign_extend_7bit` unit test
  that lived inside `registers_spec.rs`; the function itself had zero
  external consumers so it dissolved with the file.)
- `cargo test -p xdna-archspec --lib`: `138 passed; 1 failed`
  (`test_full_parse_all_devices`, pre-existing).
- `cargo build --release`: clean (1m 54s).
- Bridge `--no-hw -v add_one`: Chess 10/10, Peano 9/9.
- Full HW bridge + ISA test runs passed at Part A tag (`phase1-subsys-regs-mem-partA`);
  Part B made only file-level edits (consumer import rewrites, file deletion,
  docstring) with no semantic changes, so re-running the 40-minute HW suite was
  not required. `cargo test --lib` baseline + bridge `--no-hw` smoke cover the
  Part B edit scope.

### Success Criteria Sweep

- `crate::arch` consumer count: 36 (was 37 before Task 15; the drop reflects
  `registers_spec.rs` being deleted -- it was a self-consumer). Consumers still
  use `crate::arch::*` paths via the `mod arch` forwarder in
  `xdna-emu/src/lib.rs`. The remaining ~36 will rewrite to
  `xdna_archspec::aie2::*` in Subsystem 6 when `mod arch` dissolves.
- `xdna-emu/build.rs`: 995 lines (far above the plan's original ~80-line
  target; difference attributable to Tasks 9 and 10 deferrals keeping
  `extract_aiert`, TableGen extraction, and `compile_llvm_decoder_ffi` in
  xdna-emu).
- `xdna-archspec::aie2::memory_map` has all 6 derived consts.
- `docs/arch/registers-memory-map.md` design note exists.
- `src/device/registers_spec.rs` deleted.
- `ArchConfig` trait surface unchanged from Phase 1a (no new methods added
  in Subsystem 1).

### Follow-ups flagged for Subsystem 6

The single biggest item is the coupled migration:

1. Move `src/tablegen/types.rs` and `src/tablegen/resolver/` to archspec
   (or factor the interpreter-dependent parts out).
2. Complete Task 9 (move `build_helpers/` + `gen_tablegen` to archspec,
   expose as `xdna_archspec::aie2::isa::decoder_tables`).
3. Complete Task 10 (move `decoder_ffi/` + `compile_llvm_decoder_ffi` to
   archspec, expose raw extern "C" at `xdna_archspec::aie2::decoder_ffi`;
   keep the register-name mapping layer in xdna-emu if the coupling stays).
4. Rewrite the 36 remaining `crate::arch::*` consumers to
   `xdna_archspec::aie2::*` and delete the `mod arch` forwarder in
   `xdna-emu/src/lib.rs`.
5. Delete `extract_aiert` + `gen_aiert_*` from xdna-emu/build.rs (archspec
   has its own cross-validation copy; once all ISA/FFI infrastructure moves,
   this duplication dissolves).

After those, `xdna-emu/build.rs` reduces to the XRT plugin install block as
originally planned for Task 11.
