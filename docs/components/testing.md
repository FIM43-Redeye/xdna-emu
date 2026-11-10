# Testing Infrastructure

Test harness, suite management, NPU instruction execution, FFI bindings, and external test scripts.

Use this component reference when working on tests, the test runner, FFI, or
NPU instruction handling.

## Files

### Test Runner (`src/interpreter/test_runner.rs`)

`TestRunner` -- end-to-end kernel execution harness. Loads an XCLBIN, configures the device, runs cores, and validates output against expected values.

### Testing Module (`src/testing/`)

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports |
| `artifacts.rs` | `BuildArtifact`, `discover_build_artifacts()`, `ExampleSource` -- test artifact and xclbin discovery |
| `hardware_comparison.rs` | `CrossValidation`, `CompilerComparison` -- emulator vs hardware reference comparison |
| `host_defines.rs` | `extra_defines()` -- extra C++ compilation flags injected into test builds |
| `native_hw.rs` | Compiles and runs each test's own test.cpp on real NPU hardware (preferred HW path) |
| `npu_runner.rs` | Runs tests on real NPU hardware via npu-runner binary (fallback) |
| `npu_test.rs` | `NpuTestSource`, `TestOverrides` -- test discovery, skip/expected-fail gates |
| `opcode_collector.rs` | `OpcodeCollector` -- collects unknown opcodes when execution fails (for coverage analysis) |
| `process_control.rs` | `ProcessOutcome`, `spawn_with_timeout()`, `wait_with_timeout()` -- subprocess execution and timeout handling |
| `test_cpp_parser.rs` | Parses mlir-aie `test.cpp` files for buffer metadata (`BufferSpec`, sizes, types, input patterns) |
| `unit_test.rs` | `UnitTest`, `UnitTestBuildResult`, `discover()` -- mlir-aie unit test discovery |
| `xclbin_suite.rs` | `XclbinSuite` -- discovers and runs xclbin test files, reports `TestOutcome` |

### NPU Module (`src/npu/`)

Host-to-NPU communication protocol (XRT instruction stream).

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, `NpuOpcode` enum (Write32, BlockWrite, DdrPatch, etc.) |
| `parser.rs` | `NpuInstructionStream` -- parses binary NPU instruction sequences |
| `executor.rs` | `NpuExecutor` -- executes NPU instructions against device state (configures shim DMA, patches addresses) |
| `classify.rs` | Kernarg-role classifier: recovers semantic slot roles (Ctrlpkt, MM2S, S2MM) from the NPU instruction stream for bridge-trace-runner buffer binding |
| `cycle_cost.rs` | Control-path cycle-cost model: multi-stage latency for host→CMP→NoC→tile control packet delivery |

### FFI (`crates/xdna-emu-ffi/src/lib.rs`)

`XdnaEmuHandle` -- C-compatible foreign function interface for integrating with C/C++ applications (mock XRT library). Provides `extern "C"` functions for loading xclbins, running emulation, and reading results.

### Configuration (`src/config.rs`)

`Config` -- path management (llvm-aie, mlir-aie, XRT) loaded from `xdna-emu.toml`, environment variables, or defaults.

### External Test Infrastructure (`tests/`)

| Path | Purpose |
|------|---------|
| `tests/test_overrides.toml` | Skip gates and expected-fail annotations (replaces per-test manifest files) |
| `tests/npu-outputs/` | Captured NPU hardware reference outputs (machine-specific, gitignored) |
| `tests/mlir-aie-extracted/` | Extracted/generated test files (regenerable from mlir-aie) |
| `tests/mlir-aie/` | Pointers/snapshots into the local mlir-aie build |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/run-tests.sh` | Main test runner: runs all Cargo tests with nice -n 19 |
| `scripts/run-tests.ps1` | Windows equivalent |

## Running Tests

```bash
# All tests (recommended)
./scripts/run-tests.sh

# Library tests only (fast)
./scripts/run-tests.sh --lib
cargo test --lib

# Doc tests only (slow, loads TableGen files)
./scripts/run-tests.sh --doc

# Bridge test suite -- the primary validation path
./scripts/emu-bridge-test.sh                    # full HW + EMU dual-compiler run
./scripts/emu-bridge-test.sh --no-hw            # emulator only, no NPU needed
./scripts/emu-bridge-test.sh --no-hw add_one    # single test, EMU only
./scripts/emu-bridge-test.sh --sweep            # add event sweep across all tiles
./scripts/emu-bridge-test.sh --trace=pc-anchored ...  # add PC-anchored cycle trace comparison

# ISA suite and every other kernel suite: see "Kernel Suites" below
./scripts/isa-test.sh [--seed N] [--filter PAT] [--tag NAME]

# Standalone Rust binaries
cargo run --release --bin trace-compare -- ...   # HW vs EMU events JSON diff
cargo run --release --bin export-isa             # dump TableGen ISA tables
cargo run --release --bin npu-archspec           # archspec inspector
cargo run --release --bin vcd-compare -- ...     # aiesimulator VCD diff (--features analysis)
```

## Kernel Suites (`tools/harness`)

Every kernel we write ourselves (ISA tests, timing and collision probes, C++ instruction
tests, characterizers) runs through one runner. A **suite** is a directory of cases
(`tools/harness/suite.py`): each case is a topology (cores, tiles, buffers, objectfifos or
stream/cascade links, one or more phases) plus its sources. **Generators only write suites;
they never compile or run anything.** The runner builds (aiecc, per package, unchanged
packages reused by content hash), runs on silicon (serial: the NPU is single-tenant) or in
the emulator (parallel), and compares byte for byte.

```bash
export PYTHONPATH=tools
python3 -m harness suite build   <suite>               # parallel; --only CASE.. restricts
python3 -m harness suite hw      <suite> r1            # serial, scrubbed; the first timeout aborts the run
python3 -m harness suite emu     <suite> e1 -j 8       # FFI .so rebuilt once, then parallel
python3 -m harness suite compare <suite> r1 e1         # per buffer; exit 1 on mismatch
python3 -m harness suite hw      <suite> d5 --repeat 5 # every case 5x back to back
python3 -m harness suite repeats <suite> d5            # exit 1 if a case's repeats differ
```

**Scrub.** Silicon keeps a core's data registers from one launch to the next (only the
control and status registers and `le` reset), so a case that reads a register it never wrote
would depend on what ran before it. `suite hw` therefore launches a scrub kernel on the case's
tiles before every phase (built once per suite and tile set under `<suite>/scrub/`), and `suite
emu` starts every core from the same values (`XDNA_EMU_CORE_INIT=scrub-v<N>`); `--no-scrub`
turns both off. The pattern, its register list (from TableGen) and the kernel come from
`src/interpreter/state/scrub.rs` through the checked-in `tools/harness/scrub_kernel.json`
(`cargo run --bin scrub-kernel > tools/harness/scrub_kernel.json`; a unit test fails when it is
stale). Run records and summaries name the pattern (`scrub`), and `compare` warns when two tags
differ in it. Not scrubbed: `lr`, `sp`, `r16` (the core main restores it), data memory.

Within one emulator instance the emulator carries core state from launch to launch as silicon
does (data registers, load-port rings, sparse unit, stream read latch; `interpreter::state::carried`),
but every harness phase is its own process, i.e. a fresh emulated device, so the scrub stays.
Two tools reach what a phase cannot: `tools/harness/multi_launch.py` (+ `multi_launch_host.cpp`)
runs a list of launches in ONE process on one `xrt::device` (each its own hw context, or a second
run in the previous one), and `XDNA_EMU_STATE_DIR=<dir>` makes the emulated device keep its
carried core state, and a firmware backend its management-processor platform, across processes
(saved on device close, loaded on open; delete the files in `<dir>` to reset the device). Off by
default, so ordinary runs stay independent.

Results land in `<suite>/runs/<tag>/<case>/<core>.<buffer>.bin` plus `run.json`; tags are
never overwritten. Comparing two hardware tags (`compare <suite> r1 r2`) is the determinism
check; `--repeat N` does the same within one tag (repeats 2..N sit in `<case>/repeat_<k>/`).
HW-only work is just not running `emu`. ISA-style single-core kernels address p0 = in,
p1 = out, and `isa_suite` always gives such a kernel an input of at least 4 bytes (an
out-only core gets out in p0, and p1 is stale tile state on silicon).

C++ kernels use a `.cc` core source; the compiler is a property of the topology
(`peano`: clang++ `--target=aie2-none-unknown-elf -O2`; `chess`: `xchesscc_wrapper aie2`,
AIE2 only) because aiecc links per package. The entry symbol is renamed at compile time
(`-D`), not with objcopy, which breaks Chess objects.

| Suite | Generator | Analyzer |
|-------|-----------|----------|
| ISA: one case per batch of single-instruction tests; cascade and stream batches are two-core cases | `scripts/isa-test.sh` (wraps `tools/isa-test-gen.py`; the whole generate, build, hw, emu, compare, analyze chain) | `tools/isa-test-analyze.py <suite> [hw emu]`: per test point, and fails any that checked nothing (payload all zero, or the same for inputs that should differ) even where EMU == HW |
| Accumulator read/write timing | `tools/gen_acc_timing_probe.py generate <dir>` | `tools/analyze_acc_timing.py <dir> <tag> [--compare TAG]` |
| Vector-load timing (load -> reader, `--waw`), loads across a taken branch, same-bundle hazards, same-cycle ties | `tools/gen_vec_load_timing_probe.py generate\|branch\|hazard\|tie <dir>` | `generate`: `tools/analyze_acc_timing.py`; `branch`: `tools/gen_vec_load_timing_probe.py analyze-branch <dir> <tag> [--compare TAG]`; `hazard`, `tie`: EMU against HW with `suite compare` (each pair has its two single-instruction references in the case) |
| Accumulator collisions (single core, and cascade on two cores) | `tools/gen_collision_probe.py generate <dir>`, `tools/gen_cascade_collision_probe.py <dir>`; older probe directories convert with `tools/probe_to_suite.py` | `tools/analyze_collision_probe.py check\|report\|summary <dir> [hw emu]`, `tools/collision_points.py`, `tools/diff_probe_points.py <dir> <hw> <emu>` |
| Cascade link timing (sync word through the link, then: read release, write release, round trip; shared-memory timer phase; SEMAPHORE partners on a three-core chain) | `tools/gen_cascade_timing_probe.py generate <dir> rel_n0..24_m0 wrel_n0..8_m0 rt_w60_g0_r4 mf1_s10_k0..20 semc_d1..8 ... [--directions south east]` | `tools/gen_cascade_timing_probe.py analyze <dir> <tag>` (per case: stalls, cross-tile stamp differences, data, shared-word visibility) |
| Load/store alignment | `tools/gen_ldst_align_probe.py generate <dir>` | the analysis functions in the same file (`# -- analysis` section; a pure function of a case's output bytes) |
| Address-bit decode (gather lanes and pointers at aliased, high-bit and neighbour addresses) | `tools/gen_lut_addr_probe.py generate <dir>` | `tools/gen_lut_addr_probe.py analyze <dir> <tag> <case>` (decodes where each lane read from) |
| Loads with no memory behind them (south of row 2, north of row 5): the load-port rings | `tools/gen_port_latch_probe.py generate <dir>` (records, sweep and ring cases; every kernel flushes both rings first) | `tools/gen_port_latch_probe.py analyze <dir> <tag> <case>` (names the access every returned halfword came from); EMU against HW with `suite compare` |
| C++ intrinsics, Peano (76 cases) | `tools/instr-test-gen.py --td <llvm-aie>/llvm/include/llvm/IR/IntrinsicsAIE2.td --suite-dir <dir>` | EMU against HW with `suite compare` |
| C++ intrinsics, Chess (~3770 cases) | `tools/chess-test-gen.py --opns-header $AIETOOLS_DIR/data/aie_ml/lib/isg/me_chess_opns.h --types-header .../me_chess_types.h [--iss-types-header .../me_iss_types.h] --suite-dir <dir>` | EMU against HW with `suite compare` |
| Run-time sign of the `d` forms (UPS, SRS, crVaddSign ops; Peano and Chess suites) | `tools/gen_dynamic_sign_probe.py --suite-dir <dir>` (writes `<dir>/peano` and `<dir>/chess`) | EMU against HW with `suite compare`; per-record layout in each case's `meta` |
| SRS / UPS overflow flags `srSRS_of` / `srUPS_of` (assembly, every SRS and UPS form and addressing mode) | `tools/gen_sat_flags_probe.py --suite-dir <dir> [--only FORM ..]` | `tools/gen_sat_flags_probe.py --analyze <dir> <tag>` (the silicon model, per record) |
| Carry flag `srCarry` (assembly, every TableGen definer, adc/sbc chains, non-definers) | `tools/gen_carry_flag_probe.py --suite-dir <dir>` | `tools/gen_carry_flag_probe.py --analyze <dir> <tag>` |
| Float flags `srFPFlags` / `srF2IFlags` / `srF2FFlags` (assembly, every definer, masks, stickiness, visibility offsets; `--plan extra` adds cancellation, mask and clear timing) | `tools/gen_fp_flags_probe.py --suite-dir <dir> [--plan main\|extra] [--only FORM ..]` | `tools/gen_fp_flags_probe.py --analyze <dir> <tag>`; `suite compare` |
| Overflow-flag clear race (a `mov` to `srSRS_of`/`srUPS_of` N bundles after a saturating conversion) | `tools/gen_flag_clear_probe.py --suite-dir <dir>` | `tools/gen_flag_clear_probe.py --analyze <dir> <tag>` |
| `vlda.ups` register read cycles (late writes of crUPSSign, crSat, the shift) | `tools/gen_ups_read_cycle_probe.py --suite-dir <dir>` | `tools/gen_ups_read_cycle_probe.py --analyze <dir> <tag>` |
| Host runtime-sequence op cost (a core polls its own data memory and stamps the tile counter as `write32` / `blockwrite` / `maskwrite32` ops land; own host with optional context QoS; standalone, not a `harness` suite) | `tools/gen_fw_op_cost_probe.py generate\|build <dir>`; runner `tools/fw_op_cost_run.py <cases> <out> --tag T --dpm 1 0` (hardware only, records the clock pair before and after each launch) | `tools/fw_op_cost_report.py <cases> <runs>` |
| Core launch against the runtime sequence (each core stamps its timer at entry and spins until a host `Timer_Control` reset write runs it backwards; padding writes before or after the reset, `Topology.sequence_writes`) | `tools/gen_launch_probe.py <dir>` | `tools/gen_launch_probe.py <dir> --analyze <tag>` (each core's lead over the reset write) |
| Core lock stalls (a load in flight across a blocking `acq`: `lk_D`, `same`; where a request stalls the core: `pos`, `relpos`; acquires with a change >= 0: `eq_*`). Two cores share one lock through a topology `SharedLock` link (declared with a fixed ID and init) | `tools/gen_lock_stall_probe.py generate <dir> --cases lk_1 .. same pos relpos eq_c1v1 ..` | `tools/gen_lock_stall_probe.py analyze <dir> <tag>` (eq: grant steps per candidate acquire rule) |
| Lock overflow / underflow and requests on a frozen lock (core and tile-DMA paths, refused acquires inside a b2b window; memtile and shim lock requests, BD releases / acquires on a flagged lock, release-chain run-ahead sweeps, a memtile DMA error probe), with the flag registers, lock counts, event and DMA channel status read back over control packets to the compute, memtile and shim control ports (`tools/ctrlpkt_readback.py`: packet encoding, per-tile targets, runtime sequence, host) | `tools/gen_lock_stall_probe.py frozen-generate <dir> [--cases ..]` (`FROZEN_CASES`, and `LEFT_CASES` for the memtile / shim family), `frozen-build <dir>` | `frozen-run <dir> <tag> --backend hw|emu [--runtime release] [--only ..]`, `frozen-analyze <dir> <tag>` |
| DMA task-queue depth, the BD pipeline of a stalled channel and shim S2MM ingress (a channel parked on a lock or its stream, tasks pushed with a status read after each, chains and long BDs, a shim MM2S streaming into its own S2MM; status words read back over control packets) | `tools/gen_dma_queue_probe.py generate <dir> [--cases ..] [--pushes N] [--lengths ..]` (`cases`, `stall_cases`, `absorb_cases`), `build <dir>` | `run <dir> <tag> --backend hw|emu [--runtime release] [--only ..]`, `analyze <dir> <tag>`. Do not push past a full queue on silicon: the push blocks the writer |
| Pipeline-fill depth and MM2S BD retirement (`of_q0_rich` with per-core progress counters and rep-stamped data, read back over control packets before the drain is armed; MM2S chains into a held consumer; a memtile MM2S into an idle shim S2MM; path capacities by data race) | `tools/gen_pipeline_fill_probe.py generate <dir>` (`of_q0_rich`), `--chain core mem [--lengths ..]`, `--sink`, `--race [case ..]`; `build <dir>` | `run <dir> <tag> --backend hw|emu`, `analyze <dir> <tag>`. Every design keeps one task per channel queue |
| Tile-DMA BD acquires with a change >= 0 (hand-written one-column design, loopback MM2S -> S2MM; standalone, not a `harness` suite) | `tools/gen_dma_lock_probe.py generate\|build <dir>`; `run <dir> <tag> --backend hw\|emu` | `tools/gen_dma_lock_probe.py analyze <dir> <tag>` |
| Stream absorb (core -> core words before the producer stalls, direct and via the switch FIFO) | `tools/gen_stream_absorb_probe.py generate <dir>` | `tools/gen_stream_absorb_probe.py summarize <dir>/runs/<tag>/<case> <k>` |
| Stream round trip (core <-> core, 1..3 hops, `--fifo` through the switch FIFO): per-hop latency | `tools/gen_stream_rtt_probe.py generate <dir> [--fifo]` | `tools/gen_stream_rtt_probe.py summarize <dir> <tag>` |
| Shuffle network sweep (one Chess case) | `tests/shuffle-sweep/gen_suite.py <dir>` | `tests/shuffle-sweep/analyze.py <dir>/runs/<tag>/shuffle-sweep/core.out.bin` |
| Characterizers (`tests/*.s`: sparse mask, sparse MAC, bf16 element-wise and sub-precision, vldb.4x*), five cases | `tools/charz_suites.py <dir>` | `python3 tests/<name>.py --suite <dir> r1 [e1]` |

Chess compiles at about 6 s of wall per case at `-j16` (xchesscc runs twice per case): build
the Chess suite in slices with `--only`. `XDNA_EMU_CHESS_WRAPPER="<command> [args]"` prefixes
every Chess invocation (the `xchesscc_wrapper` compile and the Chess package's aiecc) with a
launcher such as a compile cache or server; it must run the command unchanged, and it is not
part of the package's inputs hash. Every declaration in `me_chess_opns.h` either builds or is
listed with a reason in the suite's `meta.skipped` (the generator reads `me_chess_llvm.h` and
`me_vmult_float_emulated.h` to drop declarations Chess cannot call; Chess verdicts that only a
compile can reveal are recorded in `tools/chess_gen_outcomes.json`). An operand that is a vector of
addresses (the `vldb.4x*` gathers, found from me.h's `load_4x_address` block and me_chess.lib)
is filled with offsets into a 256-byte table after the arguments, plus the table's run-time
address, so the gather reads the kernel's own input; the case metadata records the layout. The Peano `*_conf` MAC
intrinsics take a random configuration word, which the emulator models in full
([mac-conf](../fidelity-gaps/mac-conf.md)).

Typical newcomer sequence for a new instruction-level check:

```bash
export PYTHONPATH=tools
./scripts/isa-test.sh --filter VMOV --tag try1         # generate + build + hw + emu + compare + analyze
python3 tools/isa-test-analyze.py build/isa-suite --failing
```

**Determinism check.** `harness suite hw <suite> d5 --repeat 5` followed by
`harness suite repeats <suite> d5` replaces the old shell loops. On the ISA suite it is how
nondeterministic instructions were found (the cycle counter `MOV_CNTR` is skipped by the
generator for that reason; cold-start state leaking across launches was fixed by resetting
the compress and sparse state and clearing the MAC destination in the kernels).

## Test Validation Architecture

Tests are validated against **hardware reference outputs** rather than
computed expected values. The pipeline is:

1. `test_cpp_parser` extracts buffer metadata (sizes, types, input patterns)
   from mlir-aie's test.cpp files
2. `native_hw` (preferred) compiles each test's own test.cpp on real NPU
   hardware and runs it; `npu_runner` is the fallback path; either way
   the captured output lands under `tests/npu-outputs/<test_name>/`
3. `xclbin_suite` sets up emulator input buffers from `BufferSpec`, runs
   the emulation, and compares output against the hardware reference
4. `test_overrides.toml` provides skip gates and expected-fail annotations
   for tests that cannot run or have known issues
5. The bridge test suite (`scripts/emu-bridge-test.sh`) is a parallel,
   broader validation path that runs each test's own `test.exe` against
   both real NPU and the emulator XRT plugin

This replaces the previous manifest system (68 TOML files with hardcoded
expected values) with automatic test discovery and hardware-as-oracle
validation.

## Test Coverage Notes

- Unit tests are concentrated in individual modules; run `cargo test --lib` for the fast suite
- All Cargo tests run with `nice -n 19`; doc tests additionally use limited parallelism because each process loads the TableGen files
- Integration tests (real binary execution) require feature flags and pre-built test artifacts

## Bridge Test Suite (`scripts/emu-bridge-test.sh`)

The primary validation target. Exercises the full XRT bridge path:
`test.exe -> XRT -> plugin -> emulator`. Dual-compiler by default (Chess
ground truth, Peano informational). Six phases: discover, compile (parallel),
run HW (parallel -j5), run EMU (parallel -j nproc), trace comparison
(`--trace=pc-anchored`), event sweep (`--sweep`), report per-compiler matrix.

Sweep mode (Phase 5b) is two-phase: HW arm runs serially across tests
(NPU is single-tenant), EMU arm runs serial across tests with parallel
events per test (-j N). The EMU arm uses `bridge-trace-runner --batch-stdin`
with a RESET command so worker processes can re-run multiple tests
without spawning a fresh emulator each time.

`bridge-trace-runner` captures a tile state snapshot (CORE_PC/STATUS,
TIMER_LOW, perf counters, all 16 LOCK*_VALUE, DMA channel CTRL/STATUS)
on `run.wait` timeout when invoked with `--snapshot-on-timeout <dir>`.
Sweep mode wires this automatically into
`<sweep_out_dir>/<test>.<compiler>.lockstep.work/wedge-snapshots/`.

Key flags: `--chess-only`, `--peano-only`, `--no-hw`, `--compile`,
`--serial-hw`, `--sweep`, `--trace=pc-anchored`, `-v <filter>`.

### Known coverage gap: Python/IRON host path

The suite only ever exercises the **C++ host** (`test.exe`, raw XRT API). Some
upstream kernels (e.g. `add_one_objFifo`, `add_one_objFifo_elf`) also ship a
**Python/IRON host** (`test.py`, `aie.iron` + `DefaultNPURuntime.run_test` via
pyxrt) for the *same* xclbin. Both lit RUN lines derive the same variant name
(`""`, from `aie.xclbin`), so `get_run_variants` collapses them and
`get_variant_run_cmd("")` always resolves to the first command -- the C++ one.
The Python host is never run. (Before the variant-dedup fix this surfaced as a
double-enqueued job that ran `test.exe` twice.)

This is fine for *fidelity* -- the kernel compute is identical regardless of
host -- but it leaves the **IRON/pyxrt -> plugin integration** unvalidated.
Making the emulator a true drop-in for the IRON Python flow is a wanted future
feature: it needs a real *host axis* (cpp vs py) distinct from the
xclbin-derived variant, plus confirming `DefaultNPURuntime` actually routes
through `XDNA_EMU` to our plugin. Deliberate feature, not a quick flag.

### Specialized capture: `tools/bug6-trace.sh`

Single-sudo wrapper that enables the existing `amdxdna_trace`
kernel tracepoints (xdna_job, mbox_*, uc_*), runs one test under
timeout, and snapshots the trace ringbuffer + dmesg. Built for the
bug #6 (`memtile_dmas/writebd` hang) investigation; shelved 2026-05-14
with canonical pass shape captured. Re-run if writebd hangs again --
diff against `build/experiments/bug6/pass-baseline-v2.trace`.

## Trace Pipeline

Binary trace comparison between emulator and real NPU hardware. The legacy
Python tools have been superseded by mlir-aie's declarative trace IRON API
(Phase B of cycle-budget testing). Legacy tools remain in `tools/deprecated/`
for reference.

**Active pipeline:**
- `tools/mlir-trace-inject.py` -- inject trace routing into MLIR (uses mlir-aie API)
- `bridge-runner/bridge-trace-runner` -- multi-batch sweep orchestrator
- `tools/parse-trace.py` -- single-source decoder wrapping mlir-aie's `parse_trace`; emits flat events JSON, cycles scalar, raw Perfetto, or raw command stream
- `src/bin/trace_compare.rs` -- Rust binary comparison (replaced Python OOM)
- `src/trace/compare.rs` -- core comparison logic (mode 0)
- `src/trace/vcd.rs`, `src/vcd/` -- aiesimulator VCD parser and mapper

Build: `cargo build --release --bin trace-compare`

**Deprecated tools** (do not add new callers; v1 suffix avoids confusion with current top-level tools of the same base name):
- `tools/deprecated/trace-inject.py` -- legacy MLIR trace injection
- `tools/deprecated/trace-sweep-v1.py` -- pre-IRON-API multi-batch orchestrator
- `tools/deprecated/trace-trim.py` -- buffer trimming
- `tools/deprecated/trace-merge.py` -- Perfetto JSON merge
- `tools/deprecated/trace-patch-events-v1.py` -- pre-IRON-API event patching
- `tools/deprecated/trace-bridge.sh` -- shell driver superseded by `scripts/emu-bridge-test.sh`
- `tools/deprecated/trace-compare.py` -- Python comparator superseded by `src/bin/trace_compare.rs`

See `tools/deprecated/README.md`.
