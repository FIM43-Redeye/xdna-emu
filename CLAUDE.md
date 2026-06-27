# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository. It is loaded into context every session, so it stays
lean: the must-know rules and a map to deeper docs. Reference material lives
one hop away under `docs/` and `.claude/components/` -- pointers are inline.

## Workspace Setup

**Start Claude from the parent `npu-work` directory**, not from xdna-emu directly:

```bash
cd /home/triple/npu-work
claude
```

This ensures the NPU dev environment auto-activates (Python 3.13, mlir-aie,
Peano), XRT and hardware access are configured, and helpers like `npu-compile`
and `npu-run-quiet` are available.

The workspace contains:
- `xdna-emu/` - This emulator project (main focus)
- `aie-rt/` - Official Xilinx aie-rt (HAL, tests, FAL). `driver/src/` has
  register defs and DMA/lock/stream-switch implementations.
- `mlir-aie/` - MLIR-based AIE compiler, test suite, device models.
  `lib/Dialect/AIE/Util/aie_registers_aie2.json` is the AM025 register database.
- `llvm-aie/` - Peano compiler (LLVM with AIE backend, TableGen ISA definitions)

See `/home/triple/npu-work/CLAUDE.md` for environment details.

---

There is one absolutely critical rule to always keep to: **DERIVE FROM THE
TOOLCHAIN.** The open-source toolchain (aie-rt, llvm-aie, mlir-aie) is the
authoritative specification for hardware behavior. Never hardcode what can be
extracted. See "Correctness Principle" below.

## Project Vision

**xdna-emu** is an open-source, cycle-accurate emulator and visual debugger for
AMD XDNA NPUs (Ryzen AI). It fills the gap left by AMD's proprietary
aiesimulator (license-gated, CLI-only, not cycle-accurate). The goals: load and
faithfully execute real `.xclbin` binaries, debug visually (tiles, data flow,
DMA, locks, routing), validate emulator output against the real NPU, and
optionally compile kernels via Peano and run them immediately.

### Licensing and Relationship to AMD

This project is **MIT-licensed** and exists to help the AMD NPU ecosystem. The
emulator is orders of magnitude slower than real silicon -- a development tool,
not a hardware substitute. Its purpose is to lower the barrier to entry for NPU
programming, which benefits AMD by expanding their developer community.

**Source derivation policy.** All emulator code is original. Hardware behavior
is derived from these sources, in order of preference:

1. **Open-source toolchain** (aie-rt, llvm-aie, mlir-aie) -- Apache 2.0 / MIT.
   Primary and preferred source. Derive from these wherever possible.
2. **Hardware observation** -- running binaries on the real NPU we own. The
   hardware itself is ground truth.
3. **aietools** (AMD proprietary, locally installed) -- strictly a **reading
   reference** for behavior the open-source toolchain doesn't document
   (primarily vector compute semantics). Never copy code or data into this
   repo. Read, understand the hardware facts, then write original
   implementations. aiesimulator may be a debugging aid, but the real NPU is
   always ground truth.
4. **AM020/AM025 documentation** -- AMD architecture reference manuals for
   areas not covered elsewhere.

**In practice**: comment the source of behavioral knowledge as the hardware
behavior itself (e.g., "Rounding matches observed NPU output" or "BD field
layout per AM025 register database"), not as proprietary tool internals. The
knowledge is about how the silicon works; the implementation is ours.

## Current Status

See [ROADMAP.md](ROADMAP.md) for detailed status with confidence markers
(VERIFIED / OBSERVED / CLAIMED).

| Phase | Status | Summary |
|-------|--------|---------|
| [1. Core Accuracy](docs/roadmap/phase1-core-accuracy.md) | **100% ISA** | 4815/4815 ISA test points; bridge tests pass on ~75 mlir-aie kernels |
| [2. Toolchain Integration](docs/roadmap/phase2-toolchain-integration.md) | Partial | XRT plugin works; bridge tests run; Peano compilation not integrated |
| [3. Developer Experience](docs/roadmap/phase3-developer-experience.md) | GUI exists | GUI renders; debugging features not built |
| [4. Validation & Testing](docs/roadmap/phase4-validation-testing.md) | Active | Dual-compiler bridge tests, trace sweep, parallel HW |
| [5. Production Readiness](docs/roadmap/phase5-production-readiness.md) | Not started | |
| [6. Community & Ecosystem](docs/roadmap/phase6-community-ecosystem.md) | Not started | |

Run `cargo test --lib` to see the current test count. Do not rely on numbers
written in documentation -- they go stale within a session.

## Target Devices

Official device names from the xdna-driver source:

| Driver ID | Product Name | Codename | Architecture | Array Size | Status |
|-----------|--------------|----------|--------------|------------|--------|
| NPU1 | Ryzen AI | Phoenix/Hawk Point | AIE2 (XDNA) | 5 cols x 6 rows | **Primary target** |
| NPU4 | Ryzen AI 300 | Strix Point | AIE2P (XDNA2) | 5 cols x 6 rows | Planned |
| NPU5 | Ryzen AI Max | Strix Halo | AIE2P (XDNA2) | 8+ cols x 6 rows | Planned |
| NPU6 | (TBD) | Krackan | AIE2P (XDNA2) | 5 cols x 6 rows | Planned |

Array sizes include the shim tile row (row 0). NPU2/NPU3 are deprecated
prototypes, not consumer devices. We start with **Phoenix (NPU1/AIE2)** because
it's the hardware we have; AIE2P support is incremental once AIE2 is solid.

**mlir-aie device naming** (differs confusingly from driver IDs): `npu1` /
`npu1_Xcol` -> NPU1 (Phoenix, AIE2); `npu2` / `npu2_Xcol` -> NPU4 (Strix,
AIE2P); `xcvc1902` -> Versal AIE1; `xcve2802` -> Versal AIE2. Versal FPGAs are
not in scope (no local hardware), though AIE1 support may come later since the
TableGen parser handles multiple architectures.

## Correctness Principle: Derive From the Toolchain

**CRITICAL: The open-source toolchain IS the hardware specification. Derive
emulator behavior from it. Never hardcode what can be extracted.** Every
hardcoded constant, bit position, or behavioral assumption is a potential bug.
The toolchain evolves with the hardware -- if we derive from it, we evolve too.

**The rule**: before implementing any hardware feature, check whether the
toolchain already defines it. Only fall back to AM020/AM025 documentation for
things the toolchain genuinely does not cover (primarily vector operation
computational semantics).

**Authoritative sources, in priority order** (full per-source breakdown -- what
each provides, key files, where the emulator consumes it -- in
[`docs/toolchain-sources.md`](docs/toolchain-sources.md)):

1. **aie-rt** (`../aie-rt/driver/src/`, branch `xlnx_rel_v2025.2`) -- the same
   HAL that programs real silicon. Reference implementation for DMA, lock, and
   stream-switch behavior. Use the official Xilinx clone, NOT mlir-aie's
   vendored fork.
2. **AM025 register database**
   (`../mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`) -- 1,806
   registers / 6,412 bit fields, parsed by `regdb.rs`. BD field parsing is
   already fully data-driven from this (zero hardcoded bit positions in
   `bd.rs`); extend the pattern.
3. **llvm-aie TableGen** (`../llvm-aie/llvm/lib/Target/AIE/`) -- complete ISA.
   Decoding is fully TableGen-driven; semantics ~33% data-driven via SemanticOp,
   closing the gap is the goal.
4. **mlir-aie device model** (`tools/aie-device-models.json`, regenerated by
   `tools/aie-device-dump.py`) -- array topology. Fully data-driven.

**What still needs non-open-source references** (vector compute semantics,
stream-switch per-port type assignments, micro-timing) and the read-only
aietools reference paths are documented in
[`docs/toolchain-sources.md`](docs/toolchain-sources.md).

**Research guidance**: these sources are extensive. Use **Explore agents** to
navigate them rather than reading files directly and burning context. Good
queries: "How does aie-rt implement DMA channel start?", "What are the BD field
positions for shim tiles?", "What does the VMAC configuration word control?",
"How many stream switch ports does a memtile have?"

### Binary Formats
- **XCLBIN**: Container (ELF cores + PDI/CDO configuration)
- **ELF**: Per-core executables
- **CDO**: Configuration Data Objects (DMA descriptors, routing)

## What We're Emulating

**Per tile**: a VLIW compute core (256-bit vector unit), local memory (64KB
compute / 512KB mem tile), a 2-channel n-dimensional DMA engine, 64 locks, and
a circuit/packet stream switch.

**Array level**: shim tiles (DDR interface via NoC), mem tiles (shared memory
between columns), and routing via configured stream switches.

**Execution model**: cores run independently (no global clock); DMAs operate
concurrently with cores; locks synchronize between tiles; stream switches route
based on CDO configuration.

## Technical Decisions

**Language: Rust** -- memory safety for a complex state machine, excellent
binary parsing (`goblin`, `nom`), built-in profiling, modern tooling.

**GUI: egui** (via `eframe`) -- pure Rust, immediate-mode (simple state),
cross-platform.

## Component Documentation

Detailed documentation for each module lives in `.claude/components/`. Read the
relevant file when working on that area of the codebase.

| Component | File | When to Read |
|-----------|------|-------------|
| Device model | [`.claude/components/device.md`](.claude/components/device.md) | Working on tiles, array, DMA, streams, locks, host memory (`src/device/`) |
| Interpreter | [`.claude/components/interpreter.md`](.claude/components/interpreter.md) | Working on instruction decode, execution, timing, multi-core (`src/interpreter/`) |
| Parser | [`.claude/components/parser.md`](.claude/components/parser.md) | Working on XCLBIN, ELF, or CDO parsing (`src/parser/`) |
| TableGen | [`.claude/components/tablegen.md`](.claude/components/tablegen.md) | Working on ISA definitions, decoder tables, llvm-aie integration (`crates/xdna-archspec/src/aie2/isa/`, with consumers in `src/interpreter/decode/`) |
| Testing | [`.claude/components/testing.md`](.claude/components/testing.md) | Working on tests, test runner, FFI, NPU instructions, config (`src/testing/`, `src/npu/`, `src/ffi/`, `tests/`) |
| Visual | [`.claude/components/visual.md`](.claude/components/visual.md) | Working on the GUI debugger (`src/visual/`) |

Top-level source files not covered by component docs:
- `src/main.rs` -- CLI and GUI entry point
- `src/lib.rs` -- crate root, module declarations
- `src/integration/mod.rs` -- external tool integration (placeholder)

## Related Resources

- **aie-rt**: `../aie-rt/driver/src/` - Official Xilinx HAL (branch
  `xlnx_rel_v2025.2`). Reference for DMA/locks/stream-switch. (mlir-aie's
  vendored `third_party/aie-rt/` fork is NOT the emulator's reference.)
- **mlir-aie**: `../mlir-aie` - AIE compiler, test binaries, device models,
  AM025 register database JSON
- **llvm-aie**: `../llvm-aie` - Peano compiler, ISA definitions via TableGen
- **aietools**: `../amd-unified-software/aietools` - AMD proprietary tools.
  Read-only reference; see Licensing section.
- **xdna-driver**: `/home/triple/npu-work/xdna-driver` - Linux kernel driver
- **XRT**: https://github.com/Xilinx/XRT - runtime (installed at /opt/xilinx/xrt)

## Development Tools

- **XRT** (`/opt/xilinx/xrt/bin/`): `xclbinutil` (XCLBIN inspect/extract:
  `--info --input file.xclbin`, `--dump-section`), `xrt-smi` (device mgmt).
- **mlir-aie**: `llvm-objdump -d file.elf` (disassemble AIE ELFs),
  `aie-translate`, `aie-opt`.
- **aietools** (read-only reference): `aiesimulator` (cycle-accurate sim,
  debugging aid not oracle; `aie2simmsm` cycle-accurate / `aie2simmsm_func`
  functional), `elfanalyzer`, `hwanalyze`, `eventanalyze`, `xchesscc` (via
  `xchesscc_wrapper`). aietools `LD_LIBRARY_PATH` must be appended not prepended
  -- see activate-npu-env.sh.
- **RyzenAI-SW** (`/home/triple/npu-work/RyzenAI-SW/`): NPU driver source and
  examples.

## Build Commands

```bash
cargo build                     # debug
cargo build --release           # optimized
cargo run -- path/to/binary.xclbin

# Test
./scripts/run-tests.sh          # all tests (doc tests nice'd)
./scripts/run-tests.sh --lib    # fast: library tests only
cargo test --lib                # direct

# Bridge tests (dual-compiler, requires XRT + NPU)
./scripts/emu-bridge-test.sh                    # full run
./scripts/emu-bridge-test.sh --no-hw add_one    # quick EMU-only

# FFI / plugin
cargo build -p xdna-emu-ffi     # update debug .so loaded by XRT plugin
./scripts/rebuild-plugin.sh     # full release build + install

# Also: cargo bench; cargo flamegraph --release -- <xclbin> (profiling)
```

**Building for bridge tests**: `cargo build` builds the main binary but NOT the
FFI crate's cdylib `.so`. Bridge tests load `target/debug/libxdna_emu.so` (or
release). After changing emulator code, run `cargo build -p xdna-emu-ffi` to
update the `.so`, or `./scripts/rebuild-plugin.sh` for the full release cycle.
(See the Operational quick-reference -- stale `.so`s cause phantom bugs.)

**Doc tests** spawn separate processes that each load TableGen from llvm-aie;
the test script runs them `nice -n 19` with limited parallelism.

## Test Infrastructure

**Primary: bridge test suite** (`scripts/emu-bridge-test.sh`). The **XRT bridge
path is the real validation target** -- the full hardware-equivalent flow
`test.exe -> XRT -> plugin -> emulator`. Dual-compiler: every test compiles with
BOTH (Chess is ground truth; Peano failures are informational). Five phases:
discover, compile (parallel), run HW (-j5), run EMU (-j nproc), report. Flags:
`--chess-only`, `--peano-only`, `--no-hw`, `--compile`, `--serial-hw`,
`--sweep`, `--trace=pc-anchored`, `-v <filter>`. Build dirs under
`mlir-aie/build/test/npu-xrt/$name/{chess,peano}/`; results under
`build/bridge-test-results/YYYYMMDD/` (symlink `latest`).

**Backup: in-process xclbin runner** (`src/testing/xclbin_suite.rs`) runs
xclbins against the emulator without XRT, for isolated subsystem testing.
Driven from unit tests.

## Tracing Ecosystem

Binary trace comparison between emulator and real NPU hardware, all converging
to Perfetto JSON (ui.perfetto.dev).

**Division of labor.** Upstream mlir-aie owns trace *injection* (declarative
IRON API) and *decoding* (`aie.utils.trace.parse_trace`). Our local layer owns
*prep glue, sweep, matrix, regression-verification* on top. The local pieces
are the regression gate for the emulator refactor -- don't drop them in favor
of upstream-only solutions until upstream ships an equivalent 8-batch sweep +
matrix diff.

Full strategy in [`docs/trace/strategy.md`](docs/trace/strategy.md); the
six-layer pipeline tool inventory (pre-build, run, decode, compare,
matrix/regression, glue) and deprecated tools in
[`docs/trace/tooling.md`](docs/trace/tooling.md).

## XRT Plugin (`xrt-plugin/`)

Driver plugin replacing the real XDNA kernel driver. XRT loads the `.so`, which
delegates to the Rust emulator via FFI (`src/ffi/`).

**Build**: `./scripts/rebuild-plugin.sh` (debug) / `--release` (release).
Install symlinks the build output into `/opt/xilinx/xrt/lib/`.

**xdna-driver branch**: the plugin compiles SHIM sources from the sibling
`xdna-driver` tree, which must be on `emu-shim-base` -- it carries the SHIM
hooks the plugin needs (protected `m_dev_fd`, `start_col` plumbing) and pins the
`xrt` submodule to `emu-xrt-base` (virtual `scan_devices`). rebuild-plugin.sh
warns if on another branch. Branches are clean (no working-tree patches);
switch back to a PR/work branch when not building the plugin.

**Env contract**:
- `XDNA_EMU` -- presence (any value) activates the emulator; the plugin replaces
  `xrt::device(0)` so tests target the emulator with no BDF magic. Unset = real HW.
- `XDNA_EMU_RUNTIME=release|debug` -- which `.so` profile to dlopen (default `debug`).
- For HW invocations from a poisoned shell, use `env -u XDNA_EMU XDNA_EMU_RUNTIME`.

## How To Begin

1. Read [ROADMAP.md](ROADMAP.md) for the development plan and confidence markers
2. Check the relevant [phase documentation](docs/roadmap/) for current details
3. Run `cargo test --lib` to verify everything works
4. Read the component doc (`.claude/components/`) for the module you are working on

## Feature Implementation Policy

**Finish what you start.** When implementing an isolated subsystem (control
packets, DMA padding, cascade flows, etc.), build it to 100% coverage before
moving on. A half-implemented feature is worse than an unimplemented one -- it
creates false confidence and makes debugging harder because you can never be
sure whether a failure is "the feature isn't done yet" or a real bug. This
means: all test cases pass, edge cases handled, implementation matches hardware
across the full test matrix (both compilers, both HW and EMU). Only then move
to the next feature.

## Debugging Guidelines

**Match real hardware behavior.** The goal is always to reproduce what the
silicon does, not to invent workarounds or simplified approximations. If aie-rt
does something a particular way, we do it that way too.

When investigating a failing test:
1. Start from the failing assertion and work backward through the data path.
2. Do not jump to hypotheses about unrelated subsystems (e.g., don't
   investigate stream routing if the data is wrong at source memory level).
3. If unsure about hardware semantics, ask rather than guess.

**Memory watch mechanism.** Set `XDNA_EMU_WATCH` to log every memory access to
specified address ranges. Format: comma-separated `address:bytes` pairs (hex,
0x prefix optional, bytes defaults to 4). Requires `RUST_LOG=info`.

```bash
# Watch three addresses (40 bytes each) during a bridge test
XDNA_EMU=1 XDNA_EMU_WATCH=0xC000:40,0x428:40,0x400:40 RUST_LOG=info \
  ./test.exe 2>watch.log
grep "\[WATCH\]" watch.log
```

Output is cycle-correlated; DMA watches use tile-local offsets, core watches
use the full 20-bit address space (e.g., 0x70400 for local memory at 0x400).

**Correctness before performance.** Do not optimize (including multithreading)
until emulator behavior is indistinguishable from real hardware. Making wrong
answers faster helps nobody, and threading introduces its own bugs that muddy
correctness work.

## Validation

Always run `cargo test --lib` after making changes. Do not consider work
complete until tests pass. If tests were passing before your changes and are
now failing, that is a regression to fix before moving on.

**Known fidelity gaps.** Confirmed model-vs-hardware disagreements are tracked
in [`docs/known-fidelity-gaps.md`](docs/known-fidelity-gaps.md) -- check it
before investigating a suspected emulator bug (a stale wrong note once cost a
whole session), and add a row when you confirm a new one.

**Planned: differential fuzzing.** The long-term validation strategy is a logic
fuzzer that generates valid kernels, runs them on both emulator and real NPU,
and compares results. Future work -- do not start building it until hand-written
test coverage confirms baseline correctness.

## Operational Quick-Reference

The must-know-always rules. Full procedures (NPU recovery escalation chain, SMU
command map, dev-environment state, formatting enforcement) are in
[`docs/operations.md`](docs/operations.md).

- **Rebuild the `.so` before HW/bridge tests.** `cargo test --lib` does NOT
  rebuild the plugin lib; ISA/bridge tests load whatever `.so` is on disk. After
  any Rust change run `cargo build` (+ `cargo build -p xdna-emu-ffi`, +
  `--release` if exercised). Stale `.so`s cause phantom bugs.
- **One build per target, but profiles can overlap.** Don't run the same `cargo
  build` invocation twice concurrently; `cargo build` and `cargo build
  --release` together are fine (cargo locks between them).
- **Never run two hardware test suites concurrently.** Bridge and ISA tests both
  grab the NPU; in parallel they fight and both must be killed. `cargo test
  --lib` unit tests are safe alongside (no hardware).
- **Always `cargo test --lib` after changes.** A pass that regresses is a
  regression to fix before moving on.
- **Never put persistent work in `/tmp`** -- this PC reboots often and wipes it.
  Tools/scripts under `xdna-emu/{tools,scripts}/`; experiment output under
  `build/experiments/` or `~/npu-work/experiments/`. Only ephemeral logs/temp
  dirs use `/tmp`.
- **Never pipe long-running commands** (`cargo build/test`, `dmesg -w`, test
  scripts) through `tail`/`head`/`grep` -- the filter buffers until EOF and
  appears to hang. Redirect to a file and Read it, or use `run_in_background`.
  Use `tee` for long backgrounded runs you want live + logged.
- **Privileged ops use `pkexec`, never `sudo`** (sudo's interactive auth fails
  silently in background tasks). Combine multiple privileged ops into one pkexec
  call. `dmesg` is unrestricted here -- never wrap it in pkexec.
- **HW is cheap; EMU is the slow part.** Real silicon runs a kernel in
  microseconds -- a targeted HW capture + register readback is trivially fast, so
  *err toward more HW runs, not fewer*, when HW can settle a question directly.
  What's expensive is the **EMU side** (CPU-bound cycle emulation) and the
  compile step. The full suites are slow because they emulate many kernels:
  `isa-test.sh` ~5-10 min, `emu-bridge-test.sh` ~15-30 min -- run *those* once
  after a batch of fixes, don't re-run them to "check progress." That caveat is
  about the suites, NOT about touching hardware: a single capture is cheap, use
  it freely.
- **When the NPU wedges**, first try `pkexec sh -c 'modprobe -r amdxdna &&
  modprobe amdxdna'`. If that doesn't recover it, escalate per the chain in
  [`docs/operations.md`](docs/operations.md) (bridge PM-cycle -> SBR ->
  suspend/resume -> reboot). Reboot is last resort; hand it to the user rather
  than running it yourself.
