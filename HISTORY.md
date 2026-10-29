# Development history

xdna-emu, an emulator for AMD XDNA / AIE2 NPUs written in Rust, was started in
December 2025 (the GitHub repository was created on 2025-12-23). The
pre-release git history is not published. This document is a month-by-month
summary of how the project developed; the roadmap in `ROADMAP.md` and
`docs/roadmap/` describes the current state and milestone vocabulary.

Approximate commit counts per month are given for scale (about 2,400 in total).

## December 2025 (31 commits)

- First commit of an MVP, followed within days by a modular interpreter with
  the AIE2 register file and execution model.
- VLIW bundle decoding: format detection and slot extraction for the 16- to
  128-bit bundle formats, reaching full recognition of the instruction bundles
  in real kernel binaries.
- Device constants for tile types and memory sizes were checked against the
  public architecture manual.
- First DMA engine, stream router, lock and cross-tile memory-latency timing
  infrastructure, and a test runner that executes compiled kernels.
- Rapid ISA expansion (scalar division, vector comparison/bitwise/shift,
  convolution operations); the unit-test suite reached roughly 560 tests by the
  end of the month.

## January 2026 (16 commits)

- Branch delay slots implemented; the simple reference kernels then ran with
  correct results.
- A "remove hardcoding" pass centralised register and device constants.
- Instruction decoding moved to a TableGen-driven decoder built from the
  open-source llvm-aie definitions, with the regex fallback removed so
  failures are loud.
- A mock XRT library and FFI bindings were added so the mlir-aie test programs
  could be pointed at the emulator, with a harness discovering about 30
  mlir-aie tests.
- DMA gained multi-dimensional addressing, and the device model was expanded
  (register specifications, tile state, array management).

## February 2026 (158 commits)

- Work resumed in mid-February with a data-driven device model and a deferred
  load pipeline with VLIW snapshot semantics.
- Tracing: Perfetto export, trace-event coverage (port events, bank-conflict
  events, edge-detection monitors), and a trace viewer in the GUI with cursor
  navigation and drag-and-drop loading.
- Dual-compiler (Chess and Peano) test discovery and a unified test runner with
  explicit pass/skip/expected-failure status; DMA-only tests no longer count as
  false passes.
- Hardware fidelity items: cross-tile memory latency, East/West stream routing,
  cascade streams, UPS/SRS conversions with control registers, zero-overhead
  hardware loops, MemTile DMA zero-padding, and BD-chain repeat fixes.
- NPU instruction interleaving with a state-machine executor.
- A random-kernel fuzzer for differential testing against hardware was
  designed and its first pipeline built (generation, compile, emulate, compare).

## March 2026 (516 commits)

- XRT driver plugin: the emulator is built into a shared library that loads in
  place of the hardware driver, so unmodified mlir-aie test programs run
  against it. A bridge-test script runs each kernel on the emulator and,
  optionally, on real hardware, with a parallel hardware pool and automatic
  retry of collateral timeouts.
- Stream-switch pipeline delay per hop, lock-arbiter ordering fixes, and a
  single register-write bus matching hardware behaviour.
- A separate `xdna-archspec` crate began to hold the architecture description,
  cross-checked against aie-rt register constants; memory addressing was
  corrected to the hardware's directional model.
- Trace pipeline: trace-feasibility planner, trace preparation and sweep tools,
  hardware-versus-emulator trace comparison with column remapping, and VCD
  signal comparison.
- ISA test harness: an instruction classifier and assembly test-point
  generator, multi-tile MLIR generation, and branch, cascade, lock, FIFO and
  pack/unpack strategies. Passing instruction coverage climbed from the 540s to
  the 570s of 606 instructions in a week, then continued with vector, matmul,
  sparse and floating-point work.
- Large source files split into folders; the LLVM-based decoder became the
  production path for operand extraction.

## April 2026 (458 commits)

- Vector and matrix fidelity: fused instruction handlers, corrected operand
  ordering, restoring-division step semantics, and a sparse multiply-accumulate
  pipeline modelled to match hardware (from 0% to 15.7% on its test points at
  the start, continuing upward).
- Structural refactoring: the FFI layer became its own crate, and a series of
  subsystem audits moved architecture facts (registers, DMA, locks, stream
  switch topology, latency tables, shuffle modes) into `xdna-archspec`
  behind traits.
- Parser cleanup (CDO split into framing, syntax and semantics; structured
  parse errors).
- Bridge-runner improvements: async contexts, XRT kernel argument discovery,
  profile-named plugin builds, and cycle-diff classification
  (match / drift / empty) between emulator and hardware traces.
- Trace decoding gained PC-anchored modes, performance-counter-driven trace
  events, and PC-anchored set diffs; instruction lock request/release events
  were emitted.
- Release-build performance work (compiled-out logging, hoisted borrows in
  routing hot loops).

## May 2026 (390 commits)

- Cycle-cost model for NPU control instructions, and trace start pipeline with
  timer-reset synchronisation across tiles.
- Memory-tile trace injection, per-module event grounding, and bug repros
  captured as small standalone kernels.
- Core debug features: event-driven halt, single-step on event, watchpoints,
  with depth tests.
- Control-packet handling with faithful error bits, an interrupt path
  (event-generate to shim interrupt, async-error categorisation, context-state
  and recovery reporting through the plugin), and clock-control gating.
- A coverage model that records, per architectural feature, whether behaviour
  is verified, with a release gate test and generated per-architecture index.
- Calibration campaigns on real hardware: shim throughput, K-sweep kernels,
  and read/write access latency probes. A non-stalling controller refactor
  and latency-aware stream backpressure followed from these.
- The fuzzer gained a `fuzz` subcommand in the CLI.

## June 2026 (494 commits)

- Vector fuzzer: a typed operation table, runner, banking and replay, plus
  scalar and DMA (including MemTile) generators; the DMA generator reached a
  full-universe smoke gate of 81/81 cases compiling and running.
- Arithmetic fidelity: MAC pipeline latency for deferred accumulator writes,
  bf16 conversion and flush-to-zero audits, and the dominant NaN-payload
  behaviour observed on silicon.
- Simulator comparison tooling and per-kernel input specifications were added
  to the timing sweep.
- A long trace-fidelity effort: trace
  decomposition and reporting, multi-run capture with coverage union and gap
  reports, a route-graph self-model with measured provenance, falsifier tests
  for ordering and lock hand-off, and jitter clustering for hardware runs.
- Corrections found along the way: the AIE2 cascade is 512 bits wide, and
  S2MM ingress drains at memory-bus rate.
- An end-of-stream release tail on the consumer-free cadence and a hard runaway
  ceiling on the unbounded run loop in the FFI.

## July 2026 (259 commits)

- Continued skew calibration of trace timing across shim, memory tile and core
  domains, with several assumptions refuted by hardware measurements and
  recorded as such.
- Memory timing: physical bank mapping derived from the architecture layout,
  core stalls when a bank arbitration is lost, core-class priority with DMA
  urgency override, and tracing of DMA memory-pressure events.
- GUI: a tile-grid and tile-snapshot view-model, a run/pause control bar,
  route rendering, and real-time per-channel DMA progress bars.
- Live external host memory ranges mapped into the emulated address space.
- Build now fails loudly when the aie-rt-derived data is unavailable; an
  evidence-bundle ledger for NPU1 capture campaigns was defined and audited.

## August 2026 (49 commits)

- Core reset transitions preserved; native publication of compute-DMA and
  program-memory address faults.
- Trace tooling refreshed for current mlir-aie trace modes and event names,
  with task-completion-token records in decoded traces.
- Phoenix column freeze/resume behaviour classified and gated in tests.
- QoS passed through to hardware contexts in the bridge.

## September 2026 (1 commit)

- A README note on development status; otherwise a quiet month.

## October 2026 (55 commits)

- Toolchain migration: the Rust build and bridge flow moved to the aiecc C++
  driver, and bridge tests stopped treating upstream's Peano requirement
  markers as Peano-only. Trace tooling follows upstream changes (for example,
  removed objectfifo subview types).
- Fidelity fixes: the DMA control register's channel reset field is honoured,
  out-of-order BD IDs are stamped into MM2S packet headers, and BD iteration
  runs per execution; the plugin runs each command on its own compute unit's
  program image.
- Contributor guidance trimmed, the public API widened for out-of-tree
  platform crates, and the unused bytecode decoder fallback removed.
