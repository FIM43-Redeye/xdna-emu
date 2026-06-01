# aiesim Integrator — Design Spec

**Date:** 2026-06-01
**Status:** DESIGN — approved (2026-06-01); HAL independent-replay sequenced as
phase 2. Proceeding to writing-plans.
**Feasibility:** PROVEN (see
`docs/superpowers/findings/2026-06-01-aiesim-inprocess-backend-feasibility.md`).

## 1. Vision and goals

Wire AMD's `aiesimulator` into our work as a **workload-agnostic backend behind
our existing XRT plugin / FFI seam** — a "software driver for aiesim," peer to
our hand-rolled interpreter. One integration, selected at runtime, unlocking all
of:

- **Third bridge-test runtime** — HW vs. EMU vs. aiesim through the same driver
  path.
- **Post-Phoenix live functional oracle** — a second reference after the Strix
  swap retires our Phoenix hardware.
- **AIE2P / Strix bring-up** — a working model before (and alongside) new silicon.
- **Debugging cross-check + internals** — diff interpreter state against aiesim's
  at matching points; surface where each is right.

**Standing principle — never fall behind aiesim.** The interpreter is the
first-class artifact; aiesim is the pacing oracle. Every capability aiesim
exposes that we lack (see tier-3 below) is a logged item on an **interpreter
feature backlog**, not a permanent gap. The cross-check seam is how we discover
those gaps.

### Why this is feasible (summary; full capture in the findings doc)

The spike established the hard gates: the closed AIE2 cluster model
(`libaie2_cluster_msm_v1_0_0.osci.so`) loads in our own process, SystemC embeds
and runs in-process, `create_math_engine` executes, and — the biggest de-risk —
**there is no FlexLM license block**. The integration drives the *closed* ISS
through the *open* AIE-RT HAL via the `ess_*()` weak-symbol seam; no aiesimulator
launcher, no pkg-dir, no subprocess.

## 2. Architecture overview

The closed cluster, SystemC, the PS bridge, and the HAL are all C/C++, so the
backend is a thin Rust wrapper over a new C++ bridge library. The Rust side only
ever sees a small C ABI.

```
XRT plugin
  └─ FFI (Rust, crates/xdna-emu-ffi)
       └─ dyn NpuBackend
            ├─ InterpreterEngine        (Rust, existing)
            └─ AiesimBackend            (Rust, src/aiesim/backend.rs)  ── C ABI ──┐
                                                                                  ▼
                                                          libxdna_aiesim_bridge.so (C++, new)
                                                            ├─ sc_bootstrap     SystemC main + host globals, elaborate-once
                                                            ├─ aiesim_top       sc_module; builds cluster IN elaboration (fixes E513)
                                                            │     └─ MathEngine  closed cluster .so, dlopen(create_math_engine)
                                                            ├─ ps_bridge        our PSIP_ps_i3 twin: ess_*() ⇄ TLM sockets
                                                            └─ hal_driver       aie-rt HAL, -D__AIESIM__ (phase 2 — see §5)
```

## 3. The seam: `NpuBackend` trait and the three-tier boundary

Today `XdnaEmuHandle` owns `engine: InterpreterEngine` concretely; the 9 FFI
files call straight into it. We introduce a narrow trait and make the handle own
a backend:

```rust
pub struct XdnaEmuHandle {
    backend: Box<dyn NpuBackend>,   // was: engine: InterpreterEngine
    xclbin_path: Option<String>,
    // ... next_alloc_addr, free_list, async_callback stay FFI-side
}
```

The FFI uses the engine two distinct ways (43 `device()`/`device_mut()` reaches
for introspection + ~12 coarse execution ops). The boundary respects that split,
in **three tiers**:

### Tier 1 — execution spine (the trait)

`NpuBackend` is **narrow**: the cross-backend execution operations only —
`load_elf_bytes`, `step`/`run`, `sync_cores`, `reset_for_new_context`,
`flush_trace_to_host`, host-memory read/write, context state, halt reason. Both
backends implement these.

### Tier 2 — register/memory introspection (both backends, different impls)

The closed cluster is **not** a black box. The `ess_Read32` seam reads any 32-bit
register in the device address space, and in AIE2 lock counts, DMA BD/channel
status, core PC/status, and stream-switch config are *all* memory-mapped
registers — the same AM025 surface `regdb.rs` already drives. So register-backed
introspection is served by *both* backends: the interpreter reads its device
model; aiesim reads via `ess_Read32` → `transport_dbg_cb` (zero sim-time backdoor
read). This is a real cross-backend capability, **not** a downcast-to-Unsupported.

### Tier 3 — backend-exclusive extras (capability-probed)

Probed per backend; absent ⇒ a clean "unsupported" result, never a lie.

- **aiesim-exclusive** (from the `MathEngine` API, 61 exported methods): VCD
  waveforms (`add_sc_traces` / `configure_vcd_generation` / `set_vcd_dump_filename`),
  diagnostic guidance JSON (`dump_guidance_json` / `dump_fifo_guidance_json` /
  `dump_memory_violation_guidance_json`), native data-memory watchpoints
  (`dm_watchpoint_write`), event-trace (`event_trace_write` /
  `shim_event_trace_write` — the same event stream our trace ecosystem decodes),
  full state dump (`dump_state_thread` / `dump_memory` / `get_array`), per-tile
  latency injection (`add_*_tile_lat`).
- **interpreter-exclusive**: reconstructed semantic state that isn't
  register-backed (e.g. internal DMA FIFO contents, pending-syncs).

The `as_interpreter()` downcast escape hatch shrinks to **only** tier-3
interpreter-exclusives. Tiers 1–2 are honest cross-backend traits. Every
aiesim-exclusive (tier 3) that the interpreter lacks is a "never fall behind"
backlog item.

**Selection:** `xdna_emu_create` reads `XDNA_BACKEND` (`interpreter` default |
`aiesim`); the plugin's `transport_inprocess` already owns env-based config, so
no XRT-visible change.

**Trait vs. enum / downcast:** a trait (`Box<dyn>`) over an enum, so dispatch is
localized and a future third backend (real-HW passthrough, remote sim) needs no
call-site churn; the cost is the downcast hatch for tier-3, which is acceptable.

## 4. AiesimBackend internal structure

Thin Rust over a feature-gated, runtime-loaded C++ bridge. The five pieces of the
bridge:

| Piece | Responsibility |
|---|---|
| `sc_bootstrap` | Compile aietools' `sc_main.cpp`/`sc_main_main.cpp`; define the 2 host globals (`sc_stop_at_end_of_main`, `plio_complete`); own the **elaborate-once** entry (`main`→`sc_elab_and_sim`→our `sc_main`). |
| `aiesim_top` | An `sc_module` whose ctor calls `create_math_engine` *inside* live elaboration (the E513 fix), then binds sockets. Owns the `MathEngine*`. |
| `ps_bridge` | Our own `PSIP_ps_i3` twin. Provides `ess_Write32/Read32/WriteGM/ReadGM/WriteCmd`. Two bindings: host-initiator → cluster `get_ss_aximm_rd/wr` (config/MMIO in), and a host-DDR target ← cluster `shim_dma_rd/wr_socket` (DMA out). |
| `hal_driver` | The OPEN aie-rt HAL built `-D__AIESIM__`, SIM backend. **Phase 2** — see §5; not on the initial data path, but the committed next step (independent CDO replay). |
| `c_abi` | The `extern "C"` surface Rust calls: `aiesim_create` / `load_cdo` / `write_gm` / `read_gm` / `run` / `read_reg` / `read_mem` / `dump_*` / `destroy`. |

**Rust side** (`src/aiesim/backend.rs`): `AiesimBackend` implements `NpuBackend`,
translating each trait call to one C-ABI call. No SystemC or aietools types leak
into Rust.

**Repo placement:** a new top-level `aiesim-bridge/` (C++ + `CMakeLists.txt`,
mirroring `xrt-plugin/`) producing `libxdna_aiesim_bridge.so`; the Rust
`AiesimBackend` joins the existing `src/aiesim/`. The shape-A subprocess code
(`src/integration/aiesimulator.rs`) stays as the MLIR-derived fallback, untouched.

**Optionality is structural** (mechanics in §7): a cargo `aiesim` feature gate +
the bridge `.so` built only when aietools is present + runtime-loaded when
`XDNA_BACKEND=aiesim`. Default `cargo build` is unchanged.

## 5. Data path: the `ess_*()` bridge and CDO replay

### Fixed part (the PSIP_ps_i3 contract)

The bridge implements the five weak symbols as TLM transactions against the
cluster's sockets, with two bindings:

```
host → cluster   : ps_bridge initiator sockets ─► MathEngine::get_ss_aximm_rd/wr()   (config, MMIO, control)
cluster → host   : MathEngine::shim_dma_rd/wr_socket(col) ─► ps_bridge DDR target     (shim-DMA reads/writes DDR)

ess_Write32(addr,val)      → b_transport write on get_ss_aximm_wr   (or direct dm_write/proc_write/...)
ess_Read32(addr)->val      → transport_dbg_cb (zero-time backdoor read)
ess_WriteGM(addr,buf,len)  → fill host-DDR model; cluster pulls it via shim_dma during run
ess_ReadGM(addr,buf,len)   → read host-DDR model after run
ess_WriteCmd(...)          → control-channel transaction (sim start/stop/sync handshake)
```

`ess_Read32` via `transport_dbg_cb` is the keystone of tier-2 introspection: it
reads any register at zero sim-time, so all register-backed state returns through
the same `regdb.rs` addresses with no model perturbation.

### FFI spine → bridge mapping

| FFI op | bridge action |
|---|---|
| `load_xclbin`/`load_pdi` | parse → emit config op-stream → `ess_Write32`/blockwrite |
| `write_host_memory` | `ess_WriteGM` into host-DDR model |
| `execute_npu_instructions` | replay runtime-sequence ops (DMA BD config, locks) as register writes |
| `sync_cores` / `run` | `ess_WriteCmd` start → drive SystemC to quiescence → handshake (§6) |
| `read_host_memory` | `ess_ReadGM` from host-DDR model |
| `read_reg` (tier-2) | `ess_Read32` → `transport_dbg_cb` |

### CDO replay decision: parser-driven primary, HAL-driven follow-on

In AIE2 *everything is a memory-mapped register write* — core-enable, lock-init,
DMA-BD, mask-poll are all Write32/BlockWrite/MaskPoll primitives, and our
existing CDO parser already produces exactly that op-stream (it feeds the
interpreter today). So `hal_driver` is **not required on the data path**.

- **Path 2 — parser-driven (primary).** Our proven CDO parser emits ops; a thin
  shim calls `ess_*()` directly. Reuses heavily-validated code, and feeding the
  *same parsed ops* to both backends means a cross-check isolates **execution**
  divergence cleanly. This is the data path.
- **Path 1 — HAL-driven independent replay (the committed next step, phase 2).**
  Drive aie-rt's HAL, which calls `ess_*()` itself, giving a *second, independent*
  CDO interpretation that also catches bugs in our parser — strictly more rigorous
  as an oracle. Sequenced *after* the initial parser-driven backend works, not
  folded into it. One gating unknown to resolve when we take the step: whether
  aie-rt exposes a clean raw-CDO→`XAie_*` ingest (it mostly *emits* CDO; the
  consumer is normally firmware) — to be verified, and supplied if absent. This is
  a planned phase, not a maybe.

So `hal_driver` is a **phase-2** bridge piece: the initial data path is
parser-driven; the HAL independent-replay lands as the committed next step — a
deliberate "never fall behind" upgrade, sequenced after the backend is proven.

## 6. Lifecycle and process model

SystemC is **elaborate-once and process-global**: the module hierarchy is built
during elaboration, then run; it cannot be re-elaborated within a process and
cannot restart after `sc_stop()`. The interpreter, by contrast, is just Rust
state. So the backend cannot map `create`/`destroy` onto SystemC
construction/teardown.

### Model: a long-lived SystemC service thread + command queue

```
first aiesim_create  ─► spawn SystemC thread ─► sc_elab_and_sim(sc_main)
                                                   └─ sc_main: construct aiesim_top (cluster) ONCE
                                                       └─ loop { cmd = queue.wait();
                                                                 apply (LOAD_CDO / WRITE_GM / READ_GM / READ_REG = backdoor, zero-time);
                                                                 RUN = sc_start() until quiescent/complete;
                                                                 post reply }
FFI op  ─► push command, block for reply   (the plugin already serializes handles; clean synchronous handoff)
```

The cluster is constructed exactly once, inside the genuine elaboration context
the kernel sets up — which is also **how the E513 clean-instantiation gets
resolved**: the first implementation task replicates `aie_xtlm`'s pre-construction
setup *within this thread's `sc_main`*, rather than calling `create_math_engine`
from a throwaway context (what the probes did).

### `run()` semantics

Apply the submission's config via backdoor (zero-time) writes, then `sc_start()`
to advance timed simulation until the kernel signals completion (the
runtime-sequence's final `MaskPoll`/sync satisfies — `plio_complete` is the
completion flag) or a max-cycle budget elapses. Both map onto the **existing
`XdnaEmuHaltReason`** (`Completed` / `Budget`) — no new FFI surface.

### Multi-submission and teardown

Between runs, re-apply the CDO config (register writes overwrite prior state) and
re-init GM — the same logical reset the interpreter does via
`reset_for_new_context`, but without re-elaboration. `xdna_emu_destroy` resets
*logical* state and parks the service thread; real SystemC teardown
(`end_of_simulation`) happens at process exit.

### Constraints this imposes

- **The aiesim backend is a process singleton.** One SystemC sim per process. For
  the plugin/bridge-test case (one device, sequential submissions) that is exactly
  the usage; two concurrent aiesim devices in one process is unsupported by
  construction. The interpreter has no such limit.
- **All aiesim ops run on the service thread.** Matches the existing "handles
  aren't thread-safe; plugin serializes" contract — SystemC's global kernel makes
  it mandatory rather than advisory.

## 7. Build integration and license gating

Prime directive: the default `cargo build` / `cargo test --lib` stays exactly as
it is today — no aietools, no closed deps, sandbox-safe. Everything aiesim is
additive and optional, via three independent gates:

1. **Cargo `aiesim` feature (off by default).** Gates the `AiesimBackend` Rust
   code. Feature off → `XDNA_BACKEND=aiesim` returns a clean "built without aiesim
   support" error.
2. **Two-level runtime `dlopen` — zero build-time closed deps in cargo.**
   `AiesimBackend` `dlopen`s `libxdna_aiesim_bridge.so` (RTLD_GLOBAL); the bridge
   `dlopen`s the closed cluster (RTLD_GLOBAL). Nothing closed is link-time; even a
   `--features aiesim` build has no aietools dependency until the backend is
   selected at runtime. The RTLD_GLOBAL chain lets the cluster resolve our two host
   globals — the `.so` equivalent of the spike's `-rdynamic`.
3. **Bridge built separately via CMake, only when aietools present.**
   `aiesim-bridge/CMakeLists.txt` mirrors `xrt-plugin/` — uses the existing
   `FindAIETools.cmake`, finds SystemC + the cluster, applies the embedding flags
   (`-z execstack`, default-visibility host globals, system libstdc++). A
   `scripts/build-aiesim-bridge.sh` drives it, the way `rebuild-plugin.sh` builds
   the plugin. Absent aietools → bridge isn't built; runtime dlopen fails with a
   clear message.

**License gating:** `create_math_engine` + the cluster model run with **no FlexLM
block** (proven). The backend needs aietools *installed* (closed `.so`s + device
JSON + libsystemc) but **not a license server** at runtime. aietools is detected
via the same layout discovery as everything else; missing ⇒ graceful failure.

**Embedding recipe:** the full reproducible recipe (flags, host globals, libstdc++
selection, key paths) lives in the findings doc and is not duplicated here.

## 8. Arch-parameterization (AIE2 → AIE2P / AIE1)

Verified against all three arch libs:

- **Uniform C entry:** `libaie2_`, `libaie2ps_`, and `libaie_` cluster libs all
  export `create_math_engine` — same factory ABI.
- **Device models** are arch-specific: AIE2 → `data/aie_ml/devices/` (VC2802…),
  AIE2P → `data/aie2ps/devices/` (XC2VE…), AIE1 → `data/devices/` (VC1902…).

So arch-parameterization is a single small **data-driven** descriptor:

| Field | Source |
|---|---|
| cluster `.so` name | per-arch descriptor row (`aie2` / `aie2ps` / `aie`) |
| device-JSON path | discovered from aietools layout, not a hardcoded constant |
| array topology (cols, NoC masters, shim channels) | **queried from the cluster at runtime** — `get_num_cols()`, `get_num_noc_masters()`, etc. Never hardcoded. |

Arch flows in via the C ABI (`aiesim_create(arch, …)`); the Rust side already
knows target arch from the parsed xclbin/device — the same source the interpreter
uses.

**Notes:**

- The device JSON models a **full Versal array**; the NPU is a column-window. The
  mapping (start_col + ncols) comes from the CDO/partition we already parse plus
  the plugin's existing `start_col` plumbing — nothing aiesim-specific to invent.
- **AIE2 is the validated primary** (HW + verified VC2802 load). AIE2P/AIE1 are
  *structurally* supported from day one — Strix bring-up is essentially "add the
  `aie2ps` descriptor row + pin the right `XC2VE*` JSON." Pinning the exact NPU4
  device JSON is a bring-up-time task.

## 9. Testing strategy

Three tiers, matching the three optionality gates.

### Tier 1 — sandbox-safe, in `cargo test --lib` (feature-off-compatible)

With a **mock `NpuBackend`**, unit-test the FFI dispatch, the `XDNA_BACKEND`
selector, the `as_interpreter()` downcast routing, and `AiesimBackend`'s C-ABI
marshalling (command encode/decode against a mock bridge — no cluster). This is
the bulk of the new Rust code and keeps the default suite green with zero aietools
dependency.

### Tier 2 — gated bring-up correctness (aietools, feature on, outside sandbox)

1. **Hello-cluster instantiation** — the E513 fix; cluster constructs + a trivial
   kernel runs. (First implementation task; has its own test.)
2. **In-process vs. the proven ELF-swap path** — same kernel, both aiesim routes,
   **exact-match**. Isolates "wired the in-process backend correctly" from "aiesim
   itself correct."
3. **Grid cells** — {Peano, Chess} × aiesim-in-process produce correct output;
   completes the 2×3 grid *through the real seam*.

### Tier 3 — the oracle / cross-check role (gated)

- **Differential: aiesim-backend vs. interpreter** on a kernel corpus, same FFI
  seam. Compare output *and* tier-2 register/memory state at sync points;
  divergences triaged → the "never fall behind" backlog. **The Phoenix-survival
  corpus is the natural input here**, tying the two threads together.
- **aiesim as a third runtime in `emu-bridge-test.sh`** (HW vs. EMU vs. aiesim) —
  the "third bridge-test runtime" unlock and the standing integration regression
  gate.

### Validation hierarchy

Real NPU HW = ground truth; aiesim = a *second, independent* oracle
(cycle-approximate, AMD's own model); in-process-vs-swap = wiring validation.
aiesim never displaces HW as truth — it is a cross-check and a pace-setter.

## 10. First implementation task and future work

**First implementation task:** the clean, E513-free hello-cluster instantiation
(§6) — construct `MathEngine` inside the service-thread `sc_main`, replicating
`aie_xtlm`'s pre-construction setup. Everything else builds on a known-good
instantiation.

**Next step — phase 2 (committed, after the initial backend works):**

- **HAL-driven independent replay** (§5, Path 1) — a second, independent CDO
  interpretation as a stricter oracle. Sequenced after the parser-driven backend
  is proven, not folded into the first plan. Gating unknown to resolve when we
  take it: whether aie-rt exposes a clean raw-CDO→`XAie_*` ingest path (verify;
  supply if absent).

**Later (wanted, deferred on effort):**

- **Custom device-model generation** — emit our own device JSON (from
  `tools/aie-device-models.json`, the authoritative geometry) so the cluster
  models the real NPU1 geometry exactly (one memtile row, 5×6) instead of
  windowing a full Versal array. We *do* want this — it is how the NPU geometry
  gets properly right — but it is gated on first decoding the binary `XbV18.3`
  device-JSON format, which will take a while.
- **Interpreter feature backlog** — close the tier-3 gaps aiesim exposes
  (guidance JSON, memory-violation diagnostics, FIFO guidance, native watchpoints,
  event-trace, VCD) so the interpreter never falls behind aiesim.

## References

- Feasibility + embedding recipe + mechanism:
  `docs/superpowers/findings/2026-06-01-aiesim-inprocess-backend-feasibility.md`
- Peano × aiesim ELF-swap proof: `docs/aiesimulator.md`
- PS bridge model: `mlir-aie/aie_runtime_lib/AIE2/aiesim/genwrapper_for_ps.cpp`
- Real instantiation call site:
  `<aietools>/data/systemc/simlibs/aie_xtlm/aie_xtlm_v1_0_0/src/aie_xtlm.cpp`
- HAL SIM backend: `aie-rt/driver/src/io_backend/ext/xaie_sim.c`
- Phoenix-survival corpus (differential-test input):
  `docs/superpowers/specs/2026-05-31-phoenix-survival-capture-design.md`
