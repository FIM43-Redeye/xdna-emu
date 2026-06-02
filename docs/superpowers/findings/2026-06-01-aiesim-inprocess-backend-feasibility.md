# In-Process aiesim Backend — Feasibility PROVEN + Resume Notes

**Date:** 2026-06-01
**Status:** Feasibility spike complete (the decisive gates pass). Design-space
characterized. Next: the **architecture phase** (integrator design spec). This
doc is the durable capture so that phase can resume cold after a context compact.

## Vision (Maya)

A **workload-agnostic aiesim integrator** wired as a *backend behind our existing
XRT plugin / `platform_drv_emu` / FFI seam* — a "software driver for aiesim,"
peer to our hand-rolled emulator. Maya wants **all** unlocks at once (third
bridge-test runtime via the driver path, post-Phoenix live functional oracle,
AIE2P/Strix bring-up, debugging cross-check + internals), hence **maximum
versatility / workload-agnostic**, ideally **in-process, library-level, with NO
pkg-dir / launcher / subprocess**.

## FEASIBILITY VERDICT: PROVEN

In-process spike (`build/experiments/2026-06-01-aiesim-inprocess/`, probes 0–3,
gitignored) established the hard gates:
- The **closed AIE2 cluster model** (`libaie2_cluster_msm_v1_0_0.osci.so`) **loads
  in our own process**.
- **SystemC** (2.3.1 Accellera, aietools `libsystemc.so`) **embeds and runs
  in-process** (kernel banner prints from our `sc_main`).
- **`create_math_engine`** (extern "C" factory) is **callable in-process and
  executes** into SystemC module construction.
- **No FlexLM license block** — the single biggest risk, eliminated.

The only thing NOT yet clean is a fully error-free instantiation (E513, below) —
an invocation-context detail, *not* a feasibility wall.

## The in-process embedding recipe (characterized, reproducible)

To load + run the closed cluster inside a normal (modern-toolchain) host binary:

| Requirement | Why |
|---|---|
| **do NOT** mark the bridge `-z execstack` + use a **marker-cleared `libsystemc` copy** | see "Executable-stack constraint" below -- the spike's `-z execstack` was an executable-only artifact; it BREAKS the dlopen path |
| `-rdynamic` | export host globals so the dlopened cluster resolves them |
| define 2 host globals: `extern "C" { bool sc_stop_at_end_of_main=false; int plio_complete=0; }` | host-executable contract (else dlopen fails) |
| compile aietools SystemC main bootstrap (`sc_main_main.cpp` only, NOT `sc_main.cpp`) | provides `sc_elab_and_sim`→our `sc_main`; `sc_main.cpp` only adds `main()`, which a `.so` never runs |
| `sc_main` declared `extern "C"` | the bootstrap calls it C-linkage |
| use **system** libstdc++, NOT aietools' | aietools' `tps/lnx64/gcc/lib64` libstdc++ is too old (missing `GLIBCXX_3.4.3x`); system one is newer + backward-compatible |
| `LD_LIBRARY_PATH=<aietools>/lib/lnx64.o` (for cluster deps) + sanitized-libsystemc dir | cluster deps (boost-1.72/msm_cpp/etc.) live in lnx64.o; the sanitized `libsystemc` must win over the aietools one (bridge carries `DT_RPATH=$ORIGIN`, old dtags, so its co-located copy resolves first) |

### Executable-stack constraint (characterized 2026-06-01, probe4 + probe_cor)

aietools `libsystemc.so` ships marked **executable-stack (RWE)** -- but this is a
**spurious** marker (QuickThreads `.s` files lacking `.note.GNU-stack`), NOT a
real runtime need. Proven: a non-exec-stack (`GNU_STACK: RW`) executable linked
against a **marker-cleared** `libsystemc` runs `sc_thread` coroutines fine
(QuickThreads context-switch is SP/register asm, not stack code execution).

Why it matters for the in-process backend (a `.so` dlopened into a running host,
not an executable): on this hardened kernel the loader **cannot** flip a running
process's stack to executable at `dlopen` time (`EINVAL`), and dlopening from a
worker thread does not help (glibc always targets the *main* stack). Executables
escape this because the kernel sets exec-stack at `exec()` -- which is exactly why
the spike (probe executables) never saw it.

**Resolution (chosen with Maya): sanitize the marker.** The bridge build produces
a marker-cleared copy of `libsystemc.so` next to the bridge `.so` (regenerated
from local aietools at build time -- nothing modified in place, nothing aietools
committed to the repo), and the bridge carries `DT_RPATH=$ORIGIN` (old dtags) so
its co-located sanitized copy resolves ahead of the aietools one even when
`<aietools>/lib/lnx64.o` is on `LD_LIBRARY_PATH`. Result: the bridge loads + runs
in **any** host, no `-z execstack`, no consumer relink. (Alternative, rejected:
link every consumer `-Wl,-z,execstack` -- works but invasive + doesn't generalize.)
Marker patcher: `scripts/clear-execstack.py` (clears `PF_X` on `PT_GNU_STACK`).

**Key paths:**
- cluster: `<aietools>/lib/lnx64.o/libaie2_cluster_msm_v1_0_0.osci.so` (and `_func`, `_dbg` variants)
- arch-param bonus (on disk): `libaie2ps_cluster_msm...` (Strix/AIE2P), `libaie_cluster_msm...` (AIE1) → AIE2P bring-up unlock is a lib swap
- device model: `<aietools>/data/aie_ml/devices/*.json` — **NOT text JSON**; a binary/compressed device-model blob (magic `XbV18.3`). `create_math_engine` decompresses + parses it. **All shipped AIE-ML models are Versal parts (with PL), NOT the NPU** — there is no NPU device model on disk (that is the out-of-scope `XbV18.3` custom-gen work). **Stopgap device = `VE2102.json`** (the smallest: 12 cols, 4 noc tiles, ~170 stream ports vs VC2802's 38 cols / ~720). It still has cols 0-4 + the rows NPU xclbins target, so config/replay/runs work; the geometry/PL mismatch is a **fidelity caveat for the differential-vs-NPU oracle** (II-B.5), not a functional blocker. Set via `XDNA_AIESIM_DEVICE_JSON` (Rust selector has no default). Real NPU geometry awaits custom-device-model gen.
- SystemC include: `<aietools>/data/osci_systemc/include`
- SystemC main bootstrap: `<aietools>/data/osci_systemc/sc_main/{sc_main.cpp,sc_main_main.cpp}`

**Factory signature** (from `aie_xtlm.cpp` usage):
```cpp
extern "C" void *create_math_engine(const char *name, const char *device_json,
                                    bool is_fast_pm, bool is_fast_dm);
extern "C" void  destroy_math_engine(void *me);
```

## The mechanism — drive the CLOSED ISS through the OPEN HAL

The integration does **not** need the launcher or a generated `ps.so`. Three
open pieces compose:

1. **`ess_*()` weak-symbol seam.** The open AIE-RT HAL (`libxaienginecdo`,
   compiled `-D__AIESIM__`) routes *every* register/memory op through weak C
   funcs: `ess_Write32`, `ess_Read32`, `ess_WriteGM`, `ess_ReadGM`, `ess_WriteCmd`.
   The host provides them; in a real run `PSIP_ps_i3` (in `ps.so`) forwards them
   to the cluster over AXI-MM TLM. **We provide our own bridge instead.**
2. **HAL SIM backend** (`aie-rt/driver/src/io_backend/ext/xaie_sim.c`):
   `XAie_CfgInitialize(SIM)` → `XAie_Write32`/`XAie_BlockWrite32` (CDO replay) /
   `XAie_MemAllocate`+`MemSyncForDev` / `XAie_CoreEnable` / `XAie_MaskPoll` /
   `MemSyncForCPU` → all land on `ess_*()` → cluster.
3. **`create_math_engine`** factory → the closed SystemC cluster, with a
   shipped per-arch device model.

So the **`AiesimBackend`**: construct a SystemC top (cluster via
`create_math_engine` + our own `PSIP_ps_i3`-style PS bridge providing `ess_*()`,
bind AXI-MM via `me->get_ss_aximm_rd/wr()`), drive via the HAL. Maps 1:1 onto our
FFI ops: `load_pdi`→CDO replay; `write_host_memory`→`ess_WriteGM`/`MemSyncForDev`;
`run`→drive the sim to completion; `read_host_memory`→`ess_ReadGM`/`MemSyncForCPU`.

## The FFI seam (where the backend slots in)

- FFI = `crates/xdna-emu-ffi/src/`: `xdna_emu_create` / `load_xclbin` / `load_pdi`
  / `alloc_host_region` / `write_host_memory` / `read_host_memory` /
  `execute_npu_instructions` / `sync_cores` / `run` / query+diagnostics. **Hardwired
  to `InterpreterEngine` — no backend abstraction yet.**
- To add aiesim: introduce a `Backend` trait (or enum) in `XdnaEmuHandle`
  (`Interpreter | Aiesim`), dispatch each `xdna_emu_*` fn over it, add a selector
  (`XDNA_BACKEND=aiesim`) in `xrt-plugin/src/transport_inprocess.{h,cpp}`.
- Plugin call order for a run: `load_xclbin`/`load_pdi` → `alloc`+`write` BOs →
  `execute_npu_instructions` → `sync_cores` → `run` → read BOs. XRT's
  sync→run→sync ordering means the aiesim backend can run **batch per `run()`** —
  no live co-sim transactor needed.
- Reusable: `src/parser/` (xclbin/CDO/ELF parse — we already extract these),
  `src/aiesim/harness.rs` + `src/integration/aiesimulator.rs` (existing subprocess
  integration + `--input-dir/--output-dir` file-I/O mode — a fallback path).

## Design space (the shapes)

Two layers. (1) **The seam:** behind the FFI plugin (Maya's choice — driver-path
parity + arch-agnostic) vs a parallel harness. (2) **The host-driver strategy:**
- **A. aiecc-reuse** — pkg-dir from the design's MLIR via `aiecc --aiesim`, swap
  the core ELF (this is the *separately proven* Peano-x-aiesim swap; see
  `docs/aiesimulator.md`). Low risk, **not** workload-agnostic (needs MLIR + Chess).
- **B. in-process xclbin-native library backend** — the target. Workload-agnostic.
  **Now proven feasible** by this spike.
- **C. aiesim file-I/O (GMIO) mode** — middle ground, uncertain for NPU
  `runtime_sequence` designs.
Target = **B**, with **A** as the proven fallback for MLIR-derived workloads.

## RESOLVED — clean E513-free instantiation (2026-06-01, probe4)

**The crack: push one `sc_module_name` onto SystemC's name stack immediately
before `create_math_engine`, from inside a parent `sc_module`'s construction.**

Mechanism, now fully understood:
- bare `create_math_engine` from `sc_main` (no module context) → **E533**
  (`SC_ID_MODULE_NAME_STACK_EMPTY_`): the factory internally constructs a child
  `sc_module` via the **default (no-name) ctor**, which pops a name off the stack;
  with no module context at all, the stack is empty → E533.
- wrapped in a parent `sc_module` ctor (probe3) → **E513** (`sc_module.cpp:227`,
  "an sc_module_name parameter for your constructor is required"): now there *is*
  a construction context, but probe3 still pushed no name for the factory's
  internal default-ctor module → E513.
- **Fix (probe4): declare `sc_core::sc_module_name nm("math_engine");` in the
  parent ctor right before the call.** That single push is consumed by the
  factory's internal module. Result: `create_math_engine` returns a valid
  `MathEngine*` (`0x...`), the cluster fully constructs ("Array constructed /
  Shim row constructed / Mem row constructed"), `STEP1 PASS`, exit 0. AIE2 ISS
  r1p8, MTMODEL sim, all tiles enumerated. **No license check tripped; no
  launcher; no pkg-dir.** `XILINX_VITIS_AIETOOLS`/`RDI_DATADIR` were never needed.

Repro: `build/experiments/2026-06-01-aiesim-inprocess/probe4.cpp` +
`build-probe.sh` (gitignored). This recipe is what Task II.3's `aiesim_top` ctor
codifies: a parent `sc_module`, one pushed `sc_module_name`, then the factory.

## Reference files (the Rosetta stones)

- `mlir-aie/aie_runtime_lib/AIE2/aiesim/genwrapper_for_ps.cpp` — `PSIP_ps_i3` PS
  bridge: `ess_*()` → AXI-MM TLM → cluster. The model for our bridge.
- `<aietools>/data/systemc/simlibs/aie_xtlm/aie_xtlm_v1_0_0/src/aie_xtlm.cpp` —
  `create_cluster()` / the real `create_math_engine` call site (lines ~266–357).
- `aie-rt/driver/src/io_backend/ext/xaie_sim.c` + `io_backend/xaie_io.h` — HAL SIM
  backend impl + the `XAie_BackendOps` vtable.
- `<aietools>/data/osci_systemc/sc_main/{sc_main.cpp,sc_main_main.cpp}` — SystemC
  main bootstrap.
- Probes: `build/experiments/2026-06-01-aiesim-inprocess/probe{0,1,2,3}.{c,cpp}`
  (gitignored; recipe above is sufficient to reproduce).

## Resume checklist (post-compact)

This is mid-`brainstorming` (aiesim integrator design): feasibility PROVEN,
design-space characterized. Per Maya, next = **architecture phase**:
1. Present the integrator design in sections (the `Backend` trait + `AiesimBackend`
   in-process library shape; the seam wiring; the `ess_*()` bridge; arch-param for
   AIE2P), get approval.
2. Write the design spec → self-review → user review.
3. `writing-plans`. The clean-instantiation (E513) is the **first implementation
   task** in that plan.

**Also pending (separate thread):** the Phoenix-survival output-corpus
**implementation plan** (`docs/superpowers/plans/2026-06-01-phoenix-survival-corpus.md`)
is written + approved, **awaiting execution** (deferred when we pivoted to aiesim).
Its capture campaign (Task 13) is HW-gated (Phoenix).

## CRITICAL (2026-06-02): backdoor register access is a SHADOW; live registers need timed b_transport from an SC_THREAD

A de-risk probe before building DdrPatch/Sync revealed the cdo_replay II-B.2a
approach was driving nothing. Decisive evidence, three steps:

1. **probe_core** — backdoor-write `Core_Control`=Enable (offset 0x32000) on a
   compute tile, `sc_start`, read `Core_Status`/`Core_PC`/perf-counter: all FLAT.
   PC read back as `0x68635f6b` (ASCII garbage, not a 20-bit PC).
2. **probe_diag** — backdoor write+read a sentinel at 10 offsets (DM, PM, PC,
   Core_Control/Status, perf, shim BD): ALL round-trip. So the config-aximm
   `transport_dbg` (backdoor) backs a FLAT SHADOW STORE at every address — it is
   NOT the live register file (several offsets held pre-existing content distinct
   from writes, e.g. PC 0x68635f6b, BD reg 0x38).
3. **timed_probe** — the SAME write via TIMED `ps->write32` (b_transport) from
   `sc_main`: `Error: (E519) wait() is only allowed in SC_THREADs and SC_CTHREADs`
   on the first call. The initiator util's `b_transport` wait()s on the AXI
   handshake.

Scout (aie-rt `xaie_sim.c`, genwrapper, aie_xtlm) confirms: the HAL ALWAYS uses
timed `b_transport` for register access; `transport_dbg` is debug-memory-only.
Address convention is VERBATIM `(col<<25)|(row<<20)|offset` (no translation) —
addresses were right, the access METHOD was wrong.

**Conclusion:** backdoor cannot drive or observe the live cluster. Register R/W
MUST be timed `b_transport`, which requires an SC_THREAD (not `sc_main`).

**Architecture (Maya-approved 2026-06-02): unified SC_THREAD driver + sc_pause.**
SystemC 2.3.1 (aietools) has both `async_request_update()` (sc_prim_channel.h:67,
thread-safe OS→kernel handoff) and `sc_pause()` (sc_simcontext.h:131/172). The
design:
- One SC_THREAD owns ALL register access (timed b_transport) + time advance
  (`wait(N)` for RUN, `wait(1 cyc)` for STEP). `sc_main` is just
  `for(;;){ sc_start(); if(shutdown)break; wait_for_pending(); }`.
- Kernel is PAUSED between commands (driver `sc_pause()`s after draining; sc_main
  only `sc_start`s when a command is pending) → no time drift → **cycle-precise
  discrete stepping is first-class** (the debugger capability), and free-run is
  just `RUN(large N)`. Faithful: config takes effect immediately as timed writes.
- OS thread (C-ABI) rings a `Doorbell` (sc_prim_channel) via
  `async_request_update()`; its `update()` notifies the driver's cmd_event.
- Reused: command queue + reply handshake, ddr_target, cdo_replay DECODER logic,
  wire format, C-ABI. Changed: service loop moves into the SC_THREAD;
  cdo_replay/READ_REG swap backdoor→timed and `sc_start`→`wait` (now in a process
  context). II-B.2a's "gate" only round-tripped the shadow; the restructure is
  what makes it drive live hardware.
