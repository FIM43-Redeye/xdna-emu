---
class: host-firmware-dispatch
subsystem: off-array host + NPU management-firmware launch / dispatch path
posture: off-array -- the wall-clock gap between core-start and the first runtime-sequence DMA dispatch, which the emulator deliberately does not model
status: dispatch-latency documented+deferred; core-reset mechanism root-caused, Part 1 landed (`aa825ae9`); management-processor platform gaps are tracked in the private ledger named below
---

# Host / Firmware Dispatch Gaps

The wall-clock gap between core-start and the first runtime-sequence-driven DMA
dispatch. Most of this is genuinely **off-array**: the (jittery) time the host +
NPU launch path spend launching and configuring before the array does anything,
which the emulator deliberately does not model (modeling it would inject host
jitter without improving array cycle-accuracy). But the #140 SP-4a investigation
found that *part* of the EMU-side window was an on-array bug: the emulator ran
compute cores at CDO/config time because it ignored the `CORE_CONTROL` reset bit.

**This class is the seam between the array model and the host launch path** --
the reset deassert, the ~8000cy host mailbox round-trip latency, the dispatch
gates, and the core warm-up are all observed at the host/array boundary and are
not modeled inside the array. See finding
`xdna-emu-priv: research/findings/2026-07-03-sp4a-core-reset-mechanism-part1.md`.

The management-processor platform that sits between the host driver and the
array is not part of this repository; its gap ledger lives in
`xdna-emu-priv: research/firmware/host-firmware-dispatch.md`. The rows below
are the gaps that belong to the public emulator.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Dispatch latency (core-start -> first drain dispatch) | HW **~25-32k cy** (mean ~30.5k, +/-16%, occasional ~40k) + **~112cy per runtime-sequence instruction** (controller processing rate) + scales with CDO size; EMU **~673cy**. The ~30k base is host->NPU launch + management-processor startup (jittery -> off-array). | No emulator model -- deliberately un-modeled off-array latency; would sit in the controller/dispatch path (`src/npu/`) if ever added. Evidence: `build/experiments/pathA-cntr-spike/w8-drainpace/` (6 in-core cntr runs, dispatch swung 24.7k-40.7k cy). | **DOCUMENTED, deferred (2026-07-03, #140 SP-4a-discovered).** **Orthogonal to the SP-4a oracle**: the array-side fill is exactly deterministic (consA fills to **exactly 5** every run despite dispatch swinging 24.7k-40.7k cy), so the drain-pacing work is unaffected. Not yet decomposed into host-launch vs management-processor-config vs per-instruction-controller cost -- defer to a dedicated marker-probe experiment. Finding: `xdna-emu-priv: research/findings/2026-07-03-sp4a-fill-reconciliation-drain-pacing.md`. |
| Core reset-deassert at launch (`CORE_CONTROL` reset bit) | EMU derived core run-state from the ENABLE bit alone and ran cores at **CDO time (cyc 0)**; silicon holds each core `CORE_CONTROL=0x03` (ENABLE=1, RESET=1 = enabled-but-held) until **reset is deasserted at kernel launch** (observed from the host side). Nothing in the CDO or `insts.bin` clears the reset bit -- it is a side effect of the launch path outside the array. So EMU cores ran ~673cy ahead of the shim dispatch, filling the objectfifo pipeline HW keeps cold. Proven by a **1-second-invariant post-load-sleep HW sweep** (pipe cold at dispatch regardless of ~1e9-cycle pre-launch). A later audit found that the device model retained RESET but published only derived runnable-state changes to the interpreter, collapsing a repeated RESET/unreset into an ordinary pause/resume with stale PC, registers, and executor pipeline. | `src/device/state/compute.rs` (run-state now `enabled = ENABLE & !RESET` and ordered reset/enable publication); `src/interpreter/engine/coordinator.rs` and `src/interpreter/core/interpreter.rs` (architectural and executor reset); `DeviceState::release_core_resets()` (the launch seam); finding: 2026-07-03 core-reset mechanism (`xdna-emu-priv: research/findings/2026-07-03-sp4a-core-reset-mechanism-part1.md`); context-audit correction (`xdna-emu-priv: research/findings/2026-08-01-phoenix-context-repartition-proof-correction.md`) | **RESET STATE AND ENGINE EDGE FIXED; LAUNCH TIMING OPEN.** Part 1 (`aa825ae9`) made the reset bit real and named the launch seam. The 2026-08-02 correction preserves the ordered reset edge and reinitializes core execution state before re-enable. `release_core_resets()` is still called at config-completion, so **Part 2 = launch TIMING**: relocate it to the true launch point after a runtime-sequence padding HW probe pins the relationship. This root cause **supersedes** the prior drain-throttle-as-primary framing for the SP-4a offset (see the send-cadence residual in [`dma-stream-resources.md`](dma-stream-resources.md)). |
| Core register state across launches (dive node `core-reg-persistence`) | Silicon keeps a core's data registers (r, p, m/dn/dj/dc, s, ls, lc, dp, wl/wh, the accumulator quarters, q), both load-port rings, the sparse-load unit and the stream read latch from one launch to the next: in one process (several hw contexts, xclbins, launches on other tiles in between) and across processes and idle gaps up to 300 s; every launch resets the control registers (0, `crSCDEn` = `crMCDEn` = 1), the status registers (0) and `le` (0xFFFFF). EMU now does the same within one emulator instance; across processes only with `XDNA_EMU_STATE_DIR`. | `src/interpreter/state/carried.rs` (`CarriedCoreState`: what a core reset keeps; `ExecutionContext::after_reset`), `CoreState::reset` in `src/interpreter/engine/coordinator.rs` (every core reset: `Core_Control`, new hw context); power-on = `ExecutionContext::new_for_tile` (+ `XDNA_EMU_CORE_INIT`, the scrub pattern). Cross-process: `src/interpreter/engine/device_state.rs`, `XDNA_EMU_STATE_DIR=<dir>`: the FFI loads `<dir>/aie2-npu1-phoenix-hawkpoint-5x6.json` on `xdna_emu_create` and saves it on `xdna_emu_destroy`; a file of another format version or device is refused (fresh device, logged); delete it to reset the device. Evidence: `build/experiments/harness/scrub/step1` (17 launches, separate processes); `harness/core-persist/` (one process through XRT, `tools/harness/multi_launch.py`: HW r1 == r2, EMU == HW except r17-r23, which nothing in the process writes; two EMU processes with the state file == silicon's second process; `sparse/`, `latch/` probes). | **MODELLED (2026-11-06).** The harness keeps its scrub: each phase is its own process, i.e. a fresh emulated device. `lr`, `sp` and `r16` are written by crt0 / the aiecc main before any kernel code (r16 = 1 at kernel entry on both, even on a fresh device), so whether they carry is unobservable to such kernels; the model carries them as data registers. Open: the sparse unit's fill level over a launch and its reads in the underflow regime (next paragraph); a context switch between two live contexts on one column is unprobed. |
| Firmware clock domain (dive node `fw-in-loop`, 2026-11-05) | Before: the management firmware ran up to an instruction budget per array cycle, i.e. infinitely fast against the array. Now: [`src/timebase.rs`](../../src/timebase.rs) couples the two clocks by an exact rational accumulator (`Coupler`, no floats); the firmware spends whole cycles per array cycle at the price of each step (the cost model lives in `xdna-emu-priv`, `firmware-phoenix/src/timing.rs`). The pair is one named configuration (`ClockPair`: driver DPM table entries and the SMU-reported values; `XDNA_EMU_DPM=0|1`), so DPM 0 is a setting, not a code change. Simulated wall time is `array cycles / array MHz` (`ClockPair::array_cycles_to_ps`, `xdna_emu_array_cycles_to_wall_ps`, the `run:` log line). | `src/timebase.rs`; `xdna-emu-priv: firmware-phoenix/src/{timing,runtime}.rs`; evidence `build/experiments/fw-clock/fw-timing/` and `xdna-emu-priv: research/findings/2026-11-05-phoenix-firmware-op-work-and-cpi-hypothesis.md` | **MODELLED against the runtime-sequence op probes (within a few percent on every probe case, both DPM points).** Open: the once-per-run stall on silicon (about 2-3 us after the first op arrival) now lands where silicon puts it but is about half its silicon size, and costs no probe measures yet (firmware accesses outside the array window); the itemised list is in `xdna-emu-priv: research/firmware/host-firmware-dispatch.md`. Interrupt inputs a platform drives as levels follow their line every step (`XtensaBus::interrupt_levels`, `src/firmware/xtensa/bus.rs`). |

**Sparse-load unit over a launch (2026-11-06, `harness/core-persist/sparse`).** The first sparse read
of a launch returns the previous launch's line (v1, v2: EMU == HW once the unit is carried), but
silicon then reports `srCompr_uf` = 1 and refetches as if its fill level were low, and its reads in
that regime do not match the emulator's ring model even within one launch: right after `reset`, a
`peek` returns a stale line from the previous launch and the next one the reset's line (`sp3_c`);
after a few pops silicon reads lines a line behind what the emulator predicts. HW r1 == r2 on every
probe. Belongs with the sparse open row of `addressing.md` ("bytes a sparse peek reads past the
first fetched line"); not modelled.

**A second run in the same hardware context (2026-11-06, `harness/core-persist/runs/hw-same-r*`).**
A run-once core (the aiecc main ends in `done`) is not restarted by a second command on the same
hw context: silicon never completes it (XRT state 8, one TDR each, r1 == r2). The emulator
completes it, because the plugin's interpreter path resets the partition and replays the PDI on
every submit (`xrt-plugin/src/platform_emu.cpp`, `submit_cmd`). Open; not a persistence issue
(registers still carry over that replay).

**Core start order and spacing (2026-11-04, cascade work).** Silicon starts the cores of a design
one at a time in column-major order, about 1448 cycles apart with run-to-run jitter (1103-2018);
see "Core start order" in [`core-compute-timing.md`](core-compute-timing.md). The emulator now
staggers the starts at the same seam (`DeviceState::stagger_core_starts`, then
`release_core_resets` for cores still held). Note for Part 2: the current mlir-aie CDO (harness
builds, 2026-11) clears RESET itself in its init section (`CORE_CONTROL` mask write 0x2 -> 0) and
sets ENABLE in its enable section, so those cores start at their enable write rather than at a
release of a held reset; the "nothing in the CDO clears the reset bit" observation above predates
that and should be rechecked when the launch point is pinned.

**Management-core access ring (2026-11-07).** The firmware interpreter's MMU used to translate every
access at ring 0. It now follows the Xtensa ISA: the access ring is CRING (`PS.EXCM ? 0 : PS.RING`),
L32E/S32E use PS.RING, instruction fetch and the fill fast path included, and the privileged
instructions (the RM's set; RSR/WSR of special registers 64 and up) raise PrivilegedCause at
CRING != 0 (`src/firmware/xtensa/interp/ring_tests.rs`). A runtime-sequence read of a page the task's
ring may not reach now faults in the emulator as it does on silicon ("Kernel did not complete");
every other run is unchanged. MODELLED.

**Management-firmware state across processes (open).** Silicon's management firmware runs from
driver load; the emulator boots it once per emulator instance. Lazily built firmware state (for
example software-filled TLB entries) that a long-running device already holds is therefore cold in
the first command of each emulator process, and that command pays a one-time cost silicon pays once
per driver load. Within one emulator instance, later commands match. Not modelled across processes.

**Management-core instruction cache (2026-11-07).** The private per-step cost model gains an
instruction cache (hit/miss per fetch line); its geometry and refill price live with the private
cost model. Open: cold-code cost after the largest firmware paths still exceeds what the cache
explains (private ledger).
