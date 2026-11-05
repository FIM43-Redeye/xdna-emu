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
| Core register state across launches (dive node `core-reg-persistence`) | Silicon keeps a core's data registers (r, p, m/dn/dj/dc, s, ls, lc, dp, wl/wh, the accumulator quarters, q) from one launch to the next: separate processes, separate xclbins, launches on other tiles in between and idle gaps up to 300 s change nothing; every launch resets the control registers (0, `crSCDEn` = `crMCDEn` = 1), the status registers (0) and `le` (0xFFFFF). EMU builds each launch's cores fresh. | Launch reset values: `ExecutionContext::new_for_tile`. Harness workaround: the scrub kernel (`src/interpreter/state/scrub.rs`, `tools/harness/scrub.py`) puts one known value in every data register before each HW phase and the EMU starts from the same values (`XDNA_EMU_CORE_INIT`), so a case reading a register it never wrote agrees on both. Evidence: `build/experiments/harness/scrub/step1` (17 launches), seam check `scrub/step1/suite` tags s1/s2 vs se2 (byte-identical register dump). | **LAUNCH RESET VALUES MODELLED (2026-11-04); PERSISTENCE OPEN.** The EMU does not carry state between launches (each launch is its own process), and what the management firmware preserves over a context switch is unprobed; `r16` (restored by the aiecc main from its own frame) and data memory are not scrubbed. |
| Firmware clock domain (dive node `fw-in-loop`, 2026-11-05) | Before: the management firmware ran up to an instruction budget per array cycle, i.e. infinitely fast against the array. Now: [`src/timebase.rs`](../../src/timebase.rs) couples the two clocks by an exact rational accumulator (`Coupler`, no floats); the firmware spends whole cycles per array cycle at the price of each step (the cost model lives in `xdna-emu-priv`, `firmware-phoenix/src/timing.rs`). The pair is one named configuration (`ClockPair`: driver DPM table entries and the SMU-reported values; `XDNA_EMU_DPM=0|1`), so DPM 0 is a setting, not a code change. Simulated wall time is `array cycles / array MHz` (`ClockPair::array_cycles_to_ps`, `xdna_emu_array_cycles_to_wall_ps`, the `run:` log line). | `src/timebase.rs`; `xdna-emu-priv: firmware-phoenix/src/{timing,runtime}.rs`; evidence `build/experiments/fw-clock/fw-timing/` and `xdna-emu-priv: research/findings/2026-11-05-phoenix-firmware-op-work-and-cpi-hypothesis.md` | **MODELLED against the runtime-sequence op probes (within a few percent on every probe case, both DPM points).** Open: a once-per-run stall of about 2-3 us after the first op arrival on silicon that the emulator does not reproduce, and costs no probe measures yet (firmware accesses outside the array window); the itemised list is in `xdna-emu-priv: research/firmware/host-firmware-dispatch.md`. |

**Core start order and spacing (2026-11-04, cascade work).** Silicon starts the cores of a design
one at a time in column-major order, about 1448 cycles apart with run-to-run jitter (1103-2018);
see "Core start order" in [`core-compute-timing.md`](core-compute-timing.md). The emulator now
staggers the starts at the same seam (`DeviceState::stagger_core_starts`, then
`release_core_resets` for cores still held). Note for Part 2: the current mlir-aie CDO (harness
builds, 2026-11) clears RESET itself in its init section (`CORE_CONTROL` mask write 0x2 -> 0) and
sets ENABLE in its enable section, so those cores start at their enable write rather than at a
release of a held reset; the "nothing in the CDO clears the reset bit" observation above predates
that and should be rechecked when the launch point is pinned.
