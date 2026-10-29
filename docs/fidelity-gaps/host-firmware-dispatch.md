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
