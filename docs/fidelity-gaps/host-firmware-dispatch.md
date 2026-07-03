---
class: host-firmware-dispatch
subsystem: off-array host + NPU management-firmware launch / dispatch path
posture: off-array -- the wall-clock gap between core-start and the first runtime-sequence DMA dispatch; deliberately un-modeled, but one piece (core reset-deassert) turned out to be an on-array firmware side effect the model got wrong
status: dispatch-latency documented+deferred; core-reset mechanism root-caused, Part 1 landed (`1e9e6700`)
---

# Host / Firmware Dispatch Gaps

The wall-clock gap between core-start and the first runtime-sequence-driven DMA
dispatch. Most of this is genuinely **off-array**: the (jittery) time the host +
NPU firmware spend launching and configuring before the array does anything,
which the emulator deliberately does not model (modeling it would inject host
jitter without improving array cycle-accuracy). But the #140 SP-4a investigation
found that *part* of the EMU-side window was an on-array bug: the emulator ran
compute cores at CDO/config time because it ignored the `CORE_CONTROL` reset bit.

**This class is the seam to the firmware-emulation endgame** -- the reset
deassert, the ~8000cy mailbox latency, the dispatch gates, and the core warm-up
are all hardcoded firmware behaviors that the "true simulator" route would
replace by loading real management-processor firmware. See the firmware-dream
memory note and finding
[`2026-07-03-sp4a-core-reset-mechanism-part1.md`](../superpowers/findings/2026-07-03-sp4a-core-reset-mechanism-part1.md).

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Dispatch latency (core-start -> first drain dispatch) | HW **~25-32k cy** (mean ~30.5k, +/-16%, occasional ~40k) + **~112cy per runtime-sequence instruction** (controller processing rate) + scales with CDO size; EMU **~673cy**. The ~30k base is host->NPU launch + firmware startup (jittery -> off-array). | No emulator model -- deliberately un-modeled off-array latency; would sit in the controller/dispatch path (`src/npu/`) if ever added. Evidence: `build/experiments/pathA-cntr-spike/w8-drainpace/` (6 in-core cntr runs, dispatch swung 24.7k-40.7k cy). | **DOCUMENTED, deferred (2026-07-03, #140 SP-4a-discovered).** **Orthogonal to the SP-4a oracle**: the array-side fill is exactly deterministic (consA fills to **exactly 5** every run despite dispatch swinging 24.7k-40.7k cy), so the drain-pacing work is unaffected. Not yet decomposed into host-launch vs firmware-config vs per-instruction-controller cost -- defer to a dedicated marker-probe experiment. Finding: [`2026-07-03-sp4a-fill-reconciliation-drain-pacing.md`](../superpowers/findings/2026-07-03-sp4a-fill-reconciliation-drain-pacing.md). |
| Core reset-deassert at launch (`CORE_CONTROL` reset bit) | EMU derived core run-state from the ENABLE bit alone and ran cores at **CDO time (cyc 0)**; silicon holds each core `CORE_CONTROL=0x03` (ENABLE=1, RESET=1 = enabled-but-held) until **firmware deasserts reset at kernel launch**. Nothing in the CDO or `insts.bin` clears the reset bit -- it is a pure firmware side effect. So EMU cores ran ~673cy ahead of the shim dispatch, filling the objectfifo pipeline HW keeps cold. Proven by a **1-second-invariant post-load-sleep HW sweep** (pipe cold at dispatch regardless of ~1e9-cycle pre-launch). | `src/device/state/compute.rs` (run-state now `enabled = ENABLE & !RESET`); `DeviceState::release_core_resets()` (the firmware-launch seam); [finding: 2026-07-03 core-reset mechanism](../superpowers/findings/2026-07-03-sp4a-core-reset-mechanism-part1.md) | **ROOT-CAUSED, Part 1 landed (`1e9e6700`, byte-identical, 3585 lib tests).** Part 1 makes the reset bit real and names the firmware seam (called at config-completion for now -> behavior unchanged). **Part 2 = firmware-launch TIMING**: relocate `release_core_resets()` off config-completion to the true launch point; couples to an unmeasured HW fact (config-register MMIO cost in AIE-cycles) -- pin with a runtime-sequence padding HW probe. This root cause **supersedes** the prior drain-throttle-as-primary framing for the SP-4a offset (see the send-cadence residual in [`dma-stream-resources.md`](dma-stream-resources.md)). Endgame: real firmware on a simulated management processor. |
