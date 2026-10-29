---
class: permissive-vs-hw
subsystem: (cross-cutting) dataflow the emulator completes that real silicon refuses to run
posture: permissive-where-broken -- the INVERSE gap; the model is more capable than the hardware, running a structurally-valid kernel that HW wedges on
status: 1 documented (mechanism uncharacterized; HW/driver/lowering limitation)
---

# Permissive-vs-HW Gaps (the inverse gap)

The inverse of the usual fidelity gap. Here the model is *permissive where HW is
broken/limited*: a structurally-valid kernel runs to completion in the emulator
but TDRs (`ert` `state=8` timeout) on real Phoenix NPU1. These are not emulator
bugs in the usual sense -- the interpreter faithfully runs valid dataflow -- but
whether to *teach* the emulator a HW defect is a real design question, so they
are tracked here.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Task-API memtile relay TDRs on Phoenix NPU1 | A `dma_configure_task` DMA chain that **relays through a memtile** (mem-tile S2MM-in + MM2S-out) completes cleanly in the emulator but **wedges the NPU** (TDR, `state=8`). The wedge is *not* token-strategy-specific and *not* our kernel: **AMD's own** `test/npu-xrt/memtile_dmas/dma_configure_task_token` **and** `.../dma_configure_task_lock` both TDR through their **official `test.cpp`** (so it is not a runner artifact -- `bridge-trace-runner` and `test.exe` agree exactly), while the sibling `core_dmas/dma_configure_task_token` (core relay, same API) **passes**. Isolated in SP-3 bring-up (2026-06-29): minimal producer->**shim** drain + token await passes (C1); inserting a single memtile relay hop (C2) TDRs -- the memtile is the lone isolating variable. | N/A in emulator source -- the interpreter faithfully runs the valid dataflow; this is a HW/driver/`mlir-aie`-lowering limitation the model does not (and arguably should not) reproduce. Bring-up artifacts: `mlir-aie/test/npu-xrt/spike_bringup/{c1_producer_drain,c2_memtile_relay}.mlir`. | **DOCUMENTED (mechanism uncharacterized).** Root cause unknown -- task-API memtile DMA config differs from the (working) objectfifo memtile path; the lowering or driver mis-programs memtile DMA on npu1. SP-3 **pivoted to a memtile-free core-relay topology** (2026-06-29) to avoid it; do not re-chase the memtile relay on our Phoenix. Candidate upstream `mlir-aie` report. Whether to teach the emulator this TDR is an open design question (faithfully running valid dataflow is the default; modeling a HW defect is a deliberate choice). |
