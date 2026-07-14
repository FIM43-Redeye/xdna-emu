---
class: dma-stream-resources
subsystem: DMA engine + stream-switch finite-resource / consumer-pacing modeling
posture: optimistic-where-strict -- the model assumes generous or infinite resources (deep queues, unbounded token buffers, freely reusable BDs) where silicon is finite and will stall or wedge
status: send/recv port cadence substantially resolved (recv exact); 2 residual open, both HW-empirical-gated; 1 bounded peek/commit under-claim (decompression, low-impact, not HW-gated)
---

# DMA & Stream Resource Gaps

Shared root posture: our DMA model is *optimistic where silicon is strict* -- it
assumes generous/infinite resources where the hardware is finite and will stall
or wedge. The finite-resource members are **low real-workload impact** (real
compilers don't hit them; they bite synthetic / pathological kernels, e.g. our
own calibration sweeps). The one toolchain-derivable member (task-queue depth)
is fixed; the residual finite-resource gaps need **HW-empirical data** the
open-source toolchain does not expose, so they stay documented-and-deferred
rather than guessed.

The send/recv **port cadence** work is a different sub-thread (consumer-pacing,
not resource exhaustion) that turned out to be largely a stream-drain-rate and
ingress-depth modeling problem; it is now substantially resolved -- recv is
exact, send's gross defect is gone, and the residual cold-start transient was
root-caused elsewhere (the `CORE_CONTROL` reset mechanism -- see
[`host-firmware-dispatch.md`](host-firmware-dispatch.md)).

## Finite-resource gaps (optimistic-where-strict)

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| TCT token buffer | we model **unbounded**; HW has finite token backpressure that stalls the channel (`Stalled_TCT`) | `src/device/dma/token.rs` (`TokenState`; the deliberate simplification is called out in the doc comment) | **OPEN.** Documented in code. aie-rt exposes only a 1-bit `STALLED_TCT` flag, **no numeric depth** -- faithful finite modeling needs an empirically-measured HW depth (HW-gated; a guessed depth would violate derive-from-toolchain). |
| BD reuse-while-live / BD-pool over-allocation | we re-parse a BD on reuse and complete; HW wedges when a kernel needs > 16 distinct shim BDs (in-flight BD reuse) | [finding: 2026-06-06 shim BD-pool over-allocation](../superpowers/findings/2026-06-06-shim-bd-pool-overallocation-nonmonotonic-wedge.md) | **OPEN (won't-fix).** The HW wedge is *non-monotonic* in K (k8 pass, k9 wedge, k12 pass, k16 wedge) -- no clean model. Generator (`gen-shim-chain-sweep.py`) capped at K=8 instead. |

## Inter-tile port cadence (consumer-pacing)

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Inter-tile send-port cadence (consumer-pacing mid-stream backpressure) | `add_one_using_dma` memtile **send** port `PORT_RUNNING_4`: EMU `[16,16,15,8,8,1]` vs HW `[8,8,14,2,14,2,6,8,1]`. HW interleaves the send with the compute consumer's mid-stream processing stalls; the residual is the **cold-start leading 16s** (at pipeline prime EMU fills both ping-pong buffers before backpressure; HW couples to one). The **recv** port `PORT_RUNNING_0` is `[16,16,16,16]` / `STALLED [1,1,1]` == HW exactly -- **recv is closed.** | Stream/DMA consumer-drain timing model; spec [`2026-06-27-stream-switch-axi-backpressure-design.md`](../superpowers/specs/2026-06-27-stream-switch-axi-backpressure-design.md) | **SUBSTANTIALLY RESOLVED, cold-start residual superseded.** Four toolchain/HW-grounded fixes closed the gross defect (`f4009413` pipeline advance-once/cycle; `788e3d70` S2MM ingress drains at memory-bus rate -> compute recv `PORT_RUNNING_0/1` decode `[8x8]` == HW; `b5ec0404` bound ingress to one BD ahead; `eb683bc4` crossing depth `slave_fifo + hop_latency = 4+2 = 6`). Result: the never-throttles defect gone, steady-state 8-bursts match HW, both recv ports exact, data clean, full chess bridge sweep **148 pass / 0 fail, 124 diverge == baseline** (`c7c22286`). **The remaining cold-start transient was ROOT-CAUSED 2026-07-03 to the `CORE_CONTROL` reset bit, NOT ingress/crossing depth** -- see [`host-firmware-dispatch.md`](host-firmware-dispatch.md) and finding [`2026-07-03-sp4a-core-reset-mechanism-part1.md`](../superpowers/findings/2026-07-03-sp4a-core-reset-mechanism-part1.md). Detailed investigation trail: findings [`2026-06-28-send-cadence-pipeline-latency-and-ingress-depth.md`](../superpowers/findings/2026-06-28-send-cadence-pipeline-latency-and-ingress-depth.md), [`2026-06-29-coldstart-headstart-trace-baseline.md`](../superpowers/findings/2026-06-29-coldstart-headstart-trace-baseline.md). |

**Recv-path closure (2026-06-27, historical context).** The recv side reached
exact `[16,16,16,16]` via two model fixes: a per-BD accept cursor and correcting
the memtile S2MM ingress FIFO depth 2 -> 16 (DMA `s2mmChannel.buffer_depth` 12 +
master-port FIFO 4, from the decrypted aietools AIE-ML device model). Regression
oracle: `sum(PORT_RUNNING) == words` per port. Finding
[`2026-06-27-relay-fill-bd-switch-accept-coupling.md`](../superpowers/findings/2026-06-27-relay-fill-bd-switch-accept-coupling.md).
The trace-*encoding* half of the port-cadence saga (span merges, count
under-emission) is closed and lives in
[`trace-encoding.md`](trace-encoding.md).

## Bank-arbitration demand accuracy (peek vs commit)

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| `peek_bank_demand` under-claims decompression S2MM bank demand | For a decompression-enabled S2MM channel, the non-committing peek claims at most 1 word / 1 bank (the mask-word-dependent word count can't be sized without popping the stream to read the mask); the committing path (`do_transfer_cycle`) may actually call `do_transfer` up to `words_this_cycle` (4) times and record a bank on each -- so up to 4 banks touched vs 1 claimed | `src/device/dma/engine/stepping.rs` (`channel_bank_mask`'s decompression branch vs `transfer_s2mm_decompressed`) | **OPEN, bounded, low real-workload impact.** Strictly an UNDER-claim, never an over-claim: it can only miss a real bank conflict, never fabricate a phantom one or cause the arbiter to wrongly deny another requester. No current test kernel exercises DMA decompression. Not HW-empirical-gated -- the bound is a structural property of non-committing peek vs data-dependent decompression, not missing HW data. Documented in the bank-arbitration arc's Task 4 report (`.superpowers/sdd/task-4-report.md`). |
