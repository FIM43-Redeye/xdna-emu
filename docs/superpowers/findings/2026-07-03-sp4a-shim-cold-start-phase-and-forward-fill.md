# SP-4a shim S2MM cold-start: pre-transfer phase error + of_out onset is forward-fill

**Date:** 2026-07-03  **Issue:** #140 (SP-4a)  **Kernel:** of_q0_lean (traced),
EMU via bridge-trace-runner; HW via `run_hw.sh` on NPU1 Phoenix.
**Method:** instrument-first (per-cycle cold-chain state movie, since removed) +
SPAN-based of_out decode (`tools/port-span-cadence.py` reused).

## What instrument-first showed (the resume movie)

A per-cycle movie of the cold chain from shim dispatch onward (mode 1) made the
mechanism concrete:

- The shim S2MM (of_out terminal) enters `MemoryLatency(527)` at dispatch and
  decrements normally -- a **527-cycle PRE-transfer hold**, sourced exactly:
  `memory_latency(5) + channel_start(2) + shim_per_task_overhead_s2mm(179) +
  shim_ddr_cold_start_s2mm(341) = 527`.
- During that hold the shim cannot drain of_out; its 2-deep ingress jams at 2,
  of_out backs up, and the whole objectfifo chain **re-deadlocks full**
  (`4,16,4,16,4,16,4,2`) after a single release-shuffle.
- The release front, when it moves, is per-hop serialized -- each stage drains
  its full 16-word S2MM ingress FIFO (~16-25cy) before the upstream stage's
  space opens. Object-drain-dominated, not lock-pipeline latency.

The codebase already treats a pre-transfer hold on an S2MM as wrong for
**non-shim** channels (`consume_first_bd_bonus`: memtile/compute cold-start is a
POST-transfer `StartupHold`, explicitly "a pre-transfer stall would keep an S2MM
channel from draining its input stream, backpressuring upstream"). The shim S2MM
was the one channel still charging cold-start pre-transfer.

## Trustworthy of_out decode: the naive soc-gaps were the wrong metric

FINDING.md flagged the of_out cadence decode as unreliable. Root cause: the raw
`PORT_RUNNING_5` events in `events.json` are frame-records (held levels
re-checkpointed on every foreign toggle), not spans. The correct unit is the
**span**, from the oracle's Perfetto B/E output -- exactly what
`tools/port-span-cadence.py` established for the send-cadence campaign. Decoded
that way (`build/experiments/sp4a-drainthrottle/of_out_transient.py`):

| of_out (memtile slot5) | HW | EMU (throttle off) |
|---|---|---|
| first-object onset (rel shim START) | **+1126** | +75 (early dump) |
| first-burst duration | 893 | 1402 |
| steady object period | 73 (64 on + 9 off) | 69 |
| shim first STARVATION | **+13** | +1491 |

## Fix landed: relocate the residual 179 out of the pre-transfer hold

`consume_first_bd_bonus` (stepping.rs): when the shim S2MM cold-drain throttle is
armed on the first task, charge the ENTIRE cold-start (per-task overhead 179 AND
DDR cold-start 341) as the metered post-arrival drain, NOT the pre-transfer
`MemoryLatency` bonus. Previously the throttle path still left the 179 as a
pre-transfer hold (the "residual +190" starvation FINDING.md noted).

Result (throttle on, cooldown=1000 decay=0): shim first **starvation +190 ->
+11** (HW ~+13). Gated on the throttle being armed, so **throttle-off is
byte-identical** (verified: starvation stays +1491 with cooldown=0). `cargo test
--lib` 3585/0.

## The key faithfulness finding: of_out onset +1126 is FORWARD-FILL, not drain

The same HW trace pins the mechanism: the shim **starves at +13** yet of_out does
not deliver until **+1126**. Starvation = "ready, ingress empty", so the shim
sits ready for ~1113cy *waiting for data* -- of_out's SOURCE is forward-filling
the pipe, the shim is not holding off its drain. Therefore:

- The **faithful** shim S2MM cold-start (drain-hold) is the real DDR first-access
  latency (~`DMA_SHIM_DDR_COLD_START_S2MM = 341`), NOT ~1113.
- The **+1126 onset** is delivered by the OFFSET campaign (keep the pipe cold ->
  of_out forward-fills like HW), not by the throttle.
- Inflating the throttle cooldown to ~1111 to "hit +1126" would be a
  **compensator**: it fakes forward-fill via shim backpressure on EMU's
  pre-filled pipe, and would double-count once the offset campaign lands. Refused.

So the throttle's headline of_out observables (onset, and the 1402-vs-893
first-burst) are **forward-fill-gated** -- closeable only by the offset campaign,
not the drain-side throttle alone. This corrects FINDING.md's "drain-side mostly
done": the drain-side owns starvation (now +11) and the per-object drain *rate*;
of_out *onset* is offset-territory.

## Decision (Maya)

Land the 179 relocation only. Keep the throttle **env-gated** (default cooldown 0
= off) rather than ship a half-composed drain model; turn it on at the faithful
DDR value (~341, decay=0) when the offset campaign is ready to co-land. The
combinational-backpressure infra (`src/device/array/backpressure.rs`, committed
`0866b77e`) is that offset campaign.

## Reproduce

```
cargo build -p xdna-emu-ffi
cd build/experiments/sp4a-drainthrottle
XDNA_EMU_S2MM_COLD_COOLDOWN=1000 XDNA_EMU_S2MM_COLD_DECAY=0 ./run_emu.sh 1000 0 t  # starvation +11
XDNA_EMU_S2MM_COLD_COOLDOWN=0 ./run_emu.sh 0 0 off                                 # +1491 (byte-identical)
python3 of_out_transient.py out_hw_dispatch out_t     # span-based of_out onset/burst/cadence
```
