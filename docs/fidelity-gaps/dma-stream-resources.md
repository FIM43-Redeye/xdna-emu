---
class: dma-stream-resources
subsystem: DMA engine + stream-switch finite-resource / consumer-pacing modeling
posture: optimistic-where-strict -- the model assumes generous or infinite resources (deep queues, unbounded token buffers, freely reusable BDs) where silicon is finite and will stall or wedge
status: compute MM2S lock request and grant modelled 2026-11-12 (see "Compute MM2S lock request and grant": 2348 / 2348 silicon rows fit the rule, emulator probe runs 1016 / 1016; chains of short lock BDs before the waiting BD modelled 2026-11-13: three lock BDs in flight, every earlier release slot, the stall rise from the first granule, 148 / 148 fitted and 41 / 41 held-out silicon rows; open: chains of short lock-free BDs (eighth BD at W + 14, egress holds behind long heads, measured 2026-11-13, not modelled) and one five-BD row, a run's first lock BD's grant phase, memtile and shim lock timing unmeasured); MM2S hand-off word moved inside the MM2S and path capacities re-placed 2026-11-11 (see "MM2S hand-off word and path capacities": stall boundary 23 words, port counts 22 / 16, idle absorb 6, every banked race and absorb row EMU == HW); stream buffering and per-hop latency pinned by silicon and modelled 2026-11-04 (see "Stream buffering and latency": R4-D3 data race 25/25, core-core absorb, S2MM backpressure windows, round-trip slope all EMU == HW); open: core-to-core round-trip intercept (+4 cycles per direction), TCT token depth, MM2S first-offer refusal, S2MM starvation until the first word (see "Open timing rows"), memtile strided-BD cost, BD-driven (de)compression data; send/recv port cadence substantially resolved (recv exact); memtile DMA stream-bound (AM020 ch.5, HW 2026-07-15); DMA memory-pressure events driven by the staging FIFOs, both EMU == HW on the measured workloads; MM2S BD retirement modelled 2026-11-10 (a compute / memtile BD finishes when its last word is staged; no source gate before an un-armed S2MM; memtile MM2S staging 29 + 1 fetch cycle modelled 2026-11-10; open: paths into an idle shim from the north +4; next-BD lock pre-acquire modelled for the compute MM2S 2026-11-12)
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

| Gap | Model vs hardware | Where | Status / rationale | Obs. |
|-----|-------------------|-------|--------------------|------|
| TCT token buffer | we model **unbounded**; HW has finite token backpressure that stalls the channel (`Stalled_TCT`) | `src/device/dma/token.rs` (`TokenState`; the deliberate simplification is called out in the doc comment) | **OPEN, deferred 2026-11-04.** aie-rt exposes only a 1-bit `STALLED_TCT` flag, no numeric depth. Pinning it means issuing tokens the firmware does not consume until a channel raises `Stalled_TCT` -- a probe that leaves the firmware's TCT queue in an unknown state on the shared NPU (wedge risk, so not run in the 2026-11-04 wave). The probe: one tile DMA, a run of `Enable_Token_Issue` tasks with no `dma_wait`, `Stalled_TCT` (DMA status) read back per task count; depth = tasks issued before it sets. | confirmable: Stalled_TCT read-back after undrained token run |
| Shim MM2S read-ahead window | was **none**: the shim MM2S finished a BD when its last word entered the stream, so it back-pressured at the consumer's buffers and its tasks were consumer-paced; HW keeps ~3.4 KiB of reads outstanding past the stream and fires FINISHED when the last read returns | `src/device/dma/engine/stepping.rs` (`uses_read_ahead`, `retire_issued_bd`, `fire_read_completions`), `stream_io.rs` (`drain_read_ahead`), archspec `shim_mm2s_read_window_bytes` / `shim_mm2s_read_latency_cycles` | **MODELLED 2026-11-07** (see "Shim MM2S read-ahead" below). Open: per channel vs shared per shim tile (needs two MM2S channels at once); S2MM write posting unprobed; DDR latency under load (HW 308-722, model 325); first-data latency of a fresh task (HW 2.0-3.8k, model 1680, the 2026-05-27 cold start). | confirmed |
| Lock count range and arithmetic | was a **signed** 6-bit field (bit 5 sign-extended, logical -64..63, releases saturating at 63/-64), so a lock written 40 read -24: every memtile object FIFO deeper than 31 wedged (`objectfifo_deep_memtile_hold`, `objectfifo_turns_walkers`, memtile S2MM stuck acquiring). Silicon: a 0..=63 unsigned count; acquire c < 0 needs count >= -c; acquire c >= 0 needs count == c and leaves it unchanged; a release adds modulo 64 and sets the overflow/underflow bit; a flagged lock refuses every request until the bit is cleared (a Lock_value write lands but does not clear it) | `src/device/tile/locks.rs` (`Lock`), `src/device/tile/registers.rs` (`handle_lock_request`, `read_register_for_ctrl_packet`), archspec `LockValueLayout` | **FIXED 2026-11-08.** Host lock-request probe over control packets (compute and memtile, three runs, 240 response words): EMU == HW through the plugin; replayed in `lock_silicon_tests.rs`. Control-packet reads of the Lock_Request space now perform the request (they raised SLVERR). | confirmed |
| Shim mux DMA channel per port | the Mux_Config parser gave the n-th DMA-selected South port to MM2S channel n, so a mux selecting DMA on South7 alone fed MM2S ch0 and left ch1 unrouted ("no route for MM2S ch1"). Silicon ties each port to one channel: MM2S ch0 = South3, ch1 = South7; S2MM ch0 = South2, ch1 = South3 | `src/device/tile/mod.rs` (`parse_shim_mux_config`, `parse_shim_demux_config`), archspec `ShimMuxField::dma_channel` (from the AM025 field descriptions; equals aie-rt's default channel-port table) | **FIXED 2026-11-08.** Surfaced by mlir-aie #3935, whose control overlay moves control packets to MM2S ch1 when the design keeps ch0: the four `ctrl_packet_reconfig*` bridge tests pass on HW with South7 alone selected and now pass in the EMU. A DMA select on South2/6 (mux) or South4/5 (demux), which AM025 gives no DMA option, is logged NOT MODELED and connects nothing (unprobed). | confirmed (ch1 = South7 alone); confirmable: DMA select on a non-DMA port |
| DMA BD and core acquires with a change >= 0 | the equal-acquire rule (count == c, unchanged) was measured on the Lock_Request path only; the DMA path decremented on an equal acquire and treated 0 as acquire >= 1, the core path treated every change as greater-or-equal. Both are reachable from the toolchain: mlir-aie lowers `aie.use_lock(%l, Acquire, N)` to a core `acq` with +N (AIECoreToStandard) and to a BD acquire value of +N (AIERT `configureLocksInBdBlock`); `acq` takes its change from a register (TableGen `ACQ_mLockId_{imm,reg}`, `eR:$mRy`), so any value -64..63 issues | `src/device/tile/locks.rs` (`Lock::acquire_change`), `src/interpreter/execute/control.rs`, `src/device/dma/engine/locks.rs` (`submit_acquire_request`) | **FIXED 2026-11-08.** Silicon: core `acq` (`tools/gen_lock_stall_probe.py` eq_c1v1/c2v1/c0v0/c0v1/c1v2, grant step and a follow-up greater-or-equal acquire against a stepped ladder of releases from a second core) and tile-DMA BD (`tools/gen_dma_lock_probe.py` c1v1/c2v1/c0v0, a loopback MM2S -> S2MM so the grant step travels in the data, plus a count probe rescued late by a second core) all match only the Lock_Request rule; EMU == HW on every case after the fix. Data `build/experiments/lock-stall/{suite2,dma2}`. One rule (`Lock::acquire_change`) now serves all three requestors | confirmed |
| Requestors of a frozen lock | a request on a lock holding an overflow / underflow flag was refused and a refused release DROPPED (the arbiter ignored it), so a flagged lock lost releases and a core never waited on one. Silicon (`frozen` and `left` families of `tools/gen_lock_stall_probe.py`, flags read and cleared over control packets, r1 == r2, no TDR): an overflowing or underflowing release is performed (count wraps, the lock's bit is set, no hold); the next request on that lock is held until the host writes 1 to the flag and is then performed, never dropped. A core stops in front of the fourth bundle after it. A **DMA channel does not stop**: a BD release that meets the flag is parked, every later release of the channel queues behind it in order (even to an unflagged lock), and the channel runs on through the rest of its chain (compute dma_rel_c4..c6, shim chains of 1..7 BDs, memtile chains); a parked release is not a stalled BD (the earlier lock-frozen reading, "BD3 does not run", only watched BD3's witness lock). A memtile channel stops one BD later than that: with one release parked the next BD's acquire is performed, with two parked it is not (Stalled_Lock_Acq on the third BD). A BD acquire on a flagged lock holds the channel (Stalled_Lock_Acq) until the clear on every tile kind. Shim status: up to two parked releases read as an idle channel, three or more as Channel_Running with Stalled_Lock_Rel. The same flags, wrap, hold-until-clear and request results hold for memtile and shim locks driven by host Lock_Request reads (shim registers: `Locks_Overflow` 0x14120, `Locks_Underflow` 0x14128, `Lock<n>_value` 0x14000, `Lock_Request` 0x40000; memtile 0xC0420.. 0xC042C, 0xC0000, 0xD0000). LOCK_ERROR latches in Event_Status and stays after the clear: compute memory event 101 (Event_Status3 bit 5), memtile 139 (Event_Status4 bit 11), shim 74 (Event_Status2 bit 10); the memtile and shim modules also latch GROUP_ERRORS (128 / 63) at the same time. The flags do not survive into the next context | `src/interpreter/execute/control.rs` (`request_blocked`, `perform_lock_request`), `src/device/dma/engine/locks.rs` (`release_must_park`, `drain_parked_releases`, `acquire_blocked_by_parked_releases`), `src/device/dma/engine/stepping.rs` (`begin_completion`, `fire_read_completions`), `src/device/dma/engine/status.rs`, `src/device/tile/registers.rs` (shim Lock_Request space and flag reads), `src/device/state/effects.rs` (`raise_lock_error_events`) | **FIXED 2026-11-08 / 2026-11-09**: core requests hold (lock-frozen); DMA release on a flagged lock parks and the channel runs on (all three tile kinds, shim read-ahead completions included), memtile acquires stop behind two parked releases; the shim Lock_Request space and flag registers exist; control-packet reads of DMA status registers return the live status. EMU == HW on the 16 compute cases of lock-frozen, on host lock requests, flags, lock values, request results, error events and data of all 46 `left` cases (memtile and shim, `build/experiments/lock-leftovers`), and on the DMA status words but for the `Status[1:0]` field and four shim cases (a shim MM2S feeding a lock-blocked shim S2MM reads Running with backpressure in the EMU, idle on silicon: the shim loopback absorbs the whole 16..80-word transfer). Tests `a_bd_release_on_a_frozen_lock_is_parked_and_the_channel_runs_on`, `a_memtile_channel_stops_acquiring_once_two_releases_are_parked`, `a_shim_release_on_a_frozen_lock_is_parked_and_the_chain_runs_on`, `shim_locks_match_the_silicon_probe`, `test_ctrl_packet_read_of_a_dma_status_register_returns_the_channel_status`. Open: shim S2MM ingress absorption while its channel is blocked on a lock (silicon takes up to 80 words, the EMU less); `Status[1:0]` (silicon reads 0 in most stalls, 2 on a memtile stalled behind two parked releases, 1 once; the EMU always reports 2 for a running channel); Cur_BD while stalled on an acquire follows the BD prefetch (memtile one BD ahead, shim to the chain end; compute unprobed, the rule is a fit to chains of 3..7); the parked-release limits of a compute tile (none seen up to 6 BDs); a compute channel's status registers; a shim / compute channel on an invalid BD (only the memtile S2MM was read: Running + Error_BD_Invalid, 0x80800); shim S2MM channel task queue: six tasks queued back to back ran on silicon (dma_rel_c6) where the EMU's depth-4 queue deadlocks | confirmed |
| Error events reach the driver only through a programmed error network | `publish_tile_event` recorded every error event (DMA, core, lock) to the async-error sink (the driver's last-error cache and per-column rings) unconditionally. Silicon, ordinary xclbin (no error network): a memtile channel started on an unconfigured BD latches DMA_S2MM_ERROR (133) and GROUP_ERRORS (128) in Event_Status4 (0x21) with the channel at Error_BD_Invalid, the run completes normally, no TDR, nothing in the kernel log (`build/experiments/lock-leftovers` mem_dmaerr, h140); the earlier native fault runs (phoenix-native-{compute,memtile,shim}-dma-error) completed normally the same way, while the same shim event with the aie-rt `XAie_ErrorHandlingInit` PDI plus the shim L1 / L2 interrupt setup (firmware-error-network-phoenix) reached the driver and withheld completion. What aie-rt programs for that network: per-module `Event_Group_Error_Enable` (reset value already enables every error event incl. Lock_Error: 0xFFFF compute memory, 0xFFF memtile), the error broadcast channel sourced from GROUP_ERRORS with its north/east/west/south block bits, and on the shim the L1 interrupt controller (event select, IRQ_NO, enable) and the L2 NoC controller (enable, NoC IRQ id) | `src/device/state/effects.rs` (`publish_tile_event_to`, `error_network_armed`) | **FIXED 2026-11-09** (weakest form): an error event is forwarded only when some shim interrupt controller (L1 switch A or B, or L2) has an enabled input; the event itself still latches, groups, broadcasts and taps L1 as before. Test `a_dma_error_without_an_error_network_latches_the_event_but_is_not_forwarded` (fails without the gate); the five tests that expect forwarding now arm the network. Open (needs partial-network PDIs on silicon, firmware territory): which register of the network is the necessary one (L2 enable only? broadcast only?), whether LOCK_ERROR is forwarded once the network is armed (it is in the group enable; `raise_lock_error_events` still never forwards), the compute-memory GROUP_ERRORS latch (lock-frozen read only Event_Status3) | confirmed |
| BD reuse-while-live / BD-pool over-allocation | we re-parse a BD on reuse and complete; HW wedges when a kernel needs > 16 distinct shim BDs (in-flight BD reuse) | finding: 2026-06-06 shim BD-pool over-allocation (`xdna-emu-priv: research/findings/2026-06-06-shim-bd-pool-overallocation-nonmonotonic-wedge.md`) | **OPEN (won't-fix).** The HW wedge is *non-monotonic* in K (k8 pass, k9 wedge, k12 pass, k16 wedge) -- no clean model. Generator (`gen-shim-chain-sweep.py`) capped at K=8 instead. | confirmed |

## Shim MM2S read-ahead (modelled 2026-11-07, NPU1 silicon)

A shim MM2S channel reading host memory issues reads into a per-channel window of 3328 B
(`shim_mm2s_read_window_bytes`: data read but not yet accepted by its stream-switch slave port),
one 32-bit word per cycle while the window has room. Each read returns 325 cycles later
(`shim_mm2s_read_latency_cycles`) and enters the stream in issue order, every cycle whatever the
channel FSM is doing. A BD completes (lock release, FINISHED_BD; on a task's last BD FINISHED_TASK
and the TCT token) when its last read returns; the channel moves on to the next BD or queued task
as soon as that read has issued, so the next task reads behind the previous one's draining data.
STREAM_BACKPRESSURE follows the stream side and keeps tracing after FINISHED, as on HW.

The 325 is the 2026-05-27 per-task MM2S overhead re-read: it used to sit before the first read with
each word entering the stream the cycle it was read, so moving it between issue and stream leaves
every free-stream timing (first data, FINISHED) unchanged and per-task MM2S is now 0.

Evidence: the shim-inflight probe (IRON shim -> core passthrough, of_in depth 2, slow core, shim
traced; tile sizes 128 B to 4 KiB, 12 + 2 HW runs, all PASS, no TDR). `inflight.py`: bytes the core
receives after shim FINISHED_TASK, and the last backpressure release -> FINISHED.

| variant | HW after-FIN | HW rel->FIN | EMU before | EMU now |
|---|---|---|---|---|
| 1 KiB x 8 / 12 / 16 | 3008 B every run | 619, 722, 1094, 531, 575 (+4577) | 92-108 B, 214 | 3000 B, 476 |
| 4 KiB x 8 | 1656 B | 579 | 92 B, 982 | 2116 B, 476 |
| 512 B x 16 | 3008 B | 308 | 124 B, 86 | 3000 B, 348 |
| 128 B x 64 (fast core) | 21-22 tiles after FIN | n/a | 2 tiles | 22 tiles |

The window is pinned by the tile sizes: with 1 KiB tiles the release at 4 KiB undelivered lets the
final read issue, with 512 B tiles that release does not and the next one (3.5 KiB) does, and the
continuously drained 128 B point fixes it against the latency; less the ~164 B the shim -> core stream
path holds, the shim-side window lies in about [3194, 3420). The 4 KiB point's after-FIN is high
because the model's 325-cycle latency sits at the low end of the HW spread; HW DDR latency varies
run to run (outliers to 8.9k) and is not modelled. `dynamic_chain_passthrough/8` (Peano) now passes
in the EMU as on HW: its LIFO BD reuse is safe only because the shim tasks finish this early.
Bridge suite at the change (2026-11-07): Chess 90/0, Peano 106/0 (was 105/1, the one fail was that
case); trace verdicts unchanged (163 DIVERGE, 2 EMPTY, 33 NONE). Shim MM2S FINISHED edges moved
earlier wherever the shim was back-pressured (47 rows, median 322 cycles); against single HW runs
that scatter by ~2k cycles between runs, 24 moved closer and 23 farther, and the signed median
HW - EMU went from -709 to -338.
Evidence and scripts: `build/experiments/shim-inflight/` (gitignored).

## Inter-tile port cadence (consumer-pacing)

| Gap | Model vs hardware | Where | Status / rationale | Obs. |
|-----|-------------------|-------|--------------------|------|
| Inter-tile send-port cadence (consumer-pacing mid-stream backpressure) | `add_one_using_dma` memtile **send** port `PORT_RUNNING_4`: EMU `[16,16,15,8,8,1]` vs HW `[8,8,14,2,14,2,6,8,1]`. HW interleaves the send with the compute consumer's mid-stream processing stalls; the residual is the **cold-start leading 16s** (at pipeline prime EMU fills both ping-pong buffers before backpressure; HW couples to one). The **recv** port `PORT_RUNNING_0` is `[16,16,16,16]` / `STALLED [1,1,1]` == HW exactly -- **recv is closed.** | Stream/DMA consumer-drain timing model; spec `xdna-emu-priv: research/specs/2026-06-27-stream-switch-axi-backpressure-design.md` | **SUBSTANTIALLY RESOLVED, cold-start residual superseded.** Four toolchain/HW-grounded fixes closed the gross defect (`f8c4f520` pipeline advance-once/cycle; `50e2ca94` S2MM ingress drains at memory-bus rate -> compute recv `PORT_RUNNING_0/1` decode `[8x8]` == HW; `7f31ba8a` bound ingress to one BD ahead; `61eaf795` crossing depth `slave_fifo + hop_latency = 4+2 = 6`). Result: the never-throttles defect gone, steady-state 8-bursts match HW, both recv ports exact, data clean, full chess bridge sweep **148 pass / 0 fail, 124 diverge == baseline** (`c7c22286`). **The remaining cold-start transient was ROOT-CAUSED 2026-07-03 to the `CORE_CONTROL` reset bit, NOT ingress/crossing depth** -- see [`host-firmware-dispatch.md`](host-firmware-dispatch.md) and finding `xdna-emu-priv: research/findings/2026-07-03-sp4a-core-reset-mechanism-part1.md`. Detailed investigation trail: findings `xdna-emu-priv: research/findings/2026-06-28-send-cadence-pipeline-latency-and-ingress-depth.md`, `xdna-emu-priv: research/findings/2026-06-29-coldstart-headstart-trace-baseline.md`. | confirmed |

**Recv-path closure (2026-06-27, historical context).** The recv side reached
exact `[16,16,16,16]` via two model fixes: a per-BD accept cursor and correcting
the memtile S2MM ingress FIFO depth 2 -> 16 (16 is a lower bound pinned on
silicon by the [16,16,16,16] memtile recv co-capture, not a decomposition of the
ingress). Regression
oracle: `sum(PORT_RUNNING) == words` per port. Finding
`xdna-emu-priv: research/findings/2026-06-27-relay-fill-bd-switch-accept-coupling.md`.
The trace-*encoding* half of the port-cadence saga (span merges, count
under-emission) is closed and lives in
[`trace-encoding.md`](trace-encoding.md).

## Bank-arbitration demand accuracy (peek vs commit)

| Gap | Model vs hardware | Where | Status / rationale | Obs. |
|-----|-------------------|-------|--------------------|------|
| `peek_bank_demand` and decompression S2MM bank demand | The peek sizes a decompressing S2MM's demand by its first word only (the mask-word-dependent count needs the stream popped). | `src/device/dma/engine/stepping.rs` (`channel_bank_mask`) | **CLOSED 2026-11-04: exact, not an under-claim.** The memory side stops at the first granule boundary (`granule_capped_words`, the 2026-07-14 one-128-bit-port fix), so every word a granted cycle writes lies in the first word's granule and bank; `decompressing_s2mm_peek_matches_commit` checks committed == peeked banks cycle by cycle over a compressed BD. | confirmable: compressed BD over known data |
| BD-driven (de)compression data | A decompressing S2MM run by its BD consumes a whole compressed block per 4-byte step and writes only 4 bytes of it, so the buffer it fills is wrong; a compressing MM2S compresses each 4-byte word as its own zero-padded block. The direct helpers (`transfer_mm2s`/`transfer_s2mm` over a whole block, `test_s2mm_decompression_round_trip`) are right. | `src/device/dma/engine/stepping.rs` (`transfer_s2mm_decompressed`, `transfer_mm2s_compressed`) | **OPEN, found 2026-11-04** (by `decompressing_s2mm_peek_matches_commit`). No test kernel uses DMA compression. Fix: a per-channel 32-byte block buffer filled from one decoded block and drained a word per step (S2MM), and per-block compression with the compressed words queued to the stream (MM2S). The compressed format itself is unverified on silicon. | confirmable: compressed BD over known data |
| The one-granule-per-cycle cap is applied to MEMTILE DMA on inference, not measurement | The tile DMA's single 128-bit memory port was measured on a **compute** tile (`xdna-emu-priv: research/findings/2026-07-14-dma-bank-access-width.md`). `do_transfer_cycle` passes `BankLayout::MemTile` for a memtile engine, so `granule_capped_words` caps memtile S2MM transfers too -- on the inference that it is the same DMA IP and the memtile's banks are also 128-bit (`memtile::PHYSICAL_BANK_WIDTH_BITS`). No memtile capture confirms it. On a memtile this is a pure THROUGHPUT rule with no arbitration behind it: `peek_bank_demand` returns nothing for a non-compute tile, so memtile DMA never arbitrates. | `src/device/dma/engine/stepping.rs` (`granule_capped_words`, called from `do_transfer_cycle` with `self.bank_layout()`) | **Width/interleave/port DERIVED from AM020 ch.5:105/137/153; strided single-channel corner HW-confirm pending (Experiment A2).** AM020 ch.5:153 states each memtile DMA channel accesses memory "over a shared interface" (singular) -- one 128-bit port per channel, the same single-granule-per-cycle shape measured on the compute tile -- and ch.5:105/137 confirm the memtile's 16 banks are 128-bit wide with 128-bit-granularity interleave (`(addr>>4)&0xF`, wrapping every 256B). So the width, interleave, and single-port cap are architecture-derived, not guessed; the prior "inference" framing is superseded. **Exposure: a strided memtile S2MM** still moves 1 word/cycle where it moved up to 4 (memtile MM2S is exempt -- egress staging -- and contiguous transfers are unchanged). Nothing in the suite covers that shape; the realistic one is an `aie.objectfifo.link` tiling/transpose on memtile ingress. **Discriminating HW check (Experiment A2):** a memtile bank-width capture, same method as the compute-tile one (`tools/experiments/bankdisc*.py`, `e87fa5ad`) -- a memtile DMA draining a fixed buffer while a core hammers the same/another memtile bank, inverting the conflict area for bytes per access. Raised in re-review of the granule-cap fix, 2026-07-14; AM020 derivation added 2026-07-15 (finding `xdna-emu-priv: research/findings/2026-07-15-memtile-bank-access-width.md`). **HW-MEASURED 2026-07-15 (Phoenix, Experiments A1+A2, same finding):** a memtile DMA<->stream transfer is STREAM-BOUND at 1 word/cycle (32-bit AIE stream), and the memtile memory subsystem (16x 128-bit banks, 9R+9W interfaces, 30 GB/s) vastly out-bandwidths it -- so memory is NEVER the bottleneck for stream-fed DMA. A2: strided ~= contiguous (~251-258 cy, flat). A1: two DMAs same bank -> only ~10 conflict cy, ZERO mutual slowdown (validity gate passes: apart/solo = 0). The "1 vs 4 words/cycle" exposure is therefore MOOT for stream transfers -- HW is 1 word/cy regardless of stride or contention. **CLOSED -- no fidelity error (EMU check, 2026-07-15).** The same four a2_stride xclbins run under `XDNA_EMU=1` (identical pipeline, decode-parity verified) give EMU spans **flat at 258 across all strides** -- the emulator is itself stream-bound at 1 word/cycle, so the granule cap in `granule_capped_words`(`BankLayout::MemTile`) never bites for a stream-fed DMA and needs no change. **One residual, OPEN (re-measured 2026-11-04):** per 256-word BD the HW finished-BD period is 257 cycles contiguous and 264 strided (16B), the emulator 258 for both; HW also traces an 8-cycle `MM2S_STALLED_LOCK` per BD that the emulator does not emit (evidence `build/experiments/harness/dmastream/memtile/`). So the strided BD pays 7 cycles on silicon that the emulator does not charge, and the contiguous one is 1 cycle faster; the mechanism (strided granule re-fetch cost vs a lock-path difference) is not separated. | confirmed |
| A DMA channel's retry is whole-mask, so a multi-bank DMA demand has no starvation bound | The DMA's memory side declared a bank per word that opened a granule, so a strided or unaligned S2MM cycle claimed up to FOUR banks at once (`words_per_cycle` = 4) -- a DMA with four memory ports. Silicon has ONE 128-bit port and serialises the accesses across cycles. Compounding it: a denied DMA channel re-presents its WHOLE mask (`step_with_denied` skips its FSM step entirely), and a whole-mask retry of a multi-bank demand has no round-robin starvation bound. | `src/device/dma/engine/stepping.rs` (`granule_capped_words`, `channel_bank_mask`, `do_transfer_cycle`), `src/device/bank_arbiter.rs` (module doc) | **CLOSED by restricting the model to the measured hardware, not by adding a fairness device.** The memory side now stops at the first granule boundary it would cross, so a DMA demand is SINGLE-BANK by construction -- one 16-byte access per cycle, per the measured 16.0 B / 4.00 stream beats (`xdna-emu-priv: research/findings/2026-07-14-dma-bank-access-width.md`). A one-bank mask cannot be partially served, so the whole-mask retry is trivially correct and the starvation shape is structurally unreachable rather than merely unlikely. Both halves were proven live before the fix: `strided_s2mm_takes_one_granule_per_cycle` (the old model declared banks 0/2/4/6 in ONE cycle) and `bank_arbiter::dma_whole_mask_retry_starves_a_multi_bank_channel` (that demand gets ZERO grants over 400 cycles at rotor skews 2 and 3 -- an emulator livelock, the same phases that starved the core). Cost: strided/unaligned DMA cycles now move fewer words; contiguous aligned transfers are unchanged, and the `of_q0_rich` census is byte-identical. Found in the final review of the memory-stall arc, fixed 2026-07-14. | confirmed |

## DMA memory-pressure events

| Gap | Model vs hardware | Where | Status / rationale | Obs. |
|-----|-------------------|-------|--------------------|------|
| `DMA_S2MM_n_MEMORY_BACKPRESSURE` window shape | Per steady-state lock-stall window, backpressure opens **+15** cycles into the stall and closes **1** cycle after it, EMU == HW on every steady window (of_q0_rich rebuilt from its traced MLIR, 2026-11-04: HW r1/r2 and EMU all `(len 85, onset 15, held 71, end +1)`; the 2026-07-14 capture's 39 windows also +15). | `src/device/dma/engine/stream_io.rs` (`note_ingress_offer`, `input_fifo_capacity`), `src/device/array/routing.rs` (`route_tile_switches_to_dma`) | **CLOSED 2026-11-04.** The S2MM ingress is a 15-deep FIFO (`DMA_S2MM_INGRESS_FIFO_DEPTH`, which the event watches) behind a 1-word skid (`DMA_S2MM_INGRESS_SKID_DEPTH`): 16 in all (the DMA:0 master passes 16 words into a held S2MM, port probe 2026-11-11; it was a 3-word skid and 18 until then, when the sources' hand-off words were counted outside the MM2S and the core's output buffer was assumed 1 word), with the event asserting while a beat is on offer and the FIFO's level after the cycle's drain is 15. Formerly +16 (one 16-deep FIFO, onset counted at the refusal) and +2 at the close (the registered start-of-cycle level, which still gates TREADY). Evidence `build/experiments/harness/dmastream/bp/`. **Do NOT compare run totals against `of_q0_rich` captures** (the trace holds a window of the kernel's iterations; see `xdna-emu-priv: research/findings/2026-07-13-memory-stall-bank-arbitration.md`). | confirmed |
| MM2S egress staging depth (27) and the route downstream of it | `DMA_MM2S_EGRESS_FIFO_DEPTH = 27` (25 until 2026-11-11), **measured** (R4-D3, Phoenix NPU1, 2026-10-28) by data race: overwrite the source buffer while the consumer is held and count words still carrying the old pattern. The race pins the route capacity (staging + hand-off + downstream), the 2026-11-11 onset sweep and port probe the split. | `crates/xdna-archspec` (`DMA_MM2S_EGRESS_FIFO_DEPTH`), `src/device/dma/engine/stepping.rs` (`next_granule_fetch`, `staged_mm2s_cycle`), `ChannelContext::egress_staging`, rig `tools/experiments/r4d3_*.py` | **CLOSED 2026-11-04, EMU == HW on all 25 route x phase points** (staging + route; see "Stream buffering and latency"). Two defects hid the staging from the race: staged words were re-read from memory at drain time (now read at fetch time, `staged_words_carry_the_data_read_at_fetch_time`), and the MM2S output buffer was a 4-word second copy of the slave FIFO (now the 1-word hand-off word, `MM2S_OUTPUT_REGISTER_DEPTH`, which belongs to the MM2S since 2026-11-11). A memtile MM2S stages 31 (`DMA_MEMTILE_MM2S_EGRESS_FIFO_DEPTH`, 29 until 2026-11-11, see "MM2S BD retirement" and "MM2S hand-off word and path capacities"). **Open modelling question (not a measured gap):** a fetched granule's words sit in the read-to-offer pipeline for 3 cycles after their last read before they are offered, and the emulator counts them against the staging depth (27 / 31) the whole time; whether silicon counts them there or in a separate stage is not separated, since the races and the backpressure onset fix only the sums. The choice moves when a fill burst's last fetch is admitted (the seventh fetch of a burst, one cycle), which is what removed a bank-conflict event from `dynamic_pingpong_iterarg_await` and `objectfifo_turns_walkers` (see [`core-compute-timing.md`](core-compute-timing.md), "CONFLICT_DM_BANK"). | confirmed |
| `DMA_MM2S_n_MEMORY_STARVATION` cold-fetch residual | EMU 0 on `of_q0_rich` (**HW 0**, exact match). In a synthetic worst case (core hammering the DMA's bank every cycle) EMU emits a few cycles where HW emits 0. | `src/device/dma/engine/stepping.rs` (`next_granule_fetch`) | **CLOSED for the measured workloads (0 == 0 again on the 2026-11-04 of_q0_rich rebuild); one known residual, OPEN.** The event is now driven by the MM2S egress staging FIFO running dry (`DMA_MM2S_EGRESS_FIFO_DEPTH` = 25). Silicon's 0 EMERGES -- the MM2S in `of_q0_rich` loses bank arbitrations and the staging absorbs every one, and a unit test proves losing EVERY arbitration costs zero cycles and zero starvation (HW: `T_collide/T_apart` = 1.0000). Residual: a transfer's very FIRST granule fetch has an empty staging behind it, so losing THAT one does bubble the port for a cycle. Silicon apparently primes its egress before the port opens (it reported 0 starvation across 112 losses and 16 transfer starts). Bounded by transfer starts, not by denials; needs a capture that starves an egress on purpose (a core hammering the MM2S's bank from the transfer's first cycle, `MM2S_MEMORY_STARVATION` traced) to pin; not taken in the 2026-11-04 wave. | confirmed |


## Task queue, BD pipeline and shim S2MM ingress (modelled 2026-11-09, NPU1 silicon)

Probes: `tools/gen_dma_queue_probe.py` (push ramps, parked-chain `stall` sweeps, `absorb` sweeps), every
status word read over control packets; results are compared word by word against the emulator
(`src/device/dma/engine/pipeline.rs` documents the rules, `tests.rs` replays the silicon rows).

**What a channel holds.** A stalled channel keeps its stalled BD and the BDs behind it (the rest of the task's
chain, then the first BDs of the queued tasks) loaded in a short pipeline, and `Task_Queue_Size` counts only
what does not fit. The pipeline holds 3 BDs when the head is parked on a lock acquire (every tile kind),
5 when it is stalled on its stream on a compute tile or memtile, 7 on a shim; the queue behind it is 4 deep
(aie-rt `StartQSizeMax`). So 7, 9 and 11 pushes are accepted, and `Task_Queue_Size` reads 4 at the last one.
A BD joins the pipeline only while the loaded BDs fit a word budget (the BDs behind the head cost 4 words
more each); the budgets are known to within a few words (lock: 38..41 compute and memtile, 206..208 shim;
stream: 28..29 and 136..137). `Cur_BD` names the last loaded BD (at most 2 / 4 / 5 past the head).
`Status[1:0]` reads 1 (STARTING) when the pipeline is full by count, 2 (RUNNING) when the loaded BDs together
move at least a threshold number of words (lock: 21 compute and memtile, 137 shim S2MM; starved S2MM: 15 and 69;
backpressured MM2S: 57 compute and memtile, never on a shim up to 512 words), 1 when the next BD did not fit the
budget, otherwise 0. Every other stall reads 0 (lock acquire, parked releases, backpressure, starvation).

**A push into a full queue blocks the writer.** The register write itself stalls, so a runtime sequence whose
later writes would drain the queue deadlocks (the emulator halts with a deadlock diagnostic naming the tile,
channel and queue state). Whether the overflow flag is ever set on silicon is not observed. Control-packet and
CDO pushes into a full queue are still dropped with the overflow flag (blocking a stream write is unprobed).
`dma_rel_c6` queues six transfers on one shim S2MM whose head waits on its stream: well inside the 11 that fit,
and the emulator now runs it identically to silicon.

**Shim S2MM ingress.** A shim MM2S streaming into its own S2MM backpressures after 8 words while the S2MM has
no task (the stream path: slave 4 + master 4), and absorbs up to 76 words (every length to 74 and 76 do; 75 and 77 up
backpressure) while the S2MM's task is parked on a lock acquire, in order, all of it landing once the lock is
granted: the channel holds 68 words. `dma.shim_s2mm_ingress_depth` (66 in the emulator's accounting, 65 until
2026-11-11) opens once the channel holds a task. With the MM2S hand-off word inside the MM2S (below) the
emulator's no-task shim loop now holds 8 like silicon, and the parked loop still absorbs 76 (the stall boundary
moved out one word, this lump 1 up). Compute and memtile numbers (hop 8, ingress 16, egress 27) are in the next section.

**Open.** The MM2S backpressure pipeline (depth and `Cur_BD` of a chain stuck behind a stalled consumer; only the
2-word chains and the 57-word threshold are probed), the status of a TCT stall, how chain BDs and queued tasks
share the pipeline when both are present, the exact budgets and the 4-word per-BD overhead, absorb length 75.

## MM2S BD retirement and the path to an un-armed drain (modelled 2026-11-10, NPU1 silicon)

Probe: `tools/gen_pipeline_fill_probe.py` (families `chain`, `sink`, `race`, and the `of_q0_rich` readback);
data in `build/experiments/pipeline-fill/`. Every count below is stable over two or three silicon runs.

**A compute or memtile MM2S BD finishes once its last word is fetched into the egress staging**, not once it
leaves the port. A chain of K BDs of L words into an S2MM parked on a lock finishes as many BDs as fit the
whole absorb (`chain`, 0 hops, 12 BDs; BDs finished x L):

| L (words) | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|
| compute, silicon | 24 (all) | 44 | 48 | 48 | 48 | 48 |
| memtile, silicon | 24 (all) | 48 | 48 | 48 | 48 | 48 |
| emulator, port-exit rule (before) | 24 | 24 | 24 | 24 | 16 | 24 |
| emulator now | 24 | 48 | 48 | 48 | 48 | 48 |

48 is the R4-D3 0-hop absorb, so the whole path holds the finished BDs. The emulator's egress staging
(`ChannelContext::egress_staging`) now belongs to the channel: a fetched word leaves the transfer with its
TLAST, the BD completes when its last word has been read (one word per cycle, see "MM2S read bandwidth and tail
reservation" below), the next BD fetches in behind, and the staging drains one word per cycle whatever the FSM
is doing. The BD-boundary bubble (`off1`) is a port-side idle cycle after a
BD's last word (`egress_gap`). A channel whose BDs have all finished with words still staged reads Running with
Stalled_Stream_Backpressure (silicon `chain_mem_L4`), like the shim read-ahead window.

**No source gate in front of an un-armed S2MM.** A memtile MM2S chain into a shim S2MM that holds no task
finishes 48 words of BDs (`sink`, L = 8 and 16); the emulator held such a source completely (#140 SP-4a
terminal-send gate, removed) and finished 24. The opt-in `XDNA_EMU_SP4A_BP_MODE` campaign gates rest on the
same refuted premise (TREADY low end-to-end from the un-armed sink) and are left off by default.

**Path capacities by data race** (`race`: one 96-word BD, the source overwritten once the sender settled, words
read before the overwrite; prod = compute (0,2), consA = (0,3)):

| Path | Silicon | Emulator |
|---|---|---|
| prod -> own S2MM (held) | 48 | 48 |
| prod -> memtile S2MM (held) | 56 | 56 |
| memtile -> own S2MM (held) | 52 | 52 (48 before the memtile staging) |
| memtile -> prod S2MM (held) | 60 | 60 (56 before) |
| memtile -> idle shim S2MM | 48 | 48 (44 until 2026-11-11, 40 before the memtile staging) |
| prod -> idle shim S2MM | 52 | 52 (48 until 2026-11-11) |
| consA -> idle shim S2MM | 60 | 60 (56 until 2026-11-11) |

Regression evidence for the change (2026-11-10): R4-D3 all 25 points unchanged (EMU == HW); the banked
shim-queue MM2S stall probes now differ from silicon on 24 of 164 cases (50 before, none newly); absorb probes
unchanged; full bridge suite 207 pass / 0 fail on both compilers, 40 emulator traces changed against the
branch base, scored against the same silicon capture: cleanly matching edge event types 46 -> 49, level
interval-count mismatches 143 -> 149, summed edge |dt| -0.1% (`build/experiments/pipeline-fill/bridge-emu-ab.txt`).
Level rows: 9 became exact, 15 stopped being exact (`bridge-emu-ab-rows.txt`). Most of the 15 are the memtile
send port of the `memtile_dmas/dma_configure_task_*` tests, `PORT_RUNNING_4` 2/2 -> 2/3, which stays 2/3 with the
BD bubble off (`XDNA_EMU_BD_SWITCH_BUBBLE=0`), so it is not the port-side bubble (unattributed: the earlier
retirement or the removed source gate). `add_256_using_dma_op_no_double_buffering` moved most (`PORT_RUNNING_0`
6/6 -> 6/7, `PORT_RUNNING_4` 9/9 -> 9/11, shim S2MM 0 starvation 15/15 -> 15/8); with the bubble off its port
rows undershoot instead (6/5, 9/8), so the bubble's placement matters there and is not settled.

**Memtile MM2S staging (modelled 2026-11-10; 29 and 25 then, 31 and 27 since 2026-11-11 with the hand-off
word inside the MM2S).** A memtile MM2S stages 4 words more than a compute tile's staging,
and its fetched words reach the stream one cycle later (`DMA_MEMTILE_MM2S_EGRESS_FIFO_DEPTH`,
`DMA_MEMTILE_MM2S_FETCH_LATENCY_CYCLES`). Two silicon probes, two runs each identical:

- start-phase races (`race_*_loop_p1`..`_p3`, BD start 0..3 words into a granule): memtile -> own S2MM
  52/51/54/53, compute 48/47/50/49. The fetch rule gives that pattern for route capacity 54 only; the
  staging that realises it is 29 then, 31 now (see the 2026-11-11 section). Emulator now 52/51/54/53 and 48/47/50/49.
- backpressure onset (`tools/gen_path_onset_probe.py`: one 96-word BD into a lock-held S2MM, the source
  tile's DMA traced): the sustained STREAM_BACKPRESSURE rises 38 / 39 cycles after START_TASK for a
  compute / memtile source into its own S2MM and 47 / 48 one hop away; every memtile edge, the first-granule
  pulse included, is the compute one plus one cycle. Four more words downstream would move the held onset
  4 cycles, so the words are upstream of the stream port. Whether a deeper staging or a granule in flight
  holds them is not separated; the model counts the in-flight granule in the 31.

The race table's memtile rows now read 52 and 60 in the emulator (memtile -> idle shim 44: the shim
half below). Still open from the onset probe: the emulator's onset is 2 cycles early on the same tile (36 /
37) and 3 one hop away (44 / 45), it has no first-granule pulse (silicon: one 1-cycle pulse at +15, a second
at +24 one hop away), and a lock-stalled S2MM on silicon also reads STREAM_STARVATION until the first word
reaches it, which the emulator does not raise.

**Idle shim S2MM paths (closed 2026-11-11):** the paths into an idle shim S2MM from the north that held 4
more on silicon, and the shim's own loop that held one less, were the same missing word: the MM2S hand-off
belongs to the MM2S (next section). Race rows 48 / 52 / 60 and absorb thresholds (north idle 14, shim idle 8) now
read EMU == HW. The hop slope is 8 words on both sides. **Open (not modelled):** also a stalled MM2S has already acquired the locks of BDs beyond the one it works on (compute
chain one, memtile chain two: at 8-word BDs 8 and 9 acquires for 6 finished BDs, the emulator 7; the
`of_q0_rich` memtile drain one), with `Cur_BD` one BD further than the emulator's; `Status[1:0]`
reads 2 where the emulator reads 1 on a backpressured chain; a compute chain of 4-word BDs stops one BD short
(11 of 12, the next BD reading Stalled_Lock_Acq with its lock available; also a memtile chain of twenty
4-word BDs into an idle shim: 11).

## Stream wedges: switch FIFO route and core stream writes (FIXED, NPU1 silicon 2026-11-03)

Two designs wedged the emulator that silicon runs: a flow routed through the stream switch's FIFO
(`aie.flow(t, Core:0, t, FIFO:0)`, `aie.flow(t, FIFO:0, ...)`), and a core streaming more words than
its path holds into a held consumer (the R4-D3 `core_h*` designs, `tools/experiments/r4d3_*.py`).
Four independent defects, all root-caused with the original designs
(`build/experiments/r4-d3/{core_h0,core_h0_fifo,h0_d1200_d220_fifo}`):

1. **A core output beat was dropped when the Core slave port was full**
   (`device/array/routing.rs` `route_core_to_tile_switches`: `pop_stream_output_beat` ran before the
   slave `can_accept` check, so `push_with_tlast` failed and the popped word vanished). The sink saw
   25 of 128 words and waited forever (`core_h0` before: `Tier C wedge ... (1,2)ch0 Transferring(100/512)`,
   `build/experiments/harness/bugs/core_h0-prefix-wedge.txt`). The beat now leaves the output buffer
   only when the slave takes it. Test: `array::tests::core_stream_output_waits_in_its_buffer_when_the_core_slave_port_is_full`.
2. **A core blocked on a full output never woke**: `CoreStatus::WaitingStream { port }` carries no
   direction and `try_resume_stall` resumed only when stream INPUT was available. A write stall now
   carries `stream::WRITE_STALL_FLAG` and resumes when the output buffer has room
   (`interpreter/core/interpreter.rs`, test `stream_write_stall_resumes_when_the_output_drains_not_when_input_arrives`).
3. **`tile.stream_input` was unbounded** (`route_tile_switches_to_core` popped the Core master port
   unconditionally), so a held consumer never backpressured its producer. Bounded at
   `CORE_STREAM_INPUT_DEPTH` (first 1 on AM020 ch.2:143, 3 from 2026-11-04, 1 since 2026-11-11 once the
   producer's output buffer was measured at 3: the value is derived from core -> core 18, see "MM2S hand-off
   word and path capacities"). Test
   `array::tests::core_stream_input_is_bounded_and_backpressures_the_switch`.
4. **The switch FIFO did not exist**: master FIFO:0 and slave FIFO:0 were port types with no link,
   so a flow through `FIFO:0` delivered nothing. The master port is now the 16-deep FIFO
   (`stream_switch.switch_fifo_depth`, AM020 ch.2:74) and `advance_switch_fifos` moves a word per cycle
   to the paired slave port (tests `the_switch_fifo_*`).

The core output buffer depth was 4 ("a modelling choice, not a measurement"). Silicon pins the core's whole
path: a core streaming into a held DMA on its own tile writes 26 words before stalling (the word still in
`r0` included). The 2026-11-04 model split that as a 1-word output register plus a 24-word path; the port
probe of 2026-11-11 showed the Core:0 slave port passes 22 words like a DMA source's, so the core's output
buffer is 3 words (`STREAM_FIFO_DEPTH`) and 1 + 3 + 22 = 26.

**Absorb counts, silicon vs emulator** (words the source writes before it stalls on a held consumer;
`core_h*`/`h0_*` are the R4-D3 xclbins under the bridge runner, `direct`/`fifo` the harness suite
`gen_stream_absorb_probe.py`, 96 words, consumer held ~28000 cycles; HW stable over r1/r2 and the
earlier 3 runs). Emulator column unchanged by the 2026-11-11 re-placement ("MM2S hand-off word and path capacities" below; re-run that day, every row identical):

| Design | Silicon | Emulator |
|---|---|---|
| core source -> DMA sink, 0 / 1 / 2 / 3 hops | 26 / 34 / 42 / 50 | 26 / 34 / 42 / 50 |
| core source through the switch FIFO, 0 hops | 47 (+21) | 47 (+21) |
| DMA source -> DMA sink, 0 / 1 / 2 / 3 hops, BD phase 0 | 48 / 56 / 64 / 72 | 48 / 56 / 64 / 72 |
| DMA source -> DMA sink, 0 hops, with FIFO, phases 0..3 | 68 / 71 / 70 / 69 | 68 / 71 / 70 / 69 |
| core -> core, one tile apart (harness `direct`) | 18 | 18 |
| same through the producer's FIFO (harness `fifo`) | 39 (+21) | 39 (+21) |

(All four BD phases match on every DMA route too: 25 of 25 R4-D3 points.)

## MM2S hand-off word and path capacities (modelled 2026-11-11, NPU1 silicon)

Probes: `tools/gen_path_onset_probe.py` (`prod_own_nN` / `mem_own_nN` length sweep, data
`build/experiments/path-capacity/onset-len`; `*_ports` cases, data `onset-ports`), the R4-D3 rig, the start-phase
races and the shim / memtile absorb probes (`build/experiments/path-capacity/`, `shim-queue/absorb2,3`).
Reasoning and data: `build/experiments/path-capacity/PLAN.md`.

- **Stall boundary.** An MM2S into its own held S2MM first sustains STREAM_BACKPRESSURE at N = 23 words, compute
  and memtile (22 words past the stall point; the emulator: 26). 22 is also the count of words that enter the
  source's slave port (PORT_RUNNING of `DMA:0` and `Core:0` slave, N = 40; emulator 24), and 16 pass the `DMA:0`
  master port into the held S2MM (emulator 18). Slave minus master is 6 on both: slave 4 + local master 2,
  AM020 ch.2, unchanged.
- **Consequence 1: the hand-off word belongs to the MM2S.** `MM2S_OUTPUT_REGISTER_DEPTH` (1) is inside the
  channel's Running status and its STREAM_BACKPRESSURE level (and Stalled_Stream_Backpressure bit): both hold while
  that word is refused by the slave port, including with the staging already empty. The boundary then sits at the
  slave port, so an idle S2MM takes 6 words past the stall point (silicon absorb threshold 6, emulator 7).
- **Consequence 2: the races pin only the route capacity.** A race count T satisfies T == -phase (mod 4) and
  T <= C = staging + hand-off + downstream, so the start-phase patterns fix C per route, not its division. The
  division comes from the boundary above: staging 27 (compute) / 31 (memtile) + hand-off 1 + 22 = 50 / 54,
  S2MM intake 16 = FIFO 15 (pinned by the backpressure event onset +15) + skid 1. Shim: the stall boundary
  moved out one word, so `dma.shim_s2mm_ingress_depth` 65 -> 66 keeps the parked shim loop at 76.
- **Consequence 3: the core.** Core:0 slave passes 22 words like the DMA, so the core's extra words are upstream
  of it: output buffer 3 (race 26 = `r0` 1 + 3 + 22). Core input 3 -> 1 keeps core -> core at 18 (derived, not
  port-measured).

Result (all EMU == HW unless listed): R4-D3 core rows 26 / 34 / 42 / 50 and fifo 47, DMA rows (all phases and
hops, fifo route 68 / 71 / 70 / 69); race-phase compute / memtile 48/47/50/49, 52/51/54/53; shim idle 52/51/50/49
and parked 120/119/118/117; race2, shim-end, shim-chan rows; onset sweep (first sustained stall 23 on both);
port counts N = 40 (master 16, slave 22); absorb thresholds (memloop / coreloop parked 22, idle 6; north
parked 30, idle 14; shim idle 8, parked 73 / 74 / 76 pass, 77+ backpressure); core -> core 18 / 39.
**Still different:** shim absorb length 75 (silicon backpressures, emulator absorbs; unchanged oddity), and the
timing, which is a later step: the sustained-onset cycle is 34 / 35 in the emulator against silicon 38 / 39
(prod / mem, 4 early), the `DMA:0` master port's stall onset 20 / 32 against 19 / 37 (prod / mem), the memtile
slave port's 34 against 40 (N = 40), silicon's 1-cycle STREAM_BACKPRESSURE blip at +15/16 has no emulator
counterpart, and at N = 20 the emulator's slave port reports a PORT_STALLED onset (20 / 32) where silicon has none.

## MM2S read bandwidth and tail reservation (modelled 2026-11-11, NPU1 silicon)

Probes: `tools/gen_path_onset_probe.py` `prod_own_nN` / `mem_own_nN` (one MM2S BD of N words into an S2MM on the
same tile whose BD waits on a lock; cycles from the MM2S START_TASK), data `build/experiments/path-capacity/onset-len`
(N 21..27) and `onset-long` (N 1..55), and the chain / task variants `_cLxK` / `_tLxK`, data `onset-chain`. Every
row is identical over two silicon runs.

- **One word read per cycle.** FINISHED_BD = N + 8 for every N tested (1..48 compute, 1..52 memtile), independent
  of the port stalling (the port stalls at 38; N = 46 still finishes at 54). The BD is done when its last word has
  been read. Chained BDs read back to back: `c4x4` finishes at 12 / 16 / 20 / 24, `c8x2` at 16 / 24; a task started
  from idle finishes L + 8 after its START_TASK (`t4x4`: +12 each). The port-side bubble per BD boundary is separate
  (S2MM starvation pulses at 22, 27, 32 for `c4x4`: 5 cycles per 4-word BD, `egress_gap`).
- **Granule-wise visibility.** The first word is offered at the switch at 11 + min(N, 4) (compute; the memtile one
  cycle later, `memtile_mm2s_fetch_latency_cycles`): a granule enters the staging when its last word (or the BD's
  last word) has been read, so data arrives a whole granule at a time, at most one 4-word granule per 4 cycles.
- **A short tail reserves a whole granule.** The longest BD that finishes into a held own-tile S2MM is 48 words
  (compute) / 52 (memtile); 49, 50 and 53..55 never finish until the S2MM is released. Admission reserves
  `granule_words` (4) of staging for every fetch, a BD's short tail included (the emulator admitted the tail into
  the space it needed and finished 50 / 54).
- **Emulator (2026-11-11; since 2026-11-12 the memtile MM2S only, the compute MM2S runs the word clock of the next
  section).** The bank access and the admission decision stay per granule, as before (one `next_granule_fetch`
  per granule, bank demand unchanged, so the start-phase congruence and every race row are unchanged). What
  changed is when the words become available: a fetched granule's words enter `egress_staging` at once (they
  count against the staging and its admission from the fetch on) but carry `ready_at` = the cycle their last word
  is read, with reads serialised one word per cycle behind every earlier fetch (`ChannelContext::read_tail`).
  `staged_mm2s_cycle` holds the BD open (`TransferCycleResult::ReadInProgress`) until its last word has been read,
  which then releases the lock, raises FINISHED_BD and chains the next BD. A staged MM2S's chained BDs take no FSM
  bubble (`enter_chained_transfer`): the boundary idle cycle is `egress_gap` on the port side only.
- **Start and offer latencies (modelled 2026-11-11, second step; memtile only since 2026-11-12).** Two silicon-pinned offsets were leftovers of
  the older granule-at-fetch calibration. (1) First read at START_TASK + 9: a compute / memtile MM2S started from
  idle pays `bd_setup_cycles` 4 + `memory_latency_cycles` 5 and not the `channel_start_cycles` 2 term every other
  channel (S2MM, shim, locked first BD) keeps; those three constants were never isolated on silicon. (2) A granule
  is offered `mm2s_read_to_offer_cycles` (3) after its last word is read (first offer 11 + min(N, 4) against a
  last read at N + 8), plus the memtile's 1. Bank-arbitration urgency was judged on the read backlog
  (`ChannelContext::read_backlog`); the fetch backoff it overrode was removed with the word clock (next
  section), and the urgency itself with the bank arbitration rule of 2026-11-12.
- **Result.** FINISHED_BD = N + 8 for every N tested, compute and memtile; `c4x4` 12 / 16 / 20 / 24, `c8x2` 16 / 24;
  `t4x4` 12 for the first task and START + 12 for each later one; the longest finishing BD is 48 / 52; sustained
  STREAM_BACKPRESSURE at 38 (compute) / 39 (memtile) for every N >= 23 and clean at N = 21, 22; races, R4-D3 and the
  chain table unchanged. **Still different:** silicon's 1-cycle first-offer refusal (not deterministic: present in
  some tasks of a chain and not in others) and the S2MM starvation until the first word arrives
  have no emulator counterpart; both are rows in "Open timing rows" below.

## Compute MM2S word clock and lock-path start (modelled 2026-11-12, NPU1 silicon)

Probes: `tools/gen_dma_lock_probe.py` fetch cases (a 64-word compute MM2S BD parked on a lock that a core releases,
own-tile loopback, core loads swept across the grant to map the DMA's bank accesses; first-BD and chained),
`tools/gen_path_onset_probe.py` `prewake-run` (a 1-word dummy BD, then a 16-word BD) and the onset probes above.
Cycles are counted from the grant g (the cycle after STALLED_LOCK falls).

- **Silicon.** Bank accesses at g + 2, 3, 4, 5, 7, 11, 15, ..., 51 (banks alternating from bank 0), first-BD and
  chained alike; FINISHED_BD at g + 52, one cycle after the last access; a 1-word and a 4-word BD finish at g + 3.
  A core access that wins the bank at the first access (g + 2) delays the stream by one cycle and not FINISHED_BD;
  one at the last access (g + 51) delays FINISHED_BD by one cycle. The host-path numbers of the previous section
  (FINISHED_BD = N + 8, first offer 11 + min(N, 4)) hold as well.
- **Model.** A compute MM2S sequences its words (address generation) one per cycle from START_TASK + 7
  (`mm2s_word_clock_start_cycles` 3 after BD setup), lock or no lock, and on into the chained BDs behind it
  (`WordClock`, `src/device/dma/word_clock.rs`). Since 2026-11-12 the lead is counted in granules: while the
  channel is parked on a lock at most 4 started granules wait for their access to be presented
  (`mm2s_word_clock_lead_granules`), a BD's first access comes at least 2 cycles after the previous BD's last
  (`mm2s_bd_switch_access_gap_cycles`); see the next section (it replaced a 16-word lead resuming at grant + 3,
  which gave the same cells here). A granule is accessed the cycle after its last word was sequenced, from grant + 2
  (`mm2s_lock_to_access_cycles`), one access per cycle, when the staging has room; it is offered 4 cycles after
  the access (`mm2s_access_to_offer_cycles`); FINISHED_BD is the last access + 1 (`mm2s_access_to_finish_cycles`).
  A lock-started BD pays no memory latency, channel start or post-grant countdown. A denied access is retried
  the next cycle: the 2-cycle fetch backoff of 2026-07 is removed (silicon: against a dense same-bank core every
  lost granule shows as two conflicts on consecutive cycles on the same bank, 532-535 pairs per run, producer
  probe; with the backoff the emulator gave single conflicts and half the count). A BD that has not accessed
  memory yet raises no MEMORY_STARVATION.
- **Result (emulator).** The grant-relative access map, FINISHED_BD g + 52 (was g + 64 chained, g + 72 first-BD),
  the conflict at g + 51 (FINISHED_BD g + 53), and the dummy BD's start-to-finish of 3 (was 9) match silicon; a
  denied access at g + 2 delays only the stream (unit test; since 2026-11-12 the integrated emulator gives the
  core that conflict as silicon does, next section);
  every host-path guard of the previous section is unchanged; producer probe `collide`: conflicts in same-bank
  pairs, 116 per colliding transfer (silicon 115-116; before 58). With the urgency override every transfer
  collided (1740 conflicts against 1069-1074); with the bank arbitration of the next section 9 of 15 did, as on
  silicon, until the lock grant law (2026-11-12, next sections) moved the grant and none collides. Lock-synchronised variant (`fetch --sync --pre2
  32`, 34 cases): FINISHED_BD minus the previous BD's = max(64, grant - previous finish + 52) on every row, as on
  silicon (291 rows; the emulator was flat grant + 64), and a parked 4-word BD's lock-free next BD of 32 words
  finishes 21 after it (silicon 21, emulator was 32).
- **Still different (open gaps).**
  - *Bank arbitration.* Modelled since 2026-11-12 by the served-access count; see the next section. The
    urgency override this bullet described (a compute MM2S with 4 words or fewer staged beat the core) is gone.
  - *Short chained locked BDs and the grant.* Modelled since 2026-11-12 by the lock request and grant law (next
    section): the lock is requested ahead of the previous BD's finish and granted at release + 2 or + 3. Chains of
    short lock BDs before the waiting BD are modelled since 2026-11-13 (next section); six lock-free short BDs
    before it remain open there.
  - *Packet header.* The header of a lock-started compute MM2S's packet BD leaves at the grant, 6 cycles ahead of
    its payload (4 words: g + 0, then g + 6..9); on silicon header and payload are contiguous.
  - *First word at the receiving DMA.* The first word reaches the own-tile S2MM at g + 10 (silicon g + 9; the
    offer / route split is not separated), one cycle late.
  - *S2MM finish.* The S2MM finished at g + 74 against silicon's g + 78, 4 cycles early; the S2MM granule writer
    (2026-11-13, "Compute S2MM after a lock grant") moves it 4 cycles later (the sync family's S2MM finish + 73 -> + 77,
    silicon + 77); this probe was not re-scored.
  - *Choice, not measurement.* Unparked, nothing bounds the word clock's lead but the BDs and the staging
    (silicon: the ring camp needs the clock free once the parked BD presented its access; a stream-backpressured
    channel was not probed for a bound). A lock-started first BD's clock starting at START_TASK + 7 is also a
    choice.
  - Not measured, so not changed: the memtile MM2S (keeps the read-cursor rule), the shim MM2S, every S2MM lock
    path, and the host-path access position inside a granule.


## Compute MM2S lock request and grant (modelled 2026-11-12, NPU1 silicon)

Probes: `tools/gen_dma_lock_probe.py` sync family (a 4-word head BD parked on a start lock, a lock-free BD of
N = 0..40 words, a 64-word BD on lock A, which the core releases a swept number of cycles later) and chain family
(the same with a mid BD on a core-released lock, a free lock, 2-, 8- or 16-word mid BDs, 8- or 16-word heads, a
16-word waiting BD, A released by an S2MM BD, and chains of several short BDs). Cycles are counted from the head
BD's grant g0 (its STALLED_LOCK falls).

- **Silicon.** LOCK_SEL0_ACQ_GE fires exactly once per acquire of A, one cycle before STALLED_LOCK falls, and also
  when the lock was free on request. With g the grant of the channel's previous lock-acquiring BD and T the cycle
  the BD before the waiting one has its third-to-last word sequenced, the waiting BD's lock is requested at
  Q = max(g + 3, T + 3), moved off DEAD = the previous lock BD's FINISHED_BD + 4, and granted at the first
  G >= max(Q, R + 2) with (G - Q) mod 3 != 2 and G != DEAD (R the release, by the core or an S2MM). The request
  is made while the channel is still on the BDs before it: behind a 4-word lock BD and a 13-word lock-free BD a
  free lock is granted at g + 3, 5 cycles before the 13-word BD finishes. STALLED_LOCK shows on [P, G - 1] with
  P = max(S + 2, Q - 1, the previous BD's last access - 1), S the cycle the BD's first word is sequenced: it can
  rise before the BD before finishes, and a lock free on request shows one cycle. The channel side is the
  word clock with a 4-granule park (previous section's model, revised below). Revised for short chains (below):
  P = max(D, Q, the previous BD's last access) - 1 with D the cycle the BD's first granule is fully sequenced
  (S + 3 for a 4-word granule, so the same cycle there), and DEAD every earlier lock BD's FINISHED_BD + 4.
  Evidence: 2348 of 2348 rows of
  every design with at most four MM2S BDs match this rule; about 2102 of them are the rows it was fitted to and
  246 (a 2-word mid BD before 11..17-word BDs) were predicted before they were measured. What blocks one cycle in
  three is not known. One input is a fitted proxy: the core's release of a mid lock B is not traced, so the rows
  take it at the core's event 0 + its nops + 5.
- **Model.** `LockPort` (`src/device/dma/lock_port.rs`), channel state of every compute MM2S that runs the word
  clock: it picks the chain's next lock-acquiring BD, times its request from the word clock's per-word sequencing
  cycles, presents the acquire to the tile's arbiter only on the cycles the law allows, and holds the grant for
  the BD until the FSM reaches it (which then goes straight to its data phase, first access at grant + 2). The
  arbiter grants at G - 1 (ACQ_GE there) and a compute MM2S acquire does not see a release resolved in the same
  cycle (`LockRequest::sees_same_cycle_release`), which together make release + 2. The lock port drives the
  channel's STALLED_LOCK; the previous BD's last access is looked ahead one cycle. Constants:
  `mm2s_lock_request_after_grant_cycles` 3, `mm2s_lock_request_trigger_words` 3,
  `mm2s_lock_request_trigger_latency_cycles` 3, `mm2s_lock_release_after_finish_cycles` 4,
  `mm2s_lock_grant_period_cycles` 3, `mm2s_lock_grant_blocked_phase` 2, `mm2s_lock_bds_in_flight` 3, the three
  STALLED_LOCK offsets (`mm2s_lock_stall_before_first_granule_cycles` 1, `..._before_request_cycles` 1,
  `..._before_last_access_cycles` 1; all fitted) and `mm2s_lock_grant_visible_cycles` 1 (measured).
- **Short chains of lock BDs (modelled 2026-11-13).** Probes: chain families k4f (K = 1..5 four-word mids on free
  locks, A's release swept over 18 cases) and c1f..c4f (six 1..4-word mids on free locks). Three terms, each pinned
  by rows: (a) a lock BD starts sequencing no earlier than the lock release (FINISHED_BD + 4) of the lock BD three
  before it (`WordClock`, `mm2s_lock_bds_in_flight`). MEASURED: the sequencing starts implied by the rises and
  finishes -- k4f3's third mid has no STALLED_LOCK blip and finishes at 12, so its first word is sequenced at
  exactly 7, and the 64-word BD behind two, three or five mids starts at 7, 11, 19 (clock floors 72, 76, 84; from
  the fourth mid on every mid finishes 4 cycles after the one before). INFERRED: that 7 / 10 / 13 / 16 / 19 are
  the releases of the lock BD three before (a lock port holding three lock BDs between sequencing start and
  release). (b) DEAD holds every earlier lock BD's release slot: k4f5 is never granted at 20, its fourth mid's
  FINISHED_BD 16 + 4 (MEASURED; that it is that BD's release slot is the same reading as DEAD's). (c) STALLED_LOCK's
  first term is the BD's first granule fully sequenced - 1: c3f's third 3-word mid, first word at 7, blips at 8
  (MEASURED). Evidence: fitted on k4f1/2/3/5, c1f and c3f (148 / 148 rows), predicted before scoring k4f4, c2f and
  c4f (41 / 41); two or four BDs in flight, a release offset of 3 or 5, counting lock-free BDs, DEAD of the
  previous BD only, or the S + 2 rise each lose fitted rows, and none of the three terms moves an in-scope row
  (2348 / 2348). Lock-free chains of up to five 1- or 4-word BDs follow the law unchanged (kn rows; kn1x2, kn1x5,
  kn4x4 held out, 6 / 6). Emulator (integrated
  probe runs, release library): model-free, each silicon k4f row against the emulator run whose release lands on
  the same cycle after g0, 166 / 166 identical (before: 32, k4f1 only; k4f2..5 now 34 / 33 / 34 / 33); every other
  emulator scorer unchanged (sync 34 / 34, phase 428 / 428, chain 554 / 554, model-free 969 / 969 and 170 / 170,
  arb same case 113 / 174, ring 23 / 48, producer collide 1095 / 18 / 9 of 15, sticky32 456 / 143).
- **Word clock, revised (M1g).** The 16-word lead resuming at grant + 3 is replaced: the clock counts granules (a
  BD's short tail is a granule of its own), and while the channel is parked on a lock (the BD at the head of the
  accesses, or the BD it sequences into, acquires a lock and has presented no access) at most 4 started granules
  wait for their access to be presented; unparked it runs on. A BD's first access comes at least 2 cycles after
  the previous BD's last. Silicon: a long-parked channel holds 4 granules whatever their size (a 13-word BD behind
  a parked 4-word one puts the 64-word BD's FINISHED_BD at g0 + 70, where the 16-word lead gave + 69), and a
  lock-free 5..8-word BD finishes 3 cycles after a parked 4-word one, not 2. The fitted model bounds the lead
  always; the ring camp (a core camping on the bank, previous section's tests) needs the clock free once the
  parked BD has presented its access; on every row inside the lock law's scope the two readings agree (they part
  on six 1-word lock-free BDs, outside it).
- **Result (emulator).** Unit tests assert silicon cycles from the captures: every row of the measured table, 45 sync rows
  (pre-BD lengths 0..32), rows of 29 chain designs and the pins of the unmeasured tiles. Integrated runs of the
  probe designs (`--backend emu`, every case once), scored with the silicon rows' scoring on the emulator's own
  release cycles: sync sweep 34 / 34, phase sweep (pre-BD lengths 1..40) 428 / 428, chain designs inside the
  rule 554 / 554 (the core's release of B is not traced; the emulator's own core path, event + nops + 3 or 4,
  gives the same score either way). Model-free, each silicon row against the emulator run whose release lands
  on the same cycle after g0: 889 / 889 sync rows and 163 / 163 chain rows identical (grant, every STALLED_LOCK
  interval, every FINISHED_BD); with the release at FINISHED_BD + 3 more emulator runs share a silicon row's
  release cycle: 1107 / 1107 sync rows, 745 / 745 chain rows, k4f 178 / 178. Same case (no frame mapping), since the head BD's release reaches the lock unit
  at FINISHED_BD + 3 (2026-11-13, below): 26 / 34 sync rows identical (before 1 / 34), LOCK_0_REL and event 0
  equal to silicon's on 22 rows and one cycle early on 12, which are silicon's other resume phase of the held
  core (the same binary gives either run to run; the emulator's run is deterministic). On the short-chain designs (c, kn, k4f; 110 runs) the emulator follows the short-chain rule below on 108
  (before 2026-11-13: 32); the other two (six 1-word lock-free mid BDs, c1n / kn1x6) are where the parked lead and
  the always-bounded fit part (grant release + 2 against + 3).
- **Still different (open gaps).**
  - *Chains of short lock-free BDs: chain depth and egress hold* (families c1n..c4n, kn1x6 / kn4x6 and the depth
    probes, `gen_dma_lock_probe.py chain-build` with `--bd-ids`, `--fill`, `--receiver`, the S2MM's
    STREAM_STARVATION traced; open, measured 2026-11-13, not modelled). Measured on silicon:
    - the stream leaves the MM2S one word per cycle with exactly one idle cycle after each BD (the receiving S2MM's
      starvation level, every chain shape, every run), never interrupted by the late accesses below;
    - the eighth BD of a lock-free chain (chain position 7, not bd_id 7) starts sequencing at W + 14 behind a W-word
      head (18 / 22 / 30 / 38 for W = 4 / 8 / 16 / 24: STALLED_LOCK at start + 2, clock-limited FINISHED_BD at
      start + 65), and every later position two cycles after the one before behind 1-word BDs (20, 22 behind a
      4-word head; 32, 34 behind a 16-word head): the start follows the port timeline of the BD seven positions
      back, not its FINISHED_BD;
    - behind a long head the mids' accesses are held although the port keeps pace: a 16-word head puts six 1- or
      2-word mids' FINISHED_BDs at 8 10 12 14 17 25 (the law: .. 16 18), an 8-word head the sixth at 17, a 24-word
      head the fourth to sixth at 21 25 33, and 5-word mids behind a 12- / 16-word head at 17 26 32 38 / 21 30 36;
    - the waiting BD's lock request is made exactly at its start behind 1-word mids (early grants at the start in
      every such design), one cycle later per extra mid word (behind six 2- / 4-word mids: start + 1 / + 3; the set-5
      grant phases 0 / 1 / 2 / 0 mod 3 for 1- to 4-word mids);
    - with the stream sent to the S2MM one tile north, the BD-level holds (the sixth mid, the waiting BD's start and
      grant) come exactly one cycle later and the mid-drain one (the fifth mid) does not move.
    The emulator holds the eighth BD to FINISHED_BD(p - 7) + 15 nowhere (that form is refuted by the head length)
    and none of the egress holds; its rows are pinned in
    `lock_free_chains_outside_the_law_keep_the_emulators_pinned_cycles`. A five-BD row (kn4x5 t2: granted at 49, the
    law 48) and the position-6 grant phase it shows stay open as well. Candidate structure under test (not yet
    adopted): an egress counted in granule slots and a BD queue freed after the BD's last word has gone downstream;
    the route term is being measured (receivers one, two and three tiles north).
  - *A run's first lock BD* (**modelled** 2026-11-13). Probe: the first-BD lock-fetch design with START_TASK in
    the memory-module trace and the release moved one cycle at a time (`fetch-build --start-task`, 6 cases, 5 runs
    each). Silicon grants it at release + 3 exactly when (release - START_TASK) mod 3 = 1, at + 2 on the other two
    residues (30 / 30 runs); the run-to-run phase of the earlier rows is the release's jitter against the task
    start. Now the run's first lock BD is gridded from the channel's first sequenced word (START_TASK + 7, the word
    clock's start): grantable at no earlier cycle than before, blocked when (grant - that word) mod 3 = 2; the
    emulator's six cases give + 3 exactly at residue 1. The chained form (a lock-free head before the lock BD) already
    followed silicon's residue (+ 3 at residue 2, 14 / 14 rows with the release; emulator 6 / 6).
    Four sibling bridge kernels whose out MM2S starts at boot on its first lock BD and meets a core load after
    the release (`vec_nan_fp32_dense_add`, `vec_nan_bf16_dense_add`, `vec_nan_fp32_dense_sub`,
    `vec_nan_bf16_dense_sub`, Peano, three silicon samples each) show both phases on silicon: CONFLICT_DM_BANK_0
    at the last release pulse + 4 in 2 of 12 samples and at + 10 in 3; the other 7 captures carry no
    memory-module packet from the compute tile and show nothing either way.
  - *Grant phase of a lock BD whose previous BD had refused accesses* (**modelled** 2026-11-13). Probe: a 64-word
    lock BD k, a core camp refusing r = 0..3 of its accesses early (granule 6) or late (its last bank-0 granule),
    then BD k + 1 on a second core-released lock swept over six cycles (`refuse-build`, 48 cases, 3 runs). Silicon
    grants BD k + 1 at release + 3 exactly when (release - BD k's FINISHED_BD) mod 3 = 2, 144 / 144 runs; anchored on
    BD k's grant it fails 11 runs (the late r = 3 rows, whose FINISHED_BD moved by one). For a predecessor served as
    presented the law's request already has that residue; now a lock BD whose predecessor had a refused access
    counts its grid from that FINISHED_BD - 1 (`LockPort::note_finished`). Emulator: 48 / 48 cases on silicon's rule
    (before 45, not rfl3s05 / s06 / s08); 102 of the 103 silicon runs whose release lands where the emulator's does
    are identical in FINISHED_BD, grant, refusals and conflicts (the other is a silicon run that refused one access
    fewer). Producer `collide`: 1098 conflicts, 18 stalls, 9 of 15 transfers colliding (silicon 1098 / 18 / 9 and
    1103 / 20); `sticky32` 442 / 135 (silicon 443 / 136).
  - *A refused last access.* The STALLED_LOCK rise's look-ahead does not foresee a bank refusal of the previous
    BD's last access (not measured).
  - *Core-side offsets* (**fixed** 2026-11-13, next section's "Where the core's access sits"). (The ring
    probe's "release one cycle earlier at k = 0" was a build difference: the k >= 1 cases carry one more bundle
    before event 0.)
  - *A compute BD's lock release* (**fixed** 2026-11-13). In the sync designs the core waits on lock 1, which
    the head BD releases (the runtime sequence only sets the head's start lock), and reached event 0 2 or 3
    cycles early against g0. MEASURED, one trace unit: an S2MM BD's LOCK_0_REL lands at its FINISHED_BD + 3 in
    18 of 18 runs of the chain family's S2MM-releaser designs (s1n08..s1n32), where the emulator applied the
    release in the BD's last data cycle (+ 0). Across units (the core's timer read 1 ahead, as the arb rows
    show): event 0 at the head's FINISHED_BD + 5 in 1132 runs and + 6 in 567 (one in three; 483 cases with
    repeated runs show both), which is a held core arbitrating in the cycle the release arrives, issuing the
    held bundle the cycle after, and one later off its 3-cycle resume phase (the stall's start predates the
    trace, so the phase is INFERRED, not seen). That the MM2S release takes the S2MM's + 3 is INFERRED from this
    fit (no capture traces lock 1; a sync build with LOCK_1_REL in the memory unit's event list would show it).
    Now the release reaches the target tile's lock unit at FINISHED_BD + 3 (`dma.bd_lock_release_after_finish_cycles`,
    `Tile::send_dma_lock_release`), compute tiles only. Tests `a_core_held_on_a_dma_release_issues_on_silicons_cycle`,
    `a_compute_s2mm_release_reaches_the_lock_unit_three_cycles_after_its_finish`. Every compute DMA-to-core
    and DMA-to-DMA lock hand-off is 3 cycles later than before. Memtile and shim releases are unchanged
    (unmeasured).
  - *Not measured, unchanged.* Memtile and shim DMA lock timing (request, grant, STALLED_LOCK) and every S2MM
    acquire keep the FSM's path and the arbiter's same-cycle order (pinned by tests); so do the core's acquires.
    A channel with two locked BDs waiting at once, and a BD releasing its own successor's lock on one channel, are
    not measured.

## Compute-tile bank arbitration (modelled 2026-11-12, NPU1 silicon)

Probes: `tools/gen_dma_lock_probe.py` arb cases (the chained lock-fetch design with 1 to 6 consecutive core loads
or stores on one physical bank, 0 to 4 extra loads on the bank before the lock release; 56 cases, 174 rows) and
ring cases (a core camping 90 or 120 cycles on bank 0 against each of MM2S 0 / 1 and S2MM 0 / 1, two channels
together, and extra initialised words in bank 0; 15 cases, 48 rows); the producer probe (`collide`, `sticky8` to
`sticky32`). Cycles are counted from the MM2S lock grant g.

- **Silicon.** Each physical bank counts the accesses it has served: core loads and stores, the accesses of all
  four DMA channels, and the configuration image's writes, once per initialised 16-byte line (1 to 4 extra words
  in one line move the count by 1, two lines by 2). Refused requests add nothing. When the core and a DMA channel
  want the bank in the same cycle the DMA is served if the count is 2 modulo 5, the core otherwise; the loser
  retries the next cycle, so a DMA is refused at most 4 cycles in a row. With the count started at 0 at reset,
  the probe design reaches its window with 20 served accesses on bank 0 (8 + 2 image lines, 2 crt0 stores, 8
  frame stores) and 18 on bank 1, which places the core's losses at g + 7 and g + 47 on bank 0 and g + 27 on bank
  1, as measured. All four channels win on the same single residue, and serving one advances the count the
  others see (two channels with the core camping: 16 core losses exactly 5 cycles apart). A refused S2MM write
  retries the next cycle like an MM2S read, its refused cycles raise CONFLICT_DM_BANK, and
  DMA_S2MM_n_MEMORY_BACKPRESSURE does not fire for it. CONFLICT_DM_BANK marks every cycle in which two requesters
  wanted the bank (the cycle the DMA won included; two DMA channels meeting with the core silent included) and
  never a lone retry; a core loss is one MEMORY_STALL, stamped one cycle after the losing cycle's conflict.
- **Model.** `BankAccessCounts` on the tile, advanced by the arbiter (`bank_arbiter.rs`) for every access a bank
  serves and by the configuration path (`DeviceState` data-memory writes, and `AieElf::load_into` for an ELF
  loaded directly) once per 16-byte line written; the
  core-versus-DMA decision reads it against `BANK_ARBITER_PERIOD` 5 (measured) and `BANK_ARBITER_DMA_RESIDUE` 2
  (fitted once; banks 0 and 1, whose histories differ by two stores, need the same value). The count is counted,
  not offset: the emulator's own image load and prologue give 20 and 18 at the probe's window. A core access
  asks for its bank in its memory stage, issue + 4 (since 2026-11-13; below), and a lost bank cycle is a frozen
  core cycle with MEMORY_STALL in it. A bundle held in its cycle (the lock request's frozen arbitration cycle,
  a refused acquire) has issued no access for that cycle, so a camp of loads right after a release has one
  cycle without a core access (the bundle that would have issued in the arbitration cycle). The urgency override
  (`URGENT_WATERMARK`) is removed.
- **Result (emulator).** Arb and ring probes 208 of 222 rows. Since the lock grant law (previous section) the
  emulator's grant is release + 2 or + 3 as on silicon, but a row still has to be compared with a shifted case:
  silicon's phase varies run to run for the same case (m = 0: 99 rows + 2, 51 rows + 3) while the emulator's is
  fixed per case. Since the core-side placement (2026-11-13: a core access k bundles after the release meets
  its bank at release + k + 1, as on silicon) the same case, unshifted, matches 113 of 174 arb rows: every row
  whose silicon grant phase is the emulator's (113 of 113), none of the 61 with the other phase; ring 23 of 48
  (21 of the 28 at the emulator's phase; the others are the S2MM-fed and two-channel camps below). Before it:
  24 of 174 and 10 of 48 (with a shifted-case mapping 174 of 174 and 34 of 48). Producer probe `collide`
  (conflict cycles counted level-expanded, silicon 1098 / 1098 / 1103; MEMORY_STALL 18 / 18 / 20; 9 of 15
  transfers colliding, stalls per transfer 2 1 0 3 0 repeating): before the lock law 1077 conflicts (another
  count), 18 stalls; with the lock law 0 / 0 / none; with the core-side placement 603 / 24 / 12, 2 2 2 2 0;
  with the late-granule gap (next section) 1095 conflicts, 18 stalls, 9 of 15 transfers colliding,
  2 1 0 3 0 repeating, 12 of 15 transfers identical to silicon contact for contact and finish (`collide` is a
  FITTED row since then: the gap's form was set by it). With the compute BD release at FINISHED_BD + 3
  (2026-11-13): 1100 / 20 / 9; the first grant after the head BD's finish is 285 as on all three silicon runs
  (was 283), and the first transfer takes silicon r3's contacts (4 stalls) where it took r1 / r2's (2): 12 of
  15 transfers identical to silicon r3, as before to r1, the same three transfers one cycle off. Held out:
  `sticky32` (silicon 443 / 136): 456 conflicts, 143 stalls, unchanged by the gap (with the lock law, same
  count: 545 / 157); 443 / 136 with the release at FINISHED_BD + 3; no conflict run longer than 5 cycles, every 5-cycle run ends with a core stall.
  With the grid anchored on a refused predecessor's FINISHED_BD (2026-11-13, lock-grant section): `collide` 1098 / 18 / 9
  (silicon r1 and r2 exactly), `sticky32` 442 / 135.
- **Still different (open gaps).**
  - *DMA-versus-DMA order.* Which DMA channel is served when two or more want the bank in the same cycle is not
    measured (the ring probe's meetings show the conflict, not the winner); the arbiter keeps its rotor for it.
    In the two-channel camps the emulator splits the turns differently (MM2S 0 finishes at g + 52, silicon
    g + 58; with an S2MM, MM2S 0 at g + 82 against g + 62), though the 16 core losses fall where silicon's do.
  - *Producer collisions* (late-granule gap **modelled** 2026-11-13). After an access the bank refused, silicon's
    MM2S asks for its next granule no earlier than two cycles after the refused one was served when that granule
    was not accessible before (the producer fit's max(lattice slot, service + 2); in word-clock terms the
    lattice slot is the cycle the granule becomes accessible); a granule accessible earlier follows the next
    cycle (ring camps), and an access served at once leaves the next one to the word clock (every uncontended
    lock-timing row). `DMA_MM2S_LATE_GRANULE_ACCESS_GAP_CYCLES` (2, fitted), `WordClock::late_gap_allows`; test
    `the_emulated_mm2s_against_the_producer_march_matches_silicon` (types A to D, contact for contact). Still
    open: the third transfer of every five (the type-C transfer after a type B: 3 of 15) is granted one cycle
    later against the core's stores than on silicon (grant to grant 1809 then 1806 against 1808 / 1807), so
    its first contact is 15b0x2 19b1x2 instead of 15b0x3 20b1x2 and it finishes at g + 246, not 247; a 3-cycle
    phase (the lock port's grid or the core's resume phase on lk_empty) is the likely seat, not shown. The
    silicon MM2S stream backpressure at grant + 6 and + 15 (the stream goes to the shim S2MM, column 0) has no
    emulator counterpart and is not needed by the 12 matching transfers: the channel has staging slack (15 to 18
    of 27 words), so a one-cycle hold does not move its granule requests.
  - *Where the core's access sits* (**fixed** 2026-11-13). Silicon, one trace unit at a time: in the core
    module a load or store that loses its bank raises MEMORY_STALL at its issue + 4 (82 of 82 loss rows), the
    stamp a gather's frozen memory cycle has (counter reads put that freeze at issue + 4); llvm-aie gives loads,
    full-word stores and gathers MemoryCycles<[5]>. In the memory module every conflict lies on a core access
    cycle LOCK_0_REL + k + 1 + i (147 of 147 rows), which with the access at issue + 4 puts the release one
    cycle after the core's arbitration cycle. The emulator presented the access at issue (4 early), stamped
    MEMORY_STALL a cycle after the loss, and released at the arbitration cycle (1 early): 4 - 1 = the 3 seen.
    Now an issued bundle's accesses ask for their banks at issue + 4 on the cycle clock (a lost one freezes
    the core in that cycle, MEMORY_STALL stamped there) and a core release reaches the lock unit the next
    cycle (`Tile::send_core_lock_release`; cores see it when they did, the DMA from that stamp + 2). The "release 2
    early / access 3 early against event 0" figures mixed the two units' timers: across units the emulator now
    reads LOCK_0_REL - `event #0` = 4 (silicon 5) and MEMORY_STALL - conflict = 2 (silicon 1). The split itself
    rests on the gather's stamp (MEMORY_STALL in the frozen cycle) and the toolchain's E5, not on these: any split
    that fits the same-unit rows leaves equal residuals. Inferred from them: the intra-tile timer offset
    (`BROADCAST_INTRA_TILE_MEM_OFFSET`, then 2, calibrated on add_one) reads 1 on these probes; it is 1 since (see `broadcast.intra_tile_mem_offset` in the timing-constants coverage table; the memtile keeps 2 under its own constant). Unmeasured: a vst.srs / vst.conv (E7) or part-word store
    (E5, E11) arbitrates at E5 like the rest; an access lost while the core is held on a lock costs no extra
    cycle; the data itself still moves at issue.
  - *Rows outside the probes' shape.* An access in flight reaches its bank whatever the core does meanwhile
    (lock arbitration cycle, held acquire, cascade or stream hold), as the gather's memory cycle does on silicon;
    measured for the gather only.
  - *Not measured.* Neighbouring cores accessing this tile's memory, vector accesses spanning two banks, both core
    load ports at once against a DMA, a configuration write by a single-word register write (counted as one
    access, like a line), whether a run that does not reset the array keeps counting, memory-tile banks (never
    arbitrated), and why the period is 5.
## Compute S2MM after a lock grant (modelled 2026-11-13, NPU1 silicon)

Probes: `tools/gen_dma_lock_probe.py` sl family (an own-tile MM2S feeds S2MM 0, whose 64-word BD is parked on a lock the
core releases at R; chained after a lock-free 4-word head or the task's first BD; release swept over six cycles; held
word count N = 4..20; one core load k cycles after the release on either bank; a core camping on bank 0, also routed to
S2MM 1; 71 cases, 3 runs each) and `objectfifo_turns_walkers` re-traced with lock selectors (both channels' grants and
finishes and the bank conflicts on the memory-module timer, 3 runs of 41 iterations). G is the cycle STALLED_LOCK falls.

- **Silicon.** G = R + 2 or R + 3, never earlier (213 / 213 runs). The parked channel holds 16 words; it writes them a
  granule per cycle at G + 2..5, alternating banks, and the streamed granules every 4 cycles from G + 10 (a core load
  meets them there and nowhere else the sweep covers); FINISHED_BD one cycle after the last write (N = 4, 8, 12, 16, 20:
  G + 3, 4, 5, 6, 11; 64 words G + 55, chained and first BD alike); the feeding MM2S's backpressure falls at G + 6; one
  STREAM_STARVATION cycle at G + 2 when the BD wants more than it holds (N >= 20), none for N <= 16. With the camp the
  writes of bank 0 are refused up to 4 cycles, but the ingress side does not change (backpressure fall + 6, starvation
  + 2), and the finish moves to G + 57 (G + 56 in the other grant phase). The grant's phase: in walkers the grant is
  R + 3 exactly when (R - the channel's previous grant) mod 3 = 2, 119 / 119 iterations (anchored on the previous
  FINISHED_BD it fails); in sl the channel is parked before the trace starts and the anchor is not visible.
- **Model.** `S2mmPort` (`src/device/dma/s2mm_port.rs`): the S2MM acquire sees a release from the cycle after it
  arrived and holds the grant one cycle after the arbiter gives it, as the MM2S lock port, on the same 3-cycle grid
  counted from the previous grant + 2 (`dma.s2mm_lock_grid_after_grant_cycles`, fitted on walkers) or, for the task's
  first lock BD, from where a compute MM2S's word clock would start (a choice). First write at G + 2
  (`dma.s2mm_lock_to_write_cycles`), with no lock-acquire countdown, memory latency or start bonus. The granule writer
  (`DmaEngine::compute_s2mm_cycle`): a granule leaves the ingress once each of its words has been there 4 cycles
  (`dma.s2mm_ingress_to_write_cycles`, fitted: only entry-to-write is constrained), one per cycle, into a write queue of
  4 granules (`dma.s2mm_write_queue_granules`; at least 3 needed by the camp, 4 a choice) written from its head one
  per cycle, a refused write retried while the ingress side runs on; after a refused write a granule not ready before
  it comes 2 cycles later (the MM2S late-granule gap); FINISHED_BD one cycle after the last write
  (`dma.s2mm_write_to_finish_cycles`). STREAM_STARVATION while the writer runs: the BD wants words its ingress does not
  hold, the ingress has room and no word entered; one level with the lock-wait starvation (inferred: the G + 2 pulse
  is the first write freeing room the registered ready shows a cycle later). The one-BD-lookahead accept gate stays
  open while the previous BD's words are being written. Applies to every plain compute S2MM BD (lock or not);
  decompression, finish-on-TLAST and out-of-order keep the word path.
- **Result (emulator).** sl, each silicon run against the emulator run of its case at the same grant phase: release
  sweep 14 / 14, N sweep 9 / 9, k sweep 91 / 91 identical in finish, backpressure fall, starvation and both banks'
  conflict stamps (before: no run at silicon's phase, the grant was R + 0; finish 50 / 58, backpressure 5 / 13, no
  starvation); with the task anchor shifted (a local debug build, not committed) the emulator gives the other phase,
  and every one of the 213 silicon runs has an identical emulator run at its phase: fit 201 / 201, camp on S2MM 0
  6 / 6 (held out until its finish missed by one; the late-granule gap closed it).
  Walkers: S2MM grant equal on 36 / 38 / 36 of 40 iterations (before 0), FINISHED_BD + 55 / 120 / 185 against
  silicon's + 56 / 121 / 186 (before + 50 / 115 / 180), iteration 718 cycles against 719 (before 711), conflict stamps
  19 at silicon's slot (REL(out) + 6) against 0 / 1 / 1 (before 34 at + 4 / + 8). Sync family S2MM finish after the
  waiting BD's grant + 77 (22 cases), + 78, + 79 (before + 73..75). The ring rows `rgs0` / `rgs1` now match at the
  emulator's phase (ring 30 / 48, before 23).
- **Still different (open gaps).**
  - *S2MM 1 route.* The camp routed MM2S 0 -> S2MM 1: silicon's backpressure falls at G + 15, a second starvation cycle
    at G + 9, and the streamed writes one cycle later from granule 6; the emulator gives the S2MM 0 numbers (+ 8,
    + 2 only). The finish matches. Route-side, not modelled.
  - *A parked feed's restart after quiet switches (walkers' last cycle; measured 2026-11-13, not modelled).*
    Walkers: silicon's FINISHED_BDs are + 56 / 121 / 186 after the grant once its memtile has been quiet for
    248..512 cycles before the parked memtile MM2S resumes, + 55 / 120 / 185 while the memtile S2MM is still filling
    from the shim, at either grant phase; the memtile port resumes one cycle later in the same iterations (memtile
    timer) and the finish follows it. The memtile-fed probe (`gen_dma_lock_probe.py mq-*`: a parked memtile MM2S
    into a lock-held compute S2MM, 51 silicon runs) measures three levels: + 54 when a compute trace packet had
    crossed both the compute and the memtile switch 4..98 cycles before the restart, + 55 with the memtile busy (a
    shim fill flowing, or ended 100 cycles before) and the compute switch quiet 1112+ cycles, + 56 with both quiet
    (memtile 369+, compute 1131+). In every run the feed's first memtile North-master beat moves with the level
    (grant + 6 / 7 / 8, one value per level over all runs) and the finish is 48 cycles after it; one
    STREAM_STARVATION cycle at grant + 9 appears in every + 56 run only; a core load dates granule 6 at grant + 18
    at + 55 and + 19 at + 56. The + 55 / + 56 step is on the memtile side: only the memtile's activity differs
    between those runs. Inferred, fitting every run: each switch on the feed's path costs the parked feed one cycle at
    its restart once it has been quiet for its threshold (the wake-latency row below, extended to switches that hold
    parked words), memtile threshold in (100, 248], compute in (about 100, 1112]; both awake is + 54. The emulator
    gives + 55 in every state: its idle detector counts a port that holds a word as active (`begin_routing_cycle`
    seeds `cycle_active` from `has_data`), so no switch gates while the feed is parked, and its S2MM constants
    (fitted on the sl probe, whose compute switch silicon had let go quiet) already hold one such cycle; its feed
    also has 2 cycles of slack where silicon's finish follows the memtile restart one to one. A second probe
    (`mr-*`: the quiet stretch timed by the core against a compute-only, memtile-only or both-switch waker) is
    built to separate the two switches and place both thresholds.
  - *The task's first lock BD's grid* is a choice (no row shows its anchor).
  - *Lock-free first BD.* The start costs before the first write of a lock-free first BD (memory latency and start
    bonus) are unchanged; the S2MM finish of the START_TASK probe is + 79 / 80 after the release against silicon's
    + 80 / 81 (before + 75 / 76).
  - *Not measured.* Where the 4 cycles from ingress to write sit, why silicon pulses STREAM_STARVATION at G + 2, a
    write queue deeper refusal, the memtile and shim S2MM.

## Open timing rows (silicon vs emulator, 2026-11-12)

Silicon rows are the probe runs of the read-bandwidth section above (two runs of each case) and the trace and
bridge captures named in each row. Compute-tile cycles are counted from the MM2S START_TASK; the memtile is one
cycle later throughout.

| Gap | Model vs hardware | Where | Status / rationale | Obs. |
|-----|-------------------|-------|--------------------|------|
| MM2S first-offer one-cycle refusal | Silicon refuses the first offer of a BD started from idle for one cycle, visible as a 1-cycle `STREAM_BACKPRESSURE` pulse at START_TASK + 11 + min(N, 4) (+1 on the memtile; N = 1, 2, 4, 8, 16, 40: 12 / 13 / 15 / 15 / 15 / 15), and the first word reaches the S2MM one cycle later than when it is absent (19 against 18 compute, 20 against 19 memtile). Whether it is present is not predictable: present in both runs for a single BD of N = 1..8, 16 and 40 on both tiles, in the second to fourth task of four one-BD tasks (`t4x4`) on both tiles, and in the second task of two (`t8x2`) on both tiles; absent in both runs for four chained BDs (`c4x4`) on both tiles, for the first task of `t4x4` on both tiles and for two chained BDs (`c8x2`) on the memtile; present in one run and absent in the other for `c8x2` on a compute tile and for the first task of `t8x2` on both tiles. The emulator never refuses. | `src/device/dma/engine/stepping.rs` (`staged_mm2s_cycle`) | **OPEN, mechanism unknown; not modelled by heuristic.** `n4` (present) and the first `t4x4` task (absent) differ only in what the runtime sequence does after the start. Possibly related to the adaptive clock gate below (under investigation). | confirmed (presence per case) |
| S2MM `STREAM_STARVATION` while lock-stalled | A S2MM whose BD waits on a lock reads `STREAM_STARVATION` high from before the MM2S starts until its first word arrives (with the first-offer refusal present it falls at START_TASK + 19 on a compute tile and + 20 on the memtile for N >= 4, + 16 / 17 compute for N = 1 / 2; one cycle earlier when the refusal is absent). | `src/device/dma/engine/stream_io.rs` | **OPEN.** The emulator raises no starvation in that state, so the level has no counterpart in a trace of such a tile. | confirmed |
| S2MM `STALLED_LOCK` rise after START_TASK | On silicon `STALLED_LOCK` of a S2MM whose BD waits on a lock rises 9 cycles after START_TASK (6 for N = 1 and 7 for N = 2, length-dependent), three cycles after its `STREAM_STARVATION` (+6), i.e. the lock path has a BD-fetch latency (all 138 runs, both tile kinds). | The emulator raises `STALLED_LOCK` at START_TASK + 0: no BD-fetch latency on the lock path. | `src/device/dma/engine/stream_io.rs` | **OPEN.** The two levels are anchored separately; neither is modelled at its silicon offset. | confirmed |
| Adaptive stream-switch clock gating: wake latency | Which switches hold a word one cycle after an idle stretch (no-data trace probe, trace packets 135..400 idle cycles apart against 48): core trace -> memtile -> shim S2MM +2, the same route on through the next column's shim +2, memtile trace -> shim +1, shim trace -> next column's shim +0; the shim's own packets into its own S2MM +0 after thousands of quiet cycles (three bridge captures). So a quiet compute or memtile switch costs the first word one cycle, the source's own switch included, and a quiet shim switch costs nothing. The idle threshold (128 in the emulator) and the backpressure pulse the compute / memtile wake raises on silicon are separate rows. | `src/device/array/routing.rs` (`step_tile_switches_impl`), `src/device/clock_control/mod.rs` | **MODELLED 2026-11-12** for those charges: a crossing word into a quiet compute or memtile switch waits one cycle (it used to cost nothing when it arrived after the first routing pass), and a quiet shim switch does not hold trace words or words from North and West. A data word of the shim's own DMA MM2S into a quiet shim switch is still held one cycle: unmeasured, kept from the earlier model (row below). The threshold is still not measured; walkers bounds a memtile's (a charge after 248 quiet cycles, none after 2). **Open (2026-11-13):** a parked feed restarts later after quiet switches on its path ("Compute S2MM after a lock grant", the memtile-fed probe: three levels, the memtile's step measured; inferred one cycle per quiet switch, memtile threshold in (100, 248], compute in (about 100, 1112]); the emulator counts held words as activity and never gates a switch that holds one. | confirmed (trace words, North / West entries) |
| Shim switch: its own DMA's first data word after a quiet stretch | Unmeasured: whether a quiet shim switch holds the first word its own DMA MM2S offers (the entries measured so far are trace words and words from North and West, none of which it holds). | `src/device/array/routing.rs` (`step_tile_switches_impl`) | **OPEN.** The emulator holds it one cycle, the behaviour of the earlier model. | confirmable: a host-fed shim MM2S transfer after a quiet stretch against one behind a concurrent transfer |
| Memtile self-chained locked S2MM to MM2S hand-off | `runtime_cumsum` memtile (0,1): one 16-word BD, self-chained, re-streams k rows to the core out of the buffer the S2MM fills. Rows start 23 to 24 cycles apart on silicon (a 16-word run plus a gap of 7 to 8 on the S2MM port) against 22 in the emulator (gap 6). Silicon splits the first row of each group of rows 8 + 1 idle + 8 (6 of 7 groups in one capture, 7 of 7 in the other); the emulator never does. `PORT_RUNNING_4` interval count 46 to 48 (3 captures) against 38. | `src/device/dma/engine/` (memtile S2MM / MM2S lock hand-off) | **OPEN, mechanism unknown.** The count had matched by coincidence (43 before the 2026-11-11 timing steps) while the structure was further from silicon. | confirmed |
| `objectfifo_deep_memtile_hold`: emulator stalls near the end of the stream | Memtile (0,1) S2MM 0 takes the 80 x 64-word transfer from the shim (the buffer is 40 deep, so the producer stalls only when buffer b + 40 arrives before buffer b is released). Silicon: 80 `PORT_RUNNING_0` runs of exactly 64 cycles in two captures; the third counts 81, one lock-stall split in buffer 79 (16, 16 idle, 48), the same shape as the emulator's. The two captures without a stall have gaps at buffer boundaries in the producer's stream (up to about 1450 cycles, at most 34 buffers resident, so not a memtile stall; inferred upstream, host or DDR side), which the emulator's gap-free producer never has. The emulator splits one or two of the last runs into 16 cycles, 65 to 69 idle cycles, then 48 (81 and 82 runs in the two emulator versions compared). | The emulator's consumer is not behind silicon but ahead of it (about 14 cycles from the first backpressure pause on: pause 17 against 30 cycles; round trip around the memtile, core-tile S2MM and core lock -3 on silicon, -17 in the emulator); arithmetic on the measured schedules says a producer with silicon's one 76-cycle gap would not stall in the emulator at buffer 78 or 79 (by 1 cycle). | `src/device/dma/engine/` | **OPEN, mostly host timing.** Most of the difference is the producer's gaps, not the memtile; silicon does stall when no gap occurs (third capture). What is left is the consumer being about 14 cycles early after the first pause (the memtile to core-tile hand-off loop). | confirmed |

## Stream buffering and latency (modelled 2026-11-04, NPU1 silicon)

Two silicon facts settle the stream path. **Buffering:** one more tile between a source and a held
consumer adds exactly 8 words (R4-D3, above), AM020 ch.2:92's "8-deep" external-to-external
crossing: slave FIFO 4 + external master FIFO 4. **Latency:** a core-to-core round trip is 32 / 40 /
48 cycles at 1 / 2 / 3 hops (`tools/gen_stream_rtt_probe.py`, r1 == r2), +4 cycles one way per hop:
the same crossing's "4-cycle latency" and nothing for the wire. AM020's crossing table is the sum of
its two ports' latencies and depths, so the latency is time spent inside the FIFOs, not storage on top
of them. The emulator had counted it as storage (a crossing's budget was latency + master FIFO, the
inter-tile wire was a 2-cycle delay line with its own 2 words of credit): 14 words and 6 cycles per
hop.

The model now (`src/device/stream_switch/`, `src/device/array/routing.rs`):

- a word crossing a switch holds a slot of its slave port for the slave's 2 cycles
  (`STREAM_SLAVE_PORT_LATENCY`) and of its master port for the rest (`StreamPort::reserved`); a full
  master holds it, and the slave's slots, so the slave refuses its producer;
- the inter-tile wire has latency 0 (`INTER_TILE_HOP_LATENCY`) and no storage;
- sources hand off into the slave port: the core's stream output buffer (3 words since 2026-11-11, 1 before)
  and the MM2S's 1-word hand-off (`MM2S_OUTPUT_REGISTER_DEPTH`, was a 4-word second slave FIFO, and since
  2026-11-11 part of the MM2S's own Running / backpressure state); a staged MM2S word carries the data
  read when its granule was fetched;
- the S2MM ingress is 15 + 1 (above; 15 + 3 until 2026-11-11), the core's stream input 1
  (`CORE_STREAM_INPUT_DEPTH`, derived from the core -> core count: 18 - 3 - 4 - 8 = 3, less the Core master
  port's 2; it was 3 against a 1-word producer output until 2026-11-11);
- the switch FIFO route adds its 16, the FIFO slave port's 4, and a 1-word output register between
  the two (`StreamSwitch::fifo_out_regs`): +21, no latency of its own (round trip +4 on both).

Which structure inside the S2MM, the core input and the FIFO stage holds the words AM020 does not
name is not separated (only sums are observed); each is a measured total, not a fit to one point.
The TCT transit at a sync now charges one external crossing (4 cycles) per row instead of 2
(`npu/executor.rs`).

**Open: the round-trip intercept.** The emulator's round trip is 40 / 48 / 56 against 32 / 40 / 48:
the slope is exact, but every one-way trip is 4 cycles long somewhere outside the inter-tile hops --
in the core's `mov ms` / `mov r, SS` timing or in the local (Core) ports' latencies. One probe that
separates them: the same round trip with one end a DMA (a held S2MM timestamped through trace) instead
of a core.

Reproduce: `python3 tools/gen_stream_rtt_probe.py generate DIR --fifo`, `python3 -m harness suite
build|hw|emu DIR TAG`, `python3 tools/gen_stream_rtt_probe.py summarize DIR TAG` (steady round trips);
the R4-D3 points with `tools/experiments/r4d3_capture.py` (one design per call, `--emu` for the
emulator).

Reproduce: `python3 tools/gen_stream_absorb_probe.py generate DIR`, then `python3 -m harness suite
build|hw|emu DIR TAG` and `... summarize DIR/runs/TAG/<case> 96` (cycle stamps differ run to run, so
the byte compare does not apply; the summary is the absorb count and whether all 96 words arrived in
order). Before the fix the `fifo` case wedged (`wedge_recovered`, `Kernel did not complete`). The
core -> DMA wedge reproduces only with the R4-D3 xclbin (the topology harness has no core -> DMA flow).
