---
class: dma-stream-resources
subsystem: DMA engine + stream-switch finite-resource / consumer-pacing modeling
posture: optimistic-where-strict -- the model assumes generous or infinite resources (deep queues, unbounded token buffers, freely reusable BDs) where silicon is finite and will stall or wedge
status: stream buffering and per-hop latency pinned by silicon and modelled 2026-11-04 (see "Stream buffering and latency": R4-D3 data race 25/25, core-core absorb, S2MM backpressure windows, round-trip slope all EMU == HW); open: core-to-core round-trip intercept (+4 cycles per direction), TCT token depth, MM2S first-granule pulse, memtile strided-BD cost, BD-driven (de)compression data; send/recv port cadence substantially resolved (recv exact); memtile DMA stream-bound (AM020 ch.5, HW 2026-07-15); DMA memory-pressure events driven by the staging FIFOs, both EMU == HW on the measured workloads; MM2S BD retirement modelled 2026-11-10 (a compute / memtile BD finishes when its last word is staged; no source gate before an un-armed S2MM; memtile MM2S staging 29 + 1 fetch cycle modelled 2026-11-10; open: paths into an idle shim from the north +4, next-BD lock pre-acquire)
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
| `DMA_S2MM_n_MEMORY_BACKPRESSURE` window shape | Per steady-state lock-stall window, backpressure opens **+15** cycles into the stall and closes **1** cycle after it, EMU == HW on every steady window (of_q0_rich rebuilt from its traced MLIR, 2026-11-04: HW r1/r2 and EMU all `(len 85, onset 15, held 71, end +1)`; the 2026-07-14 capture's 39 windows also +15). | `src/device/dma/engine/stream_io.rs` (`note_ingress_offer`, `input_fifo_capacity`), `src/device/array/routing.rs` (`route_tile_switches_to_dma`) | **CLOSED 2026-11-04.** The S2MM ingress is a 15-deep FIFO (`DMA_S2MM_INGRESS_FIFO_DEPTH`, which the event watches) behind a 3-word skid (`DMA_S2MM_INGRESS_SKID_DEPTH`): 18 in all, as the R4-D3 data race requires, with the event asserting while a beat is on offer and the FIFO's level after the cycle's drain is 15. Formerly +16 (one 16-deep FIFO, onset counted at the refusal) and +2 at the close (the registered start-of-cycle level, which still gates TREADY). Evidence `build/experiments/harness/dmastream/bp/`. **Do NOT compare run totals against `of_q0_rich` captures** (the trace holds a window of the kernel's iterations; see `xdna-emu-priv: research/findings/2026-07-13-memory-stall-bank-arbitration.md`). | confirmed |
| MM2S egress staging depth (25) and the route downstream of it | `DMA_MM2S_EGRESS_FIFO_DEPTH = 25`, **measured** (R4-D3, Phoenix NPU1, 2026-10-28) by data race: overwrite the source buffer while the consumer is held and count words still carrying the old pattern. | `crates/xdna-archspec` (`DMA_MM2S_EGRESS_FIFO_DEPTH`), `src/device/dma/engine/stepping.rs` (`next_granule_fetch`, `staged_mm2s_cycle`), `ChannelContext::egress_staging`, rig `tools/experiments/r4d3_*.py` | **CLOSED 2026-11-04, EMU == HW on all 25 route x phase points** (staging + route; see "Stream buffering and latency"). Two defects hid the staging from the race: staged words were re-read from memory at drain time (now read at fetch time, `staged_words_carry_the_data_read_at_fetch_time`), and the MM2S output buffer was a 4-word second copy of the slave FIFO (now the 1-word hand-off register, `MM2S_OUTPUT_REGISTER_DEPTH`). A memtile MM2S stages 29 (`DMA_MEMTILE_MM2S_EGRESS_FIFO_DEPTH`, modelled 2026-11-10, see "MM2S BD retirement"). | confirmed |
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
granted: the channel holds 68 words. `dma.shim_s2mm_ingress_depth` (65 in the emulator's accounting) opens once
the channel holds a task. The compute and memtile numbers above (hop 8, ingress 18, egress 25) and the route
absorb counts are unchanged; the shim path itself is one word longer in the emulator (9, silicon 8).

**Open.** The MM2S backpressure pipeline (depth and `Cur_BD` of a chain stuck behind a stalled consumer; only the
2-word chains and the 57-word threshold are probed), the status of a TCT stall, how chain BDs and queued tasks
share the pipeline when both are present, the exact budgets and the 4-word per-BD overhead, absorb length 75, the
shim path's extra word.

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
TLAST, the BD completes when its last granule is fetched, the next BD fetches in behind, and the staging drains
one word per cycle whatever the FSM is doing. The BD-boundary bubble (`off1`) is a port-side idle cycle after a
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
| memtile -> idle shim S2MM | 48 | 44 (40 before) |
| prod -> idle shim S2MM | 52 | 48 |
| consA -> idle shim S2MM | 60 | 56 |

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

**Memtile MM2S staging (modelled 2026-11-10).** A memtile MM2S stages 29 words, not the compute tile's 25,
and its fetched words reach the stream one cycle later (`DMA_MEMTILE_MM2S_EGRESS_FIFO_DEPTH`,
`DMA_MEMTILE_MM2S_FETCH_LATENCY_CYCLES`). Two silicon probes, two runs each identical:

- start-phase races (`race_*_loop_p1`..`_p3`, BD start 0..3 words into a granule): memtile -> own S2MM
  52/51/54/53, compute 48/47/50/49. The fetch rule gives that pattern for depth 29 only (28 and 30 each
  break a phase). Emulator now 52/51/54/53 and 48/47/50/49.
- backpressure onset (`tools/gen_path_onset_probe.py`: one 96-word BD into a lock-held S2MM, the source
  tile's DMA traced): the sustained STREAM_BACKPRESSURE rises 38 / 39 cycles after START_TASK for a
  compute / memtile source into its own S2MM and 47 / 48 one hop away; every memtile edge, the first-granule
  pulse included, is the compute one plus one cycle. Four more words downstream would move the held onset
  4 cycles, so the words are upstream of the stream port. Whether a deeper staging or a granule in flight
  holds them is not separated; the model counts the in-flight granule in the 29.

The race table's memtile rows now read 52 and 60 in the emulator (memtile -> idle shim 44: the shim
half below). Still open from the onset probe: the emulator's onset is 2 cycles early on the same tile (36 /
37) and 3 one hop away (44 / 45), it has no first-granule pulse (silicon: one 1-cycle pulse at +15, a second
at +24 one hop away), and a lock-stalled S2MM on silicon also reads STREAM_STARVATION until the first word
reaches it, which the emulator does not raise.

**Open (not modelled):** every path
into an idle shim S2MM from the north holds 4 more (memtile South master or shim North slave: not separated;
the shim's own MM2S -> S2MM loop holds 8, one less than the emulator). The hop slope is 8 words on both
sides. On `of_q0_rich` neither moves the depth (its drain BDs are 32 words); a drain BD of 41..48 words
would. Also open: a stalled MM2S has already acquired the locks of BDs beyond the one it works on (compute
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
   `CORE_STREAM_INPUT_DEPTH = 1` (AM020 ch.2:143: the core stream interface has no FIFO). Test
   `array::tests::core_stream_input_is_bounded_and_backpressures_the_switch`.
4. **The switch FIFO did not exist**: master FIFO:0 and slave FIFO:0 were port types with no link,
   so a flow through `FIFO:0` delivered nothing. The master port is now the 16-deep FIFO
   (`stream_switch.switch_fifo_depth`, AM020 ch.2:74) and `advance_switch_fifos` moves a word per cycle
   to the paired slave port (tests `the_switch_fifo_*`).

The core output buffer depth was 4 ("a modelling choice, not a measurement"); AM020 gives the core
interface no FIFO and silicon pins it: a core streaming into a held DMA on its own tile writes 26 words
before stalling (EMU 30 at depth 4, 27 at the 1-word hand-off register now used; 26 since the 2026-11-04 buffering model).

**Absorb counts, silicon vs emulator** (words the source writes before it stalls on a held consumer;
`core_h*`/`h0_*` are the R4-D3 xclbins under the bridge runner, `direct`/`fifo` the harness suite
`gen_stream_absorb_probe.py`, 96 words, consumer held ~28000 cycles; HW stable over r1/r2 and the
earlier 3 runs). Emulator column at the 2026-11-04 model ("Stream buffering and latency" below):

| Design | Silicon | Emulator |
|---|---|---|
| core source -> DMA sink, 0 / 1 / 2 / 3 hops | 26 / 34 / 42 / 50 | 26 / 34 / 42 / 50 |
| core source through the switch FIFO, 0 hops | 47 (+21) | 47 (+21) |
| DMA source -> DMA sink, 0 / 1 / 2 / 3 hops, BD phase 0 | 48 / 56 / 64 / 72 | 48 / 56 / 64 / 72 |
| DMA source -> DMA sink, 0 hops, with FIFO, phases 0..3 | 68 / 71 / 70 / 69 | 68 / 71 / 70 / 69 |
| core -> core, one tile apart (harness `direct`) | 18 | 18 |
| same through the producer's FIFO (harness `fifo`) | 39 (+21) | 39 (+21) |

(All four BD phases match on every DMA route too: 25 of 25 R4-D3 points.)

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
- sources hand off through a 1-word register: the core's stream output (unchanged) and the MM2S
  (`MM2S_OUTPUT_REGISTER_DEPTH`, was a 4-word second slave FIFO); a staged MM2S word carries the data
  read when its granule was fetched;
- the S2MM ingress is 15 + 3 (above), the core's stream input 3 (`CORE_STREAM_INPUT_DEPTH`, was 1:
  the consumer side of the core -> core count, 18 - 1 - 4 - 8 = 5, less the Core master port's 2);
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
