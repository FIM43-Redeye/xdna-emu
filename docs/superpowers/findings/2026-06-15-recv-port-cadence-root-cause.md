# #140 cadence divergence: physical root-cause localization (two causes, not one)

> **RESOLVED (2026-06-15): one cause — shim DDR jitter — closed by the stochastic
> delivery model.** The cadence gap (slot0 AND slot4) is reproduced by a
> seeded-PRNG DDR burst-delivery model, calibrated against a fresh 100-run HW
> capture to the `AIE2_DDR_PHOENIX` profile (burst `[36,46]`, gap `[8,14]`):
> slot0 5.34±0.47 vs HW 5.65±0.48, slot4 7.62±0.49 vs HW 7.26±0.46 (both within
> 1σ; slot1/slot5 exact). Opt-in (default-off), per-DRAM-tunable. Full design,
> calibration table, and the irreducible-coupling caveat in
> `docs/superpowers/specs/2026-06-15-ddr-stochastic-delivery-jitter.md`. The
> investigation below stands as the root-cause record.

> **TOP / CURRENT (2026-06-15 cont., probe-confirmed — supersedes everything below).**
> A per-cycle backpressure-chain probe (`XDNA_EMU_BP_PROBE`, end of
> `coordinator::step`) on the in-process `add_one` run settled the contradictions
> in the layered sections below. The earlier "stream_out capacity is 16, MM2S
> reads all 16 ahead" explanation is **factually wrong**: MM2S gates on
> `can_push_stream_out_for_channel` at cap **4** (`stepping.rs:1218`), so it never
> holds a whole buffer.
>
> **The real mechanism — a producer/consumer race through the memtile in0
> double-buffer:**
> - **Producer = slot0** (memtile S2MM ← shim): fills at **1 word/cyc**
>   (`shim_words_per_cycle=1`, shim AXI rate). Probe: `b` climbs +4 B/cyc.
> - **Consumer = slot4** (memtile MM2S → compute): egresses at **4 words/cyc**
>   (`words_per_cycle=4`, data-memory bus rate). Probe: `b` climbs +16 B/cyc.
>
> Because the consumer is 4× faster, it empties each 16-word buffer long before
> the producer refills it, so slot0's `prod_lock` is **always already free** →
> slot0 waits exactly **1 cycle** at each buffer boundary (probe: `AcquiringLock`
> for one cycle at cyc 7906, 7922). 1-cycle blips are invisible to PORT_RUNNING
> grouping → slot0 = 1-3 bursts vs HW 6.
>
> **Why the compute-processing backpressure never reaches slot0:** at cyc 7924
> slot4 has sent 36 words while compute consumed 16 — ~**18 words in flight in the
> fabric** (memtile `so=1`, compute `si=1`). That ~18 is the *legitimate* sum of
> AM020 per-route pipeline latency (`STREAM_LOCAL_TO_LOCAL_LATENCY=3`, budget =
> `latency+master.fifo` per route) + FIFO depths across the multi-hop path +
> `stream_out`(4) + `stream_in`(2). Plus the compute in1 objfifo is 16 words.
> Total decoupling ~34 words on a **64-word** transfer. The consumer drains the
> first ~46 words fast (slot4 only stalls at `so=4` by cyc 7933), by which time
> slot0 has already received nearly everything at its own 1 word/cyc rate.
>
> **Two HW-derivable levers, both with revert history (discuss before acting):**
> 1. **MM2S stream egress = 1 word/cyc** (32-bit AXI4-Stream port; currently 4).
>    Derivable, but *proven insufficient alone*: 1 w/cyc egress just rate-matches
>    consumer==producer (~16 cyc/buffer each), which still yields ~0 producer
>    stall. (Matches the earlier route-throttle A/B: slot0 1→2.)
> 2. **Tighter producer/consumer coupling** so the compute processing rate gates
>    slot0. The fabric in-flight slack is AM020-faithful per-hop, but summed over
>    the path it decouples a 64-word transfer. NOTE: `STREAM_LOCAL_TO_LOCAL_LATENCY`
>    was bumped 3→15 once (`2026-05-11-emu-dma-pipeline-too-fast-misses-stalls`)
>    and reverted — it is load-bearing elsewhere.
>
> **Also: slot0's 6 HW bursts are partly PRODUCER-side** (shim DDR burst jitter —
> the parked BurstGate / Root cause B), not purely consumer backpressure. The DDR
> A/B (below) moved slot0 1→4 by bursting the *shim delivery*. So slot0 has BOTH a
> producer-side (DDR jitter) and consumer-side (coupling) contribution; slot4's 6
> bursts are purely consumer-side (compute processing).
>
> This is a calibration+architecture fork with three prior reverts (latency bump,
> egress throttle, gated-release). Bringing Maya in before the next change.
>
> **COUPLING SWEEP (cont. 2026-06-15, beat-accurate `cycle_beat` probe).** Maya:
> pursue the coupling rework IF correct; DDR jitter goes to a separate RNG path.
> Env-gated A/B on the in-process run (all default-off knobs):
>
> | config | slot0 | slot4 | slot1 | slot5 |
> |--------|------:|------:|------:|------:|
> | baseline | 1 | 5 | 8 | 4 |
> | 1word (MM2S egress=1) | 2 | 5 | 8 | 4 |
> | ss-latency=0 | 1 | 6 | 8 | 4 |
> | 1word + lat=0 | 2 | 6 | 8 | 4 |
> | stream_out cap 4→1 | (no change) | | | |
> | **HW band** | **5.73±0.44** | **7.07±0.25** | 8 | 4 |
>
> **Findings:**
> 1. **Lever 1 (MM2S egress 1 word/cyc) is HW-correct but near-cosmetic alone:**
>    slot0 1→2, slot4 unchanged. Rate-matching the consumer to the producer does
>    not make it *slower*, so few new producer stalls. (off1-safe: no
>    `begin_completion` change. slot1/slot5 untouched.)
> 2. **The `stream_out`-vs-slave-port double-buffer is NOT the slack** — cap 4→1
>    moved nothing. Ruled out.
> 3. **Fabric pipeline latency IS the consumer-coupling lever** (lat 3→0 moves
>    slot4 5→6) — BUT reducing it is *un-physical*: AM020 specifies 3-4 cycle
>    switch latency, and a registered pipeline of depth L genuinely buffers L
>    words. The current ~18-word fabric slack is AM020-faithful. Tuning it to 0
>    is exactly the hack to avoid.
> 4. **slot4 is physical-objfifo-capped at ~6.** Even at un-physical zero fabric
>    slack, the compute in1 objfifo (2×8=16 words) lets the producer run 2
>    buffers ahead, costing ~2 early sub-bursts. So HW's 7th burst is NOT
>    reachable by any HW-faithful slack change — it's a finer effect (initial
>    fill split?) or within the ±0.25 measurement. **known-fidelity-gap
>    candidate.**
> 5. **slot0's 2→6 gap is producer-side** (shim DDR burst jitter), to be
>    reproduced by the separate RNG path, not by consumer coupling.
>
> **Conclusion:** the consumer-coupling model is *already ~HW-faithful*; there is
> no large coupling BUG to fix. The available correct win is Lever 1 (a real
> AXI4-Stream-rate correction, small cadence effect). slot4's residual (~6 vs 7)
> is objfifo-physical-capped → known-fidelity-gap. slot0's bulk is DDR jitter →
> RNG path. Recommendation: land Lever 1 (band-sigma + lib + bridge validated),
> document the slot4 residual, move slot0 to the DDR-RNG task.
>
> **SLOT4'S 7TH BURST — RESOLVED (cont. 2026-06-15): it's the DDR jitter, same
> cause as slot0 — NOT a separate coupling effect.** Chased slot4's gap *shape*:
>
> | | early gaps | late gaps | bursts |
> |--|--|--|--|
> | HW slot4 | `[8, 9, 14]` | `[38, 65, 73]` | 7 |
> | EMU 1word+lat=0 | `[21]` (one merged) | `[55,62,55,63]` | 6 |
> | EMU + DDR gate (w=8,inter=8) | `[14, 22, 20]` | `[35, 62, 56]` | **7 ✓** |
>
> The three *late* gaps are compute-processing backpressure — EMU reproduces them
> correctly with or without DDR (the coupling model is faithful). The three
> *small early* gaps are the memtile MM2S **waiting for the 1-word/cyc shim-fed
> producer to refill the next 16-word buffer** — i.e. they're shim DDR
> burst-delivery jitter propagating through the memtile forward. Turning on the
> DDR burst gate restores them and lifts slot4 6→7 with the correct shape
> (slot1=8 / slot5=4 unchanged; slot0 overshoots to 8 only because w=8/inter=8 is
> uncalibrated). **So the ENTIRE #140 input-path cadence gap — both slot0 AND
> slot4 — collapses to ONE physical cause: shim DDR burst-delivery jitter.** The
> consumer-coupling / objfifo / fabric model needs no change. The fix is the
> DDR-RNG delivery model (Maya's scoped path): a stochastic shim-delivery jitter
> that reproduces HW's slot0 5.73±0.44 and slot4 7.07±0.25 *distributions*
> (band-matched, not fixed params). Lever 1 (1word MM2S egress) is an orthogonal
> correctness nicety, not required for cadence.
>
> SCAFFOLDING (all default-off, uncommitted): `XDNA_EMU_BP_PROBE` (coordinator
> beat probe), `XDNA_EMU_MM2S_1WORD`, `XDNA_EMU_STREAMOUT_CAP`,
> `XDNA_EMU_SS_LATENCY`. Remove or graduate when the DDR-RNG model lands.

**Date:** 2026-06-15
**Method:** in-process `add_one_using_dma` (chess), per-port push/pop probe on
every tile/port (`XDNA_EMU_ALLPORT_PROBE` scaffolding, since reverted),
cross-referenced to the variance-aware HW baseline
(`2026-06-15-port-cadence-hw-baseline.md`).

## What the probe established

The four traced memtile (0,1) event-port slots map to physical ports as
(confirmed by `event_port_selection` + first-activity temporal order):

| slot | port | DMA dir | endpoint | dataflow role | first cyc |
|------|------|---------|----------|---------------|----------:|
| slot0 | M[0] | S2MM recv | ← shim | **input** arrives from DDR | 7874 |
| slot4 | S[0] | MM2S send | → compute | **input** forwarded to compute | 7900 |
| slot1 | M[1] | S2MM recv | ← compute | result arrives from compute | 7985 |
| slot5 | S[1] | MM2S send | → shim | result drained to shim | 8059 |

(Master Dma ports drain into S2MM = receive; slave Dma ports are fed by MM2S =
send. Confirmed at `routing.rs` Step 1 / Step 6.)

Cross-referencing the HW baseline counts:

| slot | role | HW mean | EMU | verdict |
|------|------|--------:|----:|---------|
| slot0 | recv ← shim (input) | 5.73 ±0.44 | 3 | **undercounts** |
| slot4 | send → compute (input) | 7.07 ±0.25 | 5 | **undercounts** |
| slot1 | recv ← compute (result) | 8.00 det | 8 | matches |
| slot5 | send → shim (result) | 4.00 det | 4 | matches |

**This refutes the baseline doc's "send deterministic+correct, recv
undercounts" framing.** The real split is **input path undercounts, result path
matches** — one recv and one send on each side. The discriminator is the
*source* of the sub-burst gaps, and it points at **two independent root
causes**.

## FIX ATTEMPT #1 (egress-throttle + egress-gated release) — REVERTED, regresses off1

Implemented two gated pieces: (a) MM2S egress capped at 1 word/cycle
(`route_dma_to_tile_switches`), (b) defer MM2S `begin_completion`/prod_lock
release until `stream_out` for the channel drains (`stepping.rs` Transferring→
completion). In-process A/B (add_one):

- **slot4 5→7 (into HW band)**, slot1=8 / slot5=4 held, output still Pass.
- **slot0 only 1→2** (target 6): its span stays 7874–7949; the early MM2S
  buffer-drains are still too fast (slot4 early gaps `[3,3,..]` vs HW `[8,9,..]`),
  so the producer isn't backpressured from the start.
- Neither piece alone moves slot4 (gated-release-only=5, throttle-only=5); the
  pair is required.

**Why reverted:** full `cargo test --lib` → 8 failures. 3 are test-harness gaps
(isolated MM2S helper never drains `stream_out` → hang; fixable). But the
decisive one: `chained_bd_inserts_port_running_bubble_at_each_boundary` shows the
**BD-switch bubble widening 1→2 cycles**, breaking the HW-verified **off1**
(1-cycle) bubble (#26). The egress-gated release adds a residual-drain cycle at
*every* BD boundary — that's a regression in established, HW-confirmed behavior,
not a fidelity gain. **Do not land.**

**What this teaches:** "hold the buffer until egress" is too blunt — HW retires a
chained BD with only a 1-cycle bubble even though data is still draining, so the
buffer is NOT held to full egress in the no-backpressure case. The existing model
*already* egress-paces the read under genuine backpressure (the
`can_push_stream_out` stall keeps `remaining_bytes > 0`). The real lever is
narrower: under backpressure the buffer should hold (it mostly does), but the
egress-flood (≤4 words/cycle) keeps backpressure from building in the first
place — yet the throttle alone didn't surface slot4 either. The interaction is
subtler than modeled; rework must preserve off1. Deferred to the post-compact
deep-dive on slot0/#2.

## ACTUAL ROOT CAUSE (verified, 2026-06-15 cont.): MM2S frees the source buffer on read-completion, not egress

The DDR-burst story below is **refuted** — add_one's input BD programs a 256-byte
AXI burst for a 256-byte transfer (one burst; `burst_length_field=2`, confirmed
by probing `bd.burst_length`). DDR bursting cannot create slot0's 6 sub-bursts.

Per-cycle channel-FSM + lock instrumentation on the memtile input relay
(`XDNA_EMU_RELAY_PROBE`) shows the true mechanism:

- The memtile in0 objfifo is **double-buffered** (`in0_cons_buff_0/1`, 16 words,
  `prod_lock init=2`). The S2MM (slot0) fills 16-word buffers; after both are
  full it must block on `prod_lock` until the MM2S drains one.
- **EMU**: at each 16-word boundary the S2MM hits `WaitingForLock` for exactly
  **1 cycle**, then resumes (measured: Active@64B 15cy, 1cy wait, Active@128B
  15cy, 1cy wait, ...). 1-cycle gaps are invisible to PORT_RUNNING grouping →
  slot0 = 1–2 bursts.
- **HW**: the same BD-boundary waits are **7–16 cycles** (slot0 gaps
  `[16,16,7,9,13]`) → 6 visible bursts.

**Why prod_lock frees too fast:** `stream_out` capacity is **16**
(`engine/mod.rs:183`) — a whole buffer. The MM2S reads all 16 words into
`stream_out` at 4 words/cycle (~4 cycles), `begin_completion`
(`engine/stepping.rs:738`) fires, and prod_lock releases — **gated by the
memory read, not by egress.** The data still sits in `stream_out` and hasn't
crossed to the compute, but the buffer is already marked free, so the producer
S2MM never blocks. On silicon the DMA holds the source buffer until its last
beat is on the wire; under compute backpressure that hold is 7–16 cycles.

This also explains why the earlier egress-1-beat/cycle throttle didn't help:
throttling egress didn't change *when* prod_lock releases (still at read-done).

**Fix direction (to settle with Maya):** gate MM2S BD-completion / prod_lock
release on **egress** (buffer data actually leaving `stream_out`/the tile),
not on the memory read into `stream_out`. Equivalent lever: shrink `stream_out`
so the read is egress-paced. Broad blast radius (every MM2S objfifo handshake)
→ band-sigma validation + full lib suite + 15-CLEAN re-check. Related to
completed #132 (memtile double-buffer release-overlap) — likely a residual on
the read-vs-egress release boundary.

## (refuted) single root cause = smooth DDR-read delivery (propagates to both slots)

DDR burst gate (`XDNA_EMU_DDR_BURST_WORDS=16 XDNA_EMU_DDR_INTER_BURST_CYCLES=8`,
rough params for visibility) A/B on the in-process run:

| | slot0 (shim→mt) | slot4 (mt→compute) | slot1 | slot5 |
|--|-----------------|--------------------|-------|-------|
| DDR OFF | 1 burst | 5 `[28,56,63,57]` | 8 | 4 |
| DDR ON  | 4 `[9,9,9]` | **7** `[7,15,17, 56,63,57]` | 8 | 4 |
| HW target | 6±1 `[16,16,7,9,13]` | 7±1 `[8,9,14,38,65,73]` | 8 | 4 |

**One knob moved both undercounting slots in lockstep and left the matching ones
untouched.** slot4 went 5→7 (into band): its two new early gaps `[7,15,17]` are
DDR bursts propagating through the memtile forward, while the trailing
`[56,63,57]` compute-processing gaps are unchanged. slot0 went 1→4 (toward 6).
slot1=8 / slot5=4 did not move at all — the DDR read-burst never touches the
result path. **This is the proof of a single cause; the egress throttle and the
objfifo timing below were both red herrings.**

### Path forward — calibrate the BurstGate (don't repeat the 16/1 mistake)

The mechanism is the parked `BurstGate` (`src/device/dma/burst.rs`). It was
parked because the `AIE2_DDR_DEFAULT` (16/1) **smeared chained transfers — the
k8 `_diag_shim_chain_sweep` regression** (`timing.rs:160`). So the task is a
calibration: find burst params that lift slot0→6±1 and slot4→7±1 **without**
regressing the chain sweep or the 15 honest-CLEAN kernels, validated band-sigma
(never a single capture). HW's slot0 `[16,16,7,9,13]` is variable-size (AXI
burst-splitting on transfer size, per the burst.rs note), so a fixed
`burst_words` is an approximation — part of the calibration is deciding whether
fixed params suffice or the model needs the data-dependent split.

## (red herring) A/B RESULT: egress throttle is necessary but NOT sufficient

Gated 1-beat/cycle egress throttle, A/B on the in-process run:

| | slot0 | slot4 | slot1 | slot5 |
|--|------:|------:|------:|------:|
| HW target | 6±1 | 7±1 | 8 | 4 |
| EMU throttle OFF | 1* | 5 (maxpush 4) | 8 | 4 |
| EMU throttle ON | 2 | 5 (maxpush 1) | 8 | 4 |

(*slot0=1 here is the single-shot in-process run; the 15-run baseline mean is 3.)

The throttle does exactly what it says — `maxpush` drops 4→1, egress is now one
beat/cycle — but **the sub-burst count does not move** (slot4 stays 5, slot0
1→2). The clinching evidence: with the throttle on, slot4's port FIFO **sits at
0** for 39 of ~70 active cycles (push=1/pop=1 every cycle). The compute consumes
the input as fast as the memtile sends it; the FIFO never saturates, so
**backpressure never engages and no gaps form.** Throttling just spread the
flood — precisely the failure mode we set out to derisk.

## The actual root cause (slot4): input objfifo consumer doesn't hold during processing

In HW the compute core acquires an 8-word input buffer, **holds it while
processing** (~56 cy, the add loop), then releases. After the objfifo's 2
buffers (16 words) are in flight, the compute S2MM backpressures and stalls the
memtile MM2S mid-stream → the per-8-word gaps → ~8 sub-bursts.

In EMU the compute **drains its input objfifo continuously, decoupled from
processing**: it pulls input at 1 word/cycle nonstop, so the memtile feeds it
without ever backpressuring. Confirmation that production *is* processing-gated
while consumption is not: slot1 (the compute's *output*) correctly shows the
~57-cycle per-chunk processing gaps; only the *input-consume* side runs ahead.

So the slot4 fix lives in the **core ↔ input-objfifo consumer-lock timing** (the
acquire-hold-release relative to core execution), not in the DMA egress rate.
The egress throttle is a real correctness item (AXI4-Stream is 1 beat/cycle) but
belongs folded into that backpressure fix, not committed standalone — alone it
shifts timing (slot5 gaps 118→112) without buying cadence.

## (superseded) Root cause A — DMA stream egress floods (slot4, the → compute send)

`route_dma_to_tile_switches` (`src/device/array/routing.rs:809`) drains the
DMA's **entire** `stream_out` queue into the switch slave port every cycle,
bounded only by the 4-deep port FIFO:

```rust
while let Some(data) = dma.pop_stream_out_for_channel(combined_ch) {
    if slave.can_accept() { slave.push_with_tlast(...); }  // up to 4/cycle
    else { retain; break; }
}
```

Probe confirms: every MM2S send port shows **push up to 4 words/cycle**
(`maxpush=4`), while every S2MM recv master and every inter-tile slave is
strictly `maxpush=1`. The memory→`stream_out` fill is correctly 4 words/cycle
(`words_per_cycle`, 256-bit data-memory bus, `stepping.rs:924`) — but the
**egress onto the stream fabric is AXI4-Stream and is physically 1 beat (one
32-bit word) per cycle.** EMU bursts 4.

Consequence: the port FIFO is kept pre-filled, so the downstream switch pops a
beat every cycle and `cycle_beat` (→ PORT_RUNNING) never deasserts during the
compute's per-8-word consumer-lock backpressure. HW's fine 8-word objfifo gaps
(`[8,9,14]`) are masked; slot4's first burst is one continuous ~33-cycle blob
where HW shows ~3 short bursts. slot5 *also* floods but matches because its HW
gaps are coarse (BD-level, 4 sub-bursts) and survive the smoothing.

**Fix site (derivable, AXI4-Stream spec):** throttle DMA→switch-port egress to
1 word/cycle/channel. The memory-side 4-words/cycle read is unchanged; only the
fabric egress is metered. This is a hardware fact, not a tuned constant.

## Root cause B — DDR burst arrival too smooth (slot0, the ← shim recv)

slot0's HW gaps come from **DDR burst arrival jitter** (shim AXI master delivers
in bursts, not a uniform stream; STREAM_STARVATION is the dominant stochastic
axis, baseline conclusion #3). EMU delivers the input too smoothly, so the recv
port shows 3 contiguous sub-bursts vs HW's 6±1. This is the **parked BurstGate
territory** (`src/device/dma/burst.rs`, default disabled) — a *calibration*-class
change, distinct from cause A, and the one we reverted on a bad single
calibration. It must be validated band-sigma if revisited.

## Why slot1/slot5 already match

Their gaps come from sources EMU already models faithfully: slot1 from the
compute's genuine per-8-word *production* stalls (core execution timing), slot5
from coarse BD-level structure that the egress flood can't erase.

## Where this leaves us (post-A/B)

The cadence gap is **two consumer-backpressure modeling gaps**, neither an
egress-rate tweak:

1. **slot4 (→ compute):** compute input-objfifo consumer must hold its buffer
   during core processing so the memtile MM2S backpressures per-8-words. This is
   a **core↔DMA objfifo interaction** change — deeper than the stream layer, and
   the higher-value one (it's the lock-gated-delivery mechanism the toolchain
   dive predicted).
2. **slot0 (← shim):** DDR burst arrival jitter (the parked BurstGate),
   calibration-class, separate.

The egress 1-beat/cycle throttle is correct-but-cosmetic on its own; fold it
into (1) where per-beat backpressure timing actually matters.

**Lessons carried:** (a) verify the mechanism reproduces the symptom *before*
writing the fix — the throttle looked obviously right and was insufficient;
(b) any DMA-timing change is validated band-sigma, never a single capture
(BurstGate). The band-sigma comparator is the gate.
