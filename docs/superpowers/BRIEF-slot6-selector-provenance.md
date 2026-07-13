# Brief: why is slot 6 selected? -- trace the scheduler selector back to its caller

## Why this brief exists

The `a7=6` trace (finding `2026-07-12-a7-reject-provenance.md`, committed
`d5824e21`) resolved the fork: `a7=6` is **genuine firmware state** -- the current
scheduler task's slot ID. The service sequencer `0x7fc4` accepts slot IDs
`0..=5` (`Bgeui a7,6 -> 0x7fec`) and rejects the slot-6 task the scheduler made
current. This is a within-firmware contradiction: the firmware's own scheduler
selected a task its own service path refuses. Almost certainly the go-alive
re-dispatch livelock (~131x) we started this arc on.

Two causes remain, and **both are on our side of the line** (no AMD data, no
calibration):

- **(1) Upstream reconstruction divergence** -- our emulation took a wrong
  branch, miscounted, or mis-loaded somewhere upstream and selected slot 6 where
  real silicon would select `0..5`. An emulator fidelity bug; fixing it dissolves
  the livelock. This is the `0x8cae`-precedent case (a "mechanism" that was really
  a reconstruction artifact).
- **(2) Intended slot-6 service** -- real firmware genuinely reaches this service
  with a slot-6 task, the reject is normal flow, and "alive" publishes on a path
  we have not followed.

**This session traces where selector 6 comes from and decides (1) vs (2).**

## The pinned consumption (starting point)

The prior trace already pinned where selector 6 is consumed in the scheduler
body `0x2800..0x2878`:

```text
n=47960  0x2816  L32iN a6,[a1+0x14]  -> 6     (selector loaded from stack frame)
n=47966  0x2826  MovN  a10,a6        -> 6
n=47967  0x2828  Extui a3,a10,0,8    -> 6
n=47968  0x282b  Addx4 a4,a3,a4      -> SCHED + 6*4  (indexes runnable[6])
```

So selector 6 enters as `[a1+0x14]` -- a stack argument in the scheduler frame.

## The question (backward from the selector)

1. **Who writes `[a1+0x14]` (the selector argument) before `n=47960`?** Walk the
   stack slot back to the mapped caller that supplies 6. Produce the full
   producer chain: which function computed 6, from what inputs (a loop index, a
   task-count, a priority/ready-bitmap scan, a field of some object). Mark each
   edge VERIFIED (executed) vs CLAIMED (inferred).
2. **Contrast with the successful first call.** The first pass through the
   wrapper/sequencer carried `a7=0` and did NOT reject (finding controls,
   `n=53629/53632`). Find the corresponding earlier scheduler selection that
   produced selector 0, and diff the two: what differs between the run that
   selected slot 0 and the run that selected slot 6? That diff is the crux.
3. **Is this the livelock?** Determine whether the scheduler re-selects slot 6
   on each go-alive re-dispatch (i.e. the reject loops back to reselect the same
   slot-6 task). If so, confirm the loop structure -- that ties the reject to the
   ~131x re-dispatch directly.

## The (1)-vs-(2) discriminator

For each input that feeds selector 6, classify it exactly as the a7 trace did:

- **ordinary local_data written by mapped firmware** -> if the whole selector
  provenance is clean mapped-firmware computation from correct inputs, lean (2)
  intended, and the open question becomes "where does alive publish for a slot-6
  service."
- **a value that traces to a HARNESS_VIEW-supplied byte, a mis-decoded
  instruction, an off-by-one count, a wrong branch taken upstream, or any
  reconstruction seam** -> lean (1) divergence, and name the earliest point the
  selector's provenance goes wrong.

The deliverable is whichever the evidence supports, with the specific producer
chain that proves it. If (1), name the earliest corrupted input and the mapped
instruction that should have produced a different value. If (2), name the
slot-6 service's intended completion path and where it would publish alive.

## Deliverables

1. **Selector-6 producer chain** -- ordered, evidence-backed, from `[a1+0x14]`
   at `n=47960` back to the mapped caller and the inputs that yield 6, each edge
   VERIFIED/CLAIMED, with source addresses + memory classes.
2. **The selector-0 contrast** -- the successful earlier selection and a concrete
   diff explaining why one run selects 0 and the other 6.
3. **The (1)-vs-(2) verdict** -- divergence (name the earliest wrong input +
   the instruction that should differ) or intended (name the slot-6 completion/
   publish path), with the provenance that decides it.
4. **Livelock confirmation** -- whether the reject re-selects slot 6 each
   dispatch, tying it to the ~131x loop.
5. **Ranked single next step** -- derive-only.

## What "done" looks like

A written finding
(`docs/superpowers/findings/2026-07-12-slot6-selector-provenance.md`) with the
five deliverables. Plus the env-gated probe, `cargo test --lib` green (4091
baseline). Present for review -- do NOT commit.

## Execution discipline

If any run is long CPU-bound, background-and-block in ONE shell command
(`cmd & wait $!`); do not poll in a `check -> sleep -> "still running"` loop.

## Ground rules

- Derive, do NOT calibrate. No fitted constants, no hardcoded slot, no forced
  branch, no "make it pass" shim. If (1), the deliverable is the identified
  upstream defect; if (2), the identified intended path. No value injected.
- Read-only. No forcing firmware state, no `load_m2c` diff, no per-path byte
  swap. Instrument and reason.
- Test-only probe code; production `load_m2c`/`mod.rs`/`mmio.rs`/`system.rs`
  behavior unchanged.
- `cargo test --lib` stays green (4091 baseline) if you touch any probe.
- Do NOT re-open PSP-loader RE, the CPU-self-modifying-code-to-0x8cae path, or
  the below-CPU bank *mechanism* hunt (closed). This is a scheduler data-flow
  question.
- Ground everything in the Phoenix `1502_00` image. The `17f1_10` sibling is an
  untrusted, different-generation hint source -- no importing its semantics
  without a byte-level match against Phoenix.
- Present findings for review. Do NOT commit.

## Anchors

- Firmware: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`
  (SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`).
- Prior finding (the scheduler state + pinned consumption):
  `2026-07-12-a7-reject-provenance.md`. Scheduler: `SCHED=0x2250`,
  `runnable` base `SCHED+0x38`, `current` at `SCHED+0x28` (`[0x2278]`),
  `slot_id` at task `+0x08`. Selected task `0x10dfc`, slot_id `[0x10e04]=6`.
- Scheduler body: `0x2800..0x2878`; selector consumed at `[a1+0x14]` (`n=47960`).
- Successful first-call control: `a7=0` at `n=53629/53632` (no reject).
- Probe family to extend: `m2c_probe_26d4_cache_pageroot_timeline` in
  `src/firmware/boot_tests/coherence_mapper.rs`.
- Base commit: `d5824e21`.
