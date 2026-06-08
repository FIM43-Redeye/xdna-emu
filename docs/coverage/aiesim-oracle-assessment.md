# aiesim as a Validation Oracle — Capacity Assessment (pre-Strix)

**Date:** 2026-06-08. **Question:** is AMD's aiesimulator a *sufficient*
validation oracle for xdna-emu once Phoenix (NPU1 / AIE2) hardware is gone?

This is a re-evaluation, taken after the trace held-level fidelity work closed,
of where the aiesim oracle actually stands. It synthesizes the three-way timing
campaign, the control-read and broadcast-bridge fixes, the c2c-shared-memory
limitation, and the in-process backend feasibility study into one verdict, and
— most importantly — names the calibration data we must capture from Phoenix
**before the swap**, because it identifies exactly where aiesim cannot stand in
for real silicon.

Confidence markers: **VERIFIED** (proven against HW or by direct artifact),
**OBSERVED** (measured but not cross-checked against HW), **CLAIMED** (reasoned,
not yet measured). Detail lives in the linked docs; this page is the verdict.

---

## Why this matters now

The Phoenix→Strix upgrade is a **one-way door**: the swap removes our only
**AIE2 silicon** (see `project_strix_swap_replaces_phoenix`). Crucially, the swap
is not a loss of *hardware* in general — Strix *is* AIE2P silicon, so after the
upgrade we hold a real AIE2P oracle, better than any simulator. What we lose is
the ability to ground-truth the emulator's **AIE2 (Phoenix)** target against real
silicon.

That is aiesim's entire job here: **be the AIE2 oracle that outlives Phoenix.**
The whole aiesim arc (native NPU1 device file, control-read clone, broadcast
bridge, in-process backend) exists so that the emulator's AIE2 support stays
validatable after the physical Phoenix is gone. aiesim's AIE2P capability is
real but moot for us — we would never lean on a simulator for AIE2P when Strix
gives us the silicon directly.

So the question is narrow and answerable: **how thoroughly can aiesim cover AIE2,
and where does it fall short of what Phoenix gives us today?** The honest answer
is *covers most of it faithfully, falls short on two characterized micro-timing
axes — and those are exactly what we must capture from Phoenix before the swap.*

There is **no hard deadline** on the swap — our productivity gates the timing,
not a clock (`project_strix_swap_replaces_phoenix`). So the pre-swap capture
below is "do it deliberately before we choose to swap," not "race a date."

---

## Verdict in one line

aiesim is a **strong functional + trace-structure oracle and a cycle-EXACT
compute-timing oracle**, but it is **optimistic on DMA-fill latency and
lock-arbitration overhead** — the two axes only real silicon can ground. It can
replace Phoenix for *correctness and compute timing*; it **cannot** replace
Phoenix for *DMA/lock micro-timing*. Capture that micro-timing from Phoenix
before the swap.

---

## Capability matrix

| Dimension | Verdict | Confidence | Decisive evidence |
|-----------|---------|-----------|-------------------|
| **Functional correctness (values)** | Faithful across the corpus; validates **both** Peano and Chess output with no silicon | VERIFIED | Peano scalar+vector cores reach `PASS!` standalone (`aiesimulator.md:147-160`); corpus pass envelope below |
| **Trace structure / flow** | Faithful with the local broadcast bridge; event *sequence* and *count of deterministic events* match HW | VERIFIED (one kernel) | distribute_lateral: 32-event order byte-identical HW vs aiesim, trace 18719 vs 18863 B (`bcast-bridge/FINDINGS.md`; triage `:88-100`) |
| **Compute-region timing** | **Cycle-EXACT** vs HW | VERIFIED | INSTR_EVENT_0→1 span = **12297 ns in both worlds, all 4 invocations**, to the ns (`known-fidelity-gaps.md` aiesim row; `bcast-bridge/FINDINGS.md`) |
| **DMA / DDR fill latency** | **Optimistic** — under-models fill; a **constant offset, not per-iteration drift** | VERIFIED | first EVENT_0 at HW 8326 ns vs aiesim 2195 ns (~6131 ns optimistic), each invocation still exactly 12297 wide (`known-fidelity-gaps.md`) |
| **Lock-arbitration overhead** | **Optimistic** — uncontended acquires modeled as zero-stall | VERIFIED | identical lock ops both worlds (ACQ 9/9, REL 9/9), but HW trails every req with a 1 ns LOCK_STALL pulse: HW 19 = 1 startup + 18 per-txn, aiesim 3 = genuine blocking only (`known-fidelity-gaps.md`) |
| **Absolute DMA anchor timing** | Ballpark, ~same as our own emulator (neither is HW-exact) | OBSERVED | per-anchor mean-abs drift vs HW: interp +63.6%, aiesim +67.6% (`build/bridge-test-results/20260606/timing-three-way.total.txt`) — dominated by the fill-latency axis above |
| **Compute-to-compute shared memory** | **Cannot model** — neighbour shared-mem + lock handoff deadlocks; AMD's own XFAIL | VERIFIED | `04_shared_memory` XFAIL in mlir-aie; native aiesim repro; **our interpreter is the more faithful oracle here** (`2026-06-06-aiesim-aie2-cross-core-shared-memory-limitation.md`) |
| **Control-read response** | Faithful **only with** the local distinct-object clone patch (`XDNA_CLONE_BEATS`) | VERIFIED | aiesim model bug (one control block reused across beats) corrupts the header; clone patch restores route-match (`aiesimulator.md` Known Issues; `project_aiesim_native_device_file`) |

---

## The timing verdict, decomposed (the load-bearing nuance)

The single blunt number — "~68% mean drift vs HW" — is **misleading on its own**.
Per-event decomposition (the all-72 campaign cross-checked against the
HW-validated distribute_lateral capture) splits that gap into three separable
sources with *opposite* dispositions:

1. **Compute region — cycle-EXACT.** The user trace markers span 12297 ns in
   both worlds, every invocation, to the nanosecond. aiesim **is** a compute
   timing oracle. **Safe to teach the emulator.**
2. **DMA/DDR fill latency — optimistic by a constant offset.** ~6131 ns early on
   first fill, but it does **not** accumulate per iteration (each invocation
   stays exactly 12297 wide). A fixed bias, not drift. **Calibrate against HW
   only.**
3. **Lock-arbitration — optimistic per transaction.** HW charges 1 cycle even
   for an uncontended acquire; aiesim charges zero. The count gap (19 vs 3) is
   entirely this per-transaction tax. **Calibrate against HW only.**

So aiesim's drift is not "vaguely approximate everywhere" — it is *exact where
compute happens* and *biased in two specific, characterized places*. That
precision is what makes the pre-swap action tractable: we know exactly which two
numbers only Phoenix can give us.

---

## Coverage envelope (current, 2026-06-08)

The authoritative state is the **2026-06-07 re-verify** in
`aiesim-failure-triage.md`, which superseded the coarse 2026-06-05 discovery
sweep. After classification:

- **Genuine aiesim model wedges in the timeout set: ZERO.** All campaign
  "timeouts" were our own instrumentation defaults (settle 512→4096, timeout
  1200→3000 s), now fixed; the kernels PASS with budget (`:258-273`).
- **Excluded by AMD limitation:** the compute-to-compute neighbour-objectfifo
  class (`matrix_multiplication_using_cascade/buffer`). Not a fix target — our
  emulator is the oracle there.
- **Locally repaired:** control-read aliasing (clone patch); `reconfig_elf`
  PL-egress (was harness trace egress on PL via a 37 ns startup race → demoted to
  drain+warn default); distribute_lateral trace (broadcast bridge, HW-validated).
- **One genuine open FAIL:** `objectfifo_repeat/init_values_repeat` (peano) —
  fast early-stop / quiescent wedge, distinct from the settle false-trips.
  Under investigation (task #92).
- **Not aiesim's fault:** the HW-quarantined `ctrl_packet_reconfig` variants
  (FAIL on HW too).

Net: the functional envelope is **essentially the whole corpus** minus the AMD
c2c class, minus one open FAIL under investigation. That is a wide, trustworthy
oracle surface.

---

## What only Phoenix can give us — capture before the swap

The capability matrix has exactly **two VERIFIED axes where aiesim is optimistic
and the only ground truth is AIE2 silicon**: DMA/DDR fill latency, and
lock-arbitration per-transaction cost. Once Phoenix is gone we can never measure
these on AIE2 again. Therefore, **before the swap** we should capture, on NPU1
HW, the calibration corpus that pins both:

- **DMA-fill latency** — first-fill offset across BD shapes / transfer sizes /
  shim-vs-memtile sources (the ~6131 ns constant is one point; we need the curve).
- **Lock-arbitration tax** — the per-transaction stall cost across contended and
  uncontended acquires, enough to confirm the "1 cycle per request, always" rule
  generalizes.

This is exactly the **Phoenix-survival output-corpus** already spec'd and
approved, awaiting execution (`project_phoenix_survival_capture_spec_wip`). This
assessment is the *why*: it is the set of measurements that turn aiesim from a
"compute-exact, DMA-optimistic" oracle into a fully-calibrated one — and it is
HW-gated, so it must run on Phoenix while Phoenix exists. **Pre-swap priority.**

Everything else aiesim needs (functional correctness, trace structure, compute
timing) it already provides without silicon.

---

## AIE2P / Strix: not aiesim's job

For completeness, since it comes up: aietools *does* support AIE2P simulation
(binaries `aie2pssimmsm{,_func,_dbg}`, data dirs `aie2p`/`aie2ps`, license
`AIEMLv2sim` permanent → 2027.03). But this is **not** load-bearing for us.
After the swap we hold **Strix silicon, which is AIE2P** — a real hardware oracle
for that architecture. We would never substitute a simulator for the actual NPU
we own. aiesim's AIE2P capability is a fallback we don't expect to exercise.

The corollary matters for scoping aiesim work: **invest in aiesim's AIE2
coverage, not its AIE2P port.** AIE2 is the architecture about to lose its
silicon; AIE2P arrives with its own. The in-process backend, the device file,
the patches — all are correctly AIE2-targeted, and there is no reason to port
them to `aie2pssimmsm` for our validation needs. (If the emulator's AIE2P target
ever needs a *silicon-free* oracle for some reason — e.g. CI without a Strix box
— that port is available, bounded, and can be reconsidered then. It is not a
pre-swap concern.)

---

## Bottom line

| Use aiesim as the AIE2 oracle for… | …after Phoenix silicon is gone? |
|---------------------------|--------------------------|
| Value/functional correctness (Peano + Chess) | **Yes** — silicon-free, wide envelope |
| Trace structure / event flow | **Yes** — with the broadcast bridge |
| Compute-region cycle timing | **Yes** — cycle-EXACT |
| DMA-fill / lock-arbitration micro-timing | **No** — capture from Phoenix first, then aiesim runs calibrated |
| Compute-to-compute shared memory | **No** — but our emulator is the oracle there anyway |

aiesim earns its place as the post-Phoenix oracle for everything except two
characterized micro-timing axes. The single highest-value pre-swap action this
assessment surfaces is the **Phoenix-survival HW capture** of those two axes,
while the silicon to measure them still exists.
