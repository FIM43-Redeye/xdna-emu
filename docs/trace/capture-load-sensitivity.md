# Trace Capture Load Sensitivity

**Host system load contaminates HW trace cycle measurements.** A within-core-
domain cycle delta that should be host-invariant drifts by +/-1 when the host
is busy, and is cycle-exact when the host is quiet. This is a property of the
*capture pipeline under contention*, not of the silicon -- and it once led us
to record a "genuine +/-1 HW jitter" that did not exist.

This document is the durable record so we do not re-derive that phantom. It is a
sibling to [`cross-domain-skew-limit.md`](cross-domain-skew-limit.md): that one
bounds what a trace can tell us *across* timer domains; this one bounds what a
trace can tell us *within* one domain when the capturing host is loaded.

## The finding

The core-lock span `INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ` (one
core timer domain, purely core-internal) measures a fixed cycle count on a
quiet system and an occasionally-off-by-one count on a loaded one. Same kernel
binary, same machine, same analysis -- only host load changes.

## Evidence (NPU1 / Phoenix, add_one_using_dma, chess build, offset 24)

Dose-response across six capture sets, REL-ACQ cycle delta:

| Capture | Host state | REL-ACQ per run | range | off-by-one rate |
|---|---|---|---|---|
| pytest-35 (6) | quiet-ish | all 24 | 0 | 0% |
| clean (20) | quiet | all 24 | 0 | 0% |
| loaded (20) | deliberate CPU+mem load | 19x24, 1x23 | 1 | 5% |
| seg_probe (10) | I/O-storm chaos | mostly 24, one 23 | 1 | ~10% |
| pytest-36 (6) | I/O-storm chaos | 23,24,24,23,24,24 | 1 | 33% |

The controlled A/B is decisive: a 20-run capture on a quiet system is a flat
`[24]x20` (range 0); a 20-run capture under deliberate `stress-ng` CPU+memory
load on the *same* build/machine picks up a `23` (range 1). The off-by-one
*rate* tracks load monotonically: 0% quiet -> 5% synthetic (no disk) -> 33%
under a real I/O storm. A genuine HW property would not swing from 33% to 0%
with host load.

(The 22-vs-24 offset seen in older captures was a separate red herring: a core
ELF recompile at 14:57 reshuffled the schedule. Different binary, not a HW
anomaly. Always compare captures from the *same* build.)

## Mechanism

The contamination scales with how much a pair depends on host-fed DMA timing.
In the same loaded run:

| Pair | domain | quiet range | loaded range | amplification |
|---|---|---|---|---|
| `DMA_S2MM_0_START <- MM2S_0_START` | shim, DMA-mediated | 4 | 196 | ~50x |
| `PORT_RUNNING_*` | memtile | 1-2 | 1-4 | ~2x |
| `LOCK_RELEASE <- ACQUIRE` | core-internal | 0 | 1 | minimal |

The core-lock acquire waits on input data delivered by DMA. When the host is
slow to feed that DMA, the acquire occasionally grants one cycle later, nudging
the span to 23/25. Pure-DMA shim events, sitting directly on the host-fed
timing, swing by hundreds of cycles. The on-chip timer itself stays accurate;
what drifts is *when the traced events happen* relative to it, because the host
perturbs the kernel's input timing.

## What this overturns

The earlier "the core-lock span carries a genuine +/-1 jitter" conclusion was
reached entirely from captures taken during a day of host I/O chaos. It was an
artifact. Under clean capture the span is **cycle-exact** (range 0, offset 24,
20/20 runs). Two downstream corrections follow:

- The merged "cycle-exact within-domain segment" deliverable is **sound**, not
  a lucky overclaim.
- The proposed *structural-license* reframe (deny cycle-exact to the lock span
  because it "carries +/-1") was **retired** -- its premise was the phantom.

## The canary and the classification posture

DMA-mediated shim pairs are a **hypersensitive contamination detector** (~50x
amplification). More importantly, this finding sets the engine's standing
posture toward nondeterminism:

> We accept no nondeterminism unless it is verified as real (not load-induced).
> Every observed cross-run range > 0 must be **exactly classified** as one of:
> (1) load-induced capture artifact, (2) genuine HW nondeterminism, or
> (3) structurally-expected gap (cross-domain / DMA / async-CDC). Unclassified
> nondeterminism is **warned on**, never silently accepted as a cycle count.

The verification mechanism is **reproduction across independent capture
sessions** -- threshold-free, no statistical tolerance (consistent with the
`Q = 0` exact-agreement rule). A within-domain pair is treated as cycle-exact
only if range-0 holds across independent clean sessions; a single flicker
demotes it to nondeterministic, to be classified.

**Implemented (engine).** The classification aid is built into the inference
engine, not a separate subsystem. Every grounded `Gap` carries a typed `reason`
(`inference/grounding.py`): the accounted-for `cross_domain` / `async_cdc`
reasons are NOTED; the unaccounted `within_domain_nonexact` reason is surfaced
in the engine report's `warnings` list -- loud, never swallowed. The
load-vs-HW verdict stays **manual** (re-capture on a quiet host): the engine
flags, the human classifies. No autonomous classifier, no session-counting
protocol, and no durable registry were built -- they were premature; the
report-time warning is the whole canary.

## Hardware fix

A dedicated, quiet capture environment -- a mini-PC running only HW-oracle
captures, with no gaming / desktop / I/O load -- yields clean captures *by
construction* and removes this contamination class at the source. A virtualized
path (WSL2 / Hyper-V DDA passthrough) would do the opposite: it adds host-side
indirection, the very thing that contaminates capture timing.
