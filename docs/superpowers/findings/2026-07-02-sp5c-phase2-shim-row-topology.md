# SP-5c Phase 2: the shim-row `d_h` premise was silicon-refuted -- the reset's horizontal transport is fabric-routed, not shim-row

**Date:** 2026-07-02
**Issue:** #140 timer-sync arc, SP-5c Phase 2.
**Status:** Design premise (IF-1) reversed on silicon. `d_h` reframed to the free-flood
fabric measurement. Nothing was flipped (`calibrated` stays false).

## TL;DR

SP-5c Phase 2 set out to measure the reset broadcast's horizontal per-hop skew
`d_h` on the *shim-row* path (design rev4 IF-1: "the reset routes horizontal only
on the shim row; the bring-up `d_h≈4` on the AIE row was the wrong path"). We built
a block-mask kernel to force the flood onto the shim row. **On Phoenix silicon the
floods confined to their source column across 3 kernel variants / 60 clean HW runs.**
A skeptical re-derivation of `XAie_SyncTimer` showed **the premise was wrong**: the
reset distributes horizontally through the **memtile/compute fabric**, not the shim
row (which does not forward tile-to-tile E/W broadcast on AIE-ML). **The free-flood
`d_h≈4` that rev4 dismissed is, in fact, essentially the right measurement.** IF-1's
shim-row requirement is retired; `d_h` comes from the free-flood fabric capture.

## What we built and what silicon showed

Kernel `sp5_skew_r3b_pc_dh` (mlir-aie, branch `xdna-emu-cycle-budget`): the validated
free-flood R3b-PC kernel + `Event_Broadcast_Block` masks (`0xC000` = ch14|ch15) on
every compute tile (mem-mod East `0x14080` + core-mod West `0x34060`) and memtile
(switch-A E/W `0x94080`/`0x94060`), rows 1-5, cols 0-2 -- forcing any horizontal
propagation onto the (unblocked) shim row, mimicking `_XAie_SetupBroadcastConfig`.
`s1`=shim(0,0)/ch15 (START), `s2`=core(2,5)/ch14 (STOP); 6 measured cores.

Three variants, all clean on silicon (rc-0, zero TDR/IOMMU), all **confined**:

| Variant | Shim config | Result (counter_index 0..5) |
|---|---|---|
| `_dh` (initial) | shim untouched | `(0,3)`=2021/3400/2507 (free-run, varying); all others **0** |
| `_dh` + CLR | shim switch-A E/W cleared (`0x34064`/`0x34084`) | identical: `(0,3)`=3130/2042/...; others **0** |
| `_dh_swb` | shim switch-A **and** switch-B E/W cleared (`0x340A4`/`0x340C4`) | identical: `(0,3)`=1958/3039/...; others **0** |

Interpretation of the pattern (robust across 60 runs): `(0,3)` (col 0) gets ch15
(START, climbs col 0) but never ch14 (STOP) -> its counter free-runs (varying value
sampled at readback). `(2,3)` (col 2) gets ch14 but never ch15 -> STOP-without-START,
reads 0. Col-1 tiles get neither -> 0. **Both floods are confined to their source
column: the shim row carries no tile-to-tile E/W broadcast.**

For contrast, the free-flood kernel (no blocks) reaches all columns (`[137,129,121,
133,125,121]`, N=20 range-0) -- because it crosses columns via the *compute rows*,
never needing the shim row.

## Why: the reset is a two-channel FABRIC detour, not a shim-row path

Skeptical re-derivation of `XAie_SyncTimer` (aie-rt `xaie_timer.c:823-904`,
`_XAie_SetupBroadcastConfig:472-542`):

- `_XAie_SetupBroadcastConfig` is called with **only channel B** (`:850`); it blocks
  B's E/W on compute+memtiles, leaving B climbing vertically. **Channel B+1 is never
  blocked anywhere.**
- B+1 is a **free-flood distributor**: generated *once* at (0,0) (`:885`, no
  per-column trigger), unblocked on the whole array. Its job is to reach every
  column's shim tile -- which it does by **detouring through the fabric** (up to the
  memtile row, East across it, down into each shim). Each shim then re-sources B
  locally (`:860-874`) and B climbs that column.
- **The horizontal highway is the memtile/compute fabric E/W, not the shim row.** The
  two-channel design is itself the proof: if a clean shim-row-only path existed, one
  channel would suffice; the second exists precisely to flood the fabric and reach
  the shims by any route.

Our kernel confined because it blocked ch14 **and** ch15 E/W on the entire fabric --
deleting B+1's detour -- and the shim row cannot substitute.

### The shim row does not forward tile-to-tile E/W broadcast (AIE-ML)

We tested whether the shim row *can* be made to forward E/W. The shim PL tile has two
broadcast switches (A, B); one local source feeds both switches' propagation trees,
each gated by its own block bits, and AMD's own Phoenix code
(`_XAie_ErrorHandlingInitAie`, the `default` dispatch for AIE-ML) drives A and B in
lockstep to route errors E/W. We cleared **both** switches' E/W blocks on all three
shim tiles (`_dh_swb`): **still confined.** Since writes to shim(0,0) demonstrably
land (the flood *fires* from there via the same class of CSR write), this is a real
topology limit, not a write-delivery failure. Best-supported reading: aie-rt's shim
E/W config routes into the **L1/L2 interrupt controller** (a switch-B-aware consumer),
**not** tile-to-tile broadcast forwarding across the row -- so "AMD uses shim E/W on
Phoenix" and "our flood won't cross" are both true, about different consumers.

**Disclosed residual:** we did not add a block-VALUE readback to prove the block bits
read 0 after CLR (deprioritized as academic -- the reset uses the fabric regardless).
The "shim(0,0) writes land" argument gives high but not airtight confidence. If a
future need arises, the readback (VALUE regs `0x34068`/`0x34088` sw-A, `0x340A8`/
`0x340C8` sw-B) settles E1 (needs-config) vs E2 (no-wire) definitively.

## The reframe: `d_h` from the free-flood fabric measurement

Because the reset transports horizontally on the fabric, the **free-flood R3b `d_h`
is the right quantity** -- the bring-up `d_h≈4` (re-confirmed today at N=20, range-0,
`fit_residual≈7e-15`) that rev4 dismissed as "the wrong AIE-row path" is essentially
correct. IF-1's shim-row requirement is **retired**.

**One disclosed provenance caveat:** B+1's shortest detour crosses E/W at the
**memtile row** (row 1); the free-flood instrument measures **compute-row** E/W
(rows 2-5). If those per-hop costs differ, marginally. The single-scalar `d_h` model
already collapses per-row differences; a memtile-row-specific measurement is an
optional future refinement, not a blocker.

## Emulator fidelity gap (new)

`broadcast_origin_d` (`src/device/state/effects.rs:489-503`) gives **every** tile,
including the shim row, a direct E/W broadcast edge gated only by block masks. Real
Phoenix has **no functional tile-to-tile shim-row E/W broadcast edge**; a shim-sourced
horizontal broadcast costs `~2·d_v + n·d_h` (the fabric detour: up, across, down), not
`n·d_h`. Consequence: a *calibrated* `origin_D` on shim-sourced horizontal edges would
be optimistic by ~`2·d_v`. **Must be corrected before the Phase-6 flip** (route
shim-sourced horizontal through the fabric, or remove the shim E/W edge). Added to
`docs/known-fidelity-gaps.md`.

## Disposition

- **Block-shim-row Phase 2 apparatus retired.** Kernels `sp5_skew_r3b_pc_dh` (switch-A
  test) and `sp5_skew_r3b_pc_dh_swb` (switch-A+B test) kept as topology-experiment
  artifacts; the `r3b_pc_dh_*` gate/tally/observe-routing code stays (harmless, tested).
- **`d_h` = free-flood fabric measurement** (design rev5).
- **Design updated to rev5** (`specs/2026-07-01-sp5c-skew-characterization-design.md`):
  IF-1 shim-row requirement retired; decoupled-capture rationale for `d_h` withdrawn;
  memtile-vs-compute-row caveat disclosed. (The `d_v`-collapse-under-block-routing
  result stands -- it was always a separate, correct identifiability fact.)
- **Next:** Phase 3 (`d_v`) on the working free-flood + R1 instruments.

## Provenance

- Silicon: 60 clean HW runs, `build/experiments/sp5-skew/{gate_dh_n20,gate_dh_n20_clrfix,
  gate_dh_swb_n20}.log` + `task4_dh*/`.
- aie-rt: `xaie_timer.c:472-542,823-904` (single-channel block `:850`, per-column
  re-source `:860-874`, single Event_Generate `:885`); `xaie_interrupt_init.c` AIE-ML
  path `_XAie_ErrorHandlingInitAie` (shim E/W = IRQ routing); `xaiemlgbl_reginit.c:3695-
  3702` (PL 2 switches); `pm/xaie_reset*.c` (writes no broadcast regs).
- Block-mask config derivation: research passes 2026-07-02 (recorded in
  `plans/2026-07-02-sp5c-phase2-dh-capture.md` Sec.0).
- AM020 `chapter-2-aie-ml-tile-architecture.md:349-363` (compute OR-tree); interface
  tile has no E/W broadcast prose.
- Emulator: `src/device/events/broadcast.rs`, `src/device/state/effects.rs:489-503`.
