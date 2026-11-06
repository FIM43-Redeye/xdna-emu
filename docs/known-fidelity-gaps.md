# Known Fidelity Gaps

A registry of **confirmed** points where the emulator (or the aiesim oracle)
provably disagrees with real NPU hardware. Each gap is documented in detail in a
linked finding; this page and the [`fidelity-gaps/`](fidelity-gaps/) folder are
the index, so the pattern is visible in one place and we don't re-investigate the
same gap twice. (That happened once already: a stale "shim queue is 8-deep"
caveat sent a whole session chasing a dead mechanism.)

**Scope:** confirmed HW-disagreement gaps only -- behaviors where we have ground
truth that the model is wrong. This is *not* the list of not-yet-implemented
features (vector compute semantics, stream-switch per-port types, micro-timing);
those live in [`toolchain-sources.md`](toolchain-sources.md) and the
[roadmap](../ROADMAP.md).

**How these surface:** mostly via the aiesim oracle (the XRT-plugin -> aiesim
path). The count rising is the oracle doing its job -- before it, these were
invisible. Each was a deliberate "document, don't chase" call; see the per-gap
rationale in the class file.

**How this is organized.** Gaps are classified by **subsystem** (the file
boundary) -- when you're working on a subsystem, its file shows you every known
gap there so you don't re-chase one. Two categories are a genuinely *different
kind* of gap (model more capable than HW; gap in the oracle not us) and get
posture-named files instead. Each class file carries a `posture:` field in its
frontmatter, so the cross-cutting "what kind of wrong" axis stays queryable.

---

## Class directory

| Class | File | Posture | Gaps (status) |
|-------|------|---------|---------------|
| **DMA & stream resources** | [`fidelity-gaps/dma-stream-resources.md`](fidelity-gaps/dma-stream-resources.md) | optimistic-where-strict | stream buffering and per-hop latency **modelled** 2026-11-04 (crossing = slave + master FIFO, latency inside them, wire 0; R4-D3 data race 25/25, core-core absorb, S2MM backpressure windows +15/+1, round-trip slope 8/hop, all EMU == HW); open: round-trip intercept (+4 cy per direction), TCT token buffer (unbounded, probe deferred), MM2S first-granule starvation pulse, memtile strided-BD cost (HW 264 vs 257 per BD, EMU 258), BD-driven (de)compression data wrong; BD reuse/pool (won't-fix); send/recv port cadence (recv exact; send substantially resolved) |
| **Trace encoding** | [`fidelity-gaps/trace-encoding.md`](fidelity-gaps/trace-encoding.md) | encoding-artifact | held-level falling-edge (documented), count under-emission (**closed** 2026-06-27) |
| **Core compute timing** | [`fidelity-gaps/core-compute-timing.md`](fidelity-gaps/core-compute-timing.md) | needs-HW-empirical | scalar write-port collisions (**fixed** 2026-11-03: a delayed scalar write (`mov.dN`, `lda`, `mul`, vector extract/compare) is dropped by a later-issued write to the same register file through the same TableGen write unit landing in the same cycle, and two ports reaching one register OR; 510 silicon trials, EMU == HW; TableGen overpredicts across register files; open: `mov SS`/`sthb` ports unprobed); scalar-side register widths (**fixed** 2026-11-03: every register keeps its own bits, TableGen class widths narrowed by measured silicon for the shift, control and status registers, 68 ISA test points, EMU == HW; status flags: `srCarry` (**modelled** 2026-11-04: TableGen definers only, carry out for the add family, borrow for sub/sbc, sign for abs, next-bundle visibility; 1112 silicon records, EMU == HW) and `srSRS_of`/`srUPS_of` (see vector compute); `srFPFlags`/`srF2IFlags`/`srF2FFlags` **modelled** 2026-11-05, see vector compute); cascade link (**modelled** 2026-11-04 from sync-word probes: 2-word credit window, word readable 2 cycles after `vmov MCD`, held read restarts with no penalty, a pop frees its slot in the same cycle; per-side split not observable; cores start one at a time ~1448 cycles apart in CDO order (silicon jitter 1103-2018, EMU uses the median); same-column timers aligned (timer-phase probe), columns ~140 apart (not modelled); EMU == HW on 78 timing cases; found: EMU cross-tile memory visibility follows core step order, open); MEMORY_STALL mechanism **and per-iteration magnitude** root-caused + modelled (bank-arbitration arc, HW-confirmed cycle-by-cycle; 24 stalls per rep in a 47-cycle burst, EMU == HW). The "~1.9x consumer over-production" once recorded here was **FALSE and is retracted**: it compared a 16-rep EMU run against a 9-rep HW trace window (16/9 = 1.78). DMA bank-demand cadence **fixed** 2026-07-14 (16B granule, HW-measured -- producer 102->4 vs HW 1); DMA memory-pressure events closed by the staging-FIFO model; producer core-vs-DMA bank conflict count **fixed** 2026-07-15 (core-class-priority arbiter + DMA urgency/backoff, C-cycle fix -- anti-lock broken, 68->870/900 conflicts, within ~20% of HW's ~1100/1162); residual MEMORY_STALL undercount on that same probe (EMU ~0 vs HW 18-20/24) documented as a sub-cycle phase-average tail below the lockstep model's resolution, deferred (not calibrated) to the future variable-latency memory model; open: pipeline-fill depth (EMU 5 reps free-run, HW ~7), bank-arbiter resume-cycle hole (dormant/unreachable on corpus); lock-arb cost modeled, pulse emission pending |
| **Vector compute** | [`fidelity-gaps/vector-compute.md`](fidelity-gaps/vector-compute.md) | mostly-resolved | accumulator read/write timing (**resolved** 2026-10-31: one silicon-fitted rule equal to the TableGen itinerary cycles, 468 probe pairs exact; `vlda am` loads **resolved** 2026-11-04: they follow the same rule (def 7) instead of a wall-clock load commit, 2724 probe points incl. write-after-write exact, fixes the Chess cfloat `mac`/`msc_elem_8` cases; vector-register loads **resolved** 2026-11-04: the vector file's rule (every reader from distance 7), the B load channel (`vldb` forms ORed with a same-cycle W_WM write on their register bank, beaten by W_WA) and the `vldb.4x` same-bundle hazards (shared address read with `vst`/`vpack`, address handed to a load-unit-A load), 3773 probe points exact; open: a gather-redirected load's own offset and post-modify, `vldb.sparse` ties; a taken branch delays no in-flight write (**fixed** 2026-11-04, 837 probe points); fp32-input MAC via bf16 splits **closed** on the Chess `_accuracy_*` cases; Chess `mul_2x8_8x2` reads an uninitialized accumulator, not a gap), register-port collisions (**modelled** 2026-11-01: schedules TableGen forbids, where two writes share a register-file port and landing cycle or a late accumulator read meets a mover or the SRS shifter; bank/lane write resolver, read-port steal, shifter wired OR, 12 collision-probe directories exact against silicon; `VMOV_CASCADE_READ` collisions (**modelled** 2026-11-01 via the two-core harness probe: 1326 points exact, hold-at-fifth-bundle read timing, two lane rules; its `SEMAPHORE` partners `vmov MCD`, stream moves and a second read have no effect on silicon (2026-11-04, 32 cases); open: lock partners), narrow odd-bank accumulator writers drive lane 1 (2026-11-04); open: `UPS_UNIT` inseparable from `CM_WM_PORT`, unprobed `vunpack`/`vlda.am` writes unconfirmed; Chess schedules through scalar-port reservations TableGen forbids, 666 hits over 98 ELFs vs 0 over 108 Peano), source_forward alignment (benign), NaN payload regime (**resolved**), fused/standalone unpack lane map, sign and 2D/3D addressing (**resolved**, unaligned loads **resolved** 2026-11-02), fused/standalone pack lane map, `crPackSign` and `crSat` clamp rules (**resolved**, open: `crSat`/`crPackSign` values above their observable range, AIE2P pack layout), sparse MAC mask rule and `vldb.sparse.*` load model (**resolved** on NPU1; open: one unexplained `dense40`/`unprimed` read, one sparse kernel failing end to end (`vec_macsparse_4x16_16x4_alt`, Peano: mask or A rows of tile 1+, pointer and q-register timing unprobed), run-time sign of the `d` forms (crUPSSign / crSRSSign / crVaddSign) and UPS saturation (**resolved** 2026-11-04, dynsign probe suite; 2D/3D fused, `vsrsm` and every fused addressing mode **resolved** 2026-11-04, satflags probe; `vlda.ups` reads crSat with the sign and shift at issue+6, **fixed** 2026-11-04), overflow flags `srSRS_of`/`srUPS_of` (**modelled** 2026-11-04: any lane out of range before the clamp under every crSat, symmetric range drops the signed minimum, sticky, visible 3 bundles after SRS / at the UPS accumulator cycle; 5356 silicon records over 74 forms, EMU == HW), MAC configuration word (**resolved** 2026-11-03: every field and all 128 multiplier keys, complex and float-conf modes, `term_neg`; bf16 `.f` keys, `term_neg` and `shift16`, sparse 4-bit `term_neg` and dense keys on sparse instructions **resolved** 2026-11-05; see [`fidelity-gaps/mac-conf.md`](fidelity-gaps/mac-conf.md); open: integer keys on `.f` instructions, sparse bf16 `term_neg`), `vldb.compr.*` unmodelled, AIE2P sparse layout) |
| **Addressing** | [`fidelity-gaps/addressing.md`](fidelity-gaps/addressing.md) | resolved | 2D / 3D descriptor post-modify of every load and store (**resolved**: one rule on silicon, all 92 TableGen records classified); unaligned accesses (**resolved** 2026-11-02: every load and store rounds its address down to the access width, `vldb.4x*` gathers and `vldb.sparse.*` have their own rules; 328-case silicon probe); address-bit decode (**resolved** 2026-11-04: only bits 16-17 pick the direction, bits 18-31 ignored, for gather lanes and pointers alike; 196 probe accesses, fixes the Chess `load_4x*` cases with the generator now pointing address operands into the input); loads with no memory behind them (**resolved** 2026-11-06: each load port answers them from a six-entry ring per 16-byte half of the rows it fetched; gathers split lanes 0-1 / 2-3 over ports A / B; 12-case silicon probe, EMU == HW); open: count-write latency, west of column 1 (column-0 contents), `vldb.compr.*` alignment (unmodelled), stale ring bytes after a sparse line, `padds [sp]` decode (llvm-aie fix pending) |
| **Event broadcast** | [`fidelity-gaps/event-broadcast.md`](fidelity-gaps/event-broadcast.md) | fixed | shim-row E/W broadcast edge (**fixed** 2026-07-02) |
| **Host / firmware dispatch** | [`fidelity-gaps/host-firmware-dispatch.md`](fidelity-gaps/host-firmware-dispatch.md) | off-array / firmware seam | dispatch latency (deferred) and core reset-deassert (**root-caused**, Part 1 landed `aa825ae9`); core register state across launches (silicon keeps data registers, resets control/status registers and `le`; launch reset values **modelled** 2026-11-04, persistence open, harness scrub as the workaround); firmware time is now coupled to the array by the exact clock ratio (`src/timebase.rs`; the per-step cost model is private); open: the once-per-run stall and unprobed off-array costs; the Phoenix management-firmware platform gaps are tracked in `xdna-emu-priv: research/firmware/host-firmware-dispatch.md` |
| **Permissive-vs-HW** (inverse gap) | [`fidelity-gaps/permissive-vs-hw.md`](fidelity-gaps/permissive-vs-hw.md) | permissive-where-broken | task-API memtile-relay TDR (documented) |
| **aiesim oracle** | `xdna-emu-priv: research/aiesim/aiesim-oracle.md` | oracle-bound | c2c shared memory, control-read aliasing, cross-domain skew, trace micro-timing |

---

## Maintenance

When a new confirmed HW-disagreement surfaces:

1. Add a one-line row to the appropriate **class file** under
   [`fidelity-gaps/`](fidelity-gaps/), pointing at the detailed finding, and note
   whether it's fixed, documented-and-deferred, or won't-fix.
2. Update that class file's frontmatter `status:` line and this page's class
   directory row if the class's headline status changed.
3. If the gap is a genuinely new *kind* (a new subsystem, or a new posture like
   the inverse `permissive-vs-hw` category), add a new class file and a directory
   row here.

Keep the class files an index too -- current-state plus a pointer; the
blow-by-blow lives in the linked findings under
`xdna-emu-priv: research/findings`.
