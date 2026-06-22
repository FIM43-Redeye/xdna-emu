# program_path: Through-Core Dataflow Edges — Design

**Date:** 2026-06-21
**Issue:** #140 (byte-identical emulator/HW trace reports)
**Status:** Design approved (survived two adversarial reviews). Ready for implementation plan.
**Predecessor:** config-path Tier E (merged, `bf6b44b0`) — DMA-config-derived `DmaBufferRelay` + `LockPair` edges.
**Successor (separate, slated):** the active trace-experimenter loop (plan → actuate trace config → re-run → unify on one timeline).

---

## 1. Goal, ceiling, and non-goals

**Goal.** Derive the one dataflow relay that config alone cannot reach — the **compute-tile** relay
`S2MM(in1) → core → MM2S(out1)`, bridged by the core's `use_lock` acquire/release operations — so
that compute-tile trace events can be oriented the same way memtile DMA events already are.

**Why config can't reach it.** Tier E's `LockPair` links a *single* lock's releaser→acquirer when
both are DMA buffer descriptors (BDs). On the compute tile, **no BD** acquires the input lock or
releases the output lock — the **core program** does. `DmaBufferRelay` needs same-tile buffer
overlap, but the compute tile's input and output buffers are disjoint. So both Tier E edge kinds
structurally cannot bridge this tile (confirmed below). The bridging evidence lives in the core ELF,
a different source (program, not config) — hence a distinct `program_path` layer.

**Ceiling (option X, chosen).** Derive the edge statically and validate it against the emulator's
**runtime core-lock + buffer-touch enactment** (E4-style superset check). This proves the static
derivation is a sound superset of what the emulator dynamically does. **Trace-based and hardware
validation are explicitly deferred** to the trace-experimenter-loop plan, which will instrument
compute-tile ports itself and gather the data to validate end-to-end. We do not build a throwaway
manual compute-port trace here.

**Non-goals (deferred or out of scope by construction):**
- **Value-dependence / taint.** The edge claims *structural data-contact under producer/consumer lock
  ordering* — that the core had the **opportunity to relay** these bytes — NOT that the output value
  depends on the input value. This is a principled ceiling, not laziness: the oracle is hardware
  trace (timing + port activity), and no trace on any silicon can distinguish `out = in + 1` from
  `out = const` (byte-identical traces). Asserting value-dependence would assert a property the
  oracle can never confirm or refute — the same co-firing overreach the engine exists to refuse,
  relocated from timing to values. Structural contact is the strongest claim the oracle can back.
- **General through-core coverage.** Sound static analysis yields **safe false-negatives** on kernels
  with scratch/temp buffers, library/intrinsic calls the scan can't follow, or buffer handles
  arriving via stack spill rather than an immediate. This rule fires on objectFIFO-passthrough /
  simple-elementwise kernels (add_one and kin) and **blocks (does not guess)** elsewhere. The plan
  must NOT claim general coverage.
- **The active trace-experimenter loop** (config synthesis, re-run, cross-config timeline
  unification). Separate slated plan.

---

## 2. Ground truth (add_one_using_dma, compute tile (0,2))

Verified against the captured config dump and the Chess compute-core disassembly
(`main_core_0_2.elf.lst`) in `/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/`.

**Buffers** (config dump, tile-local offsets; core sees them at `0x70000 + offset`):
- `in1`  buffers at `[0x400, 0x440)` (= `[1024, 1088)`): `in1_buff_0 @ 0x400`, `in1_buff_1 @ 0x420`.
  Core-local: `0x70400 / 0x70420`.
- `out1` buffers at `[0x440, 0x480)` (= `[1088, 1152)`): `out1_buff_0 @ 0x440`, `out1_buff_1 @ 0x460`.
  Core-local: `0x70440 / 0x70460`.
- in1 and out1 are **disjoint** → `DmaBufferRelay` (E2) correctly emits nothing here.

**Locks** (per-tile; the compute tile owns locks 0–3):
- `S2MM0` writes in1: acquires lock0 (space), **releases lock1** (data-ready).
- Core: **acquires lock1 + lock2**, **releases lock0 + lock3**.
- `MM2S0` reads out1: **acquires lock3** (data-ready), releases lock2 (space).
- `{lock1 released by S2MM0} ∩ {lock3 acquired by MM2S0} = ∅` → `LockPair` (E3) correctly emits
  nothing here.

**Core body** (MLIR `input_with_addresses.mlir`): `%5 = arith.addi %4, %c1_i32; store %5` — i.e.
`out1 = in1 + 1`, genuinely value-dependent (but, per §1, we do not and cannot claim that — only
the structural contact).

**Access pattern** (the feasibility crux): buffers are accessed **register-indirect with
post-increment** — `LDA r25, [p6], #4` / `ST r31, [p7], #4` — with the base materialized far
upstream by `MOVXM p6, #0x70420` etc. A naive "load whose address ∈ range" scan matches **nothing**;
the address lives in a register. This dictates the static-analysis design (§4).

---

## 3. The derivation rule

Emit a `CoreLockRelay` edge `S2MM_w → MM2S_r` (data-ready orientation: writer-side DMA port → reader-side
DMA port) **iff ALL** of the following hold for a single compute tile:

1. **(config)** `S2MM_w` **releases** lock `L_in`; `MM2S_r` **acquires** lock `L_out`. (From captured
   BD lock fields — already in the Tier E dump.)
2. **(program — lock)** the core **acquires** `L_in` and **releases** `L_out`. (Static parse of the
   core ELF's `use_lock` ops; encoding derived from llvm-aie TableGen, via the emulator's existing
   decoder.)
3. **(program — buffer contact)** the core **reads** the buffer `S2MM_w` writes (in1's range) **and
   writes** the buffer `MM2S_r` reads (out1's range). Recovered by reaching-definition (§4), not an
   address scan.
4. **(program — ordering / dominance)** within one acquire→release region of the core:
   `acquire(L_in)` dominates the in1 load, the in1 load precedes the out1 store, and the out1 store
   precedes `release(L_out)`. (A CFG dominator query — the implementable form of "the core consumed
   in1 *before* producing out1 under these locks." Replaces the ill-defined "same loop scope," since
   the pipelined ELF has no lexical scopes.)

**Orientation.** `src = S2MM_w` (master DMA port, buffer writer) → `dst = MM2S_r` (slave DMA port,
buffer reader). Identical convention to Tier E (E2/E3). Back-pressure (the reverse, space-lock
direction) is never emitted: criterion 1 inspects S2MM **releases** and MM2S **acquires** only.

**Semantic.** The edge — and the `program_path` fact it produces — means *"the core had the
opportunity to relay these bytes from `S2MM_w`'s buffer to `MM2S_r`'s buffer under producer/consumer
lock ordering,"* NOT value-dependence. Cites and docs use "data-contact under lock ordering," never
"dataflow implies value-flow."

**Soundness shape.** This is an **intersection** of independent structural evidence — lock-pairing
(ordering) ∩ buffer-contact (which bytes) ∩ dominance (the contact happens between the locks) — the
same belt-and-suspenders philosophy as Tier E's E2∩E3. Each criterion alone is insufficient
(lock-only = the co-firing trap the first review killed; buffer-only = no ordering; either without
dominance = cross-iteration/cross-nest aliasing).

---

## 4. Static analysis design (the spike-first risk)

All in Rust, reusing the interpreter's existing TableGen-driven decoder (`src/interpreter/decode/`);
the emulator already decodes and executes these ELFs.

**P0 — SPIKE FIRST (gate the whole plan).** Before any edge code, prove on the real add_one Chess
compute ELF that criteria 3+4 are statically recoverable. The second review already half-validated
this (found the `MOVXM` immediates and confirmed they reconcile to `0x70000 + tile_offset`). The
spike formalizes it into the two analyses below and confirms they fire on add_one. **If buffer
pointers turn out non-recoverable (e.g. Peano emits stack-spilled handles), STOP and report** — do
not ship a guess. Same discipline as the Tier E BD-capture spike.

**Lock-op extraction (criterion 2).** Walk the decoded core instruction stream; identify `use_lock`
acquire/release ops and their `(lock_id, acq|rel, value)`. The decoder already exposes lock-op
predicates/params used by the interpreter to execute them
(`decode/decoder.rs::has_lock_acquire`, `execute/control.rs::get_lock_acquire_params`).

**Buffer-pointer reaching-definition (criterion 3).** For each load/store, resolve its pointer
register to a defining `MOVXM`/`MOVX` immediate by reaching-definition over the decoded stream
(post-increment strides keep the *initialization* immediate as the buffer base). Compare that
immediate against the buffer's **core-local** range, i.e. `0x70000 + config_tile_offset` where the
config range comes from the BD `base_addr` already captured in Tier E. **Address-space
reconciliation** (config tile-offset ↔ core-local `0x70000+`) is explicit, not assumed. A subset
touch (vectorized loop over part of a buffer) still has its base immediate in-range — fine for a
superset. A pointer whose base is not an immediate (stack spill, computed handle) → **cannot bound →
block (safe false-negative), reported**.

**Dominance ordering (criterion 4).** Build the core ELF's basic-block CFG (decoder already yields
control flow); answer the dominance/precedence query of §3.4 within one acquire→release region.

**P-capture.** Expose the per-core result (the lock ops + resolved buffer contacts + their ordering)
in a structure reachable from `DeviceState`, alongside how Tier E reaches `bd_configs[]`, so
`resolve_route_graph` can consume it.

---

## 5. Edge representation & engine integration

**Rust edge.** Add `EdgeKind::CoreLockRelay` to the existing `resolve_route_graph`
(`src/device/stream_switch/route_graph.rs`) — the *same* graph that holds `DmaBufferRelay` /
`LockPair`. One route graph = one source of truth; the A5/E4 validation harness and the generator
both reuse it verbatim. The edge **carries the buffer byte-range** (for dedup and deeper-pipeline
disambiguation), mirroring E2's overlap key. No separate parallel graph.

**Engine predicate (program_path made real, not cosmetic).** Today the engine collapses every
non-identity edge kind to a single `config_path` predicate (`tools/inference/ledger.py` ~L46) and
orientation is keyed solely on `config_path` (`tools/inference/rules.py` ~L40). Pooling an
opportunity-grade core edge with config-certain CDO route edges would launder confidence and make
the two unfilterable in an audit. So:
- Add `"program"` to the ledger's kind set; emit `Fact("program_path", (a, b), …)` for `program:`
  cites (distinct from `config_path`).
- Parametrize the orientation-derivation rule so it queries **both** `config_path` and `program_path`
  predicates (same `a=parent, b=child` contract, pinned by Tier E's E6), while keeping them
  separately reportable/gateable at different confidence.

**Generator.** `tools/config_extract/generator.py` emits the `program_path(a,b)` fact for each
`CoreLockRelay` edge, `a=parent` (the `S2MM_w` producer event), `b=child` (the `MM2S_r` consumer
event), with a `program:` cite. Soundness framing unchanged: reachability-gated, declines co-firing.

---

## 6. Validation (ceiling X)

Extend the runtime recorder (Tier E's `enable_lock_recording` / `take_lock_events_by_tile` lives in
the DMA lock arbiter) to also record **core-initiated** activity, as **two runtime witnesses from
different executor paths**:
- (i) **core lock handoffs** — core acquire/release, recorded in the core executor's lock path
  (`src/interpreter/execute/control.rs` → `src/device/tile/locks.rs`).
- (ii) **core buffer touches** — the actual load/store target addresses the core executes, recorded
  in the core execute path (`src/interpreter/execute/`). Genuinely separate from the lock path.

**Gate.** Every enacted `(core lock handoff ∧ core buffer touch)` is covered by a static
`CoreLockRelay` edge with matching orientation. This is a **necessary-not-sufficient superset
check** (`static ⊇ dynamic`), exactly as A5/E4 are scoped — it proves the static derivation matches
the emulator's dynamic behavior; it is **not** a hardware-fidelity oracle (that is deferred).

**Independence caveat (honest).** The static side derives lock ids *and* buffer pointers from the
**same decode pass**, so they are not two independent static signals — a decoder bug would corrupt
both identically. The real independent cross-check is **static buffer range (from the ELF) vs the
BD-config byte-range (from CDO, E2's `DmaBufferRelay` source)**: config-range ∩ ELF-access *is* two
sources. The plan states this limitation plainly rather than claiming false independence.

---

## 7. Testing

- **Rust unit** (`route_graph` tests): add_one's compute tile yields **exactly** the
  `S2MM0 → MM2S0` core-relay edge; asserts the reverse (back-pressure) edge does **not** exist; a
  scratch-buffer / non-immediate-handle fixture yields **no** edge (safe false-negative).
- **Rust validation test** (E4-style): the two-witness recorder; enacted `(lock ∧ buffer)` handoffs
  ⊆ static edges, matching orientation; a deliberately back-to-front orientation would fail the gate
  (genuine oracle, not vacuous).
- **Python** (`tools/config_extract` + `tools/inference`): generator emits the `program_path` fact;
  the engine **derives** it through the new parametrized rule (not just loads it — the C4/E6 lesson:
  a ledger that loads ≠ a ledger that derives); audit clean; `program_path` is a *distinct*
  predicate, filterable from `config_path`.
- All offline; no NPU.

---

## 8. Risks

- **R1 (gating): static buffer-pointer resolvability.** Mitigated by P0 spike-first; Chess add_one
  confirmed recoverable. Peano / stack-spilled handles → safe false-negative + report, not a guess.
- **R2: dominance recovery on heavily pipelined ELFs.** Chess unrolls/software-pipelines; the CFG
  query must operate on the linearized stream. The spike validates the dominance query fires
  correctly on add_one's pipelined body.
- **R3: narrow coverage** (§1 non-goals). Documented, not fixed — inherent to sound static analysis.
- **R4: engine-predicate change touches shared orientation code.** Keep the parametrization additive
  (both predicates queried); re-run the full Tier E / Plan 1 suite to confirm `config_path`
  derivation is unchanged.

---

## 9. Provisional task shape (to be expanded by writing-plans)

- **P0 — Spike:** prove criteria 3+4 statically recoverable on the real add_one Chess core ELF
  (lock-op extraction + buffer-pointer reaching-def + dominance). Gate: fires on add_one; report if
  blocked. *No edge code until this passes.*
- **P1 — Core lock + buffer static capture** into a `DeviceState`-reachable structure (reuses decoder).
- **P2 — `EdgeKind::CoreLockRelay`** in `resolve_route_graph`: the §3 intersection rule, carrying
  buffer range; back-pressure structurally excluded.
- **P3 — `program_path` predicate** in the engine (`ledger.py` kind set + `Fact`, `rules.py`
  orientation parametrization) + generator emits the `program:` fact.
- **P4 — Two-witness runtime validation** (E4-style): record core lock handoffs + buffer touches;
  gate static ⊇ dynamic; config-vs-ELF cross-check.
- **P5 — Engine derive test** + full-suite regression (config_path unchanged).

Trace-based / HW end-to-end validation: **deferred to the trace-experimenter-loop plan.**
