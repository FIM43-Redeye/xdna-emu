# Vector fuzzer: ledger reconcile + regression-gate hardening

**Date:** 2026-06-11
**Task:** #114 (verify Tier 1 ledger-complete) + the two infra gaps the
verification exposed.
**Decisions taken (Maya, 2026-06-11):** fresh-HW re-verify of the divergent
keys; fix BOTH infra gaps (self-healing flags + table-versioning).

---

## What the thorough re-verify found

Ran `fuzz-vector --replay` over the banked corpus with the post-#115 binary:
**24 match, 0 divergent, 45 error.**

1. **The substance is green.** All 6 keys the live ledger flags `divergent`
   (`add/Bf16x32`, `mmac_bf16/Bf16x32`, `mmac_i16/I32x16`, `mmac_i8/I32x16`,
   `mmul_bf16/Bf16x32`, `shl/I8x64`) are carried by at least one replay-MATCHED
   seed -- i.e. the current emulator matches **banked real-NPU silicon** on every
   one. Each corresponds to a fix that already landed (the 5 matmul/acc commits +
   #115). The divergences are resolved; the ledger just never reconciled.

2. **Gap 1 -- sticky divergent flags never self-heal.** `Ledger::mark_divergent`
   only inserts; `credit_keys`/`uncovered` skip flagged keys forever. A fixed key
   is permanently excluded -- the ledger can never reach 218/218 on its own. The
   `212/218, 6 divergent` headline is structurally stuck.

3. **Gap 2 -- table growth stranded the bank.** The 149->218 extension
   (`692ef54d`) inserted `t.push` calls *inside* the per-VecType loop, shifting
   existing `entry_idx` values. Replay regenerates each chain via `generate(seed)`
   under the live table, so for the 45 pre-extension seeds the regenerated keys
   (and the input pool) no longer match what was banked -- the keys-guard
   correctly refuses them. 45/69 seeds are un-replayable; the post-Phoenix replay
   gate currently covers only the 24 post-extension seeds.

4. **The regime trap is worse than a single false flag.** Credit is all-or-nothing
   per chain (`first_divergent_slice` == None gates `credit_keys(all stage keys)`).
   A fresh HW run while silicon is in the *canonical* NaN regime would fail any
   chain containing an Inf+NaN `add` stage at that slice -- re-flagging
   `add/Bf16x32` AND starving credit for every other key that chain carries
   (`mmac_bf16`, `mmul_bf16`, ...). A naive `--hw` re-verify can poison its own
   credit.

---

## The build (C -> B -> A -> rebuild -> D -> E)

### C. Regime-tolerant, type-aware comparator  [keystone; offline, unit-tested]

The NaN *payload* mantissa bits are functionally dead (the value is a NaN either
way) and silicon itself produces two regime-dependent values for them (#115).
Gating differential credit on them tests residual HW state, not emulator
correctness. Fix the comparator to mask exactly that field and nothing more.

- Replace the untyped 64-byte `first_divergent_slice` with a **type-aware** lane
  comparison that takes the chain's per-stage `out_type` (already on
  `table()[entry_idx]`).
- Per lane, for float element types (`Bf16x32`, `Float32`): if **both** sides are
  NaN (exp all-ones, mantissa != 0) **with matching sign** -> lane equal (payload
  is don't-care). Everything else exact: Inf-vs-Inf compares sign+exp,
  Inf-vs-NaN, NaN-vs-finite, and sign mismatches all still flag.
- Integer element types: exact byte compare, unchanged.
- Unit tests: 0xFF8C-vs-0xFF81 (datapath vs canonical) -> equal; +Inf-vs-NaN ->
  divergent; opposite-sign NaN (0xFF8C vs 0x7F8C) -> divergent; a real int
  mismatch -> divergent at the right slice.

**Effect:** the fuzzer is honest in *both* regimes, permanently -- it can never
again false-flag a NaN payload, on this campaign or any future one. This is the
deeper root of why the flags got stuck.

> **Fork C:** mask the dead NaN payload (recommended -- principled, scoped, and
> the only thing that makes a fresh-HW re-verify trustworthy) vs. run raw and
> hand-triage any regime re-flag. I strongly favor masking.

### B. Self-healing divergent flags  [offline, unit-tested]

- Add `--reverify`: at campaign start, move all `divergent` keys back into the
  uncovered set (clear the flags) and target them. Normal flow then re-earns each
  on silicon -- a clean chain credits + keeps it cleared; a still-divergent chain
  re-flags it. Explicit and honest: "clear the stale flags, re-earn them on HW."
- Keep a `resolved: HashMap<String, evidence>` (date + seed that cleared it) for
  audit, so a cleared flag carries its provenance.

> **Fork B:** explicit `--reverify` (recommended) vs. auto-heal on any clean hit
> that happens to touch a flagged key (more magic, less control).

### A. Durable bank format (table-versioning)  [schema-first]

Make replay reconstruct from the bank, not the live table.

- Extend `ChainRecord` to bank the **input pool bytes** and a **table-version**
  stamp (hash of the sorted key universe). `keys` is already banked and is
  table-independent.
- Replay path: if the banked table-version matches the current, regenerate as
  today (fast path). If it differs, reconstruct the `Chain` from banked
  stages+pool+keys and run the banked xclbin directly -- the emulator executes the
  compiled binary, so it needs the pool and out-byte layout, not the live table.
  Divergence localization uses banked `keys`. No more "table changed -> skip".
- New banks are durable across all future table edits.

### D. Fresh HW campaign -- re-verify the 6 keys  [HW-gated]

- Smoke first: `xrt-smi validate` (box was healthy at 15:38; not during any HW
  run). Chess-built kernels, single-device serial.
- `fuzz-vector --reverify --hw --target-hits 10` restricted to the 6 keys (round-
  robin over the cleared set). With comparator C, `add/Bf16x32` passes in either
  regime; the matmul/acc keys re-earn against live silicon.
- Acceptance: `--report` shows **218 covered, 0 divergent, 0 uncovered**. Bank the
  fresh clean seeds (durable format A).
- Capture a regime probe (a known +Inf+(-NaN) add) at campaign start and log which
  regime silicon is in, for the record.

### E. Stranded-45 disposition  [fork]

The 45 pre-extension seeds' pools are lost to the table shift; recovering them is
git-archaeology (checkout parent of `692ef54d`, regen pool, re-bank).

> **Fork E:** (a) **re-bank fresh** -- let D's campaign + format A produce durable
> banks covering the perishable keys; archive/delete the 45 stale dirs (coverage
> is key-based, and 149 keys ⊂ 218, so nothing is lost). Recommended -- no
> archaeology. (b) Git-archaeology to resurrect the exact old seeds. (c) Leave
> them; document.

---

## Sequencing & validation

1. C, B, A are pure-Rust, offline, unit-tested. `cargo test --lib` after each.
2. Rebuild release + FFI `.so` before any HW.
3. D is the only HW-gated step; run once after C/B/A are green.
4. E folds into D's banking.

`cargo test --lib` green throughout; no emulator *behavior* changes (the model is
already correct -- this is all verification-infra + comparator honesty). Commit
per part.

## Out of scope / open

- The gate flip (#113) reads the coverage-unit registry, not this ledger; it stays
  a separate explicit decision after 218/218 is clean.
- What flips the silicon NaN regime is still uncharacterized (low priority,
  payload is dead) -- see `2026-06-11-nan-payload-datapath-regime.md`.
