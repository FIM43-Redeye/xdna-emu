# Producer bank-collision C-cycle fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the emulator's producer core-vs-DMA bank arbitration faithful — core-class priority with a DMA urgency override, plus a DMA backoff-on-denial that breaks the phase anti-lock — so the dense `collide` probe lands at ~1000 conflicts / ~18–24 stalls (HW band) instead of today's 68/30, deterministically.

**Architecture:** Three coupled cycle-level changes: (1) `bank_arbiter.rs` winner selection goes from symmetric round-robin to core-class priority + urgency override; (2) `stepping.rs` DMA gains an `urgent` flag (egress FIFO near underflow) and backs off (rather than dodging on a fixed schedule) when core-priority denies its granule fetch — the backoff shifts its fetch phase so it can't stay anti-locked; (3) `coordinator.rs` threads urgency from the DMA demand into the arbiter. The FIFO depth stays 12 (do not retune).

**Tech Stack:** Rust (emulator core), `cargo test --lib`, the `producer_probe.py` HW/EMU probe + trace pipeline (Chess build, `bridge-trace-runner`, `parse-trace.py`, `producer_probe_measure.py`).

## Global Constraints

- Derive from the toolchain; never hardcode what can be extracted. No change to `banking.rs` (bank map HW/ISS-verified) or the 1-in-4 granule width.
- **No phase constant, no RNG, no FIFO-depth retune.** The anti-lock break must come from the deterministic backoff-drift mechanism, not a tuned offset. FIFO depth stays at its existing value (`model_builder.rs:~235`).
- Access-type-BLIND arbiter (no read/write field on `Requester`). The dense-read > dense-write ordering must emerge from the core presenting both load ports' real demands, not a new rule.
- Validate only against the robust anchors: `apart` 0/0, `collide` 1098–1103/18–20, `collide_read_dense` ~1162/~24. Never against intermediate densities or `collide_sticky` (HW-nondeterministic, proven by the N=10 same-binary check). `collide_read`'s "~0" is a bimodal 16/0 — not an anchor.
- Preserve consumer-side core-vs-core self-collision (~91%) and the arbiter's retry-contract + multi-bank sticky proofs.
- `cargo test --lib` must pass after every task; ISA 4815/4815 and the bridge suite are the final gate. Commit after each task. Commit messages end with "Generated using Claude Code." No emoji.

---

## File Structure

- `src/device/bank_arbiter.rs` — winner selection (class priority + urgency), demand interface (urgency), test module re-expression. One responsibility: per-physical-bank arbitration.
- `src/device/dma/engine/stepping.rs` — DMA memory-side demand: `next_granule_fetch`/egress-staging fetch gains backoff-on-denial state and surfaces `urgent` via `peek_bank_demand`.
- `src/interpreter/engine/coordinator.rs` — `arbitrate_memory_banks` (~2059): thread urgency from DMA demand into `arbitrate`.
- `tools/experiments/producer_probe.py`, `producer_probe_measure.py` — existing probe (unchanged) used for the EMU-side gate + anchor validation.
- `docs/known-fidelity-gaps.md` — new row (success: residual conflict-count phase-average note; or fallback: residual stall gap).

Tasks 1–3 build the coupled mechanism. **Task 4 is a HARD GATE** (stall-count derisk) — do not proceed to 5+ until it passes or a fallback decision is made. Tasks 5–7 finalize (tests, verification, docs).

---

### Task 1: Arbiter — core-class priority + urgency override

**Files:**
- Modify: `src/device/bank_arbiter.rs` (winner selection in `arbitrate`; demand type)
- Test: `src/device/bank_arbiter.rs` `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: nothing new.
- Produces: `arbitrate` accepts urgency per demand. Choose ONE and use it consistently downstream (Task 3): widen the demand item to `(Requester, u16 /*bank_mask*/, bool /*urgent*/)`, OR keep `(Requester, u16)` and add a second parameter `urgent: &[Requester]`. RECOMMENDED: the parallel-set form `fn arbitrate(&mut self, demands: &[(Requester, u16)], urgent: &[Requester]) -> Arbitration` — it leaves the hot `SmallVec<[(Requester,u16);5]>` demand shape and all existing call sites' tuple construction untouched, threading urgency as a small separate slice. `Arbitration` fields unchanged.

- [ ] **Step 1: Write failing tests for class priority + urgency**

Add to the test module. These encode the three new rules (core beats non-urgent DMA; urgent DMA beats core; within-class rotation among cores):

```rust
#[test]
fn core_beats_nonurgent_dma_on_a_contended_bank() {
    let mut arb = BankArbiter::new();
    let a = arb.arbitrate(
        &[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 0)],
        &[], // no urgent DMA
    );
    assert_eq!(a.contended_banks, 1 << 0);
    assert!(a.lost.contains(&Requester::S2mm(0)), "the DMA loses to the core by class priority");
    assert!(!a.core_lost(), "the core wins");
}

#[test]
fn urgent_dma_beats_the_core() {
    let mut arb = BankArbiter::new();
    let a = arb.arbitrate(
        &[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::Mm2s(0), 1 << 0)],
        &[Requester::Mm2s(0)], // MM2S is near FIFO underflow
    );
    assert!(a.lost.contains(&Requester::Core(CorePort::LoadA)), "urgent DMA forces the grant");
    assert!(a.core_lost());
}

#[test]
fn two_core_ports_still_rotate_within_the_core_class() {
    // Core-vs-core is unchanged: exactly one core port wins, and across two
    // contended cycles the winner alternates (within-class round-robin).
    let mut arb = BankArbiter::new();
    let d = [(Requester::Core(CorePort::LoadA), 1u16 << 0), (Requester::Core(CorePort::Store), 1u16 << 0)];
    let first = arb.arbitrate(&d, &[]).lost;
    let second = arb.arbitrate(&d, &[]).lost;
    assert_eq!(first.len(), 1);
    assert_ne!(first, second, "within-core rotation still alternates the loser");
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib bank_arbiter 2>&1 | tail -40`
Expected: FAIL — `arbitrate` takes one arg (arity mismatch) / new behavior absent.

- [ ] **Step 3: Implement class-priority + urgency winner selection**

In `arbitrate`, change the signature to `(&mut self, demands: &[(Requester, u16)], urgent: &[Requester])`. In the per-bank winner loop, replace the single `min_by_key` over all `wanters` ordinals with class selection:
  1. Partition this bank's `wanters` into `urgent_dma` (ordinal's requester is a DMA channel present in `urgent`), `core` (ordinals 0..NUM_CORE_PORTS), and `other_dma`.
  2. Winner class = first non-empty of [`urgent_dma`, `core`, `other_dma`].
  3. Within the winner class, keep the existing rotor rule (`min_by_key(|ord| (ord + NUM_REQUESTERS - start) % NUM_REQUESTERS)`) so within-class rotation and the retry bound are preserved. Advance `self.rotor[bank]` by +1 every contended cycle exactly as today (do NOT change the rotor advance — the module's phase-lock-avoidance argument depends on the steady +1 tick).
  4. `denied[ord] |= bit` for every wanter that is not the winner (unchanged). `contended_banks` counting unchanged (>=2 wanters).

Map an ordinal back to "is this a DMA in `urgent`?" via the `urgent` slice (compare `Requester::ordinal()`), computed once per `arbitrate` call into a small `[bool; NUM_REQUESTERS]` urgency-by-ordinal table to avoid rescanning per bank.

- [ ] **Step 4: Run the new tests + the whole arbiter module**

Run: `cargo test --lib bank_arbiter 2>&1 | tail -40`
Expected: the 3 new tests PASS. Several EXISTING tests will now FAIL (they assert symmetric fairness) — that is expected and handled in Task 5. Note which fail; do not fix them yet.

- [ ] **Step 5: Update every `arbitrate` call site to pass `urgent`**

`rg 'arbitrate\(' src/` — the production call is `coordinator.rs` `arbitrate_memory_banks` (Task 3 fills real urgency; for now pass `&[]`). Any test/bench call sites pass `&[]`. Make it compile.

Run: `cargo build 2>&1 | tail -20`
Expected: compiles.

- [ ] **Step 6: Commit**

```bash
git add src/device/bank_arbiter.rs src/interpreter/engine/coordinator.rs
git commit -m "feat(arbiter): core-class priority + DMA urgency override

Core beats non-urgent DMA on a contended bank; an urgent (FIFO-near-underflow)
DMA beats the core; within-class round-robin preserved. Existing symmetric-
fairness tests intentionally red until re-expressed (Task 5).

Generated using Claude Code."
```

---

### Task 2: DMA — urgency flag + backoff-on-denial

**Files:**
- Modify: `src/device/dma/engine/stepping.rs` (`next_granule_fetch` ~311, `channel_bank_mask` ~156, `peek_bank_demand`, `step_with_denied`/`is_mem_denied` path; per-channel state on `ChannelContext`)
- Test: `src/device/dma/engine/stepping.rs` (or the engine's test module)

**Interfaces:**
- Consumes: nothing new.
- Produces: `peek_bank_demand` (called by `coordinator.rs`) must expose, per demanded channel, whether it is `urgent`. Add a companion `fn peek_urgent(&self, layout) -> impl Iterator<Item=Requester>` OR return urgency alongside each demand — planning-local, but Task 3 consumes "which demanded DMA channels are urgent." `urgent` for an MM2S channel = its egress staging is at/below an underflow threshold (`staged_words <= URGENT_WATERMARK`, and there is still stream to feed). Define `URGENT_WATERMARK` as a named const near the staging-capacity const; start at the minimum that keeps starvation=0 in Task 4 (candidate: 4 = one granule) — this is a physical watermark, not a fitted phase.

- [ ] **Step 1: Confirm the state homes (reconnaissance done — anchors below)**

- `backoff_left: u8` is a new field on `ChannelContext` (`src/device/dma/channel.rs:295`), added next to the other per-session gates (`warm_task_index`, `prev_starving`, `controller_dispatch_index`) and reset to 0 in the SAME places those are (stop_channel / Idle re-entry — a channel reset is a fresh boot). Default 0 in the constructor.
- `staged_words` lives on the in-flight `Transfer` (read in `next_granule_fetch` at `stepping.rs:319` as `transfer.staged_words`; incremented at `stepping.rs:1599`). Urgency is a pure function of it — no new field needed on the transfer.
- `egress_staging_capacity()` returns the FIFO depth (=12); `URGENT_WATERMARK` and `BACKOFF` are new named consts near it (`stream_io.rs:431`).
- Denial is already detected by `is_mem_denied(ch_idx, denied)` (`stepping.rs:~407`), called in the `step_with_denied` path. That is the hook where `backoff_left` gets set.
Read those exact spots before editing; the field-wiring pattern is established by the neighboring gates.

- [ ] **Step 2: Write failing unit tests for the two behaviors**

(a) Urgency: a channel whose `staged_words` is at/below the watermark with stream still to feed reports urgent; a full-FIFO channel does not. (b) Backoff: after a denied cycle (`step_with_denied` names the channel), the channel does not re-present its granule demand on the immediately-following cycle for `BACKOFF` cycles (it holds), then re-presents — and the re-presented demand's phase differs (the granule fetch is attempted from a shifted cycle). Write these against a constructed `DmaEngine` mid-transfer (mirror the existing engine-test setup in this file/module). Assert on `peek_bank_demand` presence/absence across stepped cycles and on the urgency flag.

- [ ] **Step 3: Run to verify fail**

Run: `cargo test --lib dma::engine 2>&1 | tail -40` (adjust path to the engine test module)
Expected: FAIL (methods/behavior absent).

- [ ] **Step 4: Implement urgency + backoff**

- Urgency: in `peek_bank_demand` (or the new companion), a demanded MM2S channel is `urgent` iff `staged_words <= URGENT_WATERMARK && !fully_fetched`. Surface it to the coordinator.
- Backoff: add per-channel `backoff_left: u8` to `ChannelContext`. When `is_mem_denied` holds a channel this cycle (it lost arbitration), set `backoff_left = BACKOFF` (named const; candidate 2). While `backoff_left > 0`, `next_granule_fetch` returns `None` (no demand presented) and `backoff_left` decrements each cycle — UNLESS the channel is `urgent`, in which case it presents regardless (urgency overrides backoff, so a starving DMA never stalls itself out). Preserve the invariant that a presented granule demand is single-bank (the retry-contract module docs depend on it).
- The backoff is what shifts the fetch phase: by skipping presentation for BACKOFF cycles after a denial, the next presentation lands at a different point in the core's period-8 march, breaking the anti-lock.

- [ ] **Step 5: Run tests + full lib**

Run: `cargo test --lib 2>&1 | tail -30`
Expected: the new DMA tests PASS; arbiter tests still in the Task-1 state (some red, tracked). No OTHER regressions.

- [ ] **Step 6: Commit**

```bash
git add src/device/dma/engine/stepping.rs
git commit -m "feat(dma): egress-FIFO urgency flag + backoff-on-denial granule fetch

A denied MM2S backs off (holds its granule demand) for BACKOFF cycles unless
near FIFO underflow (urgent), shifting its fetch phase off the deterministic
anti-lock. Urgency = staged_words <= URGENT_WATERMARK with stream still to feed.

Generated using Claude Code."
```

---

### Task 3: Coordinator — thread urgency into arbitration

**Files:**
- Modify: `src/interpreter/engine/coordinator.rs` `arbitrate_memory_banks` (~2059–2137)
- Test: covered by the Task 4 EMU gate (this is wiring; the behavior test is end-to-end)

**Interfaces:**
- Consumes: Task 1 `arbitrate(demands, urgent)`; Task 2 urgency from `peek_bank_demand`/companion.
- Produces: the real urgency set flowing into `arbitrate`.

- [ ] **Step 1: Collect urgency in Phase A and pass it to `arbitrate`**

In `arbitrate_memory_banks`, where Phase A calls `engine.peek_bank_demand(layout)` and builds `demands`, also collect the urgent DMA requesters (from the Task-2 companion / flag) into a `SmallVec<[Requester; 4]>`. Pass it as the new second arg to `self.bank_arbiters[tile_idx].arbitrate(&demands, &urgent)`.

- [ ] **Step 2: Build**

Run: `cargo build 2>&1 | tail -20`
Expected: compiles; the `&[]` placeholder from Task 1 Step 5 is now the real set.

- [ ] **Step 3: Commit**

```bash
git add src/interpreter/engine/coordinator.rs
git commit -m "feat(coordinator): thread DMA egress-FIFO urgency into bank arbitration

Generated using Claude Code."
```

---

### Task 4: GATE — EMU stall-count derisk (STOP/decision point)

**Files:** none modified. This is a measurement + decision.

**Interfaces:** Consumes the full Task 1–3 mechanism.

- [ ] **Step 1: Build the FFI `.so` and run the `collide` + `apart` + `collide_read_dense` probes on EMU**

The probe pipeline used this session: generate MLIR (`producer_probe.py --variant collide`), Chess-build, run EMU via the emulator side, decode, `producer_probe_measure.py`. Reuse the driver at `/home/triple/.claude/jobs/*/tmp/pp_recheck.py` (or `pp_sweep.py`) which already runs the pipeline; run its EMU path (`XDNA_EMU=1 XDNA_EMU_RUNTIME=debug`, rebuild `cargo build -p xdna-emu-ffi` first). Measure CONFLICT_DM_BANK, MEMORY_STALL, MM2S_STARVATION for `collide`, `apart`, `collide_read_dense`.

- [ ] **Step 2: Evaluate against the gate**

PASS if: `collide` CONFLICT is order ~1000 (not 68), MEMORY_STALL is ~18–24, STARVATION 0; `apart` is 0/0; `collide_read_dense` CONFLICT/STALL >= `collide`. The **STALL count in ~18–24 robustly is the critical criterion** (the microsim toy was phase-fragile here).

- [ ] **Step 3: DECISION**

- **PASS** → proceed to Task 5.
- **Conflict ~1000 but STALL off-band / fragile** → do NOT tune a phase constant or the FIFO depth. Try the bounded refinements first: adjust `URGENT_WATERMARK` / `BACKOFF` to their physically-minimal values (watermark = smallest that keeps starvation 0; backoff = smallest that breaks the anti-lock). Re-measure. If STALL still off-band after that, STOP and surface it: fall back to shipping the faithful conflict count + core-priority and DOCUMENTING the residual stall gap (Task 7 covers this branch). Report to the human before choosing fallback.
- **Conflict still ~68 (anti-lock intact)** → the backoff is not shifting the phase; STOP and reassess Task 2 (the backoff may be re-aligning; verify the fetch phase actually moves across a denial).

---

### Task 5: Re-express arbiter tests as class-priority (post-gate)

**Files:**
- Modify: `src/device/bank_arbiter.rs` test module + module docs.

**Interfaces:** none new.

- [ ] **Step 1: Re-express the symmetric-fairness tests**

The tests that assert symmetric alternation between a core and a DMA (`round_robin_alternates_the_winner`, `core_is_not_starved_by_continuous_dma_contention`) now encode the WRONG model. Rewrite them as: core beats non-urgent DMA every contended cycle; a DMA is not starved because urgency escalates it before its FIFO underflows. KEEP `sticky_grants_converge_...`, `multi_bank_requester_starves_without_per_bank_stickiness`, `dma_whole_mask_retry_starves_a_multi_bank_channel`, `no_requester_starves_under_the_retry_contract`, `per_bank_arbiters_are_independent`, and the core-vs-core tests — pass `&[]` urgent to their `arbitrate` calls; they must still pass (core-vs-core is unchanged, and the retry bound holds within the core class).

- [ ] **Step 2: Add a multi-urgent-DMA starvation sweep**

The review flagged the ≥2-concurrent-urgent-DMA case as uncovered. Add a test: two DMA channels both urgent + both contending a bank against a core → each gets served within the within-DMA-class rotor bound; the core yields to urgency but is not itself starved across cycles (it wins whenever no DMA is urgent). Assert bounded waits.

- [ ] **Step 3: Update module docs**

Update the `bank_arbiter.rs` header: the anti-starvation guarantee is now core-class-priority + urgency escalation (a DMA is served before FIFO underflow), not symmetric round-robin. Keep the per-bank retry-contract section; note the rotor now rotates within the winning class.

- [ ] **Step 4: Run + commit**

Run: `cargo test --lib bank_arbiter 2>&1 | tail -30` (all green)
```bash
git add src/device/bank_arbiter.rs
git commit -m "test(arbiter): re-express fairness tests as class-priority + urgency; multi-urgent-DMA sweep

Generated using Claude Code."
```

---

### Task 6: Full re-verification + anchor validation

**Files:** none modified (verification only).

- [ ] **Step 1: `cargo test --lib`**

Run: `cargo test --lib 2>&1 | tail -30`
Expected: all pass (report the count).

- [ ] **Step 2: ISA suite**

Run: `./scripts/isa-test.sh` (per CLAUDE.md; ~5–10 min; do NOT run concurrently with the bridge suite)
Expected: 4815/4815.

- [ ] **Step 3: Bridge suite**

Run: `./scripts/emu-bridge-test.sh` (~15–30 min)
Expected: no regressions vs the pre-change baseline. Capture the summary.

- [ ] **Step 4: HW-vs-EMU anchor comparison**

Re-confirm the EMU anchors against the HW bands (HW numbers already captured this session): `collide` EMU ~1000/~18–24 vs HW 1098–1103/18–20; `collide_read_dense` EMU >= collide vs HW 1162/24; `apart` 0/0. Record the EMU-vs-HW table.

- [ ] **Step 5: Commit any verification artifacts / notes** (if a results file is produced).

---

### Task 7: Documentation

**Files:**
- Modify: `docs/known-fidelity-gaps.md`; the two producer findings' NEXT sections; `ROADMAP.md` if it carries the "producer over-collision" open item.

- [ ] **Step 1: Record the outcome**

- On full success: a known-fidelity-gaps note that the producer bank-CONFLICT count is a sub-cycle phase-average, reproduced in AGGREGATE by the deterministic backoff-drift mechanism (dense anchors match; intermediate densities are HW-nondeterministic and out of scope). Update the findings' NEXT to "landed."
- On fallback (stall gap): document the residual — faithful conflict count + core-priority land, but the exact ~18–24 stall count rides on the HW-unpinned FIFO depth / urgency watermark and is not fitted; the current arbiter's symmetric-round-robin compensation is removed in favor of the faithful mechanism.

- [ ] **Step 2: Retract the roadmap's producer over-collision open item** (point it at the landed fix / documented gap).

- [ ] **Step 3: Commit**

```bash
git add docs/
git commit -m "docs(timing): record producer bank-collision C-cycle fix + residual conflict-count phase-average gap

Generated using Claude Code."
```
