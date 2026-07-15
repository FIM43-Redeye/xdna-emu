# Phoenix DMA-Constant Captures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. NOTE: HW-capture tasks (5, 6, 7) require the real Phoenix NPU, must be serialized (never two HW users at once), and involve trace-interpretation judgment -- run those interactively, not via a mechanical subagent. The tooling tasks (2, 3, 4) and the derivation task (1) are the subagent-friendly ones.

**Goal:** Bank the Phoenix-only HW measurements that pin the memtile bank-access width/parallelism, the MM2S egress FIFO depth, and the TCT token buffer depth, before the Strix swap retires the hardware.

**Architecture:** Each experiment is an instance of the existing 6-stage AIE trace pipeline (event-time mode 0). Per experiment we add exactly two Python files -- a generator cloned from `tools/experiments/bankdisc.py` and an analyzer cloned from `bankdisc_measure.py`/`bankdisc_analyze.py`. Stages 2-5 (trace-inject, compile, run, decode) are reused unchanged. HW runs go through `bridge-runner/build/bridge-trace-runner` under `env -u XDNA_EMU`; decode via `tools/parse-trace.py --decoder ours`.

**Tech Stack:** Python 3.13 (generators/analyzers), MLIR (AIE dialect, emitted as strings), the mlir-aie compile+trace harness, Rust (`crates/xdna-archspec`, `src/device/`) for the constants.

**Spec:** `docs/superpowers/specs/2026-07-15-phoenix-dma-constant-captures.md`

## Global Constraints

Every task's requirements implicitly include these:

- **Derive-from-toolchain.** Constants come from `crates/xdna-archspec` (which derives from the toolchain/register DB) or AM020 with a cited chapter:line. Never hardcode a bit position or magic number a source already defines.
- **No calibration, no fitting.** If a capture does not pin a constant cleanly, DOCUMENT the mechanism and any observed bound -- never tune a knob or pick a value to make a number come out. (Exception path for Experiment B: escalate the setup, per its spec section, before concluding a clean pin is unreachable.)
- **More captures, not fewer.** HW is the cheap oracle. Every probe: >= 3 HW repeats per variant (`_r1/_r2/_r3`, pooled + medianed), a full sweep of the independent variable (not two endpoints), a pre-registered validity control (`idle`/`apart` reading the null), and the free grounding anchors (`PERF_CNT_2`, `INSTR_EVENT_0/1`) to prove the window did not truncate. Add a `tools/multirun-trace-campaign.py` noise campaign where a variance claim rides on the result.
- **Timebase traps** (baked into every analyzer): use `ts` never `soc`; level events (STALLED_LOCK, CONFLICT_*, BACKPRESSURE, MEMORY_STARVATION, TOKEN_STALL) compared by **interval area** from the mode-0 B/E rebuild, never decoded record counts; discrete events (FINISHED_BD, FINISHED_TASK) use the rising-edge `ts`; `START_TASK`/`FINISHED_TASK` fire ZERO times for a self-looping single-BD chain (bracket with FINISHED_BD + STALLED_LOCK falling edge).
- **HW discipline** (CLAUDE.md): HW invocations use `env -u XDNA_EMU XDNA_EMU_RUNTIME=release`; never run two HW users concurrently; no `xrt-smi` during a HW run; recover a wedged NPU with `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'` before escalating; a single capture is cheap, run freely.
- **Bank maps** (derived, not hardcoded): compute tile `physical = 2*((addr>>14)&3) + ((addr>>4)&1)` (8 banks); memtile `physical = (addr>>4)&0xF` (16 banks, interleave 256B, AM020 ch.5:137).
- **Reuse the pipeline.** Do not rebuild stages 2-5. Emit MLIR into a build dir under `mlir-aie/test/npu-xrt/spike_bringup/`, prepare with `tools/trace-prepare.py`, compile via the harness, run via `bridge-trace-runner`, decode via `tools/parse-trace.py --decoder ours`.
- No emoji. Commit messages end with "Generated using Claude Code." `cargo test --lib` green after any Rust change.

---

## File Structure

- `tools/experiments/memtile_bankwidth.py` -- generator, Experiment A (A1 two-DMA contention variants + A2 stride-sweep variants)
- `tools/experiments/memtile_bankwidth_measure.py` -- spans + conflict areas from decoded events
- `tools/experiments/memtile_bankwidth_analyze.py` -- width inversion (A1) + strided span-ratio discriminator (A2)
- `tools/experiments/mm2s_egress_depth.py` -- generator, Experiment B (fill-then-stall variants + escalation variants)
- `tools/experiments/mm2s_egress_depth_measure.py` -- STALLED_LOCK -> MEMORY_STARVATION onset delay
- `tools/experiments/tct_token_depth.py` -- generator, Experiment C (multi-task chain + throttled token route)
- `tools/experiments/tct_token_depth_measure.py` -- FINISHED_TASK count before TOKEN_STALL onset
- `docs/superpowers/findings/2026-07-15-memtile-bank-access-width.md` -- Experiment A finding
- `docs/superpowers/findings/2026-07-15-mm2s-egress-fifo-depth.md` -- Experiment B finding
- `docs/superpowers/findings/2026-07-15-tct-token-buffer-depth.md` -- Experiment C finding
- Modified: `src/device/banking.rs`, `src/device/dma/engine/stepping.rs`, `docs/fidelity-gaps/dma-stream-resources.md` (A0 derivation + any constant fix)
- Modified: `crates/xdna-archspec` `DMA_MM2S_EGRESS_FIFO_DEPTH`, `src/device/dma/token.rs` (B/C constant fixes, if pinned)

Test fixtures for the analyzers (synthetic decoded-events JSON) live beside each analyzer test.

---

## Task 1: A0 derivation updates + input-task-queue bonus check (no HW)

**Files:**
- Modify: `src/device/banking.rs:48` (comment), `:63` (comment)
- Modify: `src/device/dma/engine/stepping.rs:261` (comment block)
- Modify: `docs/fidelity-gaps/dma-stream-resources.md` (granule-cap row)
- Investigate: DMA input task-queue depth in `src/device/dma/` (is it bounded at 4?)

**Interfaces:**
- Consumes: AM020 ch.5 facts (16 banks x 128-bit @ line 105; 256B interleave @ line 137; "shared interface" @ line 153; input task queue = 4 @ lines 51/65).
- Produces: nothing code-facing; upgrades comments/docs from "inferred/unvalidated" to derived-with-citation, and a finding on the task-queue bound.

- [ ] **Step 1: Upgrade the banking.rs memtile comments.** At `src/device/banking.rs:48`, replace "Unvalidated: preserve the previous flat interleave for memtiles." with a note that AM020 ch.5:137 confirms 128-bit-granularity interleave across 16 banks wrapping every 256B -- i.e. `(addr>>4)&0xF` is derived, not a placeholder. At `:63` (`access_granule_bytes` memtile arm), cite AM020 ch.5:105 (128-bit banks) for the 16-byte granule. Keep the existing derive-from-`PHYSICAL_BANK_WIDTH_BITS` code unchanged.

- [ ] **Step 2: Upgrade the stepping.rs granule-cap comment.** At `src/device/dma/engine/stepping.rs:261`, change the memtile paragraph from "an INFERENCE, not a measurement" to: AM020 ch.5:153 ("access memory over a shared interface", singular, per channel) grounds the single-granule-per-cycle cap; the only residual is the strided single-channel corner, HW-confirmed by Experiment A2 (finding `2026-07-15-memtile-bank-access-width.md`). Do not change the code.

- [ ] **Step 3: Investigate the input task-queue bound.** Grep `src/device/dma/` for how BDs/tasks are enqueued to a channel (task queue / pending BD list). Determine whether the emulator bounds the input task queue at 4 (AM020 ch.5:51/65: "queue depth is four tasks per channel") or models it unbounded. Write a one-paragraph note of the finding into the finding doc created in Step 5 (a subsection "Input task-queue bound").

- [ ] **Step 4: Run tests.**

Run: `cargo test --lib`
Expected: PASS (comment-only changes; no behavior change). Report the count.

- [ ] **Step 5: Update the gap doc + create the finding stub.** In `docs/fidelity-gaps/dma-stream-resources.md`, rewrite the memtile granule-cap row's status from "OPEN, inferred, HW-gated" to "width/interleave/port DERIVED from AM020 ch.5; strided single-channel corner HW-confirm pending (Experiment A2)." Create `docs/superpowers/findings/2026-07-15-memtile-bank-access-width.md` with the AM020 derivation section and the input-task-queue note (the HW A1/A2 sections are filled by Task 4).

- [ ] **Step 6: Commit.**

```bash
git add src/device/banking.rs src/device/dma/engine/stepping.rs docs/
git commit -m "docs(memtile): derive bank width + interleave + port from AM020 ch.5

AM020 ch.5:105/137/153 pins the memtile 16x128-bit banks, 256B interleave, and
single shared memory interface per channel -- upgrading the granule-cap comments
and gap row from inferred to derived. Strided single-channel corner remains for
HW confirmation (Experiment A2). Notes the input task-queue=4 bound.

Generated using Claude Code."
```

---

## Task 2: Experiment A tooling -- memtile bank-width generator + analyzer

**Files:**
- Create: `tools/experiments/memtile_bankwidth.py` (generator)
- Create: `tools/experiments/memtile_bankwidth_measure.py`, `tools/experiments/memtile_bankwidth_analyze.py`
- Create: `tools/experiments/test_memtile_bankwidth_analyze.py` (fixture test)

**Interfaces:**
- Consumes: the `bankdisc.py` structure (emit(variant) -> MLIR string; VARIANTS dict; ascending-address buffer pinning); the decoded-events JSON schema from `tools/parse-trace.py --decoder ours` (records with `ev`, `ts`, module).
- Produces: `emit(variant)` for A1/A2 variants; `analyze(events_by_variant) -> {width_bytes, strided_ratio}`.

- [ ] **Step 1: Write the generator, cloning bankdisc.py deltas.** Model on `tools/experiments/bankdisc.py`. Deltas: (a) tile is a **memtile** `aie.tile(0, 1)` (row 1), no `aie.core` -- a memtile has no processor; (b) buffers via `aie.buffer(%memtile)` pinned in ascending address order; (c) bank map is `physical = (addr>>4)&0xF` (16 banks) -- put a buffer in a chosen physical bank by aligning its address so `(addr>>4)&0xF` = target. Variants:
  - **A1 contention** (`a1_collide`, `a1_apart`, `a1_idle`): a shim->memtile **S2MM** filling a buffer and a memtile->shim **MM2S** draining another buffer, both mapped to the SAME physical bank in `a1_collide`, different banks in `a1_apart`, and `a1_idle` = one channel only (floor). Use `aie.flow` + `aie.shim_dma_allocation` + `aie.dma_start(S2MM/MM2S, ...)` on the memtile DMA (`aie.memtile_dma`), locks to sequence fill/drain like bankdisc's `lk_empty`/`lk_full`.
  - **A2 stride** (`a2_stride_{4,16,32,64,128,256}`): a SINGLE memtile MM2S with a strided BD reading `word[i]` from `base + i*stride` bytes. Use `aie.dma_bd` multi-dim (wrap/stride) to express the stride; memtile BDs support 4D (K=4). Fixed `OBJ` words per run; the stride is the swept variable.
  Provide `--variant` (choices = A1 + A2 variants). Emit valid MLIR; iterate against the compiler until it compiles (the compile step is Task 4's, but smoke-compile one variant here to de-risk).

- [ ] **Step 2: Write a structural test for the generator.** `test_memtile_bankwidth_analyze.py` (or a `test_emit`): assert the emitted MLIR for `a1_collide` pins both contending buffers to the same physical bank (`(addr>>4)&0xF` equal), `a1_apart` to different banks, and each `a2_stride_S` BD encodes stride S. No compiler needed -- assert on the emitted string / computed bank indices.

Run: `python3 tools/experiments/test_memtile_bankwidth_analyze.py`
Expected: PASS.

- [ ] **Step 3: Write the measure script.** Clone `bankdisc_measure.py`: load decoded events, rebuild level-event intervals (interval-area, not counts), bracket each MM2S transfer by `MEM_TILE_DMA_MM2S_SEL0_FINISHED_BD` and the preceding `..STALLED_LOCK` falling edge, and sum `MEM_TILE_CONFLICT_DM_BANK_n` (112-127) area per variant. Output a per-variant table (durations, conflict areas).

- [ ] **Step 4: Write the analyze script + its fixture test.** Clone `bankdisc_analyze.py`. Two outputs:
  - **A1 width:** invert conflict area under the single-port round-robin model (as bankdisc does) -> bytes/access. `f` (contender occupancy) derived from the second DMA's cadence, not fitted.
  - **A2 ratio:** `span(strided)/span(contiguous)` per stride -> the 1-vs-4 discriminator.
  Write `test_memtile_bankwidth_analyze.py` with a SYNTHETIC decoded-events fixture (hand-built JSON) whose known conflict area inverts to a known width and whose known spans produce a known ratio; assert the analyzer recovers them.

Run: `python3 tools/experiments/test_memtile_bankwidth_analyze.py`
Expected: PASS (analyzer recovers the fixture's planted width and ratio).

- [ ] **Step 5: Commit.**

```bash
git add tools/experiments/memtile_bankwidth*.py
git commit -m "tools(captures): memtile bank-width generator + analyzer (Experiment A)

Generated using Claude Code."
```

---

## Task 3: Experiment B tooling -- MM2S egress-depth generator + analyzer

**Files:**
- Create: `tools/experiments/mm2s_egress_depth.py`, `tools/experiments/mm2s_egress_depth_measure.py`
- Create: `tools/experiments/test_mm2s_egress_depth_measure.py` (fixture test)

**Interfaces:**
- Consumes: `producer_probe.py`'s K-dwell machinery and its 12-word egress-FIFO model; the compute-tile MM2S structure from `bankdisc.py`.
- Produces: `emit(variant)` for the fill-then-stall + escalation variants; `measure(events) -> onset_delay_beats`.

- [ ] **Step 1: Write the generator.** Model on `producer_probe.py` + `bankdisc.py`. A **compute-tile** MM2S whose BD chain fills the egress FIFO to capacity, then stalls the memory-side fetch on a lock while the stream drains. Variants realise the spec's escalation ladder:
  - `fill_stall` (naive): mid-transfer lock stall after a fill dwell.
  - `stream_backpressure` (escalation 1): hold the downstream consumer so the FIFO fills to ceiling, then release the stream while blocking the fetch (decouple fill from drain).
  - `fetch_starve` (escalation 2): a core hammer on the MM2S source bank so the fetch loses arbitration every cycle, or a slow/strided source BD.
  - `dwell_sweep_{K}` (escalation 3): sweep the pre-stall dwell K so the FIFO reaches a range of occupancies.
  Plus a `cold` control (STARVATION only on first empty fill) and a `never_stall` control (STARVATION = 0).

- [ ] **Step 2: Write the measure script.** From decoded events, find `DMA_MM2S_0_STALLED_LOCK` onset and `DMA_MM2S_0_MEMORY_STARVATION` onset; report the delay (beats) = egress occupancy at stall-time. Read depth as the MAX stable onset-delay across the dwell sweep (the ceiling). Confirm `DMA_MM2S_0_STREAM_BACKPRESSURE` ~0 during the window (stream draining, not itself blocked).

- [ ] **Step 3: Fixture test.** `test_mm2s_egress_depth_measure.py`: synthetic events with a planted STALLED_LOCK onset and MEMORY_STARVATION onset N beats later; assert `measure` returns N and flags a window where STREAM_BACKPRESSURE overlaps (invalid).

Run: `python3 tools/experiments/test_mm2s_egress_depth_measure.py`
Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add tools/experiments/mm2s_egress_depth*.py
git commit -m "tools(captures): MM2S egress-depth generator + analyzer (Experiment B)

Generated using Claude Code."
```

---

## Task 4: Experiment C tooling -- TCT token-depth generator + analyzer

**Files:**
- Create: `tools/experiments/tct_token_depth.py`, `tools/experiments/tct_token_depth_measure.py`
- Create: `tools/experiments/test_tct_token_depth_measure.py` (fixture test)

**Interfaces:**
- Consumes: multi-task BD-chain structure (so FINISHED_TASK fires); the token-route stream-switch config.
- Produces: `emit(variant)`; `measure(events) -> tasks_outstanding_at_token_stall`.

- [ ] **Step 1: Write the generator.** A DMA channel running a **multi-task** BD chain (distinct tasks, NOT self-looping single-BD, so `FINISHED_TASK` fires) completing many small tasks while the **token-return stream route** is throttled to a slow/backpressured consumer, so completion tokens back up and `DMA_TASK_TOKEN_STALL` eventually asserts. Variants sweep task size (token rate) and the return-route throttle. Choose the tile with the clearest TOKEN_STALL event: memtile (`MEM_TILE_DMA_TASK_TOKEN_STALL=140`), memmod (`102`), or shim (`PL_DMA_TASK_TOKEN_STALL=75`) -- pick one, document why.

- [ ] **Step 2: Write the measure script.** Count `FINISHED_TASK` rising edges before the first `DMA_TASK_TOKEN_STALL` onset = outstanding tokens at stall = buffer depth. Cross-check against status-register bit[5] `Stalled_TCT` if readable.

- [ ] **Step 3: Fixture test.** `test_tct_token_depth_measure.py`: synthetic events with M FINISHED_TASK edges then a TOKEN_STALL onset; assert `measure` returns M.

Run: `python3 tools/experiments/test_tct_token_depth_measure.py`
Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add tools/experiments/tct_token_depth*.py
git commit -m "tools(captures): TCT token-depth generator + analyzer (Experiment C)

Generated using Claude Code."
```

---

## Task 5: Experiment A HW capture + finding + constant (INTERACTIVE, HW)

**Files:**
- Modify: `docs/superpowers/findings/2026-07-15-memtile-bank-access-width.md` (fill A1/A2 results)
- Modify (only if A2 ratio ~1): `src/device/dma/engine/stepping.rs` (lift the granule cap for `BankLayout::MemTile`), plus its test and the gap-doc row.

**Interfaces:**
- Consumes: the Task 2 generator/analyzer; the trace pipeline (stages 2-5).
- Produces: measured memtile access width (expect 16B) and the strided 1-vs-4 verdict.

- [ ] **Step 1: Build + capture.** For each variant: emit MLIR, `tools/trace-prepare.py` (pick the event list: `MEM_TILE_CONFLICT_DM_BANK_0..15`, `MEM_TILE_DMA_MM2S_SEL0_FINISHED_BD`/`..STALLED_LOCK`, grounding anchors), compile via the harness, run `bridge-trace-runner` under `env -u XDNA_EMU XDNA_EMU_RUNTIME=release` x3 repeats, decode with `tools/parse-trace.py --decoder ours`. Include the `a1_idle`/`a1_apart` controls and the full A2 stride sweep.

- [ ] **Step 2: Analyze.** Run `memtile_bankwidth_analyze.py` over the pooled decoded events. Confirm: `a1_apart` conflict area ~0 (validity gate), grounding anchors intact (no truncation), A1 width ~16B (cross-checks AM020), A2 ratio per stride.

- [ ] **Step 3: Interpret + finding.** A2 ratio ~4 -> cap correct, close the residual. Ratio ~1 -> cap wrong for the memtile; note the fix. Write the A1/A2 results, medians, per-repeat spread, and verdict into the finding.

- [ ] **Step 4: Constant fix (only if ratio ~1).** Lift the granule cap for `BankLayout::MemTile` in `granule_capped_words` (throughput-only; memtile DMA does not arbitrate). Add/adjust the covering test (`strided_s2mm_takes_one_granule_per_cycle` analog for the memtile). Run `cargo test --lib` (expect green). Update the gap-doc row to CLOSED with the measured direction.

- [ ] **Step 5: Commit.**

```bash
git add docs/ src/ 2>/dev/null
git commit -m "finding(memtile): HW-confirm bank width + strided single-channel parallelism

Generated using Claude Code."
```

---

## Task 6: Experiment B HW capture + finding + constant (INTERACTIVE, HW, priority)

**Files:**
- Modify: `docs/superpowers/findings/2026-07-15-mm2s-egress-fifo-depth.md`
- Modify: `crates/xdna-archspec` (`DMA_MM2S_EGRESS_FIFO_DEPTH`), `docs/fidelity-gaps/dma-stream-resources.md` row

**Interfaces:**
- Consumes: Task 3 generator/analyzer; the trace pipeline.
- Produces: measured egress FIFO depth (beats), or a characterized bound if every escalation genuinely fails.

- [ ] **Step 1: Build + capture the naive variant.** Emit `fill_stall`, prepare with events `DMA_MM2S_0_STALLED_LOCK`, `DMA_MM2S_0_MEMORY_STARVATION`, `DMA_MM2S_0_STREAM_BACKPRESSURE`, `DMA_MM2S_0_FINISHED_BD`, grounding anchors; compile; run x3 under `env -u XDNA_EMU`; decode; run `mm2s_egress_depth_measure.py`.

- [ ] **Step 2: If STARVATION does not fire, ESCALATE (do not stop).** This is a priority final gap. Work down the ladder from the spec's Experiment B section: `stream_backpressure` (decouple fill/drain), then `fetch_starve` (bank contention / slow source), then `dwell_sweep`. Capture each on HW (cheap). The onset delay from a known-full start = full depth.

- [ ] **Step 3: Pin + campaign.** Once STARVATION fires cleanly, run the dwell sweep and read the ceiling; run a `multirun-trace-campaign.py` noise campaign to establish variance (mirror the S2MM finding's zero-variance standard). Controls: `cold` (STARVATION only on first fill), `never_stall` (0).

- [ ] **Step 4: Finding + constant.** Write the measured depth, the escalation path that exposed it, per-repeat variance, and the controls into the finding. Set `DMA_MM2S_EGRESS_FIFO_DEPTH` to the measured value; update the gap-doc row from "un-derived" to "HW-pinned." If every escalation genuinely failed, document the mechanism + observed bound (no guessed number) and leave the constant with its provenance note.

- [ ] **Step 5: Test + commit.**

Run: `cargo test --lib`
Expected: PASS.

```bash
git add crates/ docs/
git commit -m "finding(mm2s): HW-pin the MM2S egress FIFO depth

Generated using Claude Code."
```

---

## Task 7: Experiment C HW capture + finding + constant/characterization (INTERACTIVE, HW)

**Files:**
- Modify: `docs/superpowers/findings/2026-07-15-tct-token-buffer-depth.md`
- Modify (if pinned): `src/device/dma/token.rs` (bound the token buffer), `docs/fidelity-gaps/dma-stream-resources.md` TCT row

**Interfaces:**
- Consumes: Task 4 generator/analyzer; the trace pipeline.
- Produces: the token buffer depth if pinned, else a characterized mechanism + bound.

- [ ] **Step 1: Build + capture.** Emit the multi-task/throttled-route variants; prepare with `DMA_TASK_TOKEN_STALL` (chosen tile's event), `FINISHED_TASK`, grounding anchors; compile; run x3 under `env -u XDNA_EMU`; decode; run `tct_token_depth_measure.py`.

- [ ] **Step 2: Sweep + interpret.** Sweep task size and return-route throttle; look for a consistent outstanding-FINISHED_TASK count at TOKEN_STALL onset. Cross-check status bit[5]. Multi-run for stability.

- [ ] **Step 3: Finding + constant (characterize-only accepted).** If a consistent depth emerges, pin it and bound `token.rs`; update the TCT gap row. If it only characterizes the TOKEN_STALL mechanism without a crisp integer, document that + the observed bound -- do NOT force a number. Either outcome is a valid completion of this characterize-only experiment.

- [ ] **Step 4: Test + commit.**

Run: `cargo test --lib`
Expected: PASS.

```bash
git add src/ docs/ 2>/dev/null
git commit -m "finding(tct): characterize the TCT token buffer depth on Phoenix

Generated using Claude Code."
```

---

## Notes on execution

- Tasks 1-4 (derivation + all tooling) are HW-free and subagent-friendly; do them first so every probe is ready before touching the NPU. Tasks 5-7 are the HW captures -- serialized, judgment-heavy, run interactively.
- The generators emit real AIE MLIR; expect to iterate a variant against the compiler until it compiles. The structural fixture tests (Steps in Tasks 2-4) do NOT need the compiler and gate the analyzer logic independently of HW.
- If the NPU wedges during a capture: `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'`, smoke-test with `xrt-smi validate` (NOT during a run), then resume.
