# Device-Model Audit & Calibration Ledger

**Status:** active (started 2026-06-27). This is the backbone for a
completeness/audit pass that lines up every per-tile hardware parameter the
emulator models against the AIE-ML device model we built (`NPU1.json`), fixes
the divergences, fills worthwhile gaps, and cross-checks the device model itself
against the physical NPU where it is our only source.

It grew out of the #140 relay-fill work: fixing the memtile S2MM ingress FIFO
depth revealed that the same value was wrong on compute tiles too, and that the
device model is a trove of per-tile facts most of which are *already* calibrated
in archspec -- so the job is a completeness pass, not a from-scratch
calibration. "I have no idea how many fixes might just emerge."

---

## 1. Methodology

**Trust model -- the governing principle.** Device-model fields do NOT map 1:1
to observable behavior. Proven by the memtile case: `s2mmChannel.buffer_depth=12`
in the model, but HW stages 16 words (the recv PORT_RUNNING decodes to a clean
`[16,16,16,16]` with 16-word BDs, which a 12-deep FIFO would split as `12+4`).
So:

- **Structural params** (counts, sizes, offsets, port maps, register layouts):
  trust `NPU1.json`, adopt freely. Discrete, verifiable, and it's our artifact.
- **Timing / depth / latency params**: treat `NPU1.json` as a *hypothesis*,
  confirm against HW before trusting. HW is the cheap oracle (a single capture +
  readback is microseconds). The field tells you *where* to look, not the answer.

**Derivation order** (from CLAUDE.md "DERIVE FROM THE TOOLCHAIN"): for each
parameter prefer (1) open-source toolchain (aie-rt / AM025 regdb / mlir-aie),
(2) hardware observation, (3) `NPU1.json` (our aiesim-oracle artifact -- same
category as a HW capture, not raw proprietary data), (4) AM020/AM025 docs. Use
`NPU1.json` to *discover* discrepancies; source the corrected value from the
highest-priority place that covers it.

**Verification markers** (per ROADMAP): VERIFIED (HW-confirmed), OBSERVED
(seen but not exhaustively), CLAIMED (model says so, unconfirmed).

**AMD doc-number mapping** (pin this -- it is load-bearing and easy to get
backwards): **AM009** = AIE1 architecture manual; **AM020** = AIE2/AIE-ML
architecture manual; **AM025** = AIE2/AIE-ML register reference. So *both* AM020
and AM025 describe AIE2, and a "384" value cited to AM020 is suspect -- 384 is the
AIE1 (AM009) cascade width; AIE2 is 512. (Verified from mlir-aie
`programming_guide/quick_reference.md`.) The cascade fix below tripped on exactly
this: code had mis-cited the AIE1 384-bit width to AM020.

---

## 2. Data sources & tooling

### `NPU1.json` -- the fact sheet
- Path: `build/experiments/aiesim-device-decrypt/NPU1.json` (**gitignored ->
  on-disk only**, survives reboots; `/tmp` does not). Built by us via
  `make_npu1.py` in the same dir, using aiesimulator as the oracle (the
  aiesim-bridge decrypt prep). Phoenix-specific: `me_arch_version r1p3`, 5 cols
  (4+1, col 0 has no shim tile), 4 compute rows + 1 mem row + shim.
- Companion AIE-ML models in that dir: `VC2802.plaintext.json` (Versal AIE-ML,
  the original decrypt), `NPU1_8col.json`, etc.
- Regenerate a flat `key = value` dump (the queryable form; the raw is nested):
  ```python
  import json
  d = json.load(open("build/experiments/aiesim-device-decrypt/NPU1.json"))
  def flat(o, p=""):
      if isinstance(o, dict):
          for k, v in o.items(): flat(v, f"{p}.{k}" if p else k)
      elif isinstance(o, list):
          for i, v in enumerate(o): flat(v, f"{p}[{i}]")
      else: print(f"{p} = {o}")
  flat(d)
  ```

### archspec touch-points (where the emulator models these facts)
- `crates/xdna-archspec/src/types.rs` -- the structs (`DmaTiming`,
  `StreamSwitchTiming`, processor/memory/lock fields).
- `crates/xdna-archspec/src/model_builder.rs` -- the concrete AIE2 values.
- `crates/xdna-archspec/build.rs` -- emits `pub const`s into the generated
  `xdna_archspec::aie2::timing` (and `gen_arch.rs` etc.).
- External inputs already wired: AM025 regdb JSON
  (`mlir-aie/.../aie_registers_aie2.json`), device-models.json
  (`tools/aie-device-models.json`, topology/counts), aie-rt headers
  (DMA/lock/port consts via `Confirmed<T>`), llvm-aie TableGen (ISA/slots).
- Consumers of the timing constants live in `src/device/` (e.g.
  `src/device/dma/engine/stream_io.rs::input_fifo_capacity`).

### Capture recipe (HW vs EMU PORT_RUNNING cadence)
The core-tile `PORT_RUNNING` plumbing now exists end-to-end (commits
`2c7cb1dd`, `925dc99d`). To measure a tile's S2MM/MM2S port cadence:
```bash
# Force event_time so CORE tiles are mode 0 (memtile/shim/memmod always are);
# rebuild_perfetto_mode0 -> clean level spans. Drop --no-hw for a HW capture.
XDNA_TRACE_MODE=event_time \
XDNA_TRACE_CORE_EVENTS="PORT_RUNNING_0,PORT_RUNNING_1" \
XDNA_TRACE_MEMTILE_EVENTS="PORT_RUNNING_0,PORT_RUNNING_4" \
  ./scripts/emu-bridge-test.sh --chess-only --trace -v add_one_using_dma
# Decode run-lengths (the durable tool):
tools/trace-port-spans.py build/bridge-test-results/latest add_one_using_dma
```
- Core port map: `PORT_RUNNING_0` = DMA ch0 S2MM (recv), `PORT_RUNNING_1` = ch0
  MM2S (send); slots 2/3 = ch1 (interleaved `channel=slot//2`, `master=even`).
- Memtile port map: slots 0..3 = S2MM ch0..3, slots 4..7 = MM2S ch0..3 (grouped).
- A clean `[N,N,...]` (N = BD words) means the recv staged the full BD; a split
  `[...,d,N-d,...]` means it backpressured at ingress depth `d`.
- `tools/trace-port-spans.py` -- the durable run-length decoder (wraps
  parse-trace.py `--out-perfetto`).

---

## 3. Delta map

NPU1.json values below are embedded (the file is gitignored, so this doc is the
durable record). Per-tile: **C**=compute, **M**=memtile, **S**=shim-NoC.

### 3a. Corroborated -- EMU already matches NPU1.json (no action; confidence)
| Param | NPU1.json | archspec | Notes |
|---|---|---|---|
| Data memory size | C 64KB / M 512KB | same | VERIFIED |
| Program memory | C 16KB | same | |
| Physical banks | C 8 / M 16 | same | |
| Bank width | 16 B (128 bit) | 128 bit | |
| Data mem latency | load 5 / store 1 | mem latency 5 | |
| Lock count | C 16 / M 64 / S 16 | same | |
| BD count (num_descriptors) | C 16 / M 48 / S 16 | same | |
| DMA channels | C 2 / M 6 / S 2 (per dir) | same | |
| Core stream FIFO in | `stream_in_buf_depth` 4 | `local_slave_fifo_depth` 4 | corroborates AM020 |
| Core stream FIFO out | `stream_out_fifo_depth` 2 | `local_master_fifo_depth` 2 | corroborates AM020 |
| Stream switch slots | `num_slots` 4 | `num_slots` 4 | |
| Vector / acc width | `cascade_bitwidth` 512 | vector/acc 512 | (cascade carries acc512) |
| BD task-start queue | `start_queue` 4 (C,M,S) | `MAX_TASK_QUEUE_DEPTH`=4 (token.rs) | aie-rt `XAIE_DMA_MAX_QUEUE_SIZE`; corrected from 8 in ee8f8f00. (Stale "8-deep" comment in channel.rs fixed this audit.) |

### 3b. Fixed (this audit)
| Param | NPU1.json | Fix | Commit |
|---|---|---|---|
| S2MM ingress depth | `StreamSwitch.fifo_depth`=16 (C,M,S all 16); `s2mmChannel.buffer_depth`=12 (C,M) | deep depth (16) now applies to compute **and** mem (was memtile-only); shim stays shallow. Const `DMA_S2MM_INGRESS_FIFO_DEPTH`. VERIFIED both tiles. | `ecc7a4a4` |
| Cascade width | `cascade_bitwidth`=512 | **Real bug, not comment-only** (ledger mis-predicted). `CASCADE_WORDS` 6->8: AIE2 cascade is 512-bit and carries a full 8-lane accumulator. The old value silently **truncated** the top 128 bits (lanes 6,7) when forwarding accumulators across tiles. All `[u64;6]` cascade types now derive from `CASCADE_WORDS`. Source: mlir-aie `AIE2TargetModel::getAccumulatorCascadeSize()`==512 (AIE1==384). Structural param -> toolchain-decisive. | (this audit) |

### 3c. Discrepancies -- EMU disagrees with NPU1.json (open)
| Param | NPU1.json | EMU now | Impact | Status |
|---|---|---|---|---|
| Bank-conflict penalty | `penalty_conflict` 4 (C,M) | `CONFLICT_PENALTY`=1 cycle (`timing/memory.rs:202`), cited to AM020 Ch2 "stall 1 cycle" | 4x on DM-bank-conflict stalls (scalar-only model). **Source conflict**: AM020 reads 1, aiesim-oracle says 4 -- could be different semantics (per-conflict vs worst-case). | HW-CONFIRM (timing) |
| Cascade FIFO depth | `cascade_fifo_depth` 2 | depth-4 entry check (`cascade.rs:94,156`; `routing.rs:276`) | After the width fix each entry is one full 512-bit transfer, so depth-4 = 4 transfers vs HW 2-deep -> likely 2x over-model. Old "4 = 2 half-register writes x 2" justification no longer holds (we push one full acc per op). | CONFIRM-THEN-FIX (likely -> 2) |
| Topology / NoC | 5 cols, col 0 shim-less (4+1) | known-imperfect | Maya: deferred, separate effort | PARKED |

### 3d. Under-modeled -- NPU1.json asserts, EMU doesn't model (open)
| Param | NPU1.json | Behavioral impact | Status |
|---|---|---|---|
| Exact S2MM ingress 16-vs-28 | `fifo_depth` 16 vs `buffer_depth`+`fifo_depth`=28 | matters only for BDs in (16,28] | DEFERRED -- needs >16-word-BD capture (greenlit "later") |
| MM2S egress buffer depth | `mm2sChannel.buffer_depth` C/M 12, **S 256** | MM2S port cadence (send/drain side) | OPEN -- item 2 (next behavioral) |
| Task-complete queue | `task_complete_queue_size` 128 (C,M,S) | EMU token buffer is **unbounded** (token.rs:335-346) -- deliberate, because aie-rt exposes only a 1-bit STALLED_TCT flag with no numeric depth. NPU1.json now *gives* that number (128). | HW-CONFIRM then bound (low impact: only bites at >128 outstanding TCTs) |
| Program-mem load latency | `ProgramMemory.latency_load` 3 (vs data 5) | No separate program-mem fetch latency field; only `data_memory_latency`=5 + `branch_penalty`=3. **Likely a non-gap**: steady-state fetch latency is pipelined/hidden; the only place it surfaces (branch refill) is already modeled as branch_penalty=3, which *coincides* with latency_load=3. | LIKELY NON-GAP (documented) |
| Shim s2mm write queue | `write_queue_depth` 17, `max_outstanding_bds` 9, `max_outstanding_transactions` 128 | shim DDR S2MM detail | OPEN (folds into the empirical shim DDR model) |
| Shim mm2s read queue | `read_resp_queue_depth` 64, `max_outstanding_bds` 8 | shim DDR MM2S detail | OPEN |

---

## 4. Open items checklist (work order)

1. [x] **Cascade width 384 -> 512** -- DONE. Predicted "comment-only"; was a real
   truncation bug. `CASCADE_WORDS` 6->8; `accumulator_to_cascade` no longer drops
   the top 2 accumulator lanes. All cascade array types now derive from the const.
   (Follow-up, low priority: `AieConstants::accumulator_cascade_bits` is a dead
   field -- claims "populated by aie-rt extractor" but is always `None`; populate
   =512 for AIE2 or delete it.)
2. [ ] **MM2S egress depth** -- model `output_fifo_capacity()` per tile type
   (C/M 12, S 256) the way `input_fifo_capacity()` now does for S2MM. Capture
   MM2S PORT_RUNNING (`PORT_RUNNING_1` core / `PORT_RUNNING_4..` memtile) to
   confirm before trusting the numbers.
3. [x] **`start_queue` / `task_complete_queue_size`** -- INVESTIGATED.
   `start_queue`=4 already MATCHES (`MAX_TASK_QUEUE_DEPTH`, aie-rt-sourced) ->
   moved to 3a; fixed a stale "8-deep" comment in channel.rs.
   `task_complete_queue_size`=128: EMU token buffer is *unbounded* (deliberate --
   no numeric depth in aie-rt). Now we have 128 from the oracle -> 3d, HW-confirm.
4. [x] **Bank-conflict penalty (4)** -- INVESTIGATED. EMU models 1 cycle
   (`CONFLICT_PENALTY`, scalar-only), cited to AM020 "stall 1 cycle"; oracle says
   4. Real source conflict, 4x impact -> 3c, HW-CONFIRM (the `CONFLICT_DM_BANK_*`
   memmod trace events + a bank-conflict-heavy kernel can settle it).
5. [x] **Program-mem latency (3) / cascade_fifo_depth (2)** -- INVESTIGATED.
   Prog-mem latency: not a separate field; likely a non-gap (branch_penalty=3
   already captures the only observable, and coincides) -> 3d, documented.
   cascade_fifo_depth: EMU depth-4 likely 2x over-models HW 2-deep -> 3c,
   confirm-then-fix.
6. [ ] **16-vs-28 discriminating capture** (deferred; see section 5).
7. [ ] **Systematic sweep** -- walk the full flat dump section by section
   (DataMemory, DMA, Locks, StreamSwitch, Processor, EventTrace) for any
   remaining param not yet classified above.
8. [ ] **Meta-validation** -- for fields where NPU1.json is the only source,
   spot-check against HW (the trust-model rule).

Each behavioral fix follows the established loop: TDD unit test (RED->GREEN) ->
`cargo test --lib` -> HW/EMU co-capture to confirm against silicon -> commit.

### Docs to reconcile as we go
- `docs/superpowers/findings/2026-06-27-relay-fill-bd-switch-accept-coupling.md`
  still contains the bogus "master-port FIFO (4) + buffer_depth (12) = 16"
  derivation (lines ~176-180; the `4` was a port INDEX). Correct it to
  `StreamSwitch.fifo_depth=16` and note the compute generalization landed.
- `docs/known-fidelity-gaps.md` -- update if any audit item closes a gap.

---

## 5. Deferred: the 16-vs-28 discriminating capture

The memtile capture proves ingress depth >= 16 (refutes 12); the compute
capture proves >= 8. Neither pins the *exact* depth because both probe kernels
have BDs <= 16. To distinguish 16 (= `fifo_depth`) from 28 (= `buffer_depth` 12
+ `fifo_depth` 16 in series) needs a kernel with a **lock-stalled S2MM recv and
a BD comfortably larger than 28** (e.g. 64 words). Design: take
`add_one_using_dma`'s structure (2-buffer ring, prod/cons locks) but enlarge the
input buffers to 64 words and slow the consumer, so the 3rd (reused) BD stages
exactly `depth` words then backpressures -- the first PORT_RUNNING run of that
BD reads the depth directly. The core-trace plumbing now makes this measurable
on compute too. Greenlit by Maya for "later, built properly."

---

## 6. Arc status (2026-06-28)

**Arc COMPLETE.** The device-model-audit branch was reviewed, merged, and
landed on `origin/master` @ `3299341a` (the 2026-06-28 consolidation commit).
The branch has been deleted local + remote.

Completed checklist items: cascade width 384->512 (item 1), S2MM ingress depth
fix (item 2a), start_queue / task_complete_queue investigation (item 3), bank-
conflict and program-mem latency investigation (items 4/5). All shipped.

**Open follow-ons** (not blocked on this arc; tracked in sections 3c/3d and
`docs/known-fidelity-gaps.md`):

- **Send-port cadence race** (consumer-pacing mid-stream backpressure): the
  inter-tile send-port gap is documented in `known-fidelity-gaps.md` as a
  follow-on sub-project. Not chased further this arc per Maya's call.
- **HW-confirm audit bucket**: `penalty_conflict` 4 vs 1 cycle (section 3c),
  `cascade_fifo_depth` likely 2x over-model (section 3c), `task_complete_queue`
  depth 128 needs HW-confirm then bounding (section 3d).
- **Systematic sweep** (items 6/7/8): the full DataMemory/DMA/Locks/
  StreamSwitch/Processor/EventTrace flat-dump sweep is deferred.

The capability that enables further measurement: core PORT_RUNNING tracing
(`XDNA_TRACE_CORE_EVENTS`, event_time) + `tools/trace-port-spans.py`.
Re-read section 1 (trust model) before adopting any timing value from NPU1.json.
