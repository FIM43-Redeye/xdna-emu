# Project Status Consolidation — 2026-06-27

Point-in-time regroup before starting the timer-sync sub-project. Reconciled
every in-flight thread against repo reality (git log, docs, experiments).
Not live state — a snapshot. `origin/master == 15592839`.

## Just banked (this session): connectivity #140

- **P1 — logical connectivity classifier** (`tools/inference/connectivity.py`):
  three-way honest status (grounded / observed_but_ungrounded / unobserved),
  wired into `assemble_timeline`. Merged.
- **P2 — headline correctness fix:** `event_map._resolve_tile_dma` had the
  compute/memtile S2MM/MM2S stream-switch direction **inverted** (latent in
  merged HW-validated code; add_one uses the shim path, two_col surfaced it).
  Fixed → the engine now **faithfully orients cross-column dataflow**. add_one
  grounding un-regressed on real HW.
- **Cross-column GROUNDING: parked.** Two adversarial reviews caught two
  assumed-not-verified indexed-mapping bugs (port direction; `LOCK_<N>_REL` is
  selection-block indexed, resets to physical lock 0). And even corrected, the
  edge would be existence-only / non-falsifiable without cross-timer-domain
  sync. Findings: `event-map-tile-dma-direction-inversion`,
  `lock-event-selection-indirection`, `compute-memmod-trace-records-no-events`.

## Live frontier / NEXT

- **Timer-sync (BROADCAST_15)** — the next sub-project. It is the real
  prerequisite for a *falsifiable* cross-column (and cross-tile / cross-module)
  grounding witness, so it unblocks more than connectivity. Prior design-spike
  specs exist to build on: `docs/superpowers/specs/2026-06-23-cross-domain-
  grounding-shared-timebase-design.md` and `…2026-06-23-cross-domain-
  reproduction-targets.md`. No implementation plan or code yet — green-field.

## Parked (real, unbuilt — resume is a deliberate choice)

- **Framework tenants 4 & 5** — diff-fuzzing tenant 4 (locks/streams under
  contention) and tenant 5 (multi-tile capstone). Steps 1–3b + coverage flips
  (#113/#126/#127) are all merged; `clean_release(Aie2)` is *intentionally* RED
  until tenants 4/5 land. DMA iter (#129) deferred (shim HW-timeout root cause
  unresolved). Work pivoted to #140; not imminent.
- **Phoenix-survival output corpus** — spec (`2026-05-31`) + 13-task plan
  (`2026-06-01`) done and approved; **unbuilt**. HW-gated; resume within the
  Phoenix window is Maya's call.

## Done / shipped (now repo-history; memory notes archived)

- **aiesim arc** — the `aiesim-cpp-bridge` branch (~80 commits) fully merged: the
  three-way (HW / interp / aiesim) harness, control-read clone patch, broadcast
  bridge, unified driver path, bespoke-sweep removal. Status + the one open gap
  (#92 `init_values_repeat`) live in `docs/coverage/aiesim-oracle-assessment.md`.
- **Coalescing / LOCK_STALL / #136** — fixed (`e6476411`, TRACE_VERDICT token);
  phoenix DDR burst model deleted (`f093b6b9`). Superseded by the inference arc.
- **Vector compute audit (Half A) / fuzzer** — vector fuzzer ledger-complete;
  per-kernel HW status in `tests/vector-verify/README.md`.
- **BUG-B** — closed at the static-feature ceiling (open follow-up: seed_1806).

## Durable constraints (unchanged)

- **Strix swap replaces Phoenix** — one-way door, no hard deadline; our
  productivity gates the timing. Exhaustive Phoenix coverage before the swap.
- **Trace infra posture** — upstream owns decoding (`parse_trace`); we own the
  sweep / matrix / regression layer. Don't reinvent decoding.

## Memory bookkeeping

7 landed notes archived to `…/memory/archive/` (aiesim ×5, coalescing, vector
Half-A); 3 updated (vector-verification, framework-arc, inference-engine); 5
kept. Active index reflects this snapshot.
