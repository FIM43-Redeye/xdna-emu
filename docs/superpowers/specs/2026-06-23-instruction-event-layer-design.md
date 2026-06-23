# Instruction-Event Layer -- Design

**Status:** approved (brainstorm + 3 HW spikes + decoder validation, 2026-06-23)
**Issue:** #140, prerequisite for the jitter-grounding rule.
**Evidence:** `docs/superpowers/findings/2026-06-23-jitter-grounding-spikes.md`
**Successor plan:** explicit jitter-robust grounding rule (consumes this layer).

## Goal

Expose core instruction-lock edge events as **oriented within-domain candidate
pairs**, so the next plan's grounding rule can ground the cycle-exact compute
segment. The spikes proved the core compute segment
`INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ` is exactly 22 cycles (range 0
over 20 runs, surviving every span outlier) -- but the static-orientation layer
does not currently produce that edge as a candidate pair, so the engine can never
see it. This plan closes that gap and nothing more.

## Why this layer (not a workaround)

The producible edges today (`event_map` orients only `PORT_RUNNING` / `DMA_*`) are
either delivery-jittery (DMA FINISHED) or genuinely +/-1..4 at the hardware
(PORT_RUNNING level events -- confirmed by the decoder validation: our decoder
matches upstream exactly, so the wobble is real silicon signal, not a decode
bug). The only proven range-0-exact segment uses instruction-lock EDGE events,
which we will need for full coverage regardless. So we build the layer directly
rather than ground a throwaway DMA-edge segment we would later discard.

## The deliverable boundary (explicit)

This plan ends when the engine's candidate pairs for add_one include the oriented
core `ACQUIRE -> RELEASE` within-domain edge, verified end-to-end. **Nothing
consumes it yet** -- the grounding rule (offset measurement, falsifier, report)
is the next plan. This plan's test is "the candidate pair is produced and
oriented correctly," not a cycle-accurate grounding result. Clean incremental
boundary, one serious thing.

## Architecture

Edge orientation for instruction-lock events does NOT come from dataflow routing
(there is no port). It comes from the **static program decode** already in
`core_relay.rs`, which scans the Chess compute-core ELF and recovers core lock
acquire/release instructions and their order. We surface, per core whose program
contains an acquire before a release, an oriented lock-order relationship between
the two trace-event types, and emit it as a ledger edge when both events are in
the configured set.

```
core_relay.rs (ELF decode: acquire-PC < release-PC)
   -> dump: oriented core lock-order fact (acquire before release)
      -> generator: emit ledger edge {a=ACQUIRE, b=RELEASE, kind=program_order}
         -> candidate_pairs_from_dump includes (RELEASE, ACQUIRE)
```

## Components

### 1. `src/device/stream_switch/core_relay.rs` (extend)

It already decodes core lock acquire/release order for the `CoreLockRelay`
3-way intersection. Add a lighter emission: for a core whose ELF contains at
least one lock-acquire instruction ordered before a lock-release instruction,
emit an oriented **lock-order** fact -- the acquire trace event precedes the
release trace event on that core. Orientation is DERIVED from the ELF
(acquire-PC < release-PC), not the hardcoded truism "acquire precedes release."
Safe false-negative if the order can't be recovered (emit nothing).

### 2. `tools/config_extract/dump_model.py` (extend)

Load the new lock-order fact from the dump (tolerate fixtures predating it ->
empty, per the established Tier-E backward-compat idiom).

### 3. `tools/inference/selfmodel.py` (extend `_MENU[0]`)

Add `INSTR_LOCK_ACQUIRE_REQ`, `INSTR_LOCK_RELEASE_REQ` to the core (pkt 0) menu.
Both are valid event names in the aie-rt events header (confirmed). They become
enumerable configured events on active core tiles.

### 4. `tools/config_extract/generator.py` + `event_map.py` (extend)

- `event_map`: recognize `INSTR_LOCK_ACQUIRE_REQ` / `RELEASE_REQ` as valid core
  events so they are not rejected. They do NOT resolve to a port -- they are
  handled by the lock-order path, not route/program port resolution.
- `generator`: when both lock events are configured on a core that carries a
  lock-order fact, emit a ledger entry `{a=<core>|INSTR_LOCK_ACQUIRE_REQ,
  b=<core>|INSTR_LOCK_RELEASE_REQ, kind="program_order", cite=<ELF source>}`.
  `kind="program_order"` is a NEW edge kind, distinct from `route` (config_path)
  and `program` (program_path through-core dataflow) -- this is intra-core
  instruction order.

### 5. `tools/inference/selfmodel.py::candidate_pairs_from_dump` (no change needed)

Already returns `(e["b"], e["a"])` for every ledger entry, so the new
`program_order` edge flows through as `(RELEASE, ACQUIRE)` automatically.

## Testing

- **Offline unit (primary deliverable):** `generate_ledger` for the add_one dump
  (with both lock events configured on the core) produces exactly one oriented
  `program_order` entry with `a = ...|INSTR_LOCK_ACQUIRE_REQ`,
  `b = ...|INSTR_LOCK_RELEASE_REQ`, and a non-empty cite.
  `candidate_pairs_from_dump` includes `(RELEASE, ACQUIRE)`.
- **Rust unit:** `core_relay` emits the lock-order fact for a core with
  acquire-before-release; emits nothing for a core without recoverable order
  (safe false-negative).
- **Menu/enumeration unit:** `enumerate_configured_events` includes the two lock
  events on active core tiles.
- **Backward-compat:** fixtures predating the lock-order fact load with no edge,
  no crash.
- `cargo test --lib` clean (Rust change); offline Python suite clean.

## Out of scope (next plan / deferred)

- The grounding rule itself: `same_domain` (keyed on `(col,row,pkt_type)` -- the
  C1 per-module-timer fix lives here, with `same_domain`), `offset_exact`,
  `ground_edge`, the falsifier (ordering + lock-handoff; additivity is vacuous),
  segment/gap report. This layer only EXPOSES the edge.
- Determinism-partition / trace_join re-keying off std.
- Graduating `validate_decoder.py` into a permanent ours-vs-upstream regression
  test (noted follow-up).
- Cross-domain timer-sync; multi-column.

## Correctness principle

The lock-order orientation is DERIVED from the Chess ELF via the existing
`core_relay` decode -- not hardcoded. Event names come from the aie-rt header.
Nothing is kernel-specific. DERIVE FROM THE TOOLCHAIN holds.
