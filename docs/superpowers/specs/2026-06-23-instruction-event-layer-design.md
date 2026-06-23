# Instruction-Event Layer -- Design (post-review rewrite)

**Status:** approved goal; rewritten 2026-06-23 after an Opus pass found the first
draft described the generator's mechanics wrong (2 CRITICAL, code-verified). Goal
and architecture unchanged; the emission path, edge kind, dump scope, and
sequencing are corrected to match the actual code.
**Issue:** #140, prerequisite for the jitter-grounding rule.
**Evidence:** `docs/superpowers/findings/2026-06-23-jitter-grounding-spikes.md`
**Successor plan:** explicit jitter-robust grounding rule (consumes this layer).

## Goal

Expose the core instruction-lock edge as an **oriented within-domain candidate
pair**, so the next plan's grounding rule can ground the cycle-exact compute
segment. The spikes proved `INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ` is
exactly 22 cycles (range 0 over 20 runs, surviving every span outlier), but the
static-orientation layer cannot currently produce that edge as a candidate pair.
This plan closes that gap and nothing more.

## Why this layer (not a workaround)

Producible edges today (`event_map` orients only `PORT_RUNNING` / `DMA_*`) are
either delivery-jittery or genuinely +/-1..4 at the hardware (level events --
confirmed by the decoder validation: our decoder matches upstream exactly, so the
wobble is real silicon, not a decode bug). The only proven range-0-exact segment
uses instruction-lock EDGE events, which full coverage needs regardless. So we
build the layer rather than ground a throwaway DMA-edge segment.

## Deliverable boundary (explicit)

This plan ends when `generate_ledger` / `candidate_pairs_from_dump` for add_one
produce the oriented core `ACQUIRE -> RELEASE` within-domain pair, verified
offline. **Nothing consumes it for grounding yet** -- `try_derives` will orient on
it (it is a `program_path` fact; see below) but the offset-measurement / falsifier
/ report is the next plan. This plan's test is "the candidate pair is produced and
oriented correctly."

## Orientation: aggregate-across-lock-IDs, derived from the ELF

`core_relay.rs` decodes the Chess core ELF into `CoreLockOp { lock_id, kind, pc }`
in program (PC) order. In add_one the first acquire and first release are on
DIFFERENT lock IDs (acquires on {1,2}, releases on {0,3}) -- there is no single
lock with both. But the trace events `INSTR_LOCK_ACQUIRE_REQ` /
`INSTR_LOCK_RELEASE_REQ` are **lock-ID-agnostic aggregates** (they fire on every
acquire / every release). So the matching static fact is also aggregate:

> **lock-order fact:** the minimum acquire PC precedes some release PC in the
> core's program (`min(acquire PC) < max(release PC)`), recovered from the ELF.

This orients the aggregate trace-event pair `ACQUIRE -> RELEASE` (parent=acquire).
It is DERIVED from the ELF decode, not the hardcoded truism "acquire precedes
release," and it is robust to the observed 15-acquire / 16-release count mismatch
(an extra release with no paired acquire does not affect min-acquire < any-release).
It is a *weaker, distinct* fact from the per-lock `CoreLockRelay` 3-way
intersection -- intentionally so, to match the aggregate trace events.

## Edge kind: reuse `program` / `program_path` (NOT a new kind)

The ledger loader hard-validates kind against a fixed allowlist
(`ledger.py:27` `_KINDS = {route,bd,lock,identity,program}`; unknown kinds raise),
the generator auditor rejects non-`{route,program}` kinds (`generator.py:359`),
and `try_derives` orients only on `config_path`/`program_path` (`rules.py:40`). A
new `program_order` kind would be rejected at load and ignored at derive. The
acquire->release edge is a program-decoded (ELF) relation, exactly what
`kind="program"` -> `program_path` already represents. **Reuse it:** emit
`kind="program"`; `try_derives` already orients on `program_path`; no loader,
audit-allowlist, or derive-union changes needed.

## Architecture (corrected to the real generator contract)

The generator does NOT emit "because both events are configured on a core with a
fact." It iterates `permutations(fired_event_keys, 2)`, resolves each end via
`resolve_event_port`, **skips any pair where an end resolves to `None`**
(`generator.py:223-224`), and orients survivors by SS-route-graph reachability.
Lock events have no port (`event_map.py:212-213` returns `None`), so they are
skipped by the existing path -- it produces zero lock edges. Therefore this plan
adds a **separate, non-port emission path**:

```
core_relay.rs (ELF: min-acquire-PC < release-PC)
  -> CoreLockUsage surfaced into TileDump  (currently computed then DISCARDED)
     -> build_dump populates a per-tile lock_order field
        -> dump JSON  ->  dump_model.py loads it  (+ fixture regenerated)
           -> generator: NEW path -- for each tile with a lock_order fact whose
              ACQUIRE+RELEASE events are in the fired set, emit
              {a=<tile>|INSTR_LOCK_ACQUIRE_REQ, b=<tile>|INSTR_LOCK_RELEASE_REQ,
               kind="program", cite=<ELF source>}, bypassing resolve_event_port
              -> audit_ledger: NEW branch that validates this entry WITHOUT
                 demanding port resolution (the existing audit re-resolves both
                 keys to route-graph nodes, generator.py:414-415, and would
                 reject a portless entry)
                 -> candidate_pairs_from_dump returns (RELEASE, ACQUIRE) as-is
```

## Components

### 1. `src/device/stream_switch/core_relay.rs` (extend)
Expose, per core whose ELF has `min(acquire PC) < max(release PC)`, an aggregate
lock-order fact (the two trace-event names + the ELF cite). It already decodes the
`CoreLockOp` PC-ordered list for `CoreLockRelay`; this is a lighter sibling
emission. Safe false-negative: emit nothing if order can't be recovered.

### 2. `examples/dump_config_json.rs` + `TileDump` + `build_dump` (Rust, extend)
`CoreLockUsage` is consumed inside `resolve_route_graph` and discarded -- it never
reaches the JSON dump. Add a `lock_order` field to the per-tile dump struct and
populate it in `build_dump` from the core_relay analysis. (This is the
serialization scope the first draft omitted.)

### 3. `tools/config_extract/dump_model.py` (extend)
Load the new `lock_order` field; tolerate fixtures predating it (empty -> no
edge), per the Tier-E backward-compat idiom.

### 4. Regenerate `tools/config_extract/fixtures/add_one_using_dma.config.json`
The offline test cannot pass on the current fixture (no lock_order field). After
the Rust change, regenerate the fixture so it carries the fact. **Explicit ordered
dependency:** Rust dump field -> regenerate fixture -> Python generator path ->
test. Regenerate the other suite fixtures too (objFifo, vector_scalar) for
consistency.

### 5. `tools/inference/selfmodel.py` (`_MENU[0]`)
Add `INSTR_LOCK_ACQUIRE_REQ`, `INSTR_LOCK_RELEASE_REQ` (aie-rt event ids 44/45,
`aie-rt/.../xaie_events_aieml.h:79-80`; `configure_batch` accepts them). **Must
land in the same change as components 1-6** -- adding them to the menu before the
emission path exists makes them fire as always-unresolved events and regresses the
loop's add_one convergence (I-5).

### 6. `tools/config_extract/generator.py` + `event_map.py` (extend)
- `event_map`: lock events already return `None` from `resolve_event_port` -- keep
  that; the new path does not use port resolution. (No "recognize but resolve"
  contradiction: they simply aren't routed through the port path.)
- `generator`: add the non-port emission path (architecture above) and the
  matching `audit_ledger` branch.

## Testing

- **Offline unit (primary deliverable):** on the REGENERATED add_one fixture,
  `generate_ledger` (both lock events in the fired set) produces exactly one
  `kind="program"` entry with `a=...|INSTR_LOCK_ACQUIRE_REQ`,
  `b=...|INSTR_LOCK_RELEASE_REQ`, non-empty cite; `candidate_pairs_from_dump`
  includes `(RELEASE, ACQUIRE)`.
- **Audit:** `audit_ledger` passes the portless lock entry (new branch), still
  rejects genuinely malformed entries.
- **Rust unit:** `core_relay` emits the aggregate lock-order fact for a core with
  acquire-before-release; nothing for a core without recoverable order.
- **Menu/enumeration:** `enumerate_configured_events` includes the two lock events
  on active core tiles.
- **No-regression:** the existing route/program edges for add_one are unchanged
  (the new path is additive); `cargo test --lib` clean; offline Python suite clean.
- **Backward-compat:** a fixture without `lock_order` loads with no lock edge, no
  crash.

## Out of scope (next plan / deferred)
- The grounding rule: `same_domain` (keyed `(col,row,pkt_type)` -- the C1
  per-module-timer fix lives there), `offset_exact` (Q=0 exact cross-run
  equality), `ground_edge` (segment vs named gap), falsifier (ordering +
  lock-handoff; additivity vacuous/dropped), segment/gap report.
- Determinism-partition / trace_join re-key off std.
- Graduating `validate_decoder.py` to a permanent regression test.
- Cross-domain timer-sync; multi-column.

## Correctness principle
Lock-order orientation is DERIVED from the Chess ELF via `core_relay`. Event names
come from the aie-rt header. Edge kind reuses the existing `program_path`
predicate. Nothing is kernel-specific. DERIVE FROM THE TOOLCHAIN holds.
