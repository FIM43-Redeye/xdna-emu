# Bridge `--trace=pc-anchored` tile discovery: aie-translate jump (2026-04-28)

## Status: deferred from A.2 housekeeping

Phase 5b' of `scripts/emu-bridge-test.sh` (added in A.2 Task 9) needs
to discover which compute tiles a test uses so it can hand them to
`trace-sweep.py --tiles`.  The current implementation parses the
test's `aie.mlir` with grep + awk:

```bash
tile_spec="$(grep -v '^[[:space:]]*//' "$mlir_src" \
  | grep -oP 'aie\.tile\(\K[0-9]+,\s*[0-9]+(?=\))' \
  | awk -F',' '{ col=$1; row=$2+0; if(row>=2) { ... } }' \
  | sed 's/ //g')"
```

A.2 Task 9 code review flagged this as "Important": text-parsing MLIR
is fragile because it can't handle:
- Block comments (`/* ... */`) -- only line comments are stripped.
- `aie.tile` ops embedded in larger expressions (rare today but legal).
- Tests that use programmatic tile construction via aie.dialects
  Python rather than literal `aie.tile(col, row)` ops in `aie.mlir`.
- Multi-line MLIR formatting where col and row are on separate lines.

The regex covers every test currently in the `npu-xrt` suite, which
is why this was deferred -- it works in practice today.

## What the replacement looks like

`aie-translate` is mlir-aie's authoritative MLIR-to-text translator.
It already exposes flag-driven extractors for various AIE structures
(routing, flows, BD layouts).  The right invocation for tiles would
be something like:

```bash
aie-translate --aie-tiles-to-json aie.mlir
```

...or whatever the tile-list extractor flag turns out to be.  We need
to:

1. **Verify whether such a flag exists.** Check `aie-translate --help`
   for `--aie-*-to-json` / `--aie-*-list` style flags.  If a tile-list
   extractor already exists, this becomes a 1-line script change.
2. **If it doesn't exist, file an upstream PR or use a Python bridge
   wrapper.**  `tools/mlir-aie-bridge.py` is the existing pattern for
   "thing I need to ask mlir-aie's Python bindings" queries; a
   `tile-list` subcommand fits naturally there.  Implementation would
   be ~10 LOC: load the MLIR via `aie.ir`, walk the device body,
   collect `aie.tile` ops, emit JSON with `[(col, row, kind), ...]`.
3. **Replace the grep+awk pipeline** with a call to whichever tool
   wins, parsing JSON instead of regex output.  Keep the row >= 2
   compute-tile filter inside the bash (or move it into the bridge
   command behind a `--filter compute` flag).
4. **Add a fallback warning** -- if the structured discovery fails for
   some reason, fall back to the regex with a "WARN: structured tile
   discovery failed, using regex fallback" line so we don't silently
   regress.

## Why the regex works today

Every test in `mlir-aie/test/npu-xrt/` produces a single `aie.mlir`
file with literal `aie.tile(col, row)` ops in canonical form (one per
line, no block comments inside the tile op).  The CI build pipeline
canonicalizes through `aie-translate` itself, so what we read from
disk is what the toolchain emitted -- we're parsing MLIR output, not
hand-written MLIR input.

This is a brittle invariant.  It holds today; it could break if:
- A test starts using `mlir-aie`'s programmatic IRON Python design
  (no on-disk `aie.mlir` to parse at all).
- A new mlir-aie release changes canonical formatting (e.g., adds
  multi-line formatting for long tile lists).
- A test introduces tiles via a non-standard op variant.

The pre-A.2-work test suite passed under the regex, but the next
suite-wide formatting change in mlir-aie could silently break sweep
discovery for the affected tests.

## Entry points when we resume

1. `scripts/emu-bridge-test.sh:2944-2950` -- the regex pipeline, with
   its own TODO comment pointing back at this finding.
2. `tools/mlir-aie-bridge.py` -- existing pattern for Python-bridge
   queries; add a `tile-list` subcommand.
3. Test: extend the existing `--trace=pc-anchored` integration on
   `add_one_using_dma` to also cover a test with a non-trivial tile
   layout (e.g., `vec_vec_add_memtile_init` if it has multiple
   compute tiles) once structured discovery lands.

## Why deferred from A.2

The sweep gate run on `add_one_using_dma` worked.  Replacing the
discovery mechanism is bigger work than the surrounding housekeeping
items (centralize constants, edge cases, doc) and didn't fit cleanly
into the 6-item batch.  Tracking it here so it doesn't get lost.

## References

- Code site: `scripts/emu-bridge-test.sh:2944-2950`
- Existing aie-translate users: `scripts/build-mlir-aie-tests.sh:42`,
  `scripts/build-mlir-aie-tests.sh:481-497`
- Bridge pattern: `tools/mlir-aie-bridge.py`
- A.2 Task 9 code review: see commit `6b7fbb2` review history
