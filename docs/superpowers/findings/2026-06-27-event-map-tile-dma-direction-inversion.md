# Finding: inverted S2MM/MM2S direction in event_map._resolve_tile_dma

**Date:** 2026-06-27
**Status:** FIXED (commit `fix(#140): correct inverted S2MM/MM2S direction`).
**Severity:** correctness bug, latent in merged + HW-validated code.

## What

`tools/config_extract/event_map.py::_resolve_tile_dma` mapped compute/memtile
DMA trace events to stream-switch ports with the direction convention
**inverted**: `S2MM → slave`, `MM2S → master`. The authoritative convention
(`src/device/stream_switch/route_graph.rs`, per `src/device/array/routing.rs`)
is the opposite — **S2MM = master switch port** (the switch drives data into the
DMA, which writes memory), **MM2S = slave switch port** (the DMA reads memory and
drives data into the switch). The shim path `_resolve_shim_dma`, in the same
file, already used the correct convention; the two DMA paths disagreed.

## Effect

Every compute/memtile DMA event landed on the *opposite* stream-switch node, so
the (sound) reachability oracle attached each event name to the wrong endpoint.
For two_col this scrambled cross-column producer/consumer orientation: the
compute **input S2MM** DMA never appeared as a consumer in any cross-column pair,
the distribute and gather directions mixed, and physically-backwards pairs
appeared (e.g. a memtile port shown downstream of a compute input DMA). This is
what made the SP1 analysis feel "two_col is missing something."

## Why latent

add_one's grounding leans on **shim** DMA events (the correct `_resolve_shim_dma`
path) plus PORT_RUNNING / lock / core events — none of which go through
`_resolve_tile_dma`. two_col is the first kernel to put a compute-tile DMA at a
cross-column endpoint, which is what surfaced it. Two unit tests
(`test_memtile_dma_{mm2s,s2mm}_maps_to_dma_port`) had *encoded* the inversion,
asserting the opposite of the shim tests for the identical concept, so the suite
stayed green.

## Fix

One line (`port_dir = "slave" if direction == "MM2S" else "master"`) + the
docstring; correct the two tests to the faithful convention; add a regression
guard (`test_tile_and_shim_dma_use_same_direction_convention`) asserting shim and
tile DMA share one convention. Verified: two_col cross-column pairs become
faithful (compute S2MM is the consumer of the memtile/shim producer); add_one
grounding un-regressed on real HW (`placed`, core segment exact, shim gaps
intact); blast radius was exactly the two inversion-encoding tests.

## Relationship to connectivity (#140)

This was the real blocker behind the cross-column grounding. With orientation
faithful, the consumer is correctly the compute input S2MM DMA (silent, circular
BD), so the conversation is honestly `unobserved` until the lock-substitution
grounding (v2 spec) gives the silent consumer an observable rel-lock proxy.
