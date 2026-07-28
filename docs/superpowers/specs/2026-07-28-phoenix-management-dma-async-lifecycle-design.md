# Phoenix Management DMA -- Asynchronous Lifecycle Design

**Date:** 2026-07-28

**Status:** Approved

## Purpose

Complete the firmware-owned management-DMA lifecycle for mode-3 transfers:
publish transferred data and result state, notify the firmware through the
configured per-lane controller source, and model the separate busy-lane drain
handshake at `lane + 0x114`.

This extends the existing functional blocking model. It does not introduce a
second DMA engine, a timing scheduler, or an AIE-array completion path.

## Derived Contract

The pinned firmware allocates three lanes at `0x27271000 + lane * 0x1000`.
Mode 3 uses the same descriptor and translated endpoint machinery as mode 0.
Command bit 0 is busy, and the low six bits of `lane + 0x100` are the result.

The runtime interrupt table registers sources `56..59` with handler `0x59f0`
and arguments `0..3`. The handler posts firmware event `(6, lane, 1)`, then
acknowledges and re-enables source `56 + lane`. Because this firmware allocates
only lanes `0..2`, normal mode-3 completions use sources `56..58`; the fourth
registered source is not consumed by this slice.

The service routine at `0x8bc8` is a separate owner-tagged drain:

1. scan allocated lanes whose owner tag matches;
2. if command bit 0 is already clear, release the software lane directly;
3. otherwise write `1` to `lane + 0x114` and wait for command bit 1; and
4. clear the lane's software allocation bit.

Therefore `+0x114` is not the normal completion notification. It acknowledges
a request to drain a lane that is still busy.

## Implementation

Rename the existing blocking-only tick to cover both supported modes and reuse
its descriptor, translation, read, and write helpers.

For a valid mode-3 transfer:

1. copy the complete byte range;
2. publish zero result at `lane + 0x100`;
3. clear command bit 0 after the data and result are visible; and
4. assert controller source `56 + lane` last.

The controller already enforces enable state and single-active-source behavior.
No parallel notification queue is added.

When firmware writes bit 0 to `lane + 0x114` while the lane is busy, set command
bit 1. A busy lane with bit 1 set is drained and must not subsequently copy data
or raise a completion source. The next firmware publication overwrites the
command word, naturally clearing the prior drain acknowledgement.

Invalid descriptors and unmapped endpoints remain busy and unnotified. No
unsupported hardware error code is invented.

## Tests

Tests first cover:

- mode-3 data and result publication before the per-lane source is observable;
- sources `56..58` for the three allocated lanes;
- a disabled source not manufacturing controller state;
- `+0x114` setting bit 1 on a busy lane and suppressing later copy/notification;
- mode-0 behavior remaining unchanged; and
- the pinned firmware guard staging the 16 KiB host range and consuming the
  source through the real firmware handler.

## Deferred

Measured latency, nonzero hardware result codes, controller priority,
source 59's owner, and drain side effects beyond the observed bit-1
acknowledgement remain unclaimed.
