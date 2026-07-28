# Phoenix Management DMA -- Shared Level Completion Design

**Date:** 2026-07-28

**Status:** Approved boundary; written correction pending review

## Purpose

Complete the pinned Phoenix firmware-owned management-DMA lifecycle without
bypassing the firmware handler. A valid mode-3 transfer publishes its data and
result, clears busy, raises the one configured completion source, and leaves
the firmware to produce the context response.

This corrects the earlier source-56 design in this file. Sources `56..59` are
AIE error-notification lanes, and the observed column-1 L2 mask write is
error-network maintenance. Neither is part of management-DMA completion.

## Pinned Evidence

The pinned tuple is Phoenix/NPU1 firmware `1502_00/npu.dev.sbin`, version
`5.5.391`, with the driver-shaped `CHAIN_EXEC_NPU` request and frozen Chess
`add_one_using_dma` instruction stream.

### Source 56 is the wrong interrupt family

The runtime table maps sources `56..59` to handler `0x59f0`. Its helper:

1. uses the four-lane map at local `0xfea0`;
2. scans AIE L1 A/B status and per-column L2 status;
3. disables and clears asserted L2 error bits; and
4. posts firmware event `(6, lane, 1)`.

The later `0x3f` write is `XAIE_ERROR_L2_ENABLE`, matching aie-rt's error
backtracking/rearm sequence. It does not consume the staged management-DMA
transaction.

### Source 76 is the completion wake

Sources `76..79` map to handler `0x5974`. For the pinned tuple, source `76`
selects firmware object 5 and event bit 7; the other three hardware lanes are
not configured for this service.

The handler first calls `0x93f0`, which reads lane 0's system aperture at
`0xbc000000` until it sees `0xdeadbeef`. The four initialized aperture
addresses are:

```text
0xbc000000  configured selector 5
0xbc800000  disabled selector 0xff
0xbd000000  disabled selector 0xff
0xbd800000  disabled selector 0xff
```

The system stub's default zero therefore means "another FIFO word" forever.
Changing only the controller source from 56 to 76 leaves source 76 active and
the handler spinning at `0xbc000000`.

Controlled GDB arms supplied source 76 without changing repository code:

- word `0`, then `0xdeadbeef`, plus level deassertion: `ResponseCompleted`;
- word `1`, then `0xdeadbeef`, plus level deassertion: identical response;
- immediate `0xdeadbeef`, plus level deassertion: identical response.

The null arm proves no invented completion payload or FIFO queue is required
for this successful path. It reaches a clean firmware `WAITI` after `0x2040`
instructions and publishes the exact I2X packet:

```text
body bytes  0x0000000c
protocol    0x0001000c
message ID  0x1d000000
opcode      0x00000018
body        0x04000003, 0x00000000, 0x03000003
```

The body matches the open driver's `AIE2_STATUS_INVALID_PARAM`,
failed-command index 0, and `AIE2_STATUS_APP_LOAD_PDI_FAIL` values. Completion
delivery exposes this response; it does not synthesize or reinterpret it.

Without deassertion, even an immediate sentinel is re-interrupted before the
dispatcher can finish, and no firmware MMIO acknowledgement occurs. The
completion peripheral must therefore drop its level as the handler drains the
aperture; this is not a controller W1C performed by firmware.

## Derived Contract

The three management-DMA lanes remain at
`0x27271000 + lane * 0x1000`. Their descriptor, translation, result, busy, and
drain semantics remain as specified by the blocking-descriptor design.

For a valid mode-3 transfer:

1. copy the complete descriptor byte range;
2. publish result zero at `lane + 0x100`;
3. clear command bit 0 after data and result are visible;
4. mark the shared completion level pending; and
5. assert controller source `76` last.

Source 76 wakes the common firmware service; it is not derived from the
management-DMA lane number. Multiple completed lanes may therefore coalesce
behind the same pending level. Per-lane command/result state remains the
authority the awakened firmware inspects.

The pending level survives temporary masking or another active controller
source. Each management-DMA tick retries source 76 while the level remains
pending.

Reading the configured completion aperture at `0xbc000000` returns the empty
sentinel `0xdeadbeef`, clears the pending level, and deasserts source 76 without
changing its enable bit. This is the acknowledgement edge required for the
firmware handler to return.

Mode 0 remains non-interrupting. Invalid/unmapped descriptors remain busy and
unnotified. A busy-lane drain acknowledgement suppresses later copy and
notification exactly as before.

## Implementation

Keep the model inside the existing `Bus` and `ManagementController`:

- replace the per-lane source-56 constant with shared source `76`;
- add one `bool` recording the shared pending completion level;
- set it after each successful mode-3 completion;
- retry `assert_source(76)` at the end of every management-DMA tick while it
  remains set;
- intercept 32-bit reads of `0xbc000000`, return `0xdeadbeef`, clear the
  pending flag, and deassert source 76; and
- add the controller's small source-deassert operation, which clears only the
  selected status/active state and preserves enable state.

No FIFO container, second DMA engine, generic interrupt-device framework,
timing scheduler, direct firmware-event injection, or response synthesis is
introduced.

## Tests

Tests are written red first and cover:

1. controller deassertion clears status/active state but preserves enable;
2. mode-3 completion publishes data, result, and clear-busy before shared
   source 76 becomes observable;
3. all three management-DMA lanes use the shared source rather than fabricated
   per-lane sources;
4. a masked or controller-blocked completion remains pending and asserts when
   the source later becomes available;
5. `0xbc000000` returns `0xdeadbeef` and deasserts source 76;
6. mode 0 and busy-lane drain behavior remain unchanged; and
7. the driver-shaped guard reaches `ResponseCompleted`, consumes X2I,
   publishes the exact three-word I2X body above, returns to `WAITI`, and
   leaves no active controller source.

The stale L2-mask assertion is removed: it observes AIE error rearming, not
completion progress. Full verification remains `cargo test --lib`.

## Evidence-Tool Correction

`m2c_probe_peek` currently reads low literal addresses through the raw
D-side `peek8` view while its documentation claims to resolve L32R literals.
For the relocated low image this is displaced from the instruction/literal
view and produced false constants during this audit.

Correct the diagnostic separately from runtime behavior so it reports the
literal/fetch view explicitly while retaining the D-side view when useful.
Add a focused regression using a known relocated literal. This prevents the
same false prior from recurring but does not participate in completion
semantics.

## Deferred

Older firmware tuples, nonzero management-DMA result codes, payload-bearing
system-FIFO records, sources `77..79`, measured latency, and priority between
simultaneous controller sources require their own evidence. None is inferred
by this pinned successful-completion slice.
