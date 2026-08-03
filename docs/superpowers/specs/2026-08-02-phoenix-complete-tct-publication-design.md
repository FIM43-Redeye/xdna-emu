# Phoenix Complete TCT Publication

**Status:** Implemented and verified (2026-08-02)
**Target:** Pinned Phoenix NPU1 firmware `1502_00`
**Goal:** Publish every TCT actor accepted by the current open mlir-aie
toolchain through the configured array fabric and the signed firmware's four
completion lanes.

## Evidence and Boundary

`AIENpuToCert.cpp` is the authoritative map from DMA direction/channel to TCT
actor. Its current Phoenix/AIE2 tables are:

| Tile | S2MM actors | MM2S actors |
|---|---|---|
| Shim | 0, 2, 3, 4 | 6, 7, 8, 9, 10, 11, 12, 13 |
| Memory | 1, 2, 3, 4, 5, 6, 7 | 16, 17, 18, 19, 20, 22, 23, 24, 25, 26 |
| Compute | 0, 1 | 6 |

The compiler rejects channels outside those tables. The emulator must derive
the tables from the live source and must not invent actors for rejected
channels.

`AIEGenerateColumnControlOverlay.cpp` establishes the transport contract. A
TCT originates at the completing tile's `TileControl:0` packet source and
reaches shim `South:0` only through a configured packet flow. Non-shim tests in
mlir-aie install that flow explicitly; the generated overlay defaults to the
shim-only flow. `keep_pkt_header=true` preserves the one-word TCT header.

The signed-firmware proofs establish the landing contract independently:
physical column `n` lands on lane `n - 1`, source `76 + lane`, and aperture
`0xbc000000 + lane * 0x00800000`. The stream header carries odd transport
parity, while the signed firmware matches the parity-free TCT key. The landing
boundary therefore removes bit 31 before publishing the word to the firmware
bus. The frozen controller-15 record did not distinguish those forms; the
controller-14 signed-firmware discriminator did.

## Chosen Design

### Build-derived actor data

Extend `xdna-archspec` generation with a small fail-fast extractor for the six
named vectors in the live `AIENpuToCert.cpp`. Emit them under
`xdna_archspec::aie2::tct`. A missing, malformed, empty, or out-of-range table
fails the build. No checked-in copy or fallback table is added.

### Token to packet

After each array cycle, visit every present Phoenix tile. For each tile with a
pending token:

1. inspect the oldest token without reordering the tile's shared token FIFO;
2. select the derived actor table from tile kind and DMA direction;
3. encode physical column, physical row, packet type 6, actor, and controller
   ID, then add odd stream-header parity; and
4. enqueue the single word with TLAST on the tile's existing TileControl output
   queue.

At most one token per tile is admitted per array cycle. A channel rejected by
the toolchain remains owned by its DMA engine rather than being guessed or
discarded.

The existing packet switch then owns routing, arbitration, backpressure, and
hop latency. A missing route must not publish a completion; the switch's
existing fail-loud packet-route path remains authoritative.

### Shim landing

After normal array routing, drain words which have actually reached each
present shim tile's `South:0` master boundary. Require a one-word packet with
TLAST, clear only transport parity bit 31, and publish it to that physical
column's existing firmware lane. No other shim South port is a completion
aperture.

The functional runtime must regard queued token output, in-flight fabric data,
and shim-boundary data as pending work. It may not report
`ArrayIdleFirmwareWaiting` while any of those can still reach firmware.

The direct `NpuExecutor` transaction-stream `WAIT_TCTS` path remains unchanged.
It does not coexist with the signed-firmware runtime and therefore must not be
unified in this slice.

## Acceptance

TDD will pin these boundaries before implementation:

1. Generated actor tables exactly match all six live toolchain vectors.
2. Shim, memory-tile, and compute-tile tokens encode the expected physical
   tile, actor, controller ID, parity, and TLAST.
3. Per-tile tokens retain issue order.
4. Without a configured packet route, a non-shim token never reaches a
   firmware lane.
5. With the toolchain-shaped route, a non-shim token traverses the existing
   switches and reaches only the correct physical-column lane as the
   parity-free firmware key.
6. Multiple columns remain isolated on sources 76 through 79.
7. Existing signed-firmware shim guards remain green, followed by
   `cargo test --lib`, `cargo test -p xdna-emu-ffi`, `cargo fmt --all --check`,
   and `git diff --check`.

## Implementation Evidence

- `3bbd0bbb` derives all six actor tables from live mlir-aie source.
- `e73a0bf1` converts eligible DMA tokens into TileControl packets and drains
  only routed shim-South0 egress.
- `eaba51ca` publishes parity-free keys at the signed-firmware boundary while
  leaving direct `NpuExecutor` waits unchanged.
- Synthetic guards cover exact actors, issue order, unsupported-channel
  ownership, configured-versus-missing non-shim routing, and parity removal.
- Pinned signed firmware completes both frozen Chess and Peano kernels through
  the routed path; the direct-response control remains green.
- `cargo test --lib`: 4,298 passed, 32 ignored, zero failed.
- `cargo test -p xdna-archspec`: 537 passed, 2 ignored, zero failed;
  `cargo test -p xdna-emu-ffi`: 100 passed, zero failed.
- `cargo fmt --all --check` and `git diff --check` pass.

## Deferred

- Unifying signed-firmware publication with direct `NpuExecutor` waits.
- Undocumented actor encodings rejected by the current toolchain.
- Finite TCT FIFO depth and hardware backpressure calibration.
- AIE2P landing and firmware-lane topology.
- Firmware/AIE clock ratio and calibrated completion timing.

Those are broader completion-model work, not prerequisites for faithful
Phoenix publication of every currently supported actor.
