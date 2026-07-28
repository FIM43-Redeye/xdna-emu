# Phoenix Shim TCT to Firmware Handoff

**Status:** Approved design
**Target:** Pinned Phoenix NPU1 firmware `1502_00` and the frozen Chess
`add_one_using_dma` artifact
**Goal:** Close the first real array-completion-to-firmware edge without
synthesizing a firmware response or assigning a firmware/AIE clock ratio.

## Evidence

The configured firmware path already loads the frozen PDI into the shared
array. Extending only the runtime iteration budget showed that the array then
executes naturally:

- output at `0x0400c000` is the expected `2..=65`;
- the engine halts after completing the kernel;
- shim tile `(1,0)` retains
  `Token { channel_id: 0, controller_id: 15 }`; and
- firmware remains in `waiti` with no I2X response.

Running the same frozen Chess artifact through the installed aiesim bridge
produced this drained completion record:

```text
[pl-tct 6] tct#1 d0=0x0020600f (token, drained)
```

Replaying exactly `0x0020600f` as the first word read from the existing
source-76 system aperture at `0xbc000000`, followed by its existing
`0xdeadbeef` empty sentinel, changed the unmodified firmware result to
`ResponseCompleted`. The firmware published the I2X response after 2,401
functional pump iterations. No response, task-done flag, or firmware event was
injected.

The word agrees with the open toolchain:

- physical tile identity is `col << 21 | row << 16`;
- shim S2MM channel 0 maps to actor 0;
- the completing shim's controller ID is 15;
- stream-header packet fields and odd parity follow
  `mlir-aie/lib/Targets/AIETargetNPU.cpp`; and
- tile packing, actor maps, and controller IDs follow
  `AIENpuToCert.cpp` and `AIETargetModel.cpp`.

Packet type 6 is grounded by the exact aiesim record and by the successful
unmodified-firmware discriminator. The open sources inspected do not name the
Phoenix management landing packet type separately, so this one field remains
hardware-behavior evidence rather than an open-source symbol.

## Chosen Design

Reuse the two endpoints that already own the state:

1. The shim DMA engine remains the sole owner and producer of task-completion
   tokens.
2. The firmware `Bus` remains the sole owner of management source 76 and the
   `0xbc000000` drain aperture.

No new interrupt controller, completion protocol, or response path is added.

### Firmware bus

Add one `VecDeque<u32>` for pending system-completion words. Publishing a word
queues it and attempts to assert the existing source 76.

The management-DMA retry point already runs after every integrated firmware
instruction. It will assert source 76 while either:

- an asynchronous management-DMA completion is pending; or
- the completion-word queue is nonempty.

Reads from `0xbc000000` behave as the observed drain loop requires:

- return the oldest queued word without deasserting the level; then
- once no words remain, return `0xdeadbeef`, clear the management-DMA
  completion level, and deassert source 76.

This preserves the already-tested management-DMA lifecycle while allowing the
same shared source and aperture to carry a real TCT record.

### Runtime handoff

After each AIE cycle, the runtime pump checks only the configured partition's
shim tile (`DeviceState::start_col`, row 0) for a channel-0 token. It must use
the channel-filtered peek/pop API so unsupported tokens are left owned by the
DMA engine rather than silently discarded.

For that token, the pump encodes the observed Phoenix TCT record from:

- the physical shim column and row;
- packet type 6;
- shim S2MM0 actor 0; and
- the token's controller ID.

The parity bit is calculated with the same odd-parity rule used by the open
toolchain. For the frozen proof this must produce exactly `0x0020600f`.

Once the word is queued, the pump must not report
`ArrayIdleFirmwareWaiting` merely because the just-completed AIE cycle halted
the engine. It gives firmware the next outer iteration to service source 76.
All existing stalled, error, unresolved-poll, unknown-instruction, and bounded
no-progress stops remain unchanged.

## Acceptance Test

TDD proceeds in three observable checks:

1. A focused encoder test requires physical shim `(1,0)`, actor 0, and
   controller 15 to produce `0x0020600f`.
2. A firmware-bus test queues that word, observes source 76, drains the word
   followed by `0xdeadbeef`, and observes source 76 deasserted. It also covers
   retry after the source becomes enabled.
3. The existing frozen configured-CU guard runs through natural completion and
   requires:
   - `RuntimePumpStop::ResponseCompleted`;
   - a matching successful EXECUTE I2X response from unmodified firmware;
   - X2I request consumption;
   - output words exactly `2..=65`;
   - no remaining shim S2MM0 token; and
   - no unresolved firmware poll, unknown instruction, engine stall, or engine
     error.

The guard retains its PDI placement, physical-column, program-memory, and
core-enable assertions. It does not assert an exact cycle or instruction
count.

## Explicitly Deferred

- TCT actor mappings beyond Phoenix shim S2MM channel 0;
- additional partitions, columns, and simultaneous completions;
- a finite completion-FIFO depth or backpressure rule;
- generalized AIE2/AIE2P packet landing;
- firmware/AIE clock-ratio and completion-latency calibration;
- Peano and additional-kernel acceptance; and
- virtual PCI and unmodified-driver integration.

These enter only after this single-token proof is green. Expanding them now
would turn one observed edge into speculative architecture.

## Verification

Run the focused encoder/bus tests through an observed RED then GREEN cycle.
Run the frozen guard with the pinned firmware and fixture present so it cannot
silently skip. Then run:

```bash
cargo test --lib
cargo test -p xdna-emu-ffi
cargo fmt --all --check
git diff --check
```
