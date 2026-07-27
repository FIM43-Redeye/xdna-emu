# Phoenix Firmware-Caused Array Mutation -- Design

**Date:** 2026-07-27

**Status:** Approved

## Purpose

Prove the first authentic control-plane transition from the pinned Phoenix
driver contract, through unmodified `1502_00` management firmware, into the
interpreter engine's existing `DeviceState`.

The proof ends when a real `CREATE_CONTEXT` request causes firmware to program
the selected array columns. It does not yet attempt kernel execution or the
unmodified-driver PCI boundary.

## Acceptance Path

```text
natural firmware boot
  -> pinned management initialization messages
  -> CREATE_CONTEXT
  -> unmodified firmware handler
  -> firmware array MMIO
  -> shared DeviceState
  -> selected physical application column reprogrammed and active
  -> genuine CREATE_CONTEXT response
```

The test starts with every array column gated. Authentic `RESUME` then ungates
all five physical columns as part of its global reset. Success requires
`CREATE_CONTEXT` array MMIO to remain within its selected physical application
column, gate and re-enable that column, and leave every unselected column's
pre-command clock state unchanged, without calling `xdna_emu_assign_partition`,
`DeviceState::assign_partition_columns`, or any other host-side stand-in.

## Chosen Approach

Use a test-only, driver-shaped management-channel harness around the existing
`Bus::host_load32`, `Bus::host_store32`, explicit source-46 assertion, and
`FirmwareProcessor::boot_to_idle_with_device`.

The harness owns transport mechanics only. Its synchronous operation:

- append a complete raw request to the firmware-published X2I ring;
- publish X2I tail after the packet is complete;
- explicitly assert management-controller source 46;
- run unmodified firmware back to its natural `waiti`;
- consume the complete I2X response;
- publish I2X head and clear host-visible interrupt status.

Its posted-request operation performs the same X2I publication and firmware
delivery but expects no immediate I2X packet. The pinned driver uses this form
for `REGISTER_ASYNC_EVENT_MSG`: the request remains outstanding and its eventual
response is the asynchronous event notification.

It does not parse opcodes, construct responses, call array helpers, or become a
production firmware API.

### Alternatives deferred

1. **Expand the public FFI now.** Deferred because no production frontend yet
   consumes post-boot BAR4 or interrupt delivery, and the causal
   tail-to-controller edge remains unknown. A test-only harness avoids freezing
   that uncertainty into the ABI.
2. **Build virtual PCI now.** Deferred because it would wrap an inner system
   that has not yet proven one firmware-caused array transition.
3. **Build the firmware/array co-scheduler now.** Deferred until this proof
   reveals the real command path and the first point that requires concurrent
   firmware, array, and host-memory progress.

## Pinned Initialization Prefix

Starting after natural alive publication and driver-style clearing of
`FW_ALIVE_OFF`, send the pinned primary driver's management messages in order:

1. `SET_RUNTIME_CONFIG(2, 1)`
2. `SET_RUNTIME_CONFIG(4, 1)`
3. `ASSIGN_MGMT_PASID(0)`
4. `SUSPEND`
5. `RESUME`
6. `SET_RUNTIME_CONFIG(1, 1)`
7. `GET_FIRMWARE_VERSION`
8. `QUERY_AIE_VERSION`
9. `QUERY_AIE_TILE_INFO`
10. post `REGISTER_ASYNC_EVENT_MSG`, once per reported column, without waiting
    for an immediate response

Then send `CREATE_CONTEXT` for a valid, deliberately narrow partition starting
at physical column 1. Phoenix advertises `first_col = 1`; physical column 0 has
no shim tile and is not an application partition start.

PSP waitmode polling and SMU clock writes occur between those messages in the
real driver but are outside this post-alive in-process slice. No PSP or SMU
internal model is added. If a firmware command itself exposes a required
architectural effect, that effect must be derived before it is modeled.

Request layouts, message IDs, and response sizes come from pinned driver commit
`216cefececd74effcd7a88350c71b99f5ef9a215`. Expected behavior comes from the
unmodified firmware and existing hardware evidence, not from a driver-specific
responder.

## Array Proof

The `CREATE_CONTEXT` check requires all of the following:

- firmware consumes the exact request and publishes a successful response;
- the response supplies a valid context ID and context-channel descriptor;
- firmware performs array-region MMIO through the borrowed `DeviceState`;
- array writes remain within the requested physical application column;
- the requested column is gated and re-enabled, ending active;
- every non-requested column keeps its pre-command clock state;
- no direct partition-assignment helper runs.

The test records the real array-MMIO sequence for later design work but pins
only behavior already licensed by aie-rt and the device model. It must not
prematurely freeze incidental write ordering or undocumented controller state.

## Evidence-Gated Seam

The sole synthetic edge remains explicit:

```text
BAR4 X2I-tail publication
  -> [unknown hardware transition]
  -> explicit source-46 assertion
  -> modeled controller and Xtensa interrupt path
```

The tail write must not automatically assert source 46. This slice may proceed
through the explicit seam, but unmodified-driver acceptance remains blocked
until hardware evidence or an authoritative controller specification closes
that edge.

## TDD and Stop Conditions

Build one command at a time in pinned order. Each command first fails in the
new lifecycle test, then becomes green through the smallest source-derived
model correction.

Stop and return to design if progress would require:

- inventing an undocumented system-register value;
- synthesizing a firmware response;
- bypassing the firmware handler with an array helper;
- adding public ABI solely for the test;
- or introducing concurrent firmware/array scheduling before a real command
  demonstrates that need.

## Verification

Focused:

```bash
cargo test --lib m2c_pinned_initialization_create_context_programs_shared_array
```

Required repository gates after the proof is green:

```bash
cargo test --lib
cargo test -p xdna-emu-ffi
cargo fmt --all --check
git diff --check
```

The real-image test is gated by the pinned `XDNA_FIRMWARE` path. Halo is not
needed for these brief local builds.

## Next Boundary

After this proof, design the smallest coordinator that advances firmware,
array execution, and the engine's existing `HostMemory` without cloning state.
That coordinator is where PDI/configuration, core launch, array completion
tokens, firmware job completion, and retirement of the 8000-cycle mailbox
residual converge.
