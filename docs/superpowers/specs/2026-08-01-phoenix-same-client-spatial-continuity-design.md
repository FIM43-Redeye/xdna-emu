# Phoenix Same-Client Spatial Continuity -- Design

**Date:** 2026-08-01

**Status:** Approved; implementation pending

## Purpose

Reproduce the smallest faithful in-process form of the physical Phoenix
`A1 -> B -> A2` observation before changing the KVM boundary or emulator
behavior. The proof asks two separate questions:

1. Does signed Phoenix firmware place each context's logical AIE transactions
   inside the physical columns assigned by the qualified open driver?
2. After B configures and executes beside A, what bounded outcome does A's
   original firmware channel produce on its second execution?

The physical observation is recorded in
[`../findings/2026-08-01-phoenix-context-repartition-proof-correction.md`](../findings/2026-08-01-phoenix-context-repartition-proof-correction.md).
It proves that A1 and B completed, then A2 stalled after its X2I tail was
published. It does not identify the responsible component, and a single run
does not license encoding the timeout as permanent hardware behavior.

## Corrected Driver Contract

This is a **same-client** proof. The pinned producer constructs one
`xrt::device` and creates both `xrt::hw_context` objects through it. At exact
qualified driver commit `216cefececd74effcd7a88350c71b99f5ef9a215`:

- `amdxdna_drm_open()` creates one device-heap allocator per DRM client;
- `aie2_hwctx_init()` attaches every hardware context from that client to the
  same first heap object;
- each context sends that same heap DMA base and size through
  `MAP_HOST_BUFFER`; and
- BO device addresses are distinct allocations inside the client's common
  device-address heap.

The faithful topology is therefore:

```text
one DRM client / one PASID value
  one 64 MiB host heap backing
  one device-visible heap range beginning at 0x04000000
    context A: distinct BO offsets, physical column 1
    context B: distinct BO offsets, physical columns 2-3
```

A and B are not memory-security domains. Sharing the client heap is intended.
The boundary under test is separation of firmware channel/context state and
context-relative placement into disjoint physical AIE columns.

## Selected Proof Boundary

Add one fixture-backed guard in the existing signed-firmware boot test module.
It must reuse:

- `PinnedMgmtChannel` and `PinnedContextChannel` for genuine management and
  application mailbox traffic;
- `pump_runtime()` for bounded firmware/array interleaving;
- `Bus::arm_probe()` / `Bus::take_probe()` for physical MMIO evidence;
- the existing frozen PDI and instruction-stream loaders; and
- the engine's existing `HostMemory` and single shared `DeviceState`.

No new production abstraction, shadow context model, alternate executor, or
test-only firmware behavior is needed for the first observation.

The test uses the installed signed image selected by `XDNA_FIRMWARE` and the
frozen mlir-aie fixtures selected by `MLIR_AIE_PATH`, under the same environment
contract as the existing configured-CU guards.

## Fixtures and Addressing

Use the already audited Chess fixtures from the physical producer:

| Context | Fixture | Requested physical partition | Encoded application columns |
|---|---|---|---|
| A | `add_one_using_dma` | start 1, width 1 | logical column 0 |
| B | `device_width` | start 2, width 2 | logical column 1 |

Allocate one 64 MiB host heap and map its identical base and size into both
firmware contexts. Place each context's PDI, instruction stream, and chained
command data at non-overlapping offsets within that heap, preserving the
driver's device-address alignment rules. Keep their ordinary input and output
BO regions distinct.

This deliberately avoids the emulator's open cross-client/PASID-selection
problem. Two different host backings behind the same device address would
model separate DRM clients, not the physical producer, and the current
address-only resolver cannot faithfully select between them.

## Staged Execution

Run the proof in this order, with a fresh probe around each configuration and
execution stage:

1. Boot the pinned signed firmware against one NPU1 `DeviceState`.
2. Initialize the management channel.
3. Create A at start 1, width 1; map the shared heap; configure A.
4. Execute A1 and verify its complete output.
5. Create B at start 2, width 2; map the same shared heap; configure B.
6. Execute B and verify its complete output.
7. Submit A2 through A's original firmware ID, mailbox channel, command
   storage, and data buffers.
8. Pump only to an explicit response or the existing finite budget, then
   classify the outcome and stop.

Do not destroy, recreate, reconnect, or replay A between A1 and A2. Those
operations did not occur in the qualified physical path before its timeout.

## Evidence and Invariants

For every stage retain enough assertion context to identify:

- firmware context ID, requested start column, and width;
- shared heap host target and device-visible offset;
- physical columns reached by the relevant application array accesses;
- X2I/I2X head and tail progression;
- `RuntimePumpStop` and the final firmware boundary; and
- output-buffer contents.

The source- and hardware-backed invariants are:

1. Both live firmware mapping records resolve the common device heap to the
   same host backing. Multiple identical candidates are not ambiguous.
2. A and B receive distinct firmware IDs and mailbox channels.
3. A's application transactions remain in physical column 1.
4. B's application transactions remain inside physical columns 2-3; its known
   logical-column-1 operations land on physical column 3.
5. A1 and B complete with their fixture-defined outputs.
6. A2 reaches a bounded, explicitly classified result.

Invariant 6 does **not** initially require either a successful response or a
timeout. The first run is characterization. After comparison with the physical
record, a later reviewed change may pin a stronger expectation.

## Failure Classification and Stop Rules

Stop at the first failed layer and review it before changing production code:

| First failure | Licensed interpretation |
|---|---|
| Shared heap fails to resolve | Harness error or regression in the already-proven same-backing translation path |
| CONFIG reaches the wrong application column | Missing or stale context-relative AIE placement |
| A1 or B cannot complete after correct CONFIG placement | Execution or context-state isolation defect |
| A2 reaches the finite budget without a response | The in-process signed-firmware/array seam reproduced the physical symptom; inspect the retained state before proposing a fix |
| A2 completes correctly | The in-process firmware/array seam does not reproduce the physical failure; proceed later to the driver-visible KVM boundary |
| Firmware reaches an unknown operation, unresolved spin, or other unclassified boundary | Incomplete harness/model support; do not reinterpret it as an A2 timeout |

No result from this first proof authorizes forcing the emulator to return a
particular A2 outcome.

## Test-Driven Implementation Order

1. Add the single fixture-backed guard using existing helpers and shared heap
   machinery.
2. Run only that guard locally with `nice -n 19`, the pinned firmware, and the
   mlir-aie fixture path.
3. Preserve the first failed invariant and its probe output without changing
   production behavior.
4. Review the result with Maya.
5. Only then design the smallest source-derived correction, if one is needed.

If every source-backed invariant already passes, there is no production change
to make in this slice. The next designed step would be correction and cautious
use of the KVM proof instead.

## Evidence Retention

The routine unit test should not create persistent files. Its assertion and
diagnostic output must contain the bounded facts above. After the first run is
reviewed, append the licensed conclusion and exact command/result to the
existing correction finding or a narrowly named successor finding.

## Alternatives Rejected

1. **Distinct host backing for A and B.** This tests separate-client/PASID
   isolation and is not faithful to the one-device physical producer.
2. **Different device-visible heap ranges.** The canonical driver initializes
   every client's heap at the same device-memory base; choosing artificial
   ranges would avoid the ownership question rather than model it.
3. **KVM first.** The guest/XRT/driver/PCI layers would make a firmware/array
   placement defect harder to localize.
4. **Add KVM logging first.** Existing in-process physical-MMIO probes already
   answer the immediate placement question with less new code.

## Explicitly Deferred

- separate-client heaps with aliased device addresses;
- PASID/context ownership propagation into device-memory translation;
- correction or execution of the currently invalid
  `--run-context-repartition` KVM mode;
- another physical NPU run;
- TDR, suspend/resume, reconnect, preemption, and recovery semantics;
- timing equivalence; and
- older Phoenix firmware versions.

These remain part of the broader driver-reachable firmware-equivalence goal,
but none is required to interpret the first same-client spatial-continuity
result.
