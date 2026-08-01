# Phoenix Context Repartition and Reconnect -- Design

**Date:** 2026-08-01

**Status:** Approved boundary; pending written review

## Purpose

Close the next driver-reachable part of Phoenix context semantics: an idle
application context must survive the driver's normal column repartitioning by
being destroyed as a firmware context, recreated, remapped, reconfigured, and
executed again without relying on stale firmware-context state.

This is a positive lifecycle proof. It does not inject a hang, force
`MGMT_ERT_BUSY`, or model the Linux runqueue inside the emulator.

## Derived Driver Contract

The pinned NPU1 driver defines six virtual contexts, six hardware contexts,
four application columns beginning at physical column 1, and dynamic spatial
partitioning. A userspace context and a firmware context are different
identities:

- `aie2_rq_add` creates a disconnected virtual context and reserves its
  requested width;
- increasing the maximum requested width schedules partition rebuilding;
- active idle contexts are marked disconnecting and stopped;
- `aie2_hwctx_stop` destroys their mailbox channel and sends
  `DESTROY_CONTEXT`;
- a later submission reconnects through `CREATE_CONTEXT`, recreates the
  mailbox channel, replays `MAP_HOST_BUFFER`, and replays `CONFIG_CU` from
  driver-owned state.

Authoritative sources are:

- `../xdna-driver/src/driver/amdxdna/npu1_regs.c`;
- `../xdna-driver/src/driver/amdxdna/aie2_ctx_runqueue.c`;
- `../xdna-driver/src/driver/amdxdna/aie2_hwctx.c`; and
- `../xdna-driver/src/driver/amdxdna/aie2_message.c`.

The emulator consumes only the commands and hardware accesses emitted by this
driver and the signed firmware. It does not reproduce the driver's scheduling
policy.

## Selected Stimulus

Use one small XRT producer that keeps two ordinary `xrt::hw_context` objects
alive:

1. **Context A:** the existing one-column Chess `add_one_using_dma` fixture.
2. Run A once and verify its ordered `2..=65` output.
3. **Context B:** the upstream two-column `device_width` fixture.
4. Construct B while A remains alive, forcing the pinned driver to rebuild
   its one-column partition layout for a two-column maximum width.
5. Run B and verify its upstream identity-output oracle.
6. Run A again through the same userspace context and buffers, with fresh
   inputs, and verify `2..=65` again.
7. Destroy both userspace contexts normally.

The producer uses public XRT only. It performs no raw DRM ioctl, mailbox write,
BAR access, context-ID selection, column assignment, or synthetic completion.
It lives with the existing vfio-user test tools rather than modifying
mlir-aie's upstream tests.

`device_width` is chosen because its `npu1_2col` metadata supplies the real
partition-width request and its kernel already has an externally observable
oracle. No synthetic xclbin or hand-edited metadata is permitted.

## Proof Layers

### 1. Source-derived transition ledger

Before adding behavior, record the exact driver state and mailbox transition
sequence for the stimulus above. The ledger must distinguish:

- stable userspace context handles;
- transient firmware context IDs and mailbox channels;
- requested width from assigned partition width;
- driver-retained heap and CU configuration; and
- the point at which partition rebuilding causes destruction and reconnect.

Only values derived from the pinned driver or observed in the run may become
assertions. In particular, firmware-ID reuse and physical-column choice must
not be guessed.

### 2. Physical NPU1 control

Run the producer once on the owned Phoenix NPU1 using XRT 2.26 and the existing
qualified response-trace candidate-module recipe recorded in
`2026-07-30-npu1-phase3a-executing-driver-qualification.md`. Do not promote the
currently installed module as evidence: its bytes are known, but its exact
source-to-bytes relationship is not qualified. Reuse the established bounded
load, capture, and restoration path.

This is ordinary known-good work with no fault injection or repetition
campaign. Retain the ordered mailbox lifecycle, output oracles, teardown
result, and explicit restoration proof.

Any timeout, TDR, firmware error, or teardown failure stops the slice; it is
not retried into a wedge.

### 3. Unmodified-driver emulator proof

Extend `scripts/phoenix-vfio-user-qemu.sh` by one explicit mode that reuses its
pinned guest kernel, open driver, XRT, firmware, vfio-user server, tuple checks,
bounded wait, and evidence retention. Carry only the new producer and its two
pinned fixture bundles into the guest.

The runner's stale XRT 2.23 package and library pins must first be replaced by
the installed, source-built XRT 2.26 tuple. Keep the existing open-driver
commit pin. Build its guest module from `git archive <pin>` so intentional
uncommitted changes in the sibling driver worktree are neither consumed nor
modified; worktree cleanliness is not provenance for an archive addressed by
commit.

The KVM run must exercise the same public-XRT sequence and match the physical
control's semantic invariants at the level visible to the driver and
userspace. Exact firmware IDs and column choices are asserted separately for
each pinned driver tuple rather than assumed equal across driver variants.

### 4. First-divergence correction

If the KVM run fails, locate the first firmware-visible divergence. Add one
focused signed-firmware or device test that is red for that missing hardware
behavior, then fix the shared hardware seam. Do not add a driver-specific
response, runqueue shadow, private context-state override, or test-only bypass.

If the existing model already passes, add no production code.

## Acceptance

The slice is complete when both physical NPU1 and KVM/vfio-user runs show:

- A succeeds before repartitioning;
- B's two-column request causes the normal driver destruction/reconnect path;
- B succeeds with its upstream output oracle;
- A succeeds again through its original userspace context;
- each reconnect receives fresh firmware-owned channel state;
- host-buffer and CU configuration are replayed by the driver, not preserved
  by an emulator shortcut;
- replayed mappings and configuration produce correct output after reconnect;
- all firmware responses and MSI-X completions are matched;
- final context destruction succeeds without `BUSY`, TDR, or leaked slots; and
- the exact platform and artifact tuples and ordered mailbox transitions are
  retained.

Local regression requires the focused tests, `cargo test --lib`, formatting,
shell syntax checks, and a clean worktree after commits.

## Test Order

1. Write the source-derived transition ledger.
2. Add a failing runner check for the unsupported lifecycle mode.
3. Build and pin the existing two-column fixture and the minimal XRT producer.
4. Run the physical control once.
5. Run the KVM proof and capture the first divergence, if any.
6. Use one focused red test before any production correction.
7. Re-run the KVM proof, then the local regression gate.

## Explicitly Deferred

- deliberately noncompleting jobs and the physical `MGMT_ERT_BUSY` trigger;
- TDR, cancellation, forced recovery, runtime suspend, and resume;
- PASID and hostile cross-context isolation;
- complete TCT-actor coverage;
- timing and clock-domain equivalence; and
- older Phoenix firmware images.

The positive reconnect path supplies the controlled baseline for the later
faulted lifecycle. Those later mechanisms are not folded into this slice.
