# Phoenix Signed-Firmware Post-TDR Replay

**Date:** 2026-08-02
**Status:** Implemented and verified

## Goal

Determine whether the existing signed-firmware and shared-array model naturally
reproduces the physical post-TDR contract:

```text
A1 completes
  -> A2 publishes but receives no response
  -> DESTROY_CONTEXT
  -> CREATE_CONTEXT
  -> MAP_HOST_BUFFER
  -> CONFIG_CU
  -> A3 completes through the replacement firmware context
```

The physical control established that the original public XRT workload and
hardware context complete A3 after the primary driver's replay. This slice
replays the same firmware-visible commands in process to locate the next model
boundary. It does not emulate the driver's TDR algorithm.

## Boundary

Extend the existing configured-CU signed-firmware harness in
`src/firmware/boot_tests/guards.rs` with one post-replay case. Reuse its pinned
Chess `add_one_using_dma` PDI, instruction stream, host heap, data BOs,
`FirmwareProcessor`, `InterpreterEngine`, management channel, and mailbox
helpers.

Do not use or modify the XRT plugin's legacy per-submit reset/PDI replay. That
path resets more often than physical hardware and cannot serve as evidence for
this contract. Do not call `reset_for_new_context`, apply a CDO directly,
manually reset or release a core, sleep, poll wall time, synthesize a response,
or introduce a recovery abstraction.

The characterization guard is the entire implementation unless it exposes a
genuine missing mechanism. A passing guard licenses documentation only; it
does not justify production-code churn.

## Exact Sequence

1. Boot the unmodified pinned `1502_00` firmware against one
   `InterpreterEngine::new_npu1()` and initialize the real management mailbox.
2. CREATE a one-column context at physical column 1, MAP the existing 64 MiB
   host heap, and CONFIGURE the frozen Chess PDI through the context channel.
3. Submit A1 with `CHAIN_EXEC_NPU`; require response `[0, 0, 0]` and ordered
   output `2..=65`.
4. Refill the same input and output BOs, submit A2 through the same context, and
   require the already-characterized deterministic nonresponse: the request is
   published, no I2X response appears, and the runtime pump reaches its finite
   array-idle or no-progress boundary.
5. While A2 remains unresolved, send management `DESTROY_CONTEXT` for its
   firmware context ID and require success.
6. CREATE the same one-column partition again and require deterministic reuse
   of the released firmware context slot. Use the newly returned context
   channel rather than retaining stale queue ownership.
7. MAP the same host heap and CONFIGURE the same unchanged PDI through that new
   context channel. Require successful responses and constrain array writes to
   physical column 1.
8. Refill the same BOs, submit A3 with the unchanged command body, and require
   response `[0, 0, 0]`, ordered output `2..=65`, and no unconsumed shim S2MM0
   completion token.
9. DESTROY the recovered context and require success.

Every transition is driven by the same management or context mailbox command
the pinned primary driver emits. Host buffer addresses, PDI bytes, instruction
bytes, and data BO addresses remain unchanged across replay.

## Interpretation

- **Full sequence passes:** the present signed-firmware/array model already
  reproduces the externally visible post-replay recovery contract. Record that
  result and make no production change. Individual reset, zeroization, PDI,
  DMA, lock, and core-state contributions remain a later causal slice.
- **DESTROY fails:** the first missing boundary is cancellation or teardown of
  an unresolved execution.
- **CREATE, MAP, or CONFIG fails:** the first missing boundary is context-slot
  reclamation or replay lifecycle.
- **A3 publishes but receives no response:** replay reaches firmware, but the
  array is not restored to a runnable state.
- **A3 responds with wrong output:** command recovery works, but memory or data
  state differs.

Stop at the first failing boundary. Do not compensate with a synthetic reset or
skip a required command. The full-sequence result alone does not identify which
individual replay operation is necessary or sufficient.

## Test Discipline

Add one `ConfiguredCuEnvelope` case and one named guard, reusing the existing
configured-CU harness. Run it with explicit paths so the real-image test cannot
silently skip:

```bash
env XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
    MLIR_AIE_PATH=/home/triple/npu-work/mlir-aie \
    nice -n 19 cargo test --lib m2c_post_tdr_replay_restores_execution \
    -- --nocapture
```

This is a characterization test, so an immediate GREEN is meaningful and must
not be forced RED by sabotaging correct behavior. If it fails naturally, retain
the failing guard and use systematic debugging before proposing the smallest
toolchain- or hardware-derived correction.

After the targeted real-image guard, run `nice -n 19 cargo test --lib`, inspect
the exact diff, update the context finding and host/firmware fidelity ledger,
and commit. No KVM, physical NPU, Halo, plugin correction, or causal-ablation
matrix belongs in this slice.

## Result

The characterization failed naturally at two derived reset boundaries before
passing the complete sequence:

1. Signed-firmware teardown emitted the aie-rt NPI protected-column shim-reset
   sequence at Phoenix system base `0xac000000`. The former system stub ignored
   it, leaving A2's shim S2MM channel active. The modeled PCSR mask/control and
   protected-column registers now apply that reset to the selected shim.
2. After the shim reset was honored, two completed link-pipeline words (`34`,
   `35`) remained queued for reset compute tile `(1,2)`. They became A3's first
   two inputs, producing the exact wrong prefix `35,36`. Column reset now drops
   ingress words targeting its reset non-shim tiles; shim reset does the same
   for its selected shim, without dropping unrelated-column traffic.

With those mechanisms modeled, `m2c_post_tdr_replay_restores_execution` passes
the exact A1 -> unresolved A2 -> DESTROY -> CREATE -> MAP -> CONFIG -> A3 ->
DESTROY sequence against the unmodified signed image. A3 returns `[0, 0, 0]`,
writes ordered output `2..=65`, and leaves no unconsumed shim S2MM0 token.
