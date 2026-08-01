# Phoenix Context Repartition Transition -- Source Derivation

**Date:** 2026-08-01

**Status:** DERIVED from pinned open XRT and NPU1 driver source; physical and
KVM observations pending

## Question

What exact driver-reachable lifecycle should a one-column Phoenix context
undergo when a live userspace client introduces a two-column context?

This derivation supplies the assertions for the approved positive
repartition/reconnect proof. It does not claim firmware IDs, timing, or
physical behavior that has not yet been observed.

## Authoritative Inputs

- XRT derives `m_col_cnt` from `aie_partition.ncol`, then creates a DRM context
  with `num_tiles = m_col_cnt * core_rows`:
  `../xdna-driver/src/shim/hwctx.cpp`.
- The KMQ context constructor retains each xclbin's PDI/CU configuration in
  the userspace context:
  `../xdna-driver/src/shim/kmq/hwctx.cpp`.
- NPU1 defines `first_col=1`, `ctx_limit=6`, `hwctx_limit=6`, and
  `temporal_only=0`:
  `../xdna-driver/src/driver/amdxdna/npu1_regs.c`.
- The driver converts `num_tiles` back to `orig_num_col`, owns the virtual
  context and CU table, and adds it to the runqueue:
  `../xdna-driver/src/driver/amdxdna/aie2_ctx.c`.
- Partition selection, rebuild, disconnect, and reconnect are in
  `../xdna-driver/src/driver/amdxdna/aie2_ctx_runqueue.c`.
- Firmware-context creation, host-buffer replay, CU replay, and destruction
  are in `../xdna-driver/src/driver/amdxdna/aie2_hwctx.c` and
  `../xdna-driver/src/driver/amdxdna/aie2_message.c`.

The selected fixtures carry real widths:

- Chess `add_one_using_dma`: one logical column;
- `device_width`: `AIEDevice.npu1_2col`, two logical columns.

## Initial Runqueue Geometry

Phoenix exposes four ordinary application columns `[1,4]`. With no wider
reservation, `rq_part_reinit` chooses width 1 and creates:

| Partition | Physical columns | `max_hwctx` |
|---|---:|---:|
| P0 | `[1,1]` | 1 |
| P1 | `[2,2]` | 1 |
| P2 | `[3,3]` | 1 |
| P3 | `[4,4]` | 1 |

`max_hwctx` is integer division `6 / 4`, so this layout connects at most four
contexts concurrently even though firmware and driver both expose six total
context slots. Extra virtual contexts wait in partition runqueues.

Default public-XRT QoS contains no recognized priority value, so
`qos_to_rq_prio` normalizes it to the normal-priority queue.

## Exact Positive Transition

Assume no other contexts and the default `wait_update_parts=true`.

| Step | Userspace action | Driver state and physical assignment | Firmware-visible traffic |
|---:|---|---|---|
| 1 | Construct A | A is `DISCONNECTED`; width-1 reservation added | None |
| 2 | Submit A | A becomes `DISPATCHED`, selects least-used P0, then becomes `CONNECTED` at `[1,1]` | `CREATE_CONTEXT(start=1,num=1,unused=0)` -> `MAP_HOST_BUFFER` -> `CONFIG_CU` -> execute |
| 3 | A completes | A remains `CONNECTED`; `submitted == completed` | Normal execute completion |
| 4 | Construct B | Width-2 reservation makes current width 1 stale; runqueue pauses. A becomes `DISCONNECTING`, is drained, then becomes `DISCONNECTED` | `DESTROY_CONTEXT(A)` |
| 5 | Rebuild | With no connected contexts, partitions become P0=`[1,2]`, P1=`[3,4]`; each has `max_hwctx = 6 / 2 = 3`. B construction returns only after this because `wait_update_parts=true` | None |
| 6 | Submit B | B selects empty P0 and becomes `CONNECTED` at `[1,2]` | `CREATE_CONTEXT(start=1,num=2,unused=0)` -> `MAP_HOST_BUFFER` -> `CONFIG_CU` -> execute |
| 7 | B completes | B remains `CONNECTED` at `[1,2]` | Normal execute completion |
| 8 | Submit A again | A selects less-used P1 and becomes `CONNECTED` at `[3,4]`; its requested width remains 1, so the second assigned column is explicitly unused | `CREATE_CONTEXT(start=3,num=2,unused=1)` -> `MAP_HOST_BUFFER` -> replayed `CONFIG_CU` -> execute |
| 9 | A completes | A remains `CONNECTED` at `[3,4]` | Normal execute completion |

The second A execution uses the original userspace context handle and its
driver-retained heap/CU state, but it necessarily uses a newly created
firmware context and mailbox channel. The emulator must learn the new mapping
only from the replayed firmware commands.

## Bounded Teardown

The producer will explicitly destroy A before B:

1. Deleting A sends `DESTROY_CONTEXT(A-reconnected)` and removes its width-1
   reservation. B's width-2 reservation keeps the two-column layout intact.
2. Deleting B sends `DESTROY_CONTEXT(B)`. With no width reservations left,
   the empty runqueue returns to its default one-column geometry without
   another firmware context to destroy.

Together with A's destruction during expansion, the complete positive
lifecycle therefore contains exactly three successful firmware-context
destroy operations. It also contains exactly three create/map/config groups:
initial A, B, and reconnected A.

## Assertions Licensed Before Observation

The pinned source licenses assertions for:

- userspace A retaining its handle across the firmware destroy/recreate;
- the state sequence `DISCONNECTED -> DISPATCHED -> CONNECTED ->
  DISCONNECTING -> DISCONNECTED -> DISPATCHED -> CONNECTED` for A;
- initial A assignment `[1,1]`;
- rebuilt assignments B=`[1,2]`, A=`[3,4]` for this isolated normal-priority
  ordering;
- create request widths and unused-column counts `(1,1,0)`, `(1,2,0)`, and
  `(3,2,1)`;
- replay of `MAP_HOST_BUFFER` and `CONFIG_CU` on every firmware-context
  creation; and
- three create/map/config groups and three destroy operations.

## Still Requires Observation

Do not assert before the physical or KVM run establishes it:

- which numeric firmware context IDs are allocated or reused;
- exact mailbox channel/MSI-X indices after each reconnect;
- host timing or firmware/array cycle timing;
- response ordering beyond each mailbox's protocol guarantees; or
- behavior under a pending command, timeout, TDR, suspend, or fault.

This clean transition is the control for the later faulted lifecycle. It does
not explain the historically observed physical `MGMT_ERT_BUSY`.
