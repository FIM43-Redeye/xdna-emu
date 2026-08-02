# Phoenix Context Repartition Proof Correction

**Date:** 2026-08-01

**Status:** **INVALIDATED PREMISE; SAME-CONTEXT CONTROL RESOLVES B CONFOUND**

## Verdict

The proposed normal repartition/reconnect proof does not exercise a real path
in the qualified Phoenix driver. Its lifecycle ledger was derived from the
legacy `src/driver/amdxdna` implementation. The qualified trace module was
built from the primary `drivers/accel/amdxdna` implementation at exact commit
`216cefececd74effcd7a88350c71b99f5ef9a215`.

That primary driver keeps A alive at physical column 1 and allocates the
two-column B beside it at physical columns 2-3. The one bounded physical run
confirmed that lifecycle: A1 completed, B completed, and there was no destroy
or reconnect between them. A's second submission was accepted by its original
mailbox but received no interrupt, worker dispatch, response, or head advance;
the normal 2,000 ms TDR path recovered A's context. No retry was made.

The missing same-context `A1 -> A2` control has now been run once on the same
pinned tuple, without constructing B. A1 completed; A2 published through the
same context mailbox and received no response before the normal TDR path
recovered that context. B is therefore not required for the observed
nonresponse, and the earlier A-B-A run is not evidence of spatial
multi-context interference. The physical trace still cannot reveal the AIE
core's internal PC, so the finite program's modeled teardown remains a causal
explanation rather than a directly observed silicon state.

## Source Correction

The qualification record explicitly excludes the dirty legacy tree from the
candidate build. The invalidated design and transition ledger nevertheless
cited that tree's runqueue implementation.

At the exact primary-driver commit:

- `drivers/accel/amdxdna/npu1_regs.c` sets `first_col=1`,
  `col_align=COL_ALIGN_NONE`, and `hwctx_limit=6`;
- `aie2_hwctx_col_list()` in `aie2_ctx.c` therefore offers every valid start
  column for a requested width;
- `get_free_partition()` in `aie2_solver.c` selects the first candidate whose
  column range does not intersect the occupied bitmap;
- `allocate_partition()` shares only an existing partition with exactly the
  same start and width when no free partition exists;
- `aie2_xrs_load()` immediately applies the selected start/width and calls
  `CREATE_CONTEXT`; and
- only resource release calls `aie2_xrs_unload()` and `DESTROY_CONTEXT`.

With a five-column Phoenix envelope, the isolated allocation is forced:

| Context | Requested width | Candidate starts | Occupied before allocation | Selected columns |
|---|---:|---|---|---|
| A | 1 | 1, 2, 3, 4 | none | 1 |
| B | 2 | 1, 2, 3 | 1 | 2-3 |

There is no ordinary repartition, relocation, disconnect, or reconnect in
this allocation path. The primary driver's explicit stop/restart and
create/map/config replay instead belong to TDR and suspend/resume paths.

## Frozen Control-Stream Audit

The existing `NpuInstructionStream` format and the repository's instruction
walker decode the pinned files without ambiguity:

| Fixture | SHA-256 | Operations | Encoded logical columns |
|---|---|---:|---|
| A `add_one_using_dma/chess/insts.bin` | `ee49b0a66c53d3952604460fe83fab879f38f1dad6cb70a994fc4422aa285896` | 8 | column 0 only; TCT column 0 |
| B `device_width/chess/insts.bin` | `f6b358372f584f0f0c220ae3dcc83066ae8922d9a15617ca84f3472d4a787941` | 10 | column 1 only; TCT column 1 |

The streams are in logical column space. The bytes alone do not prove where
the signed firmware ultimately directs B's column-1 transactions. If the
context start is applied, B targets physical column 3 and is disjoint from A;
if it is not applied or the active translation is stale, B targets physical
column 1 and can disturb A. Distinguishing those cases is now the first useful
offline/KVM boundary; it must not be replaced with an assumption.

## Physical Observation

Campaign root:

```text
build/experiments/npu1-firmware-evidence/physical-context-repartition-20260801-01/
```

Pinned tuple:

- qualified response-trace module SHA-256
  `723e557ac72d74e97822386cc8d544598380394a1d09c03de708738b41ad2371`;
- signed Phoenix firmware SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`;
- XRT packages `xrt-base=2.26.0`, `xrt-npu=2.26.0`, and
  `xrt_plugin-amdxdna=2.26`; and
- normal `tdr_timeout_ms=2000`, `force_cmdlist=Y`, and D0 pin.

Observed order:

| Step | Driver/firmware observation | Userspace oracle |
|---:|---|---|
| 1 | A `CREATE_CONTEXT(start=1,num=1,unused=0)` -> firmware ID 5 / mailbox 136 -> MAP -> CONFIG -> `CHAIN_EXEC_NPU [0,0,0]` | `PHOENIX_REPARTITION_A1_PASS` |
| 2 | B `CREATE_CONTEXT(start=2,num=2,unused=0)` -> firmware ID 4 / mailbox 135 -> MAP -> CONFIG -> `CHAIN_EXEC_NPU [0,0,0]` | `PHOENIX_REPARTITION_B_PASS` |
| 3 | A2 posts request `0x1d000002` to the original ID 5 / mailbox 136; no response-side lifecycle follows | `A did not complete` |
| 4 | TDR destroys A, recreates the same geometry and ID/channel, then replays MAP and CONFIG successfully | producer exits 1 |
| 5 | Stack unwind destroys B and A successfully | device remains responsive |

There was no `MGMT_ERT_BUSY`, no destroy/recreate between A and B, and no
mailbox rejection of A2. The failure boundary is after A2's request tail is
published and before any firmware response becomes host-visible.

Sealed evidence hashes:

| File | SHA-256 |
|---|---|
| `raw/trace.log` | `f8f1d8c1cc1f5fc399502b70913bddda422670d9b93d5c3c826e2011cb6348a0` |
| `raw/dmesg-delta.log` | `3daeacd75f93ff6597cce89dbbdff9a5239c3c76df1f64c972eec470b52235b1` |
| `raw/stdout.log` | `b122d795e2269db7ea7f6a5a15a035a49ee919e85b58a190b7d450fc0eaf5158` |
| `restoration.log` | `0dce9ea0c0cc47906f147f412b42c3f8c17083e009339b957b36bb08c8d6c6a2` |

## Same-Context Physical Control

Campaign root:

```text
build/experiments/npu1-firmware-evidence/physical-same-context-repeat-20260802-01/
```

The producer at commit `a642180be745b7aee63f7a327279773ff4b1817c`
constructed one A `Workload`, one `xrt::hw_context`, and no B object. The run
used the same qualified response-trace module, signed firmware, XRT 2.26
packages, Chess A fixture, 2,000 ms TDR, `force_cmdlist=Y`, and D0 pin as the
preceding campaign. No retry was made.

| Step | Driver/firmware observation | Userspace oracle |
|---:|---|---|
| 1 | `CREATE_CONTEXT` -> firmware ID 5 / mailbox 136 -> MAP -> CONFIG | one live A context |
| 2 | A1 posts execute request `0x1d000001`; IRQ, worker, response, head advance, and fence completion follow | `PHOENIX_CONTEXT_REPEAT_A1_PASS` |
| 3 | A2 posts execute request `0x1d000002` through the same mailbox; no IRQ, worker, response, or head advance follows | `A did not complete` |
| 4 | After 4.072 seconds, TDR signals the fence, destroys ID 5, recreates the same context, and replays MAP and CONFIG | producer exits 1 |
| 5 | Stack unwind destroys the recovered context successfully | device remains responsive |

The private trace instance retained all 60 of 60 entries. Its mechanical
ledger contains two execute tails but one execute response; every create, map,
config, and destroy tail has a response. No B construction marker, second
userspace context, mailbox rejection, or IOMMU fault appears.

Loading the qualified module also failed one order-10 allocation for its
optional firmware-log buffer. The driver reported that logging degradation,
then initialized and produced the complete command/response trace above. It is
recorded as an incidental infrastructure event, not a semantic result and not
a reason to retry the one-shot control.

The transaction wrapper restored the normal module and device but returned 97
because Bash's redirected `$(< file)` substitutions yielded empty verifier
strings. The executed script is preserved byte-for-byte; a corrected copy uses
`cat`. Independent live checks confirmed the normal module hash and
`srcversion`, PCI binding, device node, no holders, and `power/control=on`.
`SHA256SUMS` and `post-run-attestation.txt` seal the correction and receipts.

## Signed-Firmware In-Process Reproduction

The approved same-client guard now reproduces the physical ordering against
the unmodified signed image and one shared `DeviceState`:

1. A receives firmware ID 5 at physical column 1, configures, and completes.
2. B receives firmware ID 4 at physical columns 2-3, configures, and completes.
3. B's logical-column-1 writes land at physical column 3. Its completion uses
   the signed firmware's lane 2: source 78 and aperture `0xbd000000`.
4. A2 publishes through its original channel and reaches the finite proof
   bound without a response or X2I-head advance, matching the physical
   externally observable boundary.

The completion correction is not an inferred routing convenience. Live signed
firmware state assigns physical columns 1-4 to sources 76-79 and apertures
`0xbc000000`, `0xbc800000`, `0xbd000000`, and `0xbd800000`; after B's creation,
the lane-2 owner/selector is firmware context 4. The previous one-lane model
delivered B's token through source 76 to A's firmware object, preventing B from
completing.

The in-process A2 trace rules out the earlier application-channel
pre-consumption hypothesis in the model. Source 37 is acknowledged, the
management DMA stages the command, and the live application path reaches the
command interpreter already captured in
[`2026-05-22-chain-exec-npu-silent-drop-captured.md`](2026-05-22-chain-exec-npu-silent-drop-captured.md).
It then programs A's physical-column-1 shim input and output DMAs. The input
DMA completes a second transfer; the output DMA remains starved and produces
no second task-completion token.

The frozen toolchain artifacts explain why that modeled output cannot appear:

- `add_one_using_dma/aie.mlir` contains a finite outer loop that consumes its
  eight object-FIFO chunks and then executes `aie.end`;
- the Chess map places the core body at `0x00e0..0x032f`, `_fini` at
  `0x0370..0x041f`, and `__cxa_finalize` at `0x0420..0x04ef`;
- after A1 responds, the modeled core is still in teardown at PC `0x0480`;
- B gates column 1 without changing that PC; and
- A2 ungates the column, after which the core finishes teardown and reaches
  the `DONE` instruction at PC `0x00bc` without executing the kernel body a
  second time.

This is a complete causal explanation for the in-process nonresponse and makes
the guard unsuitable as proof of spatial state loss. The physical same-context
control now reproduces the same externally visible nonresponse without B, but
cannot observe these core states; the internal finite-program explanation
therefore remains model-derived rather than directly measured on silicon.

### Core-reset fidelity correction exposed by the audit

Reapplying the A PDI as a counterfactual exposed a separate emulator bug. The
CDO asserts `CORE_CONTROL.RESET`, sets `ENABLE` while RESET remains asserted,
and the firmware-launch seam later deasserts RESET. aie-rt exposes reset,
unreset, and enable as distinct operations. `DeviceState` previously published
only the derived runnable boolean, so a reset/unreset sequence drained as an
ordinary disable/enable pair and left the interpreter PC, registers, and
in-flight executor state intact.

Core-control publication now retains the ordered RESET transition. The engine
resets the interpreter, executor pipeline, architectural/timing context, and
per-core bookkeeping before applying the following enable. The focused
`core_control_masked_reset_restarts_engine_context_before_release` test reproduces the
actual CDO-to-firmware sequence. This correction does not make an already-used
DMA/lock graph reentrant and is not used to explain the original A2 run, which
contains no second CONFIG.

## Restoration

The original system module was restored through normal `modprobe` resolution:

- path `/lib/modules/7.1.5-custom+/kernel/drivers/accel/amdxdna/amdxdna.ko`;
- SHA-256
  `9b403eb8d34f0a66f385e6918bba1ebf86da5b527393280047588196b2d16297`;
- live `srcversion=77910A99EDBD0B6C78C8053`;
- `/dev/accel/accel0` present; and
- PCI `power/control=on` restored.

## Licensed Conclusions and Remaining Unknowns

The two physical runs and the signed-firmware reproduction license these
conclusions:

1. This producer does not trigger normal repartition/reconnect in the pinned
   primary driver.
2. Two disjoint driver contexts can be created and each can complete its first
   workload.
3. Returning to A immediately after B can stall after mailbox publication and
   trigger recoverable TDR on this exact tuple.
4. The current KVM runner's repartition-specific lifecycle assertions are
   invalid and must not be run or treated as an oracle.
5. In the in-process signed-firmware model, B's transactions are rebased to
   physical columns 2-3 and A2 reaches application dispatch before stalling on
   a missing second array completion.
6. The finite A fixture confounds the same-client guard: its modeled A2
   nonresponse cannot establish spatial state loss.
7. The identical physical A fixture also stalls on A2 in one unchanged context
   without B. B is not necessary for the nonresponse, so the A-B-A result does
   not establish spatial multi-context interference.

This still does **not** establish the silicon's internal core state or prove
that program finality is its cause. It establishes the driver-reachable
external contract for this pinned finite fixture: a second execute can be
accepted without producing a second completion, and normal TDR recovers the
context. On-silicon physical placement of B's transactions also remains
unobserved. Reconnect characterization should use a real primary-driver path
such as suspend/resume or bounded TDR and remain a separate slice.
