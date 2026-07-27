---
name: Phoenix pinned driver command and lifecycle audit
description: Exact read-only inventory of the Phoenix host-to-NPU surface and lifecycle at xdna-driver 216cefe, separating the primary staging driver, packaged legacy driver, and development-only paths.
type: finding
---

# Phoenix Pinned Command and Lifecycle Audit

Date: 2026-07-27

External primary firmware tuple, pinned by the approved goal design rather
than derived from the driver commit:

`/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin`,
SHA-256
`d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

Driver:
`216cefececd74effcd7a88350c71b99f5ef9a215`

Target: Phoenix / NPU1, PCI ID `1022:1502`, management protocol 5.8

## Verdict

The pinned primary driver can issue exactly 20 distinct Phoenix mailbox
opcodes. That count includes the two direct-execution opcodes selected by the
real `force_cmdlist=false` module-parameter path; the default path uses
`CHAIN_EXEC_NPU`.

The complete pinned host surface is wider than the firmware mailbox:

- three NPU1 PSP commands (`VALIDATE`, `START`, `RELEASE_TMR`);
- four NPU1 SMU commands (`POWER_ON`, `POWER_OFF`, `SET_MPNPUCLK_FREQ`,
  `SET_HCLK_FREQ`);
- firmware-alive and mailbox-descriptor publication in SRAM;
- management and per-context X2I/I2X rings, doorbells, MSI-X interrupts, and
  DMA-visible buffers; and
- the 20 mailbox opcodes below.

The repository also ships a second, out-of-tree driver as
`amdxdna_legacy.ko`. It adds no production Phoenix opcode to the primary
surface. It omits the primary production routes for `GET_TELEMETRY` and
`AIE_RW_ACCESS`, while adding debug/development-only probes. Those probes are
inventory, not part of the frozen primary acceptance gate.

This audit freezes stimuli and observable lifecycle. It does not make the
driver an oracle for NPU behavior, and it does not license emulating driver
bugs.

## Provenance and Completeness Rule

The sibling driver worktree was dirty during this audit. Every source fact
below was read from the exact commit object with `git show` or `git grep`;
none comes from the checked-out files.

The pinned repository explicitly ships two independent trees:

| Tree | Packaged module | Role |
|---|---|---|
| `drivers/accel/amdxdna/` | `amdxdna.ko` | primary upstream/staging driver |
| `src/driver/amdxdna/` | `amdxdna_legacy.ko` | compatibility/bring-up driver |

Evidence:
`README.md:184-202` and `CMake/native.cmake:7-11` at the pinned commit.

For the primary tree, the audit:

1. resolved PCI ID `0x1502` through `dev_npu1_info` to `aie2_ops`;
2. applied the additive NPU1 5.7/5.8 feature table;
3. enumerated every `MSG_OP_*`, `DECLARE_AIE_MSG`, direct opcode assignment,
   and mailbox send;
4. traced each constructor to a boot, PM, UAPI, scheduler, debug-BO, timeout,
   teardown, or recovery caller;
5. included non-default but supported driver configuration, notably
   `force_cmdlist=false`;
6. excluded constructors with no caller and paths stopped before transport by
   a missing NPU1 feature; and
7. audited PSP, SMU, mailbox transport, response, timeout, and teardown code
   separately.

The legacy tree was then checked for a production delta rather than treated as
another authority.

## Non-Mailbox Host Surface

### NPU1 apertures used by the lifecycle

All addresses below are NPU device addresses from
`drivers/accel/amdxdna/npu1_regs.c:16-47,105-138`.

| Function | Address or range | Host mechanism |
|---|---:|---|
| PSP waitmode | `0x03010034` | BAR0 read/poll |
| PSP interrupt | `0x03010090` | BAR0 write |
| SMU interrupt | `0x03010094` | BAR0 write |
| PSP command/status | `0x030100a0` | BAR0 read/write |
| PSP argument 0 / response | `0x030100a4` | BAR0 read/write |
| PSP argument 1 | `0x030100a8` | BAR0 write |
| SMU command | `0x030100ac` | BAR0 write |
| SMU response | `0x030100b0` | BAR0 read |
| SMU argument / output | `0x030100b4` | BAR0 read/write |
| PSP argument 2 | `0x030100bc` | BAR0 write |
| firmware-alive record | `0x030bf000` | BAR2 SRAM |
| management rings | firmware-published descriptor | BAR2 SRAM |
| mailbox head/tail/IRQ registers | firmware-published descriptor | BAR4 |

The concrete post-alive ring and doorbell envelope is separately pinned in
`2026-07-27-phoenix-post-alive-observability.md`.

### PSP command surface

`aie_psp.c:18-25,59-174` defines the protocol. NPU1 passes no certificate
firmware to the PSP configuration, so generic `VALIDATE_CERT` is not part of
this tuple.

| Command | Value | Reachable use | Completion contract |
|---|---:|---|---|
| `VALIDATE` | `1` | every hardware start | poll ready, write command and physical firmware buffer, notify, poll ready, require response 0 |
| `START` | `2` | after successful validation | argument 0 is `COPY_FW=1`; same ready/response protocol |
| `RELEASE_TMR` | `3` | hardware stop and startup unwind | same ready/response protocol; failure is logged during stop |

Ready and waitmode polling use a 20 ms interval and one-second timeout.
`SUSPEND` mailbox success is followed by a PSP waitmode-bit poll; it is not a
PSP command.

### SMU command surface

`aie_smu.c:15-139` defines the generic commands. NPU1's `set_dpm` callback
uses direct clock commands, so generic soft/hard-DPM commands `0x7` and `0x8`
are not NPU1-reachable.

| Command | Value | Reachable use |
|---|---:|---|
| `POWER_ON` | `0x3` | hardware start, after a defensive power-off |
| `POWER_OFF` | `0x4` | first step of hardware start and final step of stop |
| `SET_MPNPUCLK_FREQ` | `0x5` | initial PM setup, power-mode changes, resolver DPM, stop |
| `SET_HCLK_FREQ` | `0x6` | immediately after each MP-NPU clock command |

Each command clears response, writes argument and command, pulses the interrupt
register, polls for a nonzero response for up to one second, and accepts only
response `1`. Frequency commands also return the accepted frequency through
the output register.

## Primary Mailbox Surface

The NPU1 feature table accepts protocol 5.7 and, at protocol 5.8, adds only
`AIE2_NPU_COMMAND` and the locally verified `AIE2_RW_ACCESS` feature:
`drivers/accel/amdxdna/npu1_regs.c:68-81`.

“Management” rows use the shared synchronous management channel. “Context”
rows use the per-context completion queue and asynchronous callbacks.
Request/response sizes exclude the 16-byte mailbox header.

| Opcode | Command | Route | Request -> response | Pinned trigger or gate |
|---:|---|---|---:|---|
| `0x002` | `CREATE_CONTEXT` | management | 28 -> 76 | resource-resolver load and context restart; response supplies firmware context ID and CQ/MSI descriptors |
| `0x003` | `DESTROY_CONTEXT` | management | 4 -> 4 | context unload, close, timeout recovery, suspend, and create cleanup |
| `0x004` | `GET_TELEMETRY` | management | 16 -> 16 | production telemetry UAPI for all declared telemetry types |
| `0x007` | `SYNC_BO` | context | 24 -> 4 | debug-BO device-to-host synchronization |
| `0x00c` | `EXECUTE_BUFFER_CF` | context | 80 -> 4 | direct `ERT_START_CU` when `force_cmdlist=false` |
| `0x00d` | `QUERY_COL_STATUS` | management | 20 -> 8 | production AIE-status query |
| `0x00e` | `QUERY_AIE_TILE_INFO` | management | 4 -> 48 | mandatory startup query; result cached |
| `0x00f` | `QUERY_AIE_VERSION` | management | 4 -> 8 | mandatory startup query; result cached |
| `0x010` | `EXEC_DPU` | context | 160 -> 4 | direct `ERT_START_NPU` when `force_cmdlist=false` |
| `0x011` | `CONFIG_CU` | context | 132 -> 4 | CU configuration and context restart |
| `0x014` | `CONFIG_DEBUG_BO` | context | 20 -> 4 | production debug-BO attach/detach |
| `0x018` | `CHAIN_EXEC_NPU` | context | 24 -> 12 | explicit command chain, or any single CU/NPU submission with default `force_cmdlist=true`; requires protocol 5.8 |
| `0x101` | `SUSPEND` | management | 4 -> 4 | firmware reset sequence, system/runtime stop, and startup unwind |
| `0x102` | `RESUME` | management | 4 -> 4 | firmware reset sequence after suspend |
| `0x103` | `ASSIGN_MGMT_PASID` | management | 4 -> 4 | mandatory startup initialization with PASID 0 |
| `0x106` | `MAP_HOST_BUFFER` | management | 20 -> 4 | initial 64 MiB context heap mapping and restart |
| `0x108` | `GET_FIRMWARE_VERSION` | management | 4 -> 20 | mandatory startup query; result cached |
| `0x10a` | `SET_RUNTIME_CONFIG` | management | 12 -> 4 | NPU1 config types 2=1 and 4=1 at initialization, and type 1 for clock gating |
| `0x10c` | `REGISTER_ASYNC_EVENT_MSG` | management/async | 12 -> 8 | one initial registration for each index below `total_col`; conditionally re-registered after a successfully parsed actionable event |
| `0x203` | `AIE_RW_ACCESS` | management | 24 -> 8 | production register/memory read/write routes; requires the pinned local 5.8 feature addition |

Primary source evidence:

- request construction and transport:
  `drivers/accel/amdxdna/aie2_message.c`;
- context scheduler and lifecycle callers:
  `drivers/accel/amdxdna/aie2_ctx.c`;
- UAPI and hardware lifecycle callers:
  `drivers/accel/amdxdna/aie2_pci.c`;
- wire structures:
  `drivers/accel/amdxdna/aie2_msg_priv.h`; and
- AIE register/memory ownership guards:
  `drivers/accel/amdxdna/aie.c`.

### Defined but outside the pinned primary surface

| Opcode or family | Classification |
|---|---|
| `0x012 CHAIN_EXEC_BUFFER_CF`, `0x013 CHAIN_EXEC_DPU` | selected only when `AIE2_NPU_COMMAND` is absent; not selected by the pinned 5.8 firmware, but candidates for the later older-firmware manifest |
| `0x10b GET_RUNTIME_CONFIG` | constructor exists, but no in-tree caller or installed callback |
| `0x104 INVOKE_SELF_TEST`, `0x301 GET_PROTOCOL_VERSION` | enum/structure declarations only in the primary tree; no sender |
| `0x115 ADD_HOST_BUFFER` | NPU1 lacks the feature; heap expansion is rejected before transport. The initial mapping helper can change a second chunk to this opcode, but NPU1's heap maximum and chunk size are both 64 MiB, so that second chunk is unreachable |
| `0x113 UPDATE_PROPERTY`, `0x114 GET_APP_HEALTH` | callers exist, but NPU1 feature checks stop them before transport |
| `0x10f`, `0x110`, `0x111` | firmware-trace family; no NPU1 feature or installed callbacks |
| `0x116`, `0x117`, `0x119`, `0x11c` | firmware log, device revision, coredump, and clock calibration; feature-gated off for NPU1 |
| preempt/ELF command-slot forms | rejected because NPU1 lacks `AIE2_PREEMPT` |
| all `AIE4_MSG_OP_*` values | different device family; NPU1 binds `aie2_ops` |

## Transport, Error, and Teardown Contract

The driver exposes several distinct observable failure classes:

| Condition | Driver-visible consequence |
|---|---|
| management response arrives with exact expected size and status 0 | synchronous success |
| management response status is nonzero | `-EINVAL`; channel remains present |
| management response body has the wrong size | callback returns `-EINVAL`; RX worker marks the channel bad; future sends return `-EPIPE` |
| no management response within 5000 ms | `-ETIME`; driver stops and destroys the management channel |
| invalid I2X tail, tombstone placement, size, message ID, or unmatched ID | RX worker marks channel bad; a waiter not completed by the malformed packet can subsequently time out |
| X2I ring remains full | send polls for only 100 us and returns the poll error |
| channel teardown with requests outstanding | every outstanding callback is invoked with `(NULL, 0)` before channel storage is destroyed |

The nominal `TX_TIMEOUT=2000 ms` passed to `xdna_mailbox_send_msg()` is not
used. The actual ring-space timeout is the hard-coded 100 us poll in
`amdxdna_mailbox.c:224-229`.

Context responses have opcode-specific handling:

- direct execution requires one status word;
- chain execution requires exactly `{status, fail_cmd_idx,
  fail_cmd_status}`;
- sync/debug-BO commands store the returned status for the waiting caller;
- `CONFIG_CU`'s completion callback only drops the PM reference and does not
  inspect firmware status; and
- the async callback records the response fields without validating them; the
  worker parses the DMA buffer and only stores the error and re-registers when
  parsing succeeds, the count is in range, and backtracking produces a
  nonzero column bitmap. Re-registration failure only warns.

The wire status namespace is defined in
`drivers/accel/amdxdna/aie2_msg_priv.h:51-89`: AIE errors
`0x01000001+`, management-ERT errors `0x02000001+`, app-ERT errors
`0x03000001+`, and RTOS/input errors `0x04000001+`. Those names define
possible response values, not proof that every status is reachable from every
opcode.

## Ordered Lifecycle

### Probe and boot

The primary NPU1 order is:

```text
PCI enable/map BARs/allocate MSI-X
  -> SMU POWER_OFF
  -> SMU POWER_ON
  -> PSP VALIDATE
  -> PSP START(COPY_FW)
  -> poll firmware-alive SRAM
  -> consume firmware-published management descriptor
  -> start management MSI channel
  -> SET_RUNTIME_CONFIG(2,1)
  -> SET_RUNTIME_CONFIG(4,1)
  -> ASSIGN_MGMT_PASID(0)
  -> SUSPEND
  -> poll PSP waitmode
  -> RESUME
  -> set MP-NPU and H clocks
  -> SET_RUNTIME_CONFIG(1,1)
  -> GET_FIRMWARE_VERSION
  -> QUERY_AIE_VERSION
  -> QUERY_AIE_TILE_INFO
  -> REGISTER_ASYNC_EVENT_MSG once per column
```

`UPDATE_PROPERTY` and clock calibration are invoked by shared initialization
code but are feature-gated into no-message success for NPU1.

Any mandatory initialization failure unwinds the channel, PSP, SMU, and PCI
layers. Once the management channel exists, the unwind attempts `SUSPEND`
before stopping it.

### Context create, use, and destroy

```text
resolver chooses columns
  -> CREATE_CONTEXT
  -> create per-context CQ/MSI channel from response
  -> MAP_HOST_BUFFER(initial 64 MiB heap)
  -> optional CONFIG_CU
  -> submit/sync/debug operations
  -> stop per-context channel
  -> DESTROY_CONTEXT
  -> free per-context channel
```

There is no host-buffer-unmap opcode in this surface. `DESTROY_CONTEXT` is the
firmware-side teardown boundary.

Failed IRQ allocation, channel allocation/start, or other local setup after a
successful `CREATE_CONTEXT` normally sends `DESTROY_CONTEXT`. Initial mapping
failure releases the resolver resource, which also destroys the context.

One primary-driver cleanup gap is present:
`xrs_allocate_resource()` calls the firmware-creating load callback before its
DPM callback. If the DPM callback then fails, `free_node` removes only the
solver node and never invokes unload. The driver therefore omits
`DESTROY_CONTEXT` for the firmware context it just created
(`aie2_solver.c:299-344`). This is a legitimate omission in an input trace,
not behavior for the emulator to reproduce internally.

### Submission and timeout recovery

Default `force_cmdlist=true` converts both single and multi-command CU/NPU
jobs to `CHAIN_EXEC_NPU`. Setting the supported module parameter false sends
single CU/NPU jobs as `EXECUTE_BUFFER_CF` or `EXEC_DPU`; an explicit command
chain still uses `CHAIN_EXEC_NPU`.

On a scheduler timeout, Phoenix cannot query `GET_APP_HEALTH` because the
feature is absent. Recovery is per-context:

```text
stop scheduler
  -> DESTROY_CONTEXT
  -> restart scheduler
  -> CREATE_CONTEXT
  -> MAP_HOST_BUFFER
  -> optional heap additions (unavailable on NPU1)
  -> optional CONFIG_CU
```

It is not a PSP/SMU/full-firmware reset. Restart errors are logged without
transactional rollback.

The NPU1 `WAIT_CMD` UAPI path sends no firmware command because `aie2_ops`
provides no `cmd_wait`; the generic path returns `-EOPNOTSUPP`.

### Suspend, resume, close, and removal

For suspend, each client waits up to two seconds for its last fence and
ignores that wait result. The shared stop helper then stops the scheduler,
destroys its firmware context, and restarts the scheduler before device stop
performs:

```text
SET_RUNTIME_CONFIG(clock gating)
  -> SUSPEND
  -> poll PSP waitmode
  -> stop management mailbox
  -> PSP RELEASE_TMR
  -> set NPU1 clocks to DPM level 0
  -> SMU POWER_OFF
```

Resume repeats the full hardware-start sequence, then recreates each surviving
host context with `CREATE_CONTEXT`, heap mapping, and CU configuration. If one
context restart fails, already restarted contexts remain and later contexts
are neither retried nor rolled back.

Normal context destruction, file close, and PCI removal converge on resource
release and `DESTROY_CONTEXT` before device finalization.

### Power-mode transitions

A power-mode change first issues the two NPU1 SMU clock commands, then sends
runtime config type 1 for clock gating. If the runtime-config command fails,
the physical clock change remains but the driver's `pw_mode` field is not
updated. This is another observable partial transition, not an atomic
transaction.

## Packaged Legacy Driver Delta

The legacy Makefile unconditionally adds `-DAMDXDNA_DEVEL`, and its debugfs
file is additionally guarded by `CONFIG_DEBUG_FS`. Those build facts do not
turn diagnostic probes into the production hardware contract.

For NPU1 protocol 5.8, the legacy optional-message table contains only
`CHAIN_EXEC_NPU`; its feature table contains only `AIE2_NPU_COMMAND`.

| Operation | Primary `amdxdna.ko` | Legacy `amdxdna_legacy.ko` |
|---|---|---|
| `0x004 GET_TELEMETRY` | production UAPI | compiled only under `AMDXDNA_DEVEL` |
| `0x203 AIE_RW_ACCESS` | production 5.8 route | explicitly returns `-EOPNOTSUPP` because the optional table lacks it |
| `0x301 GET_PROTOCOL_VERSION` | no sender | `CONFIG_DEBUG_FS` health-check test only |
| `0x104 INVOKE_SELF_TEST` | no sender | `CONFIG_DEBUG_FS` test only |
| `0x001 REGISTER_PDI`, `0x00a UNREGISTER_PDI`, `0x00b LEGACY_CONFIG_CU` | absent | `AMDXDNA_DEVEL` only |

The remaining production mailbox operations overlap the primary surface,
including direct execution when its corresponding `force_cmdlist` path is
selected. The legacy tree therefore supplies useful compatibility stimuli and
lifecycle variation, but no unique production opcode that expands the primary
hardware-command set.

## Existing External Hardware Anchors

These observations come from the named repository captures, not from driver
commit `216cefe`. The source audit says what can be sent; silicon observations
still determine the expected behavior.

| Surface | Existing observation |
|---|---|
| post-alive ordinary request | one `GET_TELEMETRY` transaction pins request-tail -> IRQ -> receive-worker -> response-head ordering; see `2026-07-27-phoenix-post-alive-observability.md` |
| `GET_TELEMETRY` | Phoenix accepts the command but provides only a small real counter prefix and a large `0xff` placeholder region; see `2026-05-06-npu1-msg-op-capability-survey.md` and `2026-05-26-phoenix-fw-1.5.6.399-diff.md` |
| `CHAIN_EXEC_NPU` | a controlled ctrl-packet workload can execute and then wait forever for an array-completion event, causing TDR recovery; see `2026-05-22-chain-exec-npu-silent-drop-captured.md` |
| `AIE_RW_ACCESS` | compute-tile access is hardware-verified; known unowned/memtile access can wedge Phoenix, and the pinned primary driver contains a guard; see `2026-05-26-aie-rw-access-memtile-wedge-mechanism.md` |
| gated NPU4-only commands | sampled commands return a four-byte `INVALID_COMMAND` response on Phoenix rather than implementing the advertised larger success response; see `2026-05-06-npu1-msg-op-capability-survey.md` |

## What This Audit Freezes

The initial contract corpus now has three explicit layers:

1. **Primary required surface:** the NPU1 PSP/SMU/BAR/mailbox lifecycle and all
   20 configuration-reachable mailbox opcodes above.
2. **Legacy compatibility corpus:** production paths from the packaged
   secondary driver, tagged separately; it does not expand the opcode set.
3. **Development inventory:** legacy debugfs and `AMDXDNA_DEVEL` operations,
   excluded from the primary gate unless a later, explicit hardware-verifiable
   requirement promotes one.

Expected values, failure statuses, side effects, and timing still require
firmware-byte derivation or controlled silicon capture per matrix row. A
driver-side rejection proves only that the pinned driver does not transmit a
request; it is not proof that the firmware lacks the operation.

The remaining architecture decision is therefore not “which driver do we
emulate.” We must decide how the emulator represents the PSP/SMU/mailbox and
the currently opaque post-tail controller-to-Xtensa interrupt transition while
preserving natural unmodified-firmware execution and leaving every
unobservable controller detail explicitly unknown.
