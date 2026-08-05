---
name: Phoenix AIE_RW_ACCESS uses a legacy wire layout incompatible with the current driver
description: Offline source and signed-firmware audit proving that Phoenix FW 1.5.5.391 interprets opcode 0x203 bytes 4 and 5 as physical row and column, while the current amdxdna driver sends firmware context ID and relative row there. This aliases every request to the wrong physical tile and explains the 2026-08-05 column-clock wedge and prior misleading Timer_Low results.
type: finding
---

# Phoenix `AIE_RW_ACCESS` wire-layout mismatch

## Verdict

**CONFIRMED:** Phoenix firmware `1.5.5.391` implements opcode `0x203`, but it
does **not** implement the request ABI used by the current amdxdna driver. The
two sides assign different meanings to bytes 4 through 6:

| Payload byte | Current driver (NPU4 ABI) | Phoenix FW 1.5.5.391 |
|---:|---|---|
| 4 | firmware context ID | physical row |
| 5 | context-relative row | physical column |
| 6 | context-relative column | ignored |
| 7 | reserved | ignored |

Consequently, enabling `AIE2_RW_ACCESS` in the NPU1 feature table was unsafe.
The current driver validates one requested tile, then Phoenix accesses a
different physical tile. The NPU1 feature must be disabled under this ABI.

The 2026-08-05 failure was therefore **not a read of the shim
`Column_Clock_Control` register**. With the normal first Phoenix firmware
context ID of 5, the request addressed physical tile `(column 0, row 5)` at
offset `0xFFF20`. That offset is a shim-only privileged register and does not
decode on a row-5 compute tile. The firmware then stopped answering the
mailbox request.

This finding supersedes the tile identities and causal interpretations in:

- `docs/archive/findings/2026-05-05-aie-rw-access-firmware-actually-supported.md`
- `docs/superpowers/findings/2026-05-06-npu1-msg-op-capability-survey.md`
- `docs/archive/findings/2026-05-06-aie-rw-access-tile-claim-authorization.md`
- `docs/archive/findings/2026-05-07-aie-rw-access-memtile-dm-half-impl.md`
- `docs/superpowers/findings/2026-05-26-aie-rw-access-memtile-wedge-mechanism.md`
- `docs/superpowers/findings/2026-05-26-aie-rw-access-not-a-cycle-probe.md`

The old observations remain useful as observations of *some* physical AIE
accesses. They do not identify the tiles the reports named.

## Audit boundary

This is an offline audit of the exact failed tuple:

- Phoenix/NPU1
- signed firmware `1.5.5.391`, management protocol `5.8`
- installed XRT `2.26.0`
- the transient XRT-2.26-qualified amdxdna candidate based on
  `216cefececd74effcd7a88350c71b99f5ef9a215`
- the candidate's only added instrumentation was a behavior-neutral mailbox
  response tracepoint

No hardware request, XRT device open, privileged operation, or recovery action
was performed during this audit. The failed read was deliberately not retried
after reboot.

Failure artifacts are under:

`build/experiments/phoenix-pm-clock-characterization/20260805T021725Z-column-clock-state/qualified-driver/`

The pre-restore log records the exact terminal sequence at lines 64277 onward:
mailbox completion timeout, `aie2_rw_aie_reg` `-62`, tile-read `-62`, context
destroy `-19`, then the IOMMU fault cascade.

## End-to-end request path

### 1. Runner

After `run.wait()`, the uncommitted clock-characterization path in
`bridge-runner/bridge-trace-runner.cpp` calls:

```cpp
get_aie().read_aie_reg(
    aie_pid, aie_ctxid, args.column_clock_col, 0,
    XAIEMLGBL_PL_MODULE_COLUMN_CLOCK_CONTROL);
```

For the failed run, this requested relative column 0, row 0, offset
`0xFFF20`.

### 2. XRT 2.26

`xrt/src/runtime_src/core/common/api/xrt_device.cpp::read_aie_reg` takes the
new `query::aie_read` path. It forwards `pid`, userspace context ID, `col`,
`row`, `offset`, and size 4 unchanged. It calls `get_abs_col()` only in the
old-query fallback, which was not selected here.

The installed `libxrt_coreutil.so.2` SHA-256 was
`d6a6ea581c95d4c6c09732f7ba2a4d09b4b4a76f5f13657da4a00a9a6e42ca90`,
matching the installed 2.26 manifest.

### 3. UAPI and driver validation

The new query becomes `DRM_AMDXDNA_AIE_TILE_READ` with an
`amdxdna_drm_aie_tile_access`. `drivers/accel/amdxdna/aie.c`:

- resolves the userspace PID and context ID to an owned live hardware context;
- checks the requested relative column against `hwctx->num_col`;
- checks the requested row against device metadata; and
- invokes `msg_ops.rw_reg` with the requested row and column.

Those checks are real, but they constrain the coordinates as the current
driver understands them. They cannot protect an older firmware ABI that reads
different bytes as coordinates.

### 4. Current amdxdna serialization

`drivers/accel/amdxdna/aie2_msg_priv.h` defines:

```c
struct aie_rw_access_req {
    enum aie2_access_type type;
    __u8 ctx_id;
    __u8 row;
    __u8 col;
    __u8 reserved;
    union { /* ... */ };
} __packed;
```

`aie2_message.c::aie2_rw_aie_reg` fills `ctx_id` from
`hwctx->fw_ctx_id`. `DECLARE_XDNA_MSG_COMMON` zero-initializes the complete
request, so neither uninitialized padding nor a short payload is involved.

For a register read, the six little-endian words are:

```text
[ 2,
  fw_ctx_id | (requested_row << 8) | (requested_col << 16),
  aie_offset,
  0,
  0,
  0 ]
```

The current layout entered the open driver in commit `b9686b4` on 2026-02-06
and was enabled only for NPU4 firmware 6.24. The equivalent current
`drivers/accel` port is commit `1ebb85c`.

Our commit `d91ad8f5cfa976e1874eb4eaadeacacacd76f910` later advertised the same
feature for NPU1 protocol 5.8. Its evidence was a raw Timer_Low response whose
coordinates were decoded using the wrong layout.

### 5. Phoenix signed-firmware decode

The exact pinned firmware export is:

`/home/triple/npu-work/ghidra-projects/npu-fw/analysis-xtensa/`

The opcode wrapper `FUN_08ad98c4`:

- validates non-null request/response pointers;
- requires a 24-byte payload;
- maps a DRAM buffer only for memory access types 0 and 1; and
- calls `FUN_08ade104` for all four access types.

`FUN_08ade104` reads:

```text
type   = u32(payload[0..4])
row    = payload[4]
column = payload[5]
offset = u32(payload[8..12])
```

For register-read type 2, its direct load address is:

```text
array_base + column * 0x02000000 + row * 0x00100000 + offset
```

The decompiler shows this at `decompiled.c:8825-8891`; the independent Xtensa
disassembly shows the same byte loads and shifts in `disasm.txt:12190-12320`.
There is no context-ID read or context-based coordinate translation in this
handler.

This is the natural packed layout of an older request containing the exact
aie-rt `XAie_LocType` physical `(row, column)` pair. The official aie-rt type
is defined as `u8 Row; u8 Col;` in `driver/src/global/xaiegbl.h`:

```c
struct phoenix_aie_rw_access_req {
    u32 type;
    u8 row;
    u8 col;
    u16 reserved;
    union { /* same 16-byte payload */ };
};
```

The complete 24-byte Phoenix request contract is:

| Offset | Memory access (types 0/1) | Register access (types 2/3) |
|---:|---|---|
| `0x00` | `u32 type` | `u32 type` |
| `0x04` | `u8 row` | `u8 row` |
| `0x05` | `u8 column` | `u8 column` |
| `0x06` | `u16` ignored/reserved | `u16` ignored/reserved |
| `0x08` | `u64 dram_addr` | `u32 aie_offset` |
| `0x0c` | continuation of `dram_addr` | `u32 write_value` |
| `0x10` | `u32 aie_offset` | ignored |
| `0x14` | `u32 size` | ignored |

### Embedded aie-rt contract

The worker does not merely resemble aie-rt. Its indirect targets resolve to
the embedded functions:

- `0x08B08B54`: `XAie_CfgInitialize`
- `0x08B09138`: `XAie_DataMemBlockWrite`
- `0x08B092F0`: `XAie_DataMemBlockRead`

The function names and error strings are present in the signed image, and the
implementations match the official `aie-rt/driver/src/memory/xaie_mem.c`
control flow. The worker constructs this `XAie_Config` on its stack:

| Field | Phoenix value |
|---|---:|
| `AieGen` | 3 |
| `BaseAddr` | `0x9C000000` |
| `ColShift` | 25 |
| `RowShift` | 20 |
| `NumRows` | 6 |
| `NumCols` | 5 |
| `ShimRowNum` | 0 |
| `MemTileRowStart` | 1 |
| `MemTileNumRows` | 1 |
| `AieTileRowStart` | 2 |
| `AieTileNumRows` | 4 |

The packed literals `0x05061419` and `0x02010100`, plus the following byte
value 4, independently encode the shifts and row topology. This agrees with
the toolchain-derived Phoenix 5-column by 6-row physical envelope.

### Operation semantics

The four type values are:

| Type | Operation | Firmware behavior |
|---:|---|---|
| 0 | memory read | `XAie_DataMemBlockRead`: tile data memory to the mapped management buffer |
| 1 | memory write | `XAie_DataMemBlockWrite`: mapped management buffer to tile data memory |
| 2 | register read | raw 32-bit load from physical tile base plus offset |
| 3 | register write | raw 32-bit store to physical tile base plus offset |

For memory operations, the wrapper first resolves the management PASID and
maps `dram_addr` for `size` bytes. The embedded aie-rt path accepts compute
tiles and memory tiles, rejects shim tiles, bounds-checks the complete byte
range against the derived tile data-memory size with 64-bit arithmetic, and
supports arbitrary byte alignment and byte counts. All nonzero aie-rt return
codes are collapsed to the generic firmware status `0x04000003`.

Register operations are materially less defensive. They perform a direct
32-bit Xtensa load or store with no tile-kind, ownership, offset-range, or
alignment validation. This explains the asymmetric failure mode: an invalid
memory request can return an error, while a raw register access to a
non-decoding address can stop the firmware before it posts a response.

### Response contract

The wrapper always posts an eight-byte response:

```c
struct phoenix_aie_rw_access_resp {
    u32 status;
    u32 reg_read_value;
};
```

Its status values are:

- `0`: success
- `0x04000001`: null pointer, wrong request length, or other invalid input
  buffer condition
- `0x04000003`: worker initialization, operation, or aie-rt failure

Only a successful type-2 register read initializes the second word. It is
reserved/undefined for the other operations and on errors.

## Exact failed target

The pinned firmware allocates context IDs in the deterministic order
`[5, 4, 3, 2, 1, 0]`; the first context in this run was ID 5. This is
independently pinned by:

- the nearest CREATE_CONTEXT response before the failed request in
  `dmesg-before-restore.log:63555`, whose second response word is
  `00000005`; and
- `src/firmware/boot_tests/guards.rs`, which executes the unmodified signed
  image and asserts the same six-ID allocation order.

The failed request bytes were therefore:

```text
02 00 00 00  05 00 00 00  20 ff 0f 00  00 00 00 00  00 ... 00
type=read    current: ctx=5,row=0,col=0    offset=0xFFF20
             Phoenix: row=5,col=0
```

Phoenix computed:

```text
array_base + 0 * 0x02000000 + 5 * 0x00100000 + 0x000FFF20
= array_base + 0x005FFF20
```

For the pinned `0x9C000000` array base, that is `0x9C5FFF20`: physical
column 0, row 5, offset `0xFFF20`.

The AM025 register database defines `Column_Clock_Control` at `0xFFF20` in
the shim PL module and describes it as privileged and TMR-protected. It is not
a row-5 compute-tile register. aie-rt also treats column clock control as the
privileged backend operation `XAIE_BACKEND_OP_SET_COLUMN_CLOCK`, not as an
ordinary tile-register query.

**Strong inference:** the synchronous firmware load to the non-decoding
compute-tile address received no completion, so the handler never posted its
mailbox response. This matches the exact five-second mailbox timeout and the
known direct-load implementation. We lack a physical AXI trace or firmware
exception record, so the no-completion mechanism is not labeled direct
observation.

## Reinterpretation of earlier experiments

The mismatch explains the otherwise inconsistent May results. For the usual
firmware context ID 5:

| Reported intent | Current request bytes 4/5/6 | Phoenix physical target | Corrected interpretation |
|---|---|---|---|
| Raw debugfs `(col 0,row 2)` Timer_Low with `ctx=0` | `00 02 00` | `(col 2,row 0)` | Successful **shim** Timer_Low read, not compute row 2 |
| Production `(col 0,row 2)` core Timer_Low | `05 02 00` | `(col 2,row 5)` | Successful access to a different compute tile |
| Production `(col 0,row 2)` compute-memory Timer_Low | `05 02 00` | `(col 2,row 5)` | Valid compute-memory module on the aliased tile |
| Production `(col 0,row 0)` shim Timer_Low | `05 00 00` | `(col 0,row 5)` | Compute-core Timer_Low shares offset `0x340F8`, explaining the apparently valid shim result |
| Production `(col 0,row 1)` memtile Timer_Low | `05 01 00` | `(col 1,row 5)` | Invalid memtile offset `0x940F8` on a compute tile, explaining the wedge |
| 2026-08-05 `(col 0,row 0)` column clock | `05 00 00` | `(col 0,row 5)` | Invalid shim-only offset `0xFFF20` on a compute tile, explaining the wedge |

The old conclusion that Timer_Low advanced by a firmware-handler artifact is
not licensed: the reads sampled other physical tiles. The old conclusion that
runtime-sequence writes were invisible is also not licensed because the CDO
and AIE_RW_ACCESS paths targeted different tiles. Direct AIE_RW_ACCESS
write/read behavior on one correctly identified tile is now covered through
the signed-firmware emulator seam described below; equivalent validation on
real Phoenix silicon remains open.

The current NPU1 memtile guard in amdxdna commit `dd6c95a` is therefore not a
root-cause fix. It checks `requested_row == memtile_row`, but Phoenix uses the
firmware context ID as the physical row and the requested row as the physical
column. The guard happened to prevent one known bad alias while leaving many
others reachable, including the column-clock failure.

## Safety and fix disposition

### Required immediate fix

Remove the NPU1 `AIE2_RW_ACCESS` feature advertisement introduced by
`d91ad8f`. This restores the fail-closed upstream posture: XRT receives
`-EOPNOTSUPP` instead of sending an incompatible message to firmware.

The uncommitted runner's `--read-column-clock` path must not be used on NPU1
through this API. The same applies to its performance-counter reads until the
driver feature is disabled or a separately proven compatibility path exists.

### Why merely swapping bytes is not an automatic fix

A Phoenix compatibility serializer could send physical `row,col` in bytes
4 and 5. That is technically straightforward, but the legacy firmware handler
does not accept or enforce a firmware context ID. The current public API is
context-scoped and uses relative columns; a driver shim would have to:

1. translate the relative column through the live context's physical
   partition;
2. validate the complete physical tile and register/memory range;
3. preserve isolation from other contexts and reserved tiles entirely in the
   driver; and
4. version the wire layout by firmware family/protocol.

That is a security and contract design, not a field-order patch. It should be
reviewed separately before any re-enablement.

The RE does narrow that design considerably. The current driver already
resolves the caller's PID/context to a live `amdxdna_hwctx`, validates the
context-relative column against `hwctx->num_col`, and records the allocated
physical start in `hwctx->start_col`. The legacy physical coordinate is
therefore derived, not guessed:

```text
physical_column = hwctx->start_col + context_relative_column
physical_row    = requested_row
```

Memory and register compatibility have different safety boundaries:

| Path | What can be made safe in the driver | Remaining blocker |
|---|---|---|
| memory read/write | live-context lookup, relative-column bounds, `start_col` translation, physical topology checks; embedded aie-rt enforces tile kind and data-memory bounds | real-hardware validation of the compatibility serializer |
| register read/write | the same ownership and coordinate translation | a driver-side register-address policy strong enough to exclude absent modules, holes, unaligned accesses, and offsets that escape the tile window |

The current Phoenix memtile guard is based on the old false premise that a
context-owned memtile access itself lacks firmware support. The signed handler
explicitly calls `XAie_DataMemBlockRead/Write`, whose accepted tile kinds
include memory tiles. Correctly serialized real-hardware validation is still
needed before claiming powered/context-owned memtile behavior, but the guard
is neither the root fix nor an accurate statement of the firmware contract.

The register policy must be settled before generic NPU1 register access is
re-enabled. Phoenix firmware does not provide the validation that newer
firmware appears to own, and the kernel cannot depend at runtime on mlir-aie's
AM025 JSON. A generated toolchain-derived allowlist, conservative per-module
ranges, or leaving register access disabled are distinct designs; silently
adding only a row/column swap is not acceptable.

### Better clock-observation seam

For the current clock-characterization work, observe the column-clock write
through the already authenticated signed-firmware/emulator seam or instrument
the known context-create sequence that writes physical shim offsets
`0x9C0FFF20`, `0x9E0FFF20`, and so on. Do not use opcode `0x203` as a generic
privileged-register path on Phoenix.

### Signed-firmware emulator validation

`m2c_signed_firmware_legacy_aie_rw_access_round_trips_compute_and_memtile_memory`
boots the unmodified pinned firmware, creates a physical-column-1 context, and
sends correctly serialized type-1/type-0 requests through the real management
mailbox and embedded aie-rt paths. It verifies eight-byte write and readback
round trips independently on physical `(column 1, row 2)` compute memory and
`(column 1, row 1)` memtile memory.

The first RED run exposed a loader-model gap before the handler reached array
memory: firmware startup copied each task template from
`0xb0027100..0xb0027ef7`, while the emulator treated that range as an empty
system aperture. Those bytes match signed-image offsets
`0x27100..0x27ef7` exactly. The M2c loader now exposes the signed image through
the corresponding PSP read view at `0xb0000000`; the bus-level guard bounds the
view to the loaded image and confirms those reads no longer enter `SysStub`.
With that shared loader seam corrected, both tile kinds pass without any
opcode-specific firmware or driver shim.

## Unknowns retained

- Whether Phoenix firmware `1.5.6.399` retains the same legacy layout.
- Which older Phoenix firmware versions expose opcode `0x203` and with which
  layout.
- What a correctly serialized legacy request to the real shim
  `Column_Clock_Control` would do. The failed run never tested it, and its
  privileged/TMR semantics make an ad-hoc live test inappropriate.
- Whether a safe, context-isolated legacy compatibility API is desirable
  upstream. The current evidence supports disabling the feature, not silently
  broadening the driver's physical access surface.
- The exact driver-side register-address policy. This is now the principal
  compatibility design fork, not the wire layout.
- Dynamic signed-firmware coverage of correctly serialized register read/write
  types 2 and 3. Memory read/write types 0 and 1 are now covered on compute and
  memory tiles; the raw-register path needs a deliberately safe offset before
  it is exercised dynamically.
