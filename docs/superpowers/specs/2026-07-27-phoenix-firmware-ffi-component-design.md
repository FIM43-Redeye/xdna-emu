# Phoenix Firmware FFI Component Seam -- Design

**Date:** 2026-07-27

**Status:** Concept approved; written-spec review pending

## Purpose

Expose the already-proven Phoenix management-firmware processor through a
small public C ABI so a future virtual PCI frontend can drive the real
firmware state instead of synthesizing driver-specific responses.

This is a component gate toward the Phoenix firmware-equivalence goal. It is
not unmodified-driver acceptance and does not make the current XRT SHIM plugin
such a frontend.

## Fidelity Boundary

The seam exposes only behavior already implemented and grounded:

- explicit loading of a caller-supplied `$PS1` firmware image;
- reset-to-idle execution of that image against the interpreter engine's sole
  `DeviceState`;
- host-visible 32-bit access through the firmware-programmed BAR2 SRAM aliases.

It does not:

- discover firmware through an environment variable or filesystem path;
- emulate the PSP or SMU;
- invent BAR4-to-Xtensa register aliases or interrupt delivery;
- parse or answer management commands in the FFI or plugin;
- wire the existing XRT SHIM plugin into the firmware path;
- claim an unmodified kernel-driver end-to-end result.

The current open evidence proves the BAR2/BAR4 host contract, but not the
internal bridge from a BAR4 X2I-tail write to an unknown controller transition,
active source 46, and Xtensa interrupt bit 0. `0x27200170` is an unrelated
internal queue and must not be used as that bridge.

## State Ownership

`XdnaEmuHandle` gains:

```rust
firmware: Option<FirmwareProcessor>
```

The FFI handle is already the owner of backend-adjacent host state. Keeping
the optional processor there avoids adding interpreter-only operations to the
cross-backend `NpuBackend` trait.

Firmware execution is supported only when `backend.as_interpreter_mut()`
succeeds. Aiesim and other backends return an explicit unsupported error.

Loading a new image atomically replaces the prior firmware processor after
parsing and Phoenix load-map validation succeed. It does not reset or replace
the array `DeviceState`. Context reset likewise does not reboot firmware:
firmware survives ordinary context lifecycles on hardware. Destroying the
handle drops both.

## C ABI

Add these declarations to `include/xdna_emu.h` and matching Rust exports:

```c
XdnaEmuResult xdna_emu_load_firmware(
    XdnaEmuHandle* handle,
    const uint8_t* firmware_data,
    uint64_t firmware_size
);

XdnaEmuResult xdna_emu_boot_firmware(
    XdnaEmuHandle* handle,
    uint64_t max_instructions
);

XdnaEmuResult xdna_emu_firmware_read_host_sram32(
    XdnaEmuHandle* handle,
    uint32_t device_address,
    uint32_t* value_out
);

XdnaEmuResult xdna_emu_firmware_write_host_sram32(
    XdnaEmuHandle* handle,
    uint32_t device_address,
    uint32_t value
);
```

Addresses are firmware device addresses such as `0x030bb000`, not BAR-relative
offsets. A later PCI frontend owns BAR-resource translation.

The read uses an output pointer because zero is a valid hardware value.

## Loading and Validation

`xdna_emu_load_firmware`:

1. validates the handle, pointer, and `u64 -> usize` length conversion;
2. rejects a backend that cannot expose the interpreter engine;
3. copies and parses the explicit bytes with `FirmwareImage::parse`;
4. validates that every fixed Phoenix PSP-load slice is present before
   indexing it;
5. constructs `FirmwareProcessor::try_load_m2c`;
6. installs it on the handle only after all prior steps succeed.

The core gains a fallible Phoenix loader for the public trust boundary.
Existing known-good internal callers may keep the infallible wrapper rather
than mechanically changing every diagnostic test.

No SHA is baked into the implementation. The validation test pins the primary
image SHA externally; older authoritative images must be loadable through the
same byte-oriented API.

## Boot

`FirmwareProcessor` gains `boot_to_idle_with_device`. Both it and the existing
standalone `boot_to_idle` use one internal loop; the only variation is whether
each CPU step receives a borrowed `DeviceState`.

`xdna_emu_boot_firmware`:

1. rejects a zero instruction budget;
2. rejects a missing firmware processor or non-interpreter backend;
3. borrows the interpreter engine's existing `DeviceState`;
4. runs the unmodified firmware to its natural wait or a bounded failure;
5. returns `Success` only when the report reaches idle.

On failure, the processor remains available for inspection and
`xdna_emu_get_error` reports the stop state, PC, instruction count, unresolved
spin, or unknown opcode.

The FFI does not enable the diagnostic `HostMailbox` completion agent and does
not force firmware progress.

## Host SRAM

The read and write functions delegate directly to the processor bus's
`host_sram_load32` and `host_sram_store32`.

They therefore use the firmware-programmed alias geometry and the
reset-preconfigured, four-byte I2X slot-15 alive mapping. They do not copy the
descriptor into a second shadow buffer.

An unconfigured alias retains the bus model's existing dropped-write /
zero-read behavior. A missing processor is an API error.

## Error Mapping

| Condition | Result |
|---|---|
| Null handle | `InvalidHandle` |
| Null input/output pointer | `NullPointer` |
| Length cannot fit `usize` | `ParseError` |
| Invalid or too-short firmware | `ParseError` |
| Missing loaded processor | `ExecutionError` |
| Unsupported backend | `ExecutionError` |
| Zero boot budget | `ExecutionError` |
| Boot stops before natural idle | `ExecutionError` |

Every non-success path sets the thread-local error text.

## Tests

Implementation follows RED/GREEN:

1. A core test proves `boot_to_idle_with_device` runs the same boot algorithm
   while borrowing, rather than cloning or moving, the array state.
2. FFI boundary tests cover null pointers, malformed/truncated firmware,
   missing firmware, zero budget, and unsupported backend behavior.
3. A real-image FFI test, gated by `XDNA_FIRMWARE`, passes the exact image bytes
   through the public load function, then boots through the public boot
   function and checks:
   - descriptor magic `0x55504e5f` at `0x030bb020`;
   - alive pointer `0x030bb000` at `0x030bf000`;
   - driver-style clearing of `0x030bf000`;
   - an array-register sentinel outside the firmware's touched aperture
     survives, proving the engine's `DeviceState` was borrowed rather than
     replaced.
4. The C signatures are type-checked and documented in the public header.

Required local gates:

```bash
cargo test -p xdna-emu-ffi
cargo test --lib
cargo fmt --all --check
git diff --check
```

The firmware-gated tests run with the pinned `XDNA_FIRMWARE` path. These are
brief local builds; Halo is not needed.

## Documentation Corrections

The implementation milestone also corrects stale architecture text that:

- still says PSP/SMU column power blocks natural firmware boot;
- says `Bus` owns a second `DeviceState`;
- labels `0x27200170/174/178` as the management mailbox;
- implies the current SHIM plugin can validate the unmodified kernel driver.

The corrected documentation must distinguish:

- proven PSP handoff effects from unmodeled PSP internals;
- proven BAR2/BAR4 host behavior from the unresolved BAR4-to-controller
  translation;
- the raw FFI component gate from the later virtual PCI driver boundary.

## Next Evidence Milestone

After this component gate is green, capture a genuine post-alive management
transaction. That capture can pin the host-visible envelope, but the missing
internal chain requires a non-halting management-Xtensa trace or authoritative
controller specification:

```text
BAR4 X2I-tail publication
  -> unknown controller transition
  -> active controller source 46
  -> Xtensa interrupt bit 0
  -> firmware event (6,4)
```

Until that evidence exists, the mailbox-fabric operation remains explicitly
unimplemented.
