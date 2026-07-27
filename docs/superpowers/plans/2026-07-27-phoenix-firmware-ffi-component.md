# Phoenix Firmware FFI Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the proven Phoenix management-firmware processor through a validated C ABI that boots against the interpreter engine's existing array state and exposes the genuine firmware-programmed host SRAM aliases.

**Architecture:** Keep `InterpreterEngine` as the sole owner of `DeviceState`, store one optional `FirmwareProcessor` on `XdnaEmuHandle`, and borrow the engine's device for each firmware CPU step. Reuse the existing `crates/xdna-emu-ffi/src/firmware.rs` module and existing `FirmwareImage`, `FirmwareProcessor`, and bus alias methods; add no backend trait methods, mailbox responder, or plugin wiring.

**Tech Stack:** Rust 2021, existing xdna-emu core and FFI crates, C11 public header, Cargo tests, system `sha256sum` for the firmware-gated test.

## Global Constraints

- Test-driven development: add each failing test before its implementation and record the RED result.
- Derive behavior from the pinned firmware, open driver, and existing bus model; do not invent BAR4-to-Xtensa interrupt routing.
- The caller supplies firmware bytes explicitly; production code must not read a firmware environment variable or filesystem path.
- Do not enable `HostMailbox`, synthesize command responses, force firmware progress, or wire the SHIM plugin.
- Support only the interpreter backend and return an explicit error for every other backend.
- Keep `DeviceState` solely owned by `InterpreterEngine`; firmware borrows it only while stepping.
- Keep the existing infallible `FirmwareProcessor::load_m2c` wrapper for known-good internal research callers.
- Validate all raw pointers, integer conversions, image bounds, and output pointers at the public C boundary.
- A failed replacement load must preserve the previously loaded processor.
- A context reset must preserve firmware state; handle destruction drops it.
- The primary firmware-gated test must pin SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`.
- Required final gates are `cargo test -p xdna-emu-ffi`, `cargo test --lib`, `cargo fmt --all --check`, and `git diff --check`.

---

### Task 1: Fallible Phoenix Loading and Borrowed-Device Boot

**Files:**
- Modify: `src/firmware/image.rs`
- Modify: `src/firmware/mod.rs`
- Test: `src/firmware/image.rs`
- Test: `src/firmware/boot_tests/guards.rs`

**Interfaces:**
- Consumes: `FirmwareImage::parse(&[u8])`, `Cpu::step_with_device(&mut Bus, &mut DeviceState)`, and the existing `boot_to_idle` algorithm.
- Produces: `FirmwareProcessor::try_load_m2c(FirmwareImage) -> Result<FirmwareProcessor, FirmwareError>` and `FirmwareProcessor::boot_to_idle_with_device(&mut self, &mut DeviceState, u64) -> IdleReport`.

- [x] **Step 1: Add image-bound RED tests**

Add these cases beside the existing `FirmwareImage` parser tests:

```rust
#[test]
fn rejects_declared_payload_beyond_the_supplied_file() {
    let mut raw = build_image(&[0u8; 4]);
    let declared = raw.len() as u32 + 1;
    raw[0x14..0x18].copy_from_slice(&declared.to_le_bytes());
    let err = FirmwareImage::parse(&raw).unwrap_err();
    assert!(matches!(err, FirmwareError::SizeMismatch { .. }), "got {err}");
}

#[test]
fn rejects_declared_payload_shorter_than_the_header() {
    let mut raw = build_image(&[]);
    raw[0x14..0x18].copy_from_slice(&0x10u32.to_le_bytes());
    let err = FirmwareImage::parse(&raw).unwrap_err();
    assert!(matches!(err, FirmwareError::SizeMismatch { .. }), "got {err}");
}
```

- [x] **Step 2: Run the parser tests and record RED**

Run:

```bash
cargo test --lib firmware::image::tests::rejects_declared_payload
```

Expected: both tests fail because `FirmwareImage::parse` currently accepts any declared payload size after checking the magic.

- [x] **Step 3: Validate the declared payload interval**

After reading `payload_size`, reject values outside `HEADER_END..=raw.len()`:

```rust
let payload_size = u32::from_le_bytes(raw[SIZE_OFFSET..SIZE_OFFSET + 4].try_into().unwrap());
let declared = payload_size as usize;
if !(HEADER_END..=raw.len()).contains(&declared) {
    return Err(FirmwareError::SizeMismatch { header: payload_size, file: raw.len() });
}
```

Keep the full supplied bytes in `FirmwareImage` for container diagnostics. The Phoenix loader separately copies only `image.bytes()[..image.payload_size()]` into the bus, so the inert signature trailer is never treated as loadable content.

- [x] **Step 4: Add fallible-load and borrowed-boot RED tests**

Add a non-firmware-gated truncation test and a firmware-gated borrowed-device test:

```rust
#[test]
fn m2c_loader_rejects_an_image_without_segment_b() {
    let mut raw = vec![0u8; SEG_B_FILE_START as usize - 1];
    raw[0x10..0x14].copy_from_slice(b"$PS1");
    let declared = raw.len() as u32;
    raw[0x14..0x18].copy_from_slice(&declared.to_le_bytes());
    let image = FirmwareImage::parse(&raw).expect("container header");
    let err = match FirmwareProcessor::try_load_m2c(image) {
        Ok(_) => panic!("truncated Phoenix load map was accepted"),
        Err(error) => error,
    };
    assert!(matches!(err, FirmwareError::Truncated { .. }), "got {err}");
}

#[test]
fn m2c_boot_with_device_borrows_and_preserves_array_state() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(path).expect("read firmware");
    let image = FirmwareImage::parse(&raw).expect("parse firmware");
    let mut processor = FirmwareProcessor::try_load_m2c(image).expect("load Phoenix firmware");
    let mut device = crate::device::DeviceState::new_npu1();
    device.write_tile_register(4, 0, 0x000f_ff20, 1);

    let report = processor.boot_to_idle_with_device(&mut device, 200_000);

    assert!(report.reached_idle, "firmware did not reach idle: {report:?}");
    assert_eq!(device.read_tile_register(4, 0, 0x000f_ff20), 1);
}
```

- [x] **Step 5: Run the loader and borrowed-boot tests and record RED**

Run:

```bash
XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
  cargo test --lib m2c_loader_rejects_an_image_without_segment_b
XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
  cargo test --lib m2c_boot_with_device_borrows_and_preserves_array_state
```

Expected: compilation fails because `try_load_m2c` and `boot_to_idle_with_device` do not exist.

- [x] **Step 6: Add the fallible Phoenix loader**

Make the signed payload length the input to the PSP load map and validate both fixed slices before indexing:

```rust
pub fn try_load_m2c(image: FirmwareImage) -> Result<Self, FirmwareError> {
    let image_len = image.payload_size() as usize;
    let initialized_data_end =
        (M2C_INITIALIZED_DATA_VADDR + LOW_VMA_FILE_OFFSET + M2C_INITIALIZED_DATA_LEN) as usize;
    let needed = initialized_data_end.max(SEG_B_FILE_START as usize);
    if image_len < needed {
        return Err(FirmwareError::Truncated {
            offset: image_len,
            needed,
            got: image.bytes().len(),
        });
    }

    let loaded_bytes = image.bytes()[..image_len].to_vec();
    // Existing load_m2c body, using loaded_bytes for the bus and
    // image_len as u32 for psp_load_map.
    Ok(Self { cpu, bus, entry: RESET_ENTRY, symbols, host_mailbox: HostMailbox::new() })
}

pub fn load_m2c(image: FirmwareImage) -> Self {
    Self::try_load_m2c(image).expect("known-good Phoenix firmware image")
}
```

Do not change the PSP offsets or image-derived boot state.

- [x] **Step 7: Share one boot loop between standalone and borrowed-device stepping**

Move the existing loop body into:

```rust
fn boot_to_idle_on(
    &mut self,
    max_instrs: u64,
    mut step_cpu: impl FnMut(&mut Cpu, &mut Bus) -> Step,
) -> IdleReport
```

Keep the loop body byte-for-byte equivalent except for replacing:

```rust
let step = self.cpu.step(&mut self.bus);
```

with:

```rust
let step = step_cpu(&mut self.cpu, &mut self.bus);
```

Expose the two entry points:

```rust
pub fn boot_to_idle(&mut self, max_instrs: u64) -> IdleReport {
    self.boot_to_idle_on(max_instrs, Cpu::step)
}

pub fn boot_to_idle_with_device(
    &mut self,
    device: &mut crate::device::DeviceState,
    max_instrs: u64,
) -> IdleReport {
    self.boot_to_idle_on(max_instrs, |cpu, bus| cpu.step_with_device(bus, device))
}
```

- [x] **Step 8: Run Task 1 GREEN gates**

Run:

```bash
XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
  cargo test --lib firmware::image::tests
XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
  cargo test --lib m2c_loader_rejects_an_image_without_segment_b
XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
  cargo test --lib m2c_boot_with_device_borrows_and_preserves_array_state
XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
  cargo test --lib m2c_boot_publishes_alive_state_through_host_sram
```

Expected: all selected tests pass and the pre-existing standalone boot guard remains green.

- [x] **Step 9: Commit Task 1**

```bash
git add src/firmware/image.rs src/firmware/mod.rs src/firmware/boot_tests/guards.rs
git commit -m "feat(firmware): expose fallible shared-device boot"
```

---

### Task 2: Public Firmware Component C ABI

**Files:**
- Modify: `crates/xdna-emu-ffi/src/lib.rs`
- Modify: `crates/xdna-emu-ffi/src/firmware.rs`
- Modify: `crates/xdna-emu-ffi/tests/tier_c_completeness.rs`
- Modify: `include/xdna_emu.h`

**Interfaces:**
- Consumes: `FirmwareProcessor::try_load_m2c`, `FirmwareProcessor::boot_to_idle_with_device`, `Bus::host_sram_load32`, `Bus::host_sram_store32`, and `NpuBackend::as_interpreter_mut`.
- Produces: `xdna_emu_load_firmware`, `xdna_emu_boot_firmware`, `xdna_emu_firmware_read_host_sram32`, and `xdna_emu_firmware_write_host_sram32`.

- [x] **Step 1: Add C-ABI signature RED checks**

Extend `tier_c_completeness.rs`:

```rust
#[test]
fn firmware_component_signatures_match_the_public_contract() {
    type FnLoad = unsafe extern "C" fn(*mut XdnaEmuHandle, *const u8, u64) -> XdnaEmuResult;
    type FnBoot = unsafe extern "C" fn(*mut XdnaEmuHandle, u64) -> XdnaEmuResult;
    type FnRead = unsafe extern "C" fn(*mut XdnaEmuHandle, u32, *mut u32) -> XdnaEmuResult;
    type FnWrite = unsafe extern "C" fn(*mut XdnaEmuHandle, u32, u32) -> XdnaEmuResult;

    let _: FnLoad = xdna_emu_load_firmware;
    let _: FnBoot = xdna_emu_boot_firmware;
    let _: FnRead = xdna_emu_firmware_read_host_sram32;
    let _: FnWrite = xdna_emu_firmware_write_host_sram32;
}
```

- [x] **Step 2: Add FFI behavior RED tests**

In `crates/xdna-emu-ffi/src/firmware.rs`, add a test builder whose declared payload includes the fixed Phoenix slices:

```rust
fn synthetic_m2c_image() -> Vec<u8> {
    let mut raw = vec![0u8; 0x2d100];
    raw[0x10..0x14].copy_from_slice(b"$PS1");
    let declared = raw.len() as u32;
    raw[0x14..0x18].copy_from_slice(&declared.to_le_bytes());
    raw
}
```

Add tests for these exact contracts:

```rust
fn last_error() -> String {
    let mut buffer = [0i8; 256];
    let len = unsafe { crate::xdna_emu_get_error(buffer.as_mut_ptr(), buffer.len() as u64) };
    let bytes = unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), len as usize) };
    String::from_utf8(bytes.to_vec()).expect("UTF-8 LAST_ERROR")
}

#[test]
fn firmware_api_rejects_nulls_missing_state_and_zero_budget() {
    let bytes = synthetic_m2c_image();
    assert_eq!(
        unsafe { xdna_emu_load_firmware(std::ptr::null_mut(), bytes.as_ptr(), bytes.len() as u64) },
        XdnaEmuResult::InvalidHandle,
    );

    let handle = unsafe { xdna_emu_create() };
    assert_eq!(
        unsafe { xdna_emu_load_firmware(handle, std::ptr::null(), bytes.len() as u64) },
        XdnaEmuResult::NullPointer,
    );
    let mut untouched = 0xfeed_face;
    assert_eq!(
        unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut untouched) },
        XdnaEmuResult::ExecutionError,
    );
    assert_eq!(untouched, 0xfeed_face);
    assert_eq!(
        unsafe { xdna_emu_boot_firmware(handle, 1) },
        XdnaEmuResult::ExecutionError,
    );
    assert_eq!(
        unsafe { xdna_emu_load_firmware(handle, bytes.as_ptr(), bytes.len() as u64) },
        XdnaEmuResult::Success,
    );
    assert_eq!(
        unsafe { xdna_emu_boot_firmware(handle, 0) },
        XdnaEmuResult::ExecutionError,
    );
    assert_eq!(
        unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, std::ptr::null_mut()) },
        XdnaEmuResult::NullPointer,
    );
    assert!(!last_error().is_empty());
    unsafe { xdna_emu_destroy(handle) };
}

#[test]
fn failed_load_preserves_the_existing_processor_and_array() {
    let bytes = synthetic_m2c_image();
    let handle = unsafe { xdna_emu_create() };
    unsafe {
        (*handle)
            .backend
            .as_interpreter_mut()
            .expect("interpreter")
            .device_mut()
            .write_tile_register(4, 0, 0x000f_ff20, 1);
    }

    assert_eq!(
        unsafe { xdna_emu_load_firmware(handle, bytes.as_ptr(), bytes.len() as u64) },
        XdnaEmuResult::Success,
    );
    assert_eq!(
        unsafe { xdna_emu_firmware_write_host_sram32(handle, 0x030b_f000, 0x1234_5678) },
        XdnaEmuResult::Success,
    );

    let malformed = [0u8; 0x18];
    assert_eq!(
        unsafe { xdna_emu_load_firmware(handle, malformed.as_ptr(), malformed.len() as u64) },
        XdnaEmuResult::ParseError,
    );

    let mut value = 0;
    assert_eq!(
        unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut value) },
        XdnaEmuResult::Success,
    );
    assert_eq!(value, 0x1234_5678);
    assert_eq!(
        unsafe {
            (*handle)
                .backend
                .as_interpreter_mut()
                .expect("interpreter")
                .device_mut()
                .read_tile_register(4, 0, 0x000f_ff20)
        },
        1,
    );
    unsafe { xdna_emu_destroy(handle) };
}

#[test]
fn context_reset_preserves_loaded_firmware_state() {
    let bytes = synthetic_m2c_image();
    let handle = unsafe { xdna_emu_create() };
    assert_eq!(
        unsafe { xdna_emu_load_firmware(handle, bytes.as_ptr(), bytes.len() as u64) },
        XdnaEmuResult::Success,
    );
    assert_eq!(
        unsafe { xdna_emu_firmware_write_host_sram32(handle, 0x030b_f000, 0xcafe_babe) },
        XdnaEmuResult::Success,
    );
    assert_eq!(
        unsafe { crate::xdna_emu_reset_context(handle, 0) },
        XdnaEmuResult::Success,
    );
    let mut value = 0;
    assert_eq!(
        unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut value) },
        XdnaEmuResult::Success,
    );
    assert_eq!(value, 0xcafe_babe);
    unsafe { xdna_emu_destroy(handle) };
}
```

Use the existing test backend to prove unsupported-backend rejection precedes parsing:

```rust
#[test]
fn firmware_api_rejects_non_interpreter_backend_before_parsing() {
    let handle = unsafe { xdna_emu_create() };
    unsafe {
        (*handle).backend = Box::new(crate::backend::mock::MockBackend::default());
    }
    let malformed = [0u8; 1];
    assert_eq!(
        unsafe { xdna_emu_load_firmware(handle, malformed.as_ptr(), malformed.len() as u64) },
        XdnaEmuResult::ExecutionError,
    );
    assert!(last_error().contains("does not support firmware execution"));
    unsafe { xdna_emu_destroy(handle) };
}
```

- [x] **Step 3: Add the firmware-gated end-to-end RED test**

Add a test that skips only when `XDNA_FIRMWARE` is absent:

```rust
#[test]
fn public_ffi_boots_pinned_firmware_and_exposes_genuine_alive_state() {
    let Ok(path) = std::env::var("XDNA_FIRMWARE") else {
        eprintln!("skip: set XDNA_FIRMWARE");
        return;
    };
    let digest = std::process::Command::new("sha256sum")
        .arg(&path)
        .output()
        .expect("run sha256sum");
    assert!(digest.status.success());
    assert_eq!(
        String::from_utf8(digest.stdout)
            .expect("sha256sum output")
            .split_whitespace()
            .next(),
        Some("d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e"),
    );
    let bytes = std::fs::read(path).expect("read firmware");
    let handle = unsafe { xdna_emu_create() };
    unsafe {
        (*handle)
            .backend
            .as_interpreter_mut()
            .expect("interpreter")
            .device_mut()
            .write_tile_register(4, 0, 0x000f_ff20, 1);
    }

    assert_eq!(
        unsafe { xdna_emu_load_firmware(handle, bytes.as_ptr(), bytes.len() as u64) },
        XdnaEmuResult::Success,
    );
    assert_eq!(
        unsafe { xdna_emu_boot_firmware(handle, 200_000) },
        XdnaEmuResult::Success,
    );

    let mut value = 0;
    assert_eq!(
        unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_b020, &mut value) },
        XdnaEmuResult::Success,
    );
    assert_eq!(value, 0x5550_4e5f);
    assert_eq!(
        unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut value) },
        XdnaEmuResult::Success,
    );
    assert_eq!(value, 0x030b_b000);
    assert_eq!(
        unsafe { xdna_emu_firmware_write_host_sram32(handle, 0x030b_f000, 0) },
        XdnaEmuResult::Success,
    );
    assert_eq!(
        unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut value) },
        XdnaEmuResult::Success,
    );
    assert_eq!(value, 0);
    assert_eq!(
        unsafe {
            (*handle)
                .backend
                .as_interpreter_mut()
                .expect("interpreter")
                .device_mut()
                .read_tile_register(4, 0, 0x000f_ff20)
        },
        1,
    );
    unsafe { xdna_emu_destroy(handle) };
}
```

- [x] **Step 4: Run Task 2 tests and record RED**

Run:

```bash
cargo test -p xdna-emu-ffi firmware_component_signatures_match_the_public_contract
cargo test -p xdna-emu-ffi firmware_api_rejects_nulls_missing_state_and_zero_budget
XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
  cargo test -p xdna-emu-ffi public_ffi_boots_pinned_firmware_and_exposes_genuine_alive_state
```

Expected: compilation fails because the handle field and four exported functions do not exist.

- [x] **Step 5: Add firmware ownership to the handle**

Add:

```rust
use xdna_emu_core::firmware::FirmwareProcessor;
```

and:

```rust
pub(crate) firmware: Option<FirmwareProcessor>,
```

Initialize it to `None` in `xdna_emu_create`. Do not touch it in `xdna_emu_reset_context`.

- [x] **Step 6: Implement atomic byte-oriented loading**

Implement `xdna_emu_load_firmware` in the existing FFI firmware module with this validation order:

```rust
if handle.is_null() {
    set_last_error("xdna_emu_load_firmware: null handle".to_string());
    return XdnaEmuResult::InvalidHandle;
}
if firmware_data.is_null() {
    set_last_error("xdna_emu_load_firmware: null firmware_data".to_string());
    return XdnaEmuResult::NullPointer;
}
let Ok(size) = usize::try_from(firmware_size) else {
    set_last_error("xdna_emu_load_firmware: firmware_size does not fit usize".to_string());
    return XdnaEmuResult::ParseError;
};
let handle = &mut *handle;
if handle.backend.as_interpreter_mut().is_none() {
    set_last_error("xdna_emu_load_firmware: backend does not support firmware execution".to_string());
    return XdnaEmuResult::ExecutionError;
}
let bytes = std::slice::from_raw_parts(firmware_data, size);
let processor = match FirmwareImage::parse(bytes).and_then(FirmwareProcessor::try_load_m2c) {
    Ok(processor) => processor,
    Err(error) => {
        set_last_error(format!("xdna_emu_load_firmware: {error}"));
        return XdnaEmuResult::ParseError;
    }
};
handle.firmware = Some(processor);
XdnaEmuResult::Success
```

The assignment must remain after every fallible operation.

- [x] **Step 7: Implement bounded borrowed-device boot**

Validate the budget and split-borrow the handle fields:

```rust
let XdnaEmuHandle { backend, firmware, .. } = &mut *handle;
let Some(processor) = firmware.as_mut() else {
    set_last_error("xdna_emu_boot_firmware: no firmware loaded".to_string());
    return XdnaEmuResult::ExecutionError;
};
let Some(engine) = backend.as_interpreter_mut() else {
    set_last_error("xdna_emu_boot_firmware: backend does not support firmware execution".to_string());
    return XdnaEmuResult::ExecutionError;
};
let report = processor.boot_to_idle_with_device(engine.device_mut(), max_instructions);
if !report.reached_idle {
    set_last_error(format!(
        "xdna_emu_boot_firmware: stopped before idle: pc={:#010x}, instructions={}, \
         unresolved_spin={:?}, unknown_op={:?}",
        report.last_pc,
        report.instrs_executed,
        report.unresolved_spin,
        report.unknown_op,
    ));
    return XdnaEmuResult::ExecutionError;
}
XdnaEmuResult::Success
```

Reject `max_instructions == 0` before borrowing state. Do not enable `HostMailbox`.

- [x] **Step 8: Implement direct host-SRAM access**

For both accessors, validate the handle, output pointer where present, backend support, and loaded processor. Read and write only through:

```rust
processor.bus.host_sram_load32(device_address)
processor.bus.host_sram_store32(device_address, value)
```

Write `value_out` only after all validation succeeds. Do not translate BAR offsets or copy SRAM into handle-owned storage.

- [x] **Step 9: Declare and document the four C functions**

Add the exact approved declarations to `include/xdna_emu.h` after the PDI-loading API. State that addresses are Phoenix device addresses, that loading copies caller bytes, that boot is bounded, and that the SRAM functions use firmware-programmed aliases.

- [x] **Step 10: Run Task 2 GREEN gates**

Run:

```bash
XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
  cargo test -p xdna-emu-ffi
cc -std=c11 -Wall -Werror -fsyntax-only -include include/xdna_emu.h -x c /dev/null
```

Expected: every FFI test passes, including the pinned real-image path, and the public header is valid C11.

- [x] **Step 11: Commit Task 2**

```bash
git add crates/xdna-emu-ffi/src/lib.rs \
  crates/xdna-emu-ffi/src/firmware.rs \
  crates/xdna-emu-ffi/tests/tier_c_completeness.rs \
  include/xdna_emu.h
git commit -m "feat(ffi): expose Phoenix firmware component"
```

---

### Task 3: Correct the Firmware Wiring Record

**Files:**
- Modify: `src/firmware/mod.rs`
- Modify: `crates/xdna-emu-ffi/src/firmware.rs`
- Modify: `docs/arch/firmware-array-plugin-wiring.md`
- Modify: `docs/fidelity-gaps/host-firmware-dispatch.md`

**Interfaces:**
- Consumes: the proven natural boot, shared-device processor, open-driver BAR2/BAR4 map, and current SHIM-only plugin boundary.
- Produces: an architecture record that clearly separates proven PSP handoff effects, proven host apertures, and the unresolved BAR4-to-controller interrupt bridge.

- [ ] **Step 1: Update source-level module contracts**

Change `src/firmware/mod.rs` to say that array MMIO is routed through a borrowed `DeviceState` and the current missing boundary is management-mailbox interrupt delivery.

Change the FFI firmware-module preamble to distinguish:

- the new real-processor component API;
- the retained `xdna_emu_assign_partition` synthetic SHIM hook;
- the fact that the hook is not the unmodified-driver firmware path.

- [ ] **Step 2: Supersede stale architecture claims with current evidence**

In `docs/arch/firmware-array-plugin-wiring.md`:

- mark the 2026-07-07 record as historical and superseded by the 2026-07-27 state;
- state that unmodified 1502_00 now reaches scheduler `waiti` and publishes `_NPU`;
- state that the modeled PSP handoff comprises load offset `0x5c`, low overlay, initialized D-side data, segment B at `0x08b00000`, reset MMU state/page table, and reset-preconfigured I2X slot 15;
- replace the old claim that `Bus` owns `DeviceState` with per-step borrowing from `InterpreterEngine`;
- replace the old `0x27200170/174/178` management-mailbox identification with the proven BAR2/BAR4 contract:

```text
BAR2 0x03080000:
  +0x3b000 descriptor
  +0x3c000 X2I ring
  +0x3d000 I2X ring
  +0x3f000 FW_ALIVE

BAR4 0x030c0000:
  +0x2c000 X2I tail
  +0x2c004 X2I head
  +0x2d000 I2X tail
  +0x2d004 I2X head
  +0x2d008 IOHUB status/clear
```

- state that `0x27200170/174/178` is an unrelated earlier internal queue;
- state that the unresolved chain is BAR4 X2I-tail publication to slot-14 pending state to controller source 46 to Xtensa interrupt 0;
- state that the existing plugin is an XRT SHIM replacement and cannot validate an unmodified kernel driver; that proof requires a virtual Phoenix PCI frontend below the driver.

- [ ] **Step 3: Update the fidelity-gap row**

Change the go-alive row in `docs/fidelity-gaps/host-firmware-dispatch.md` so its remaining work names the raw public component FFI as complete and the virtual PCI driver boundary plus BAR4 interrupt bridge as incomplete. Do not claim current XRT plugin routing can prove driver equivalence.

- [ ] **Step 4: Run the stale-claim audit**

Run:

```bash
rg -n "Bus owns|0x27200170|27200170|SMU/PSP column power|SHIM plugin|unmodified kernel driver" \
  src/firmware/mod.rs crates/xdna-emu-ffi/src/firmware.rs \
  docs/arch/firmware-array-plugin-wiring.md docs/fidelity-gaps/host-firmware-dispatch.md
```

Expected: each surviving hit is explicitly marked historical, unrelated, synthetic, or incomplete; none is presented as the current firmware path.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/firmware/mod.rs crates/xdna-emu-ffi/src/firmware.rs \
  docs/arch/firmware-array-plugin-wiring.md \
  docs/fidelity-gaps/host-firmware-dispatch.md
git commit -m "docs(firmware): correct boot and mailbox wiring"
```

---

### Task 4: Full Local Verification and Component Handoff

**Files:**
- Verify only; modify a Task 1-3 file only if a gate exposes a defect.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: a clean, committed component milestone ready for the post-alive hardware capture.

- [ ] **Step 1: Run formatter and focused FFI suite**

Run:

```bash
cargo fmt --all
XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
  cargo test -p xdna-emu-ffi
```

Expected: formatter completes and every FFI test passes.

- [ ] **Step 2: Run the full required library gate**

Run from the activated NPU environment:

```bash
source /home/triple/npu-work/toolchain-build/activate-npu-env.sh
XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
MLIR_AIE_PATH=/home/triple/npu-work/mlir-aie \
AIE_RT_PATH=/home/triple/npu-work/aie-rt/driver/src \
LLVM_AIE_PATH=/home/triple/npu-work/llvm-aie \
  cargo test --lib
```

Expected: zero failures.

- [ ] **Step 3: Run static gates**

Run:

```bash
cargo fmt --all --check
cc -std=c11 -Wall -Werror -fsyntax-only -include include/xdna_emu.h -x c /dev/null
git diff --check
git status --short --branch
```

Expected: formatter, header, and whitespace checks pass; the worktree contains no uncommitted changes.

- [ ] **Step 4: Review the exact component diff against the approved design**

Run:

```bash
git diff 1e880796..HEAD -- \
  src/firmware/image.rs src/firmware/mod.rs src/firmware/boot_tests/guards.rs \
  crates/xdna-emu-ffi/src/lib.rs crates/xdna-emu-ffi/src/firmware.rs \
  crates/xdna-emu-ffi/tests/tier_c_completeness.rs include/xdna_emu.h \
  docs/arch/firmware-array-plugin-wiring.md \
  docs/fidelity-gaps/host-firmware-dispatch.md
```

Confirm from the diff:

- no environment or filesystem firmware discovery in production code;
- no SHA check in production code;
- no `NpuBackend` expansion;
- no copied `DeviceState` or shadow SRAM;
- no `HostMailbox` enablement;
- no BAR4/controller inference;
- all non-success C paths set `LAST_ERROR`;
- failed loads preserve prior processor state;
- the genuine descriptor/alive values are asserted only after natural idle.

- [ ] **Step 5: Advance to the next evidence milestone**

Keep the long-running firmware-equivalence goal active. The next work item is a hardware-grounded capture of one genuine post-alive management transaction that identifies:

```text
BAR4 X2I-tail publication
  -> subordinate slot-14 pending state
  -> active controller source 46
  -> Xtensa interrupt bit 0
  -> firmware event (6,4)
```

Do not implement that bridge until the capture pins the missing register alias and subordinate source.
