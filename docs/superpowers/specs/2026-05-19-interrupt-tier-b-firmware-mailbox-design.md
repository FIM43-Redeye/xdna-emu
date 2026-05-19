# Interrupt Tier B — Firmware Async-Event Host Delivery (Plumbing + INSTR_ERROR)

**Status:** Spec, awaiting plan.
**Predecessor:** [Tier A interrupt closeout](2026-05-19-interrupt-l2-closeout-design.md) (shipped 2026-05-19, commits `8638f7a..b0115e3` on `dev`).
**Tracking context:** [findings note](../findings/2026-05-19-interrupt-tier-b-firmware-delivery.md).

## 1. Purpose & scope

Tier A landed the AIE interrupt path to the firmware-notify boundary (events
generate, L1 latches, broadcasts propagate, L2 raises NPI lines). Tier B is the
firmware async-event mailbox path that surfaces those errors to the host —
the surface a real XRT consumer actually reads via
`ioctl(DRM_AMDXDNA_GET_ARRAY, DRM_AMDXDNA_HW_LAST_ASYNC_ERR)`.

The XRT plugin replaces the kernel driver entirely, so there is no
`amdxdna` module loaded against the emulator. The user-facing contract
that matters is the ioctl, and our plugin currently returns
`num_element = 0` for it (`platform_emu.cpp:1189-1193`, comment: "No async
errors in emulation"). Tier B replaces that no-op with real synthesized
errors derived from Tier A's event-generation path.

### 1.1 In scope (this spec)

- **Plumbing for the full async-error pipeline**:
  - A `DeviceState::async_errors: AsyncErrorSink` with three output surfaces.
  - Per-column ring buffers in driver-wire format (`aie_err_info` + `aie_error[]`).
  - A last-error cache (mirrors driver `amdxdna_async_err_cache`).
  - Optional push callback for observers (visual debugger, tests).
  - Driver-mirror categorization tables (event_id → category → err_code).
  - FFI surface (5 new symbols).
  - Plugin wire-up: `DRM_AMDXDNA_HW_LAST_ASYNC_ERR` reads the cache.
- **`INSTR_ERROR` (event 69) as the demonstrating producer.** Tier A already
  routes this event; Tier B's effects.rs hook also records it as an async error.
- **Tests:** unit per layer + one control-packet-driven integration test.

### 1.2 Out of scope (follow-ups, separate specs)

- **Detection producers for events Tier A does not currently fire.** Driver
  categorizes ~50 error events across core/mem/memtile/shim modules
  (`aie2_error.c:89-150`). Each missing producer (DMA bounds, parity, ECC,
  lock-acquire-on-released, stream overruns, etc.) becomes its own
  detection-spec follow-up. They all reuse the plumbing landed here by
  calling `device.async_errors.record_error(...)` from their detection site.
- **Real-clock timestamp mode.** Cycle-as-microsecond is used (deterministic,
  matches trace path); a wall-clock mode is left for later.
- **Bridge test addition.** No new bridge fixture in this spec; the first
  detection-spec follow-up that adds a real producer can add a bridge test.
- **Tier C TDR / context-recovery.** Tier B records async errors; emulating
  the driver's `aie2_rq_handle_idle_ctx` periodic timer & context restart
  path is a separate Tier C effort.

## 2. Architectural decisions (locked during brainstorming)

| Choice | Decision | Rationale |
|---|---|---|
| **Boundary depth** | Both surfaces: cache (for current ioctl) + raw mailbox ring (for future real-driver attachment). | Maya: emu is designed to wire into XRT as a faux NPU; ring fidelity supports that future, cache satisfies today's consumer. |
| **Trigger scope** | All error events the driver categorizes (full coverage, plumbing-only here; producers per-class follow). | "YAGNI only works if it's genuinely not in scope; the scope of a full emulator is BIG." |
| **Hook point** | Event-generation in `apply_tile_local_effects` (alongside existing Tier A calls), NOT after L1 latch or at L2 sink. | Mirrors HW: firmware mailbox path is independent of AIE L1/L2 enable; an error fires both paths in parallel. Approach 2 silently skips errors when L1 isn't enabled (wrong); Approach 3 loses event_id at the L2 boundary (would need re-derivation). |
| **Push callback** | Yes, registerable C function pointer. | Cheap to add now; non-trivial to retrofit. Visual debugger and tests benefit directly. |
| **Validation** | Unit tests + one control-packet-driven integration test. No bridge test. | Faithful to how real consumers drive the path (control-packet event-generation), validates end-to-end without depending on detection-spec follow-ups. |
| **Categorization tables location** | `crates/xdna-archspec/src/aie2/async_errors.rs` (data). | Tables are hardware/driver constants, not emulator runtime state. Same pattern as `trace_events::core_events`. |
| **Tables encoding** | `&[EventCategory]` const slices, linear scan. | Elegance / DRY over speed; called once per error fire, never hot. Match-arm form deferred. |
| **Ring storage** | `Box<[u8; ASYNC_BUF_SIZE]>` per column. | Byte layout pinned for wire-format integrity. |
| **Timestamp** | `ts_us = cycle / 1000` (treats simulated cycle as nanoseconds at ~1 GHz). | Deterministic; unit math is correct for the eventual wall-clock mode without re-spec. Non-determinism is the killer; cycle-derived time avoids it. |
| **Plugin symbol resolution** | `resolve_required` for `xdna_emu_get_last_async_error`. | Fail-loud on stale `.so` rather than silent "no errors" reports — emu staleness has caused phantom bugs before; the rebuild-plugin discipline depends on loud failure. |
| **Cache granularity** | Single `amdxdna_async_error` per handle, mutex-protected, last-write-wins. | Matches driver: one `amdxdna_async_err_cache` per device. |
| **Per-column rings** | One 8 KB ring per column (5 cols on NPU1). | Matches driver: one `async_event` slot per column registered with firmware. |
| **INSTR_ERROR categorization** | Add event 69 to `CORE_EVENT_CAT` as `AieErrorCategory::Instruction` (emu-specific extension; driver's table omits it). | Driver's `aie_get_error_category` returns `UNKNOWN` for event 69, but mlir-aie's event table names it `INSTR_ERROR` and our `raise_instr_error` already fires it. Categorizing as Instruction is more correct per architectural naming; documented inline as an emu-specific divergence so a future driver-table update doesn't silently re-introduce ambiguity. |

## 3. Architecture

A new `async_errors` subsystem in `xdna-emu-core::device` sits parallel to
the existing Tier A `interrupts` subsystem. When an error-category event is
generated, the effect-application path emits a record to the subsystem
*in addition to* the existing L1/L2 latch. The subsystem maintains three
output surfaces:

1. **Last-error cache** — one `amdxdna_async_error` per handle,
   mutex-protected, last-write-wins. Mirrors driver
   `amdxdna_async_err_cache`. Consumed by the plugin's
   `DRM_AMDXDNA_HW_LAST_ASYNC_ERR` ioctl.
2. **Per-column mailbox rings** — five 8 KB buffers (one per column on
   NPU1), holding `aie_err_info` header + `aie_error[]` payload entries
   in driver-wire format. Byte-compatible with what firmware would DMA
   into the host-allocated async-event message buffer. Unused by today's
   plugin; reserved for any future real-driver attachment.
3. **Push callback** — optional C function pointer registered via FFI,
   invoked synchronously when a record is added.

FFI exposes getters for surfaces 1 and 2, a registration function for 3,
plus a clear helper. The XRT plugin's
`DRM_AMDXDNA_HW_LAST_ASYNC_ERR` ioctl handler reads surface 1. Surface 2
lives unused for now but is wire-correct.

```
                  ┌─────────────────────────────────────────────┐
                  │ apply_tile_local_effects (is_event_generate) │
                  │   - em.generate_event(event_id)              │
                  │   - tile.seed_broadcasts_for_event(event_id) │  [Tier A]
                  │   - tile.tap_l1_interrupt(event_id)          │  [Tier A]
                  │   - if is_error_event(event_id, tile.kind):  │  [Tier B, NEW]
                  │       device.async_errors.record_error(...)  │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │            AsyncErrorSink                    │
                  │  ┌──────────────┐  ┌──────────────────────┐  │
                  │  │  Last-error  │  │ Per-column rings     │  │
                  │  │  cache       │  │ (5 × 8KB, wire fmt)  │  │
                  │  └──────────────┘  └──────────────────────┘  │
                  │  ┌──────────────────────────────────────────┐ │
                  │  │ Newly-recorded queue (for FFI drain)     │ │
                  │  └──────────────────────────────────────────┘ │
                  └─────┬────────────────┬──────────────────┬───┘
                        │                │                  │
              ┌─────────▼─────┐ ┌────────▼─────────┐ ┌─────▼────────────┐
              │ FFI getter:   │ │ FFI getter:      │ │ FFI callback     │
              │ get_last_async│ │ read_async_event │ │ (synchronous on  │
              │ _error        │ │ _ring(col, ...)  │ │  record_error)   │
              └────┬──────────┘ └─────────┬────────┘ └──────┬───────────┘
                   │                      │                  │
         ┌─────────▼────────┐     ┌───────▼─────┐    ┌───────▼────────┐
         │ XRT plugin ioctl │     │ (unused;    │    │ Visual debugger│
         │ DRM_AMDXDNA_HW_  │     │ reserved    │    │ / test harness │
         │ LAST_ASYNC_ERR   │     │ for future  │    │                │
         │                  │     │ driver shim)│    │                │
         └──────────────────┘     └─────────────┘    └────────────────┘
```

## 4. Components

| Unit | Path | Responsibility |
|---|---|---|
| `categorize` data | `crates/xdna-archspec/src/aie2/async_errors.rs` (NEW) | Driver-mirror const tables (events → `AieErrorCategory`, module → `AmdxdnaErrorModule`), `event_to_category(event_id, mod_type) -> Option<AieErrorCategory>`, `is_error_event(event_id, mod_type) -> bool`, `build_err_code` and `build_ex_err_code` const fns (mirror driver macros `AMDXDNA_ERROR_CODE_BUILD` / `AMDXDNA_ERROR_EXTRA_CODE_BUILD`). |
| `AsyncErrorRecord` types | `src/device/async_errors/types.rs` (NEW) | `#[repr(C)]` mirrors of `aie_error` (12 B), `aie_err_info` header (12 B), `amdxdna_async_error` (24 B) plus compile-time size assertions and `MAX_ERRORS_PER_RING` constant (= 681). |
| `AsyncErrorSink` | `src/device/async_errors/sink.rs` (NEW) | Owns cache, per-column rings, newly-recorded drain queue, optional Rust-side observer hook. `record_error(col, row, mod_type, event_id, cycle)` is the single mutation entry point. `clear()` zeros everything. |
| `AsyncErrorSink` module root | `src/device/async_errors/mod.rs` (NEW) | Re-exports `types`, `sink`. Two-module structure mirrors `device/interrupts/`. |
| `DeviceState` integration | `src/device/state/mod.rs` (one field add) + `src/device/state/effects.rs` (one branch in `is_event_generate`) | Adds `async_errors: AsyncErrorSink` field; calls `record_error` from the existing effect-application code. |
| FFI surface | `crates/xdna-emu-ffi/src/async_errors.rs` (NEW) + register in `lib.rs` | 5 new symbols (Section 7). |
| `XdnaEmuHandle` extension | `crates/xdna-emu-ffi/src/lib.rs` | One field: `async_callback: Option<(XdnaEmuAsyncErrorCallback, *mut c_void)>`. |
| Plugin ioctl wire-up | `xrt-plugin/src/platform_emu.cpp:~1189` | Replace the `arg.num_element = 0;` no-op with a call to `xdna_emu_get_last_async_error`. |
| Plugin symbol bind | `xrt-plugin/src/transport_inprocess.cpp` (one `resolve_required`) + `transport_inprocess.h` (declare typedef + member) | Resolves the new FFI symbol. |

**Why `archspec` for tables:** event→category mapping is identical for every
NPU1 device, derived from the driver's static tables in `aie2_error.c`.
Lives in archspec next to other AIE2 architectural constants. Same pattern
as `trace_events::core_events`.

**Why single `AsyncErrorSink` (not split per-tile):** cache and rings are
device-scope; per-column rings are keyed by col, not owned by per-tile
state. Single owner avoids cross-tile mutation that would force every
error-firing site to know about array layout.

## 5. Data flow

A core executes a `MOVE` to an invalid memory address. The interpreter's
instruction error handler calls `raise_instr_error`, which already calls
`generate_event(INSTR_ERROR)` on the core EventModule and seeds Tier A's
L1/L2 path. The next dispatch tick processes the pending event-generate
effect:

```rust
// src/device/state/effects.rs -- in apply_tile_local_effects, is_event_generate arm
em.generate_event(event_id);                                          // existing
tile.seed_broadcasts_for_event(event_id);                             // existing, Tier A
tile.tap_l1_interrupt(event_id);                                      // existing, Tier A

// NEW, Tier B:
let mod_type = tile.tile_kind.to_aie_module_type();
if xdna_archspec::aie2::async_errors::is_error_event(event_id, mod_type) {
    let cycle = self.array.current_cycle();
    self.async_errors.record_error(tile.col, tile.row, mod_type, event_id, cycle);
}
```

`record_error` does three things atomically:

```rust
fn record_error(&mut self, col: u8, row: u8, mod_type: AieModuleType, event_id: u8, cycle: u64) {
    // 1. Build the wire-format aie_error record (12 B) and append to col's ring.
    let record = AieError { row, col, mod_type: mod_type as u32, event_id, ..Default::default() };
    if let Err(Overflow) = self.rings[col as usize].push(record) {
        // Set ret_code so a consumer can detect the dropped tail.
        self.rings[col as usize].header_mut().ret_code = RET_CODE_OVERFLOW;
    }

    // 2. Categorize and update the last-error cache.
    let category = event_to_category(event_id, mod_type)
        .expect("is_error_event gated this; bug if None");
    let error_num = category_to_error_num(category);
    let module = mod_type_to_amdxdna_module(mod_type);
    let err_code = build_err_code(Severity::NonFatal, module, Class::Aie, error_num);
    let ex_err_code = build_ex_err_code(row, col);
    let ts_us = cycle / 1000;  // see Section 2: deterministic, ~1 GHz scaling
    self.cache = Some(AmdxdnaAsyncError { err_code, ts_us, ex_err_code });

    // 3. Queue for FFI drain (push-callback path).
    self.newly_recorded.push_back(self.cache.unwrap());
}
```

**Drain path (FFI → callback bridge, design option (b) per Section 5 of brainstorming):**

After each `engine.step()` in `xdna_emu_run`, the FFI loop drains newly-recorded
records and fires the registered callback if present:

```rust
// crates/xdna-emu-ffi/src/execution.rs -- inside xdna_emu_run loop, after handle.engine.step()
if let Some((cb, user_data)) = handle.async_callback {
    let device_state = handle.engine.device_mut();
    for record in device_state.async_errors.drain_newly_recorded() {
        cb(&record as *const _ as *const XdnaEmuAsyncError, user_data);
    }
}
```

Pattern mirrors `flush_trace_to_host` — the FFI layer drives observation
between engine steps.

**Reset semantics:**

- `xdna_emu_reset_context` (existing) clears `async_errors` alongside other
  per-context state.
- `xdna_emu_clear_async_errors` (new) clears just the sink without
  touching tile state — for tests that want to re-arm observation.

**Cycle-as-timestamp:** documented in the `record_error` source and in the
FFI doc. `ts_us = cycle / 1000` is unit-correct for the eventual
wall-clock mode (silicon runs at ~1 GHz, so cycle≈ns, cycle/1000≈µs); a
real-clock mode is a separate spec.

## 6. Wire format

Driver-source-of-truth types, mirrored byte-exact:

```rust
// crates/xdna-emu-core/src/device/async_errors/types.rs

/// Mirrors `struct aie_error` in `xdna-driver/src/driver/amdxdna/aie2_error.c:56-64`.
/// 12 bytes. NOT packed -- driver comment: "Don't pack, unless XAIE side changed".
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct AieError {
    pub row: u8,
    pub col: u8,
    pub reserved_0: u16,
    pub mod_type: u32,       // AieModuleType: CORE=0, MEM=1, PL=2, MEM_TILE=3
    pub event_id: u8,
    pub reserved_1: u8,
    pub reserved_2: u16,
}
const _: () = assert!(std::mem::size_of::<AieError>() == 12);

/// Header for the ring buffer; followed by err_cnt × AieError.
/// Mirrors `struct aie_err_info` in `aie2_error.c:66-71`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct AieErrInfoHeader {
    pub err_cnt: u32,
    pub ret_code: u32,
    pub rsvd: u32,
}
const _: () = assert!(std::mem::size_of::<AieErrInfoHeader>() == 12);

pub const ASYNC_BUF_SIZE: usize = 8 * 1024;     // SZ_8K from driver
pub const MAX_ERRORS_PER_RING: usize =
    (ASYNC_BUF_SIZE - std::mem::size_of::<AieErrInfoHeader>())
    / std::mem::size_of::<AieError>();           // = 681

pub const RET_CODE_OVERFLOW: u32 = 1;            // emu-defined; driver treats nonzero as error

/// Mirrors uapi `struct amdxdna_async_error` in
/// `xdna-driver/include/uapi/drm/amdxdna_accel.h:610-617`. 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct AmdxdnaAsyncError {
    pub err_code: u64,
    pub ts_us: u64,
    pub ex_err_code: u64,
}
const _: () = assert!(std::mem::size_of::<AmdxdnaAsyncError>() == 24);
```

**Per-column ring:**

```rust
pub struct AsyncRing {
    bytes: Box<[u8; ASYNC_BUF_SIZE]>,
}

impl AsyncRing {
    pub fn header(&self) -> &AieErrInfoHeader { /* transmute at offset 0 */ }
    pub fn header_mut(&mut self) -> &mut AieErrInfoHeader { /* transmute at offset 0 */ }
    pub fn push(&mut self, e: AieError) -> Result<(), Overflow> {
        let cnt = self.header().err_cnt as usize;
        if cnt >= MAX_ERRORS_PER_RING { return Err(Overflow); }
        // write e at offset 12 + cnt * 12
        // increment err_cnt
        Ok(())
    }
    pub fn read_into(&self, dst: &mut [u8]) -> usize {
        let used = 12 + (self.header().err_cnt as usize) * 12;
        let n = used.min(dst.len());
        dst[..n].copy_from_slice(&self.bytes[..n]);
        n
    }
    pub fn clear(&mut self) { self.bytes.fill(0); }
}
```

**Encoding helpers** (in archspec — direct port of driver macros from `amdxdna_error.h`):

```rust
pub const fn build_err_code(severity: u64, module: u64, class: u64, error_num: u64) -> u64 {
    // Mirrors AMDXDNA_ERROR_CODE_BUILD from amdxdna_error.h:100-111
    ((severity & SEVERITY_MASK) << SEVERITY_SHIFT)
        | ((module & MODULE_MASK) << MODULE_SHIFT)
        | ((class & CLASS_MASK) << CLASS_SHIFT)
        | error_num
}

pub const fn build_ex_err_code(row: u8, col: u8) -> u64 {
    // Mirrors AMDXDNA_ERROR_EXTRA_CODE_BUILD from amdxdna_error.h:139-141
    (((col as u64) & EXTRA_COL_MASK) << EXTRA_COL_SHIFT)
        | (((row as u64) & EXTRA_ROW_MASK) << EXTRA_ROW_SHIFT)
}
```

**Categorization tables** (direct port of `aie2_error.c:89-150`):

```rust
pub struct EventCategory { pub event_id: u8, pub category: AieErrorCategory }

pub const CORE_EVENT_CAT: &[EventCategory] = &[
    EventCategory { event_id: 55, category: AieErrorCategory::Access },
    EventCategory { event_id: 56, category: AieErrorCategory::Stream },
    EventCategory { event_id: 57, category: AieErrorCategory::Stream },
    EventCategory { event_id: 58, category: AieErrorCategory::Bus },
    EventCategory { event_id: 59, category: AieErrorCategory::Instruction },
    EventCategory { event_id: 60, category: AieErrorCategory::Access },
    EventCategory { event_id: 62, category: AieErrorCategory::Ecc },
    EventCategory { event_id: 64, category: AieErrorCategory::Ecc },
    EventCategory { event_id: 65, category: AieErrorCategory::Access },
    EventCategory { event_id: 66, category: AieErrorCategory::Access },
    EventCategory { event_id: 67, category: AieErrorCategory::Lock },
    EventCategory { event_id: 70, category: AieErrorCategory::Instruction },
    EventCategory { event_id: 71, category: AieErrorCategory::Stream },
    EventCategory { event_id: 72, category: AieErrorCategory::Bus },
    // Note: INSTR_ERROR = 69 is NOT in the driver table. Driver derives it
    // via the "unknown" fallback at categorize() time. We treat it as
    // AieErrorCategory::Instruction in our category mapping table (added
    // as a Tier B emu-specific entry, documented inline) so the cache
    // record decodes meaningfully without an "Unknown" sentinel.
];

pub const MEM_EVENT_CAT: &[EventCategory] = &[ /* events 88-101 per aie2_error.c:89-103 */ ];
pub const MEMTILE_EVENT_CAT: &[EventCategory] = &[ /* events 130-139 per aie2_error.c:122-132 */ ];
pub const SHIM_EVENT_CAT: &[EventCategory] = &[ /* events 64-71 per aie2_error.c:134-150 */ ];

pub fn event_to_category(event_id: u8, mod_type: AieModuleType) -> Option<AieErrorCategory> {
    let table = match mod_type {
        AieModuleType::Core => CORE_EVENT_CAT,
        AieModuleType::Mem => MEM_EVENT_CAT,
        AieModuleType::MemTile => MEMTILE_EVENT_CAT,
        AieModuleType::Pl => SHIM_EVENT_CAT,
    };
    table.iter().find(|e| e.event_id == event_id).map(|e| e.category)
}

pub fn is_error_event(event_id: u8, mod_type: AieModuleType) -> bool {
    event_to_category(event_id, mod_type).is_some()
}

// Mirrors driver `aie_cat_err_num_map` (aie2_error.c) -- direct port of the
// driver table at plan time. Lookup pattern identical to event_to_category.
// Returns AMDXDNA_ERROR_NUM_UNKNOWN on no match (driver default at
// aie2_error.c:182).
pub fn category_to_error_num(cat: AieErrorCategory) -> AmdxdnaErrorNum;

// Mirrors driver `aie_mod_amdxdna_err_mod_map` (aie2_error.c:161-165).
// 3 entries (CORE/MEM/PL). Returns AMDXDNA_ERROR_MODULE_UNKNOWN on no match.
pub fn mod_type_to_amdxdna_module(mod_type: AieModuleType) -> AmdxdnaErrorModule;
```

`Option` return is load-bearing: callers gate on `is_error_event` first;
inside `record_error` the `expect("is_error_event gated this; bug if None")`
turns table inconsistency into a loud panic during development.

## 7. FFI surface

Five new C symbols in `crates/xdna-emu-ffi/src/async_errors.rs`. All follow
existing FFI conventions (null-handle check returns sentinel, last_error
thread-local on failures, opaque handle, copy-on-read).

```c
// === Surface 1: cache reader (consumed by plugin ioctl) =====================

typedef struct XdnaEmuAsyncError {
    uint64_t err_code;
    uint64_t ts_us;
    uint64_t ex_err_code;
} XdnaEmuAsyncError;  // 24 bytes -- matches amdxdna_async_error layout

// Returns 1 if a record is populated and copied to *out;
//         0 if no errors recorded since last reset;
//        -1 null handle, -2 null out.
int32_t xdna_emu_get_last_async_error(
    XdnaEmuHandle* handle,
    XdnaEmuAsyncError* out);

// === Surface 2: mailbox ring reader (future kernel-driver attachment) =======

// Copies up to buf_size bytes from column `col`'s ring into `buf`. The
// copied bytes are aie_err_info header (12 B) + err_cnt * aie_error (12 B
// each) in driver-wire format.
// Returns number of bytes copied; 0 if ring is empty (err_cnt == 0);
//        -1 null handle, -2 invalid col, -3 null buf.
int64_t xdna_emu_read_async_event_ring(
    XdnaEmuHandle* handle,
    uint32_t col,
    uint8_t* buf,
    uint64_t buf_size);

// Returns 1 if err_cnt > 0 for the given column; 0 if empty;
//        -1 null handle, -2 invalid col. Polls without copying.
int32_t xdna_emu_async_event_pending(
    XdnaEmuHandle* handle,
    uint32_t col);

// === Surface 3: push callback ==============================================

typedef void (*XdnaEmuAsyncErrorCallback)(
    const XdnaEmuAsyncError* record,
    void* user_data);

// Register a callback invoked synchronously the moment a record is added.
// Pass NULL to unregister. user_data is round-tripped to each invocation.
// Thread-safety: callback fires from whichever thread is executing the
// emu run loop (matches "handles are not thread-safe" contract).
// Returns 0 on success, -1 on null handle.
int32_t xdna_emu_set_async_event_callback(
    XdnaEmuHandle* handle,
    XdnaEmuAsyncErrorCallback callback,
    void* user_data);

// === Reset helper ==========================================================

// Clears all per-column rings and the cache; does NOT touch L1/L2 latch
// state (Tier A) or any other tile state.
// Returns 0 on success, -1 on null handle.
int32_t xdna_emu_clear_async_errors(XdnaEmuHandle* handle);
```

**FFI-internal callback bridge:** the engine doesn't know about C function
pointers. The FFI layer stores the callback on `XdnaEmuHandle` and drains
the `async_errors.newly_recorded` queue between `engine.step()` calls in
`xdna_emu_run` (Section 5). Engine stays free of FFI concerns; pattern
mirrors `flush_trace_to_host`.

## 8. Plugin wire-up

Smallest possible C++ change. Two files:

```cpp
// xrt-plugin/src/platform_emu.cpp -- replace the existing case (~line 1189).
case DRM_AMDXDNA_HW_LAST_ASYNC_ERR: {
  if (arg.buffer_size < sizeof(amdxdna_async_error))
    shim_err(EINVAL, "get_info_array: buffer too small for async error");

  amdxdna_async_error rec{};  // 24-byte uapi struct
  auto& transport = m_drv->transport();
  int32_t got = transport.sym_get_last_async_error_(
      transport.handle(),
      reinterpret_cast<XdnaEmuAsyncError*>(&rec));

  if (got == 1) {
    std::memcpy(arg.buffer, &rec, sizeof(rec));
    arg.num_element = 1;
  } else {
    arg.num_element = 0;  // no record yet; same surface as today's no-op
  }
  break;
}
```

```cpp
// xrt-plugin/src/transport_inprocess.cpp -- add one resolve_required line.
sym_get_last_async_error_ = resolve_required<fn_get_last_async_error>(
    "xdna_emu_get_last_async_error");
```

```cpp
// xrt-plugin/src/transport_inprocess.h -- declare typedef + member.
using fn_get_last_async_error = int32_t (*)(XdnaEmuHandle*, XdnaEmuAsyncError*);
// ...
fn_get_last_async_error sym_get_last_async_error_;
```

**`resolve_required` rationale:** an old emu `.so` without the symbol
will fail to load loudly. Pre-release silent-staleness has caused phantom
bugs before; loud failure is the convention here. Driven by the same
discipline as `rebuild-plugin.sh` / `-refresh_dkms`.

The other four FFI symbols (`read_async_event_ring`, `async_event_pending`,
`set_async_event_callback`, `clear_async_errors`) are unused by the plugin
in this spec. The existing FFI completeness test will list them as
"exported but not consumed" (informational `eprintln!`, not a failure).
That is the correct state — they exist for future consumers.

## 9. Testing

Five layers, each tested in isolation, plus one integration test that
crosses all of them.

### 9.1 Unit tests by layer

| Layer | Tests | Location |
|---|---|---|
| **Wire format** | Compile-time: `AieError` size == 12, `AieErrInfoHeader` == 12, `AmdxdnaAsyncError` == 24, `MAX_ERRORS_PER_RING == 681`. Runtime: round-trip a payload through `AsyncRing::read_into` bytes and parse back, verifying byte positions of every field. | `src/device/async_errors/types.rs` |
| **Categorization** | For every entry in every driver category table: `event_to_category(event_id, mod_type) == Some(expected_category)`. Negative: `event_to_category(99, Core) == None` (event 99 is mem, not core). `is_error_event(INSTR_ERROR=69, Core) == true`. | `crates/xdna-archspec/src/aie2/async_errors.rs` |
| **Encoding** | `build_err_code(NonFatal, Aie, Aie, N)` bit-unpacks to the same fields. `build_ex_err_code(row=2, col=3) == expected`. | Same archspec module |
| **Sink behavior** | `record_error` populates cache and ring. Second call overwrites cache (last-write-wins) and appends to ring (`err_cnt += 1`). Overflow at MAX_ERRORS_PER_RING+1 sets `ret_code = OVERFLOW`. `clear()` zeros both. Per-column independence: record in col 1 leaves col 3 untouched. | `src/device/async_errors/sink.rs` |
| **Drain queue** | `drain_newly_recorded()` returns records added since last drain in FIFO order; empty after drain; preserves all fields. | Same sink module |
| **FFI** | `xdna_emu_get_last_async_error`: returns 0 on fresh handle, 1 after a record. `xdna_emu_read_async_event_ring`: returns the right bytes per column, 0 on empty, -2 on invalid col. `xdna_emu_async_event_pending` agrees with read result. `xdna_emu_set_async_event_callback`: fires synchronously on record, user_data round-trips, NULL unregisters cleanly. `xdna_emu_clear_async_errors`: zeros all rings + cache. | `crates/xdna-emu-ffi/src/async_errors.rs` |
| **FFI completeness** | Existing auto-discovery test picks up `xdna_emu_get_last_async_error` as required (plugin uses `resolve_required`); fails loudly if missing. | `crates/xdna-emu-ffi/src/lib.rs` (existing test, no change) |

### 9.2 Integration test — control-packet-driven INSTR_ERROR fixture

A new unit test in the testing module that drives event generation from
"outside the engine" the same way a host harness would:

1. Build a `DeviceState` via the standard test fixture, enable a compute
   core on tile (1,2), advance one dispatch tick.
2. Construct a minimal control packet that writes register
   `Event_Generate` on tile (1,2) with `event_id = INSTR_ERROR = 69`.
   The control-packet path is already used by bridge tests.
3. Submit it, step until dispatch settles.
4. Assert:
   - `device.async_errors.last_cache().is_some()`
   - cache `err_code` decodes to `(severity=NonFatal, module=AIE_CORE, class=Aie)`
     matching the driver's INSTR_ERROR category
   - cache `ex_err_code == (2 << 8) | 1` (row=2, col=1)
   - `device.async_errors.ring(1).err_cnt == 1`
   - the ring's first `aie_error` decodes to `event_id=69, row=2, col=1, mod_type=AIE_CORE`
   - Tier A's L1 latch also fired (read L1 status register on shim col 1 — proves
     both paths fire in parallel, neither swallows the other)

This is the "end-to-end at the emu boundary" test that closes the loop
without a bridge fixture or deliberately-broken kernel. Uses the same
control-packet path real consumers take.

### 9.3 What's NOT in test scope

- **No new bridge test.** The plugin ioctl path will be exercised by ad-hoc
  manual test once this spec lands. When the first detection-spec follow-up
  adds a real error producer (DMA bounds, etc.), it can add a bridge test
  that fires the path end-to-end.
- **No HW vs EMU comparison test.** Real silicon's TDR recovery / context
  restart path isn't modeled here (separate Tier C effort); bridge HW
  comparison for error reporting is meaningless until that exists.
- **No multi-error overflow stress test.** Single-error path is the primary
  case for INSTR_ERROR; overflow handling has a unit test in §9.1 that
  covers the failure mode.

## 10. Open follow-ups (out of scope for this spec, but tracked)

| Follow-up | Sketch | Trigger |
|---|---|---|
| **Tier B detection spec — DMA errors** | Wire emu-side detection for DMA bounds violations + lock acquire on free pair; fire events 65/66/67 (shim) or 97-101 (mem); flow through this spec's plumbing. | After Tier B Spec 1 lands. |
| **Tier B detection spec — stream errors** | Detection for stream switch parity / packet errors; events 56/57/71 (core), 64/68 (shim), 135-138 (memtile). | After Tier B Spec 1 lands. |
| **Tier B detection spec — ECC / parity** | Cache parity / memory ECC simulation (emu has no bit-flip simulation today); events 62/64 (core), 88/90 (mem), 130/132 (memtile). | Lower priority — bit flips are not a current pain point. |
| **Tier B exception type** | Driver also handles `ASYNC_EVENT_TYPE_EXCEPTION` separately from errors (`aie2_msg_priv.h:398`). Add when a producer exists. | When a use case appears. |
| **Tier C — TDR & context restart** | Emulate driver's `aie2_rq_handle_idle_ctx` periodic timer + context restart path; needed to validate driver-level error recovery against the emu. | Separate Tier C spec. |
| **Wall-clock timestamp mode** | Add an opt-in mode where `ts_us` comes from `SystemTime::now()` instead of `cycle / 1000`. Non-deterministic; for "looks real to users" cases. | When asked. |
| **Multi-handle async-error testing** | Multi-process / multi-context scenarios where two handles fire errors concurrently. | When the consumer pattern emerges. |

---

**References:**
- Tier A spec: [`2026-05-19-interrupt-l2-closeout-design.md`](2026-05-19-interrupt-l2-closeout-design.md)
- Tier B host-boundary findings: [`../findings/2026-05-19-interrupt-tier-b-firmware-delivery.md`](../findings/2026-05-19-interrupt-tier-b-firmware-delivery.md)
- Driver async-error implementation: `xdna-driver/src/driver/amdxdna/aie2_error.c` (esp. 56-150 for types/tables, 278-289 for callback, 364-410 for allocation)
- Driver uapi: `xdna-driver/include/uapi/drm/amdxdna_accel.h:607-617` (struct), `:692` (ioctl param)
- Driver error encoding: `xdna-driver/src/driver/amdxdna/amdxdna_error.h:100-141`
- Plugin current no-op: `xrt-plugin/src/platform_emu.cpp:1189-1193`
- FFI completeness test: `crates/xdna-emu-ffi/src/lib.rs:832-918`
