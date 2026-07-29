# Phoenix vfio-user Driver Boundary Implementation Plan

**Goal:** Run the frozen Phoenix `add_one_using_dma` artifact through normal
guest XRT, the unmodified pinned primary `amdxdna.ko`, a vfio-user PCI
function, the real pinned management firmware, and the existing shared AIE
array emulator.

**Architecture:** Extend the existing `HostMemory`, firmware `Bus`,
`pump_runtime`, and C ABI at their current shared seams. Add one
single-threaded C frontend using pinned libvfio-user. Prove live no-IOMMU guest
physical mappings before trusting PSP boot, then gate the work with driver
probe and the frozen Chess and Peano kernels.

**Execution:** The primary agent executes this plan serially in the existing
`firmware-priors` worktree. No subagents. Each production change follows an
observed failing test and is committed only after its focused GREEN gate.

**Approved design:**
[`2026-07-28-phoenix-vfio-user-driver-boundary-design.md`](../specs/2026-07-28-phoenix-vfio-user-driver-boundary-design.md)

## Fixed Constraints

- Firmware:
  `/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin`,
  SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`.
- Driver commit:
  `216cefececd74effcd7a88350c71b99f5ef9a215`,
  `drivers/accel/amdxdna`.
- QEMU: `10.2.1`.
- libvfio-user commit:
  `37491ed9af828fc161238dacd82e83ea35a09f87`.
- The first guest has exactly 2 GiB shared RAM, no vIOMMU,
  `amdxdna.force_iova=0`, and reserves
  `0x60000000..0x70000000` with `memmap=256M$1536M`.
- Before XRT opens the accelerator, the guest writes and reads back
  `0x10000000@0x60000000` through the unique `carveout` debugfs node.
- No patched driver, patched QEMU, XRT emulator plugin, mailbox responder, or
  `xdna_emu_assign_partition` call belongs in the first acceptance path.
- A narrow QEMU or driver shim remains allowed only if the direct-map proof
  disproves the approved physical-address path.
- The frontend is single-threaded. It adds no scheduler, worker, mutex, BAR
  shadow, or mailbox-opcode implementation.
- Persistent sources and evidence stay below this repository. Ephemeral clone
  and socket state may use ignored `build/` paths, never `/tmp`.
- Halo is reserved for a genuinely heavy guest-kernel or image build. Rust
  tests, C compilation, libvfio-user, and brief QEMU probes run locally.

## Worktree Test Environment

The linked worktree cannot use sibling-path inference because its parent is
`.worktrees/`. Run Rust gates with:

```bash
PATH=/home/triple/npu-work/mlir-aie/ironenv/bin:$PATH \
PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
MLIR_AIE_PATH=/home/triple/npu-work/mlir-aie \
LLVM_AIE_PATH=/home/triple/npu-work/llvm-aie \
AIE_RT_PATH=/home/triple/npu-work/aie-rt/driver/src \
cargo test --lib
```

Do not change build-script path inference merely to hide this worktree
property.

---

## Task 1: Live External Host Memory

**Files:**

- Modify and test: `src/device/host_memory.rs`
- Modify and test: `src/firmware/mmio.rs`

**Existing seams to reuse:**

- `HostMemory::{read_bytes, write_bytes, region_at, clear}`
- `Bus::{registered_host_target, management_dma_host_target}`

### RED

- [ ] Add focused `HostMemory` tests proving:
  - writes through `HostMemory` mutate the caller-owned byte buffer;
  - caller writes are immediately visible through `HostMemory`;
  - reads and writes cross adjacent external mappings;
  - `contains_range` accepts one named region or a contiguous union of
    external mappings;
  - overlap and zero-length mappings are rejected;
  - unmap requires the exact base and length;
  - post-unmap access falls back to existing sparse-page behavior;
  - `clear` forgets mappings without freeing caller memory; and
  - named-region behavior is unchanged.
- [ ] Run only the new tests and retain the compiler/test failure caused by
  missing external-map operations.

### GREEN

- [ ] Add one private external-region record to `HostMemory` containing only
  guest base, length, and a non-owned `NonNull<u8>`.
- [ ] Add unsafe map, exact unmap, and shared `contains_range` operations.
- [ ] Reject address/length overflow and overlap between external mappings.
  Named-region metadata may overlap an external mapping because the external
  bytes are the preferred backing store.
- [ ] Make byte reads and writes prefer external mappings and chunk at mapping
  boundaries. Preserve sparse pages everywhere else.
- [ ] Keep the external list private and use a linear scan. The first guest
  has a handful of RAM mappings; a tree is warranted only if measured map
  counts make the scan material.
- [ ] Replace the two firmware range checks that require one
  `MemoryRegion` with `HostMemory::contains_range`.
- [ ] Run:

```bash
cargo test --lib device::host_memory
cargo test --lib firmware::mmio
cargo test --lib
```

- [ ] Commit:

```text
feat(memory): map live external host ranges
```

---

## Task 2: Host-Memory and General Firmware-Word C ABI

**Files:**

- Modify and test: `crates/xdna-emu-ffi/src/memory.rs`
- Modify and test: `crates/xdna-emu-ffi/src/firmware.rs`
- Modify: `crates/xdna-emu-ffi/src/lib.rs`
- Modify: `include/xdna_emu.h`
- Modify: `crates/xdna-emu-ffi/tests/tier_c_completeness.rs`

**Existing seams to reuse:**

- `NpuBackend::host_memory_mut`
- `Bus::{host_load32, host_store32}`
- existing handle, pointer-validation, result, and `LAST_ERROR` patterns

### RED

- [ ] Lock these exact C signatures in `tier_c_completeness.rs`:

```c
XdnaEmuResult xdna_emu_map_host_memory(
    XdnaEmuHandle*, uint64_t, uint8_t*, uint64_t);
XdnaEmuResult xdna_emu_unmap_host_memory(
    XdnaEmuHandle*, uint64_t, uint64_t);
XdnaEmuResult xdna_emu_firmware_read_host32(
    XdnaEmuHandle*, uint32_t, uint32_t*);
XdnaEmuResult xdna_emu_firmware_write_host32(
    XdnaEmuHandle*, uint32_t, uint32_t);
```

- [ ] Add Rust FFI tests for null handle/data/output pointers, zero length,
  guest-address overflow, host-size conversion overflow, overlap, inexact
  unmap, and missing firmware.
- [ ] Add one direct-coherence test in which a caller-owned buffer changes
  through existing emulator host-memory writes and caller changes are read
  back by the emulator.
- [ ] Observe the compile/link failures before adding exports.

### GREEN

- [ ] Add map/unmap wrappers which validate the public trust boundary and call
  Task 1 directly. Do not add mapped guest RAM to the backend's named runtime
  argument list.
- [ ] Add general firmware word wrappers which delegate to the loaded
  processor's existing bus. Keep the current SRAM-only wrappers unchanged.
- [ ] Return existing result variants and detailed thread-local errors; do not
  add a new public error taxonomy.
- [ ] Export the functions from Rust and the public C header.
- [ ] Run:

```bash
cargo test -p xdna-emu-ffi memory
cargo test -p xdna-emu-ffi firmware
cargo test -p xdna-emu-ffi --test tier_c_completeness
```

- [ ] Commit:

```text
feat(ffi): expose live host mappings
```

---

## Task 3: Genuine I2X MSI-X Edges

**Files:**

- Modify and test: `src/firmware/phoenix_mailbox.rs`
- Modify and test: `src/firmware/mmio.rs`

**Existing seams to reuse:**

- the one `PhoenixMailboxRegisters` store for host and firmware aliases
- `Bus::{host_store32, region_store32}`
- the derived 16-channel stride and alias translation

### RED

- [ ] Add address-mapping tests for all 16 host and firmware I2X status
  aliases.
- [ ] Add bus tests proving:
  - a firmware zero-to-nonzero I2X status store sets exactly the channel bit;
  - a repeated nonzero store creates no second edge;
  - draining the pending mask does not alter mailbox data;
  - host clear followed by firmware publication creates a new edge; and
  - ordinary mailbox words never create MSI-X edges.
- [ ] Run the focused tests and retain the missing-state failures.

### GREEN

- [ ] Add one pure status-address-to-channel helper beside the mailbox geometry.
- [ ] Add one pending bitmask to `Bus`.
- [ ] Route both host and firmware mailbox writes through one shared write
  helper so edge detection cannot diverge by caller.
- [ ] Set the channel bit only when the authoritative status word changes
  from zero to nonzero. Add one draining accessor.
- [ ] Do not infer completion from tail pointers or synthesize interrupts.
- [ ] Run:

```bash
cargo test --lib firmware::phoenix_mailbox
cargo test --lib firmware::mmio
```

- [ ] Commit:

```text
feat(firmware): publish genuine mailbox interrupts
```

---

## Task 4: Bounded Firmware-Service C ABI

**Files:**

- Modify and test: `crates/xdna-emu-ffi/src/firmware.rs`
- Modify: `include/xdna_emu.h`
- Modify: `crates/xdna-emu-ffi/tests/tier_c_completeness.rs`

**Existing seams to reuse:**

- `pump_runtime`
- `RuntimePumpStop`
- the interpreter-backend downcast already used by firmware boot
- Task 3's pending-mask drain
- the existing firmware lifecycle status bit 0

### RED

- [ ] Lock the exact struct layout and service signature in the C ABI test:

```c
typedef struct {
    XdnaEmuResult result;
    uint32_t pending_msix_mask;
    int quiescent;
    int wait_mode;
} XdnaEmuFirmwareServiceStatus;

XdnaEmuFirmwareServiceStatus xdna_emu_service_firmware(
    XdnaEmuHandle*, uint64_t, uint64_t);
```

- [ ] Add focused tests for invalid handle, missing firmware, unsupported
  backend, natural wait, bounded progress, wait-mode reflection, one-shot
  MSI-X drain, unresolved firmware poll, unknown instruction, engine stall,
  and engine error.
- [ ] Observe RED before adding the service export.

### GREEN

- [ ] Call `pump_runtime(..., |_, _| false)` with the caller's two guards.
- [ ] Map `ArrayIdleFirmwareWaiting` to successful quiescence and
  `NoProgressExhausted` to successful non-quiescence.
- [ ] Map unresolved polls, unknown instructions, array stalls, and array
  errors to `ExecutionError` with the existing detailed last-error channel.
- [ ] Drain pending MSI-X bits only into a successful status. Reflect lifecycle
  bit 0 without forcing it.
- [ ] Add no timer, new scheduler, or background thread.
- [ ] Run:

```bash
cargo test -p xdna-emu-ffi firmware
cargo test -p xdna-emu-ffi --test tier_c_completeness
cargo test --lib firmware::runtime
```

- [ ] Commit:

```text
feat(ffi): service live firmware runtime
```

---

## Task 5: Minimal Phoenix Controller Frontend

**Files:**

- Add: `tools/phoenix-vfio-user/phoenix_vfio_user.c`
- Add: `tools/phoenix-vfio-user/build.sh`

No extra library, framework, test binary, or configuration format is needed.
The frontend binary's `--self-test` mode exercises its static controller
helpers without QEMU.

### RED

- [ ] Write `--self-test` assertions first for:
  - BAR0/BAR2/BAR4 address and bounds translation;
  - aligned, split, and one-/two-byte-tail little-endian BAR access;
  - PSP and SMU notify `0 -> 1` edge behavior;
  - the exact accepted PSP and SMU command set;
  - PSP validation reading live mapped physical bytes;
  - malformed order and unmapped firmware becoming fatal;
  - cold handle replacement and active-map replay; and
  - reset-state clearing.
- [ ] Run the build/self-test and retain its initial compile/assertion failure.

### GREEN

- [ ] Make `build.sh`:
  - verify required tools and development packages;
  - clone libvfio-user only under ignored `build/deps/`;
  - verify the exact pinned commit;
  - build it with Meson/Ninja; and
  - compile the single frontend against `include/xdna_emu.h` and the debug FFI
    library.
- [ ] Implement one compact frontend state object: libvfio-user context,
  emulator handle, active maps, BAR0 words, power/start/fatal state.
- [ ] Implement BAR transport as byte spans over existing 32-bit firmware
  accesses. Do not parse BAR2/BAR4 words.
- [ ] Implement only the approved driver-reachable PSP/SMU envelope.
- [ ] Recreate the handle and replay exact live maps for release-TMR, FLR,
  device reset, and disconnect.
- [ ] Make fatal callback failures latch one error which terminates the event
  loop after returning from the callback.
- [ ] Run:

```bash
tools/phoenix-vfio-user/build.sh
build/tools/phoenix-vfio-user/phoenix-vfio-user --self-test
```

- [ ] Commit:

```text
feat(vfio-user): add Phoenix controller frontend
```

---

## Task 6: PCI, DMA Registration, and MSI-X Wiring

**Files:**

- Modify and self-test:
  `tools/phoenix-vfio-user/phoenix_vfio_user.c`

### RED

- [ ] Extend `--self-test` to inspect the constructed PCI config space and
  assert the pinned identity, BAR flags and sizes, PCIe/FLR capability, and
  16-vector MSI-X layout.
- [ ] Add callback-level tests for DMA map/unmap bookkeeping, protection
  checks, exact unmap, reset replay, and one trigger per returned service bit.
- [ ] Observe failures before registering the libvfio-user machinery.

### GREEN

- [ ] Configure PCI `1022:1502`, subsystem `f111:0005`, revision 0, class
  `0x118000`, and exactly the four approved BARs.
- [ ] Register trapped BAR0/BAR2/BAR4 callbacks and libvfio-user-owned BAR1
  MSI-X table/PBA.
- [ ] Register DMA callbacks that require non-null directly mapped pointers and
  mirror exact ranges into both the frontend list and emulator FFI.
- [ ] Register the same cold-reset helper for FLR, device reset, and
  disconnect.
- [ ] Implement one nonblocking attach/run/service/poll loop, refreshing the
  poll fd after attach and `ENOTCONN`.
- [ ] Trigger every returned vector exactly once through `vfu_irq_trigger`;
  failure is fatal.
- [ ] Run the frontend self-test again.
- [ ] Commit:

```text
feat(vfio-user): wire Phoenix PCI transport
```

---

## Task 7: QEMU Direct-Mapping Proof

**Files:**

- Add: `scripts/phoenix-vfio-user-qemu.sh`
- Modify as needed:
  `tools/phoenix-vfio-user/phoenix_vfio_user.c`

The script initially implements only a `--map-smoke` acceptance mode. It gains
driver and kernel execution modes only after this gate is GREEN.

### RED

- [ ] Add a guest/server nonce exchange covering:
  - non-null direct mappings;
  - callback address equals GPA;
  - bidirectional immediate visibility;
  - inclusion of `0x60000000..0x70000000`;
  - protection flags; and
  - exact unmap.
- [ ] Launch stock QEMU 10.2.1 with 2 GiB shared memfd RAM, no vIOMMU, and the
  vfio-user function. Retain the first failing transcript under
  `build/experiments/phoenix-vfio-user/`.

### GREEN or Architecture Stop

- [ ] If every property holds, check the exact passing QEMU command into the
  script and record its tuple.
- [ ] If any property fails, stop. Do not add a bounce buffer. Bring the
  evidence to Maya and choose a narrow QEMU extension or driver shim before
  changing architecture.
- [ ] Commit a passing smoke harness only:

```text
test(vfio-user): prove live guest physical mappings
```

---

## Task 8: Pinned Driver Probe

**Files:**

- Modify: `scripts/phoenix-vfio-user-qemu.sh`
- Add evidence under ignored:
  `build/experiments/phoenix-vfio-user/`

### Acceptance

- [ ] Verify every pinned component and reject dirty/patched QEMU or primary
  driver sources used for the guest.
- [ ] Boot with `memmap=256M$1536M`, default `force_iova=0`, and no vIOMMU.
- [ ] Capture `lspci -nnvv`, driver logs, and MSI-X allocation.
- [ ] Wait for the unique driver-created `carveout` node, write
  `0x10000000@0x60000000`, and require identical readback before XRT open.
- [ ] Require successful firmware validate/start, alive/init/query traffic,
  16 MSI-X vectors, and `/dev/accel/accel0`.
- [ ] Require no emulator XRT plugin and no legacy driver.
- [ ] Commit the reproducible passing harness:

```text
test(vfio-user): probe pinned Phoenix driver
```

---

## Task 9: Frozen Chess and Peano Completion

**Files:**

- Modify: `scripts/phoenix-vfio-user-qemu.sh`
- Update after verified success:
  `docs/roadmap/phase2-toolchain-integration.md`
- Update after verified success:
  `docs/known-fidelity-gaps.md`

### Chess Gate

- [ ] Run the already frozen Chess `add_one_using_dma` artifact through guest
  XRT and require:
  - output `2..=65`;
  - successful XRT command completion;
  - genuine X2I consumption and I2X publication;
  - genuine context-vector MSI-X delivery;
  - no synthetic response path; and
  - clean teardown.

### Peano Gate

- [ ] Run the already validated Peano form through the identical guest path
  with the same requirements.

### Final Verification

- [ ] Run:

```bash
cargo test -p xdna-emu-ffi
cargo test --lib
cargo fmt --all --check
git diff --check
tools/phoenix-vfio-user/build.sh
build/tools/phoenix-vfio-user/phoenix-vfio-user --self-test
scripts/phoenix-vfio-user-qemu.sh --map-smoke
scripts/phoenix-vfio-user-qemu.sh --driver-probe
scripts/phoenix-vfio-user-qemu.sh --run-frozen chess
scripts/phoenix-vfio-user-qemu.sh --run-frozen peano
```

- [ ] Re-run `cargo test --lib` after all documentation changes.
- [ ] Record exact current test counts and tuple evidence; do not copy stale
  counts into documentation.
- [ ] Commit:

```text
docs(firmware): record full driver-boundary proof
```

## Completion Boundary

This plan is complete only when both frozen compiler artifacts pass through
the real guest driver boundary. A clean driver probe, successful firmware
boot, or passing Rust/C self-test is progress evidence, not completion.

Older Phoenix firmware versions and any IOVA/PASID/AIE2P expansion begin only
after this pinned tuple is GREEN.
