# Phoenix vfio-user Driver Boundary -- Design

**Date:** 2026-07-28

**Status:** Architecture approved, including PA/carveout correction; written-spec review pending

## Purpose

Expose the existing Phoenix firmware and array emulator as a real PCI function
to an unmodified Linux guest. The guest must use the normal XRT userspace stack
and the pinned primary `amdxdna.ko`; neither may know that the device is
simulated.

This closes the missing driver boundary:

```text
guest application
  -> normal XRT XDNA SHIM
  -> unmodified amdxdna.ko
  -> vfio-user PCI function
  -> real npu.dev.sbin in xdna-emu
  -> shared AIE array emulator
```

Driver probe is a diagnostic gate, not completion. The first completed slice
must run the already-validated `add_one_using_dma` artifact through that whole
path and return the expected output and command completion.

## Pinned Validation Tuple

The first acceptance result is reproducible against this tuple:

- Firmware:
  `/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin`
  - size: 248,592 bytes
  - SHA-256:
    `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`
- Driver repository commit:
  `216cefececd74effcd7a88350c71b99f5ef9a215`
  - the accepted module is built from `drivers/accel/amdxdna`;
  - `src/driver/amdxdna` and `amdxdna_legacy.ko` are not substitutes.
- QEMU:
  `10.2.1` (`Debian 1:10.2.1+ds-1ubuntu3.1`).
- libvfio-user:
  `37491ed9af828fc161238dacd82e83ea35a09f87`
  from `https://gitlab.com/qemu-project/libvfio-user.git`.

libvfio-user's public API and ABI are not stable. The full commit ID is
therefore part of the validation tuple, not a minimum version.

The virtual function reproduces the locally observed Phoenix identity:

| Field | Value |
|---|---:|
| Vendor / device | `1022:1502` |
| Subsystem vendor / device | `f111:0005` |
| Revision | `0` |
| Class | `0x118000` (signal processing controller) |
| BAR0 | 512 KiB, 32-bit, non-prefetchable |
| BAR1 | 8 KiB, 32-bit, non-prefetchable |
| BAR2 | 256 KiB, 64-bit, prefetchable |
| BAR4 | 256 KiB, 32-bit, non-prefetchable |
| MSI-X | 16 vectors; table in BAR1 at `0`, PBA at `0x1000` |

The first function advertises PCI Express endpoint and FLR capabilities plus
MSI-X. It does not advertise PASID. The first gate deliberately uses the
driver's existing physical-address-plus-carveout bring-up mode, with no guest
IOMMU, and must not claim an IOVA, SVA, or PASID contract that the frontend
does not yet implement.

## Selected Mechanism

Add one small, single-threaded C server under
`tools/phoenix-vfio-user/`. It links the existing `libxdna_emu.so` C ABI and
the pinned libvfio-user.

libvfio-user owns:

- the Unix-socket vfio-user protocol;
- PCI configuration-space machinery;
- trapped BAR dispatch;
- DMA map/unmap notifications;
- MSI-X eventfds and mask/PBA behavior; and
- device, connection-loss, and FLR reset callbacks.

The frontend owns only the Phoenix state at the PCI boundary. It must not parse
mailbox opcodes, synthesize mailbox responses, configure the array on the
driver's behalf, or call the legacy `xdna_emu_assign_partition` SHIM hook.

The rejected alternatives remain rejected for this first gate:

- no QEMU fork or custom QEMU PCI device;
- no kernel fake-PCI or endpoint-controller driver;
- no patched `amdxdna.ko` or out-of-tree address-translation shim in the
  acceptance guest;
- no reuse of the existing XRT emulator plugin as acceptance;
- no in-process vfio-user protocol implementation; and
- no copied or mirrored guest-memory shadow.

This is a scope choice, not a permanent ban on shims. A narrow QEMU extension
or driver compatibility shim is allowed later if simultaneous PSP physical
access and ordinary guest IOVA access are required. That work needs its own
derived contract and acceptance test; it is not speculative scaffolding in
the physical-address gate.

## State Ownership

| State | Sole owner |
|---|---|
| PCI config, BAR metadata, MSI-X table/PBA | libvfio-user |
| PSP/SMU scratch-register envelope | C frontend |
| Active guest-physical-to-pointer list | C frontend, mirrored into the emulator through FFI |
| BAR2 SRAM aliases and BAR4 mailbox registers | firmware `Bus` |
| Management Xtensa state | `FirmwareProcessor` |
| Guest DMA contents | QEMU shared guest RAM |
| AIE tiles, DMA, streams, locks, and host-memory view | existing `InterpreterEngine` |

There is one `XdnaEmuHandle`, one libvfio-user context, and one event loop.
Neither library is called concurrently, so no new mutex or worker thread is
needed.

## PCI and BAR Contract

The driver-visible device bases remain the open NPU1 values:

| BAR | Device base | Frontend behavior |
|---|---:|---|
| BAR0 | `0x03000000` | PSP/SMU lifecycle registers |
| BAR1 | MSI-X only | handled by libvfio-user |
| BAR2 | `0x03080000` | firmware-programmed SRAM aliases |
| BAR4 | `0x030c0000` | mailbox rings and control words |

BAR0, BAR2, and BAR4 are registered as trapped memory regions, with no mmap
backing file. BAR2 and BAR4 accesses translate `BAR base + offset` to a Phoenix
device address and call the general firmware host-access ABI below.

The pinned driver uses `readl`/`writel` for control words and
`memcpy_toio`/`memcpy_fromio` for ring payloads. The BAR callback therefore
accepts any in-range byte span:

- aligned four-byte chunks use one 32-bit emulator access;
- longer spans split into four-byte chunks;
- one- and two-byte tails use little-endian read/modify/write; and
- a span that overflows or crosses the BAR boundary fails.

This is transport adaptation only. The callback does not interpret the word it
is moving.

## Minimal FFI Extension

The current SRAM-only functions stay available for compatibility. Add the
smallest general operations needed by the PCI frontend:

```c
XdnaEmuResult xdna_emu_firmware_read_host32(
    XdnaEmuHandle* handle,
    uint32_t device_address,
    uint32_t* value_out
);

XdnaEmuResult xdna_emu_firmware_write_host32(
    XdnaEmuHandle* handle,
    uint32_t device_address,
    uint32_t value
);

XdnaEmuResult xdna_emu_map_host_memory(
    XdnaEmuHandle* handle,
    uint64_t address,
    uint8_t* data,
    uint64_t size
);

XdnaEmuResult xdna_emu_unmap_host_memory(
    XdnaEmuHandle* handle,
    uint64_t address,
    uint64_t size
);

typedef struct {
    XdnaEmuResult result;
    uint32_t pending_msix_mask;
    int quiescent;
    int wait_mode;
} XdnaEmuFirmwareServiceStatus;

XdnaEmuFirmwareServiceStatus xdna_emu_service_firmware(
    XdnaEmuHandle* handle,
    uint64_t max_iterations,
    uint64_t firmware_budget
);
```

The general host-word functions delegate to the existing
`Bus::host_load32`/`host_store32`; they do not add a second BAR model.

The service function reuses `pump_runtime(..., |_, _| false)`. It does not add
a second scheduler:

- `quiescent = 1` means firmware is at a natural wait, the array is idle, and
  no just-published task-completion token requires another turn;
- `quiescent = 0` with `Success` means the bounded budget expired and the
  caller should service again;
- unresolved firmware polls, unknown instructions, array stalls, and array
  errors return `ExecutionError` and detailed thread-local error text;
- `pending_msix_mask` drains genuine pending firmware I2X edges; and
- `wait_mode` reflects the firmware bus's existing lifecycle-status bit 0.

The budgets are termination guards, not modeled hardware timing.

## Live Guest DMA

### Why the first gate is physical

The pinned driver's `aie_psp.c::psp_alloc_fw_buf` explicitly obtains a host
physical address with `virt_to_phys`; that controller path bypasses the normal
NPU buffer IOMMU. In contrast, `amdxdna_iommu.c::amdxdna_dma_map_bo` installs
ordinary device heaps and shared BOs in the forced IOVA domain. QEMU 10.2.1's
`hw/vfio/listener.c` exposes either RAM sections at guest physical addresses or
the PCI device's IOMMU mappings, while `hw/vfio-user/container.c` forwards that
one selected address space to the server. It does not concurrently publish an
IOMMU-bypass GPA view for PSP.

Loading a host-side firmware copy would evade the command being validated.
Instead, the first gate follows the driver's own documented fallback:
`amdxdna_pci_drv.c` permits open without PASID because the user may select
"pa + carveout later," and `amdxdna_cbuf.c` provides that contiguous physical
allocator.

### Physical-address launch

The guest is launched with 2 GiB of shared RAM and no virtual IOMMU. The kernel
command line reserves 256 MiB at physical address `0x60000000`:

```text
memmap=256M$1536M
```

After the primary driver has probed, but before XRT first opens the accelerator,
the guest harness writes:

```text
0x10000000@0x60000000
```

to the sole driver-created `carveout` debugfs node. `amdxdna.force_iova`
remains false. This selects the pinned driver's explicit PA-plus-carveout
bring-up path: the PSP firmware buffer remains an ordinary guest physical
allocation, while all later device-accessible BOs come from the reserved
physical aperture. The NPU1 XRT path initially creates one 64-MiB device heap;
the remaining carveout space is available for command and shared BOs.

Keeping guest RAM below `0x80000000` is intentional. The existing
firmware-derived management-DMA translation accepts a bit-31-decorated address
only when exactly one of its decorated or undecorated forms is live. With no
RAM above bit 31, a real low guest-physical target cannot ambiguously match both
forms.

QEMU's reservation is a guest E820 property; the underlying shared RAM remains
present in QEMU. With no client IOMMU, libvfio-user's DMA registration callback
therefore supplies guest physical ranges, their lengths and protections, and
directly mapped `vaddr` pointers. The protocol field remains named `iova`, but
its value is a GPA in this launch.

### Emulator memory view

`HostMemory` gains one external-region list alongside its existing sparse pages
and named regions. It remains the sole host-memory API:

- an external region contains a guest-physical base, length, and non-owned data
  pointer;
- external mappings do not become ordinary named `MemoryRegion`s;
- one shared `contains_range` check accepts a span covered by either one named
  region or a contiguous union of external mappings;
- firmware host access and management-DMA validation use that shared check
  instead of assuming one named region contains the entire span;
- reads and writes prefer matching external regions and operate on live QEMU
  RAM;
- byte spans may cross page and adjacent external-region boundaries;
- sparse pages retain their existing behavior outside external ranges; and
- clearing or unmapping drops references without freeing QEMU-owned memory.

The FFI trust boundary rejects:

- a null pointer;
- a zero length;
- address or host-size overflow;
- overlapping ranges; and
- an unmap that does not exactly match a live base and length.

The pointer remains valid until the matching unregister callback. The
single-threaded frontend is already quiesced while libvfio-user invokes that
callback, so the emulator cannot access a range concurrently with its removal.

The frontend also keeps its own compact active-map list. This is needed to
replay still-valid maps when a PCI reset replaces the emulator handle.

### Direct-mapping proof gate

Before wiring PSP or firmware service, a QEMU smoke test must prove that:

1. shared guest RAM produces non-null libvfio-user `vaddr` mappings;
2. the mapping is readable and writable;
3. callback addresses match guest physical addresses;
4. the reserved `0x60000000..0x70000000` aperture is included;
5. writes through the server mapping are immediately visible in the guest and
   vice versa; and
6. unmap removes the exact range.

If QEMU supplies an indirect or insufficiently protected region, the callback
latches a fatal frontend error and the process exits after returning from the
callback. The callback is `void`, so the design does not pretend it can reject
the map in-band. A bounce or mirror implementation is not an automatic
fallback. A QEMU extension or driver shim remains a permitted new architecture
decision if the physical path proves insufficient.

## BAR0 PSP and SMU Envelope

Only the driver-reachable controller envelope is modeled. PSP and SMU internals
are not useful to NPU execution and are not emulated.

### Registers

The pinned primary driver maps these Phoenix device addresses:

| Address | Role |
|---:|---|
| `0x03010034` | PSP wait-mode status |
| `0x03010090` | PSP notify |
| `0x03010094` | SMU notify |
| `0x030100a0` | PSP command / ready status |
| `0x030100a4` | PSP argument 0 / response |
| `0x030100a8` | PSP argument 1 |
| `0x030100ac` | SMU command |
| `0x030100b0` | SMU response |
| `0x030100b4` | SMU argument / output |
| `0x030100bc` | PSP argument 2 |

The frontend initializes PSP status bit 31 (`READY`) and processes a command
only on the driver's observed notify transition `0 -> 1`.

### PSP

The pinned driver uses:

- `VALIDATE = 1`;
- `START = 2`;
- `RELEASE_TMR = 3`; and
- `START` argument 0 equal to 1.

Its Phoenix configuration uses argument-2 mask `GENMASK(23, 0)` and notify
value 1.

`VALIDATE` combines argument 0 and argument 1 into the 64-bit guest physical
address and uses the low 24 bits of argument 2 as the driver-supplied aligned
buffer size. The frontend verifies that the full span is live, copies those
bytes once from the external guest mapping, and passes them to
`xdna_emu_load_firmware`. Trailing allocator padding is accepted by the
existing `$PS1` parser, matching the driver's real 64-KiB-aligned PSP buffer.

`START` validates the expected argument shape and calls
`xdna_emu_boot_firmware` with the already-proven 200,000-instruction safety
budget. Success means the exact supplied image reached natural idle; only then
does the frontend restore ready status and response zero.

`RELEASE_TMR` replaces the emulator handle with a cold one and replays active
guest mappings. The following SMU power-off therefore sees a reset NPU, while a
later power-on/validate/start sequence boots a fresh processor.

Unexpected PSP order, malformed arguments, unmapped firmware memory, parse
failure, or boot failure is a fatal validation error. Exact PSP failure-code
parity is deferred until it is captured from hardware; the frontend must not
invent a mailbox-success path after such an error.

### SMU

The pinned driver reaches commands:

- power on/off: `3` / `4`;
- set management-NPU and H clocks: `5` / `6`; and
- soft/hard DPM level: `7` / `8`.

On the observed `0 -> 1` notify transition, the frontend:

- records the power state for commands 3 and 4;
- returns the requested clock argument through the output register for commands
  5 and 6;
- accepts the two DPM commands; and
- writes response 1 (`SMU_RESULT_OK`).

No clocks are applied to emulator timing. This is the open driver's controller
contract, not a claim about internal SMU execution or a firmware/AIE clock
ratio.

The BAR0 wait-mode bit is updated only from
`XdnaEmuFirmwareServiceStatus.wait_mode`, which is in turn derived from the
existing firmware lifecycle register. It is not forced merely because the
driver polls it.

## Firmware Mailbox and MSI-X

BAR4 already has one sparse register model for all 16 channels. Host X2I tail
publication asserts the source selected by the existing channel geometry.

Add one pending MSI-X bitmask to the firmware bus. For each channel:

```text
X2I tail base       = 0x030d0000 + channel * 0x2000
I2X status address  = 0x030d1008 + channel * 0x2000
MSI-X vector        = channel
```

The management channel is therefore vector 14 with I2X status
`0x030ed008`; the first context channel is vector 5 with I2X status
`0x030db008`.

When firmware changes an I2X status word from zero to nonzero, the bus sets the
matching bit. Repeated nonzero stores do not create duplicate edges. The host
driver clears the status word after draining the ring; the ring pointers and
status remain authoritative, while MSI-X remains only the wakeup hint.

After every bounded service call, the frontend invokes `vfu_irq_trigger` once
for each returned bit. libvfio-user owns masks, eventfds, and the pending-bit
array. An interrupt-trigger failure is fatal rather than silently dropping a
completion.

## Event Loop

Create the libvfio-user context with
`LIBVFIO_USER_FLAG_ATTACH_NB`. The one thread performs:

1. attach or reattach the QEMU client;
2. process at most one available vfio-user request with `vfu_run_ctx`;
3. if firmware is started, run one bounded firmware/array service call;
4. trigger every returned MSI-X vector;
5. continue immediately while service is non-quiescent; and
6. otherwise poll the current `vfu_get_poll_fd`.

The poll fd is refreshed after attach and after `ENOTCONN`, as required by the
pinned library. No periodic timer drives simulated work.

Firmware boot runs synchronously from the PSP `START` command. Normal mailbox
and array work runs after the triggering BAR request returns to the event loop.

## Reset and Disconnect

The same reset helper handles `VFU_RESET_DEVICE`, `VFU_RESET_PCI_FLR`, and
connection loss:

1. stop calling the firmware service function;
2. destroy the current `XdnaEmuHandle`;
3. create a fresh interpreter handle;
4. clear BAR0 command state, pending interrupts, and power state; and
5. replay every still-live external mapping into the new handle.

For connection loss, the pinned libvfio-user first invokes unregister callbacks
for every old-client range and only then invokes the `VFU_RESET_LOST_CONN`
callback. The unregisters remove those ranges from both the frontend list and
the current handle; the following cold reset therefore has no old-client maps
to replay before reattach. Ordinary firmware suspend/resume is not a PCI reset:
it continues through the real mailbox lifecycle and wait-mode contract.

## Dependency and Build Layout

Do not vendor libvfio-user or add another submodule. A small build script pins
the full commit above, clones it under ignored `build/deps/`, verifies the
checked-out SHA, builds it with its native Meson/Ninja flow, then compiles the
single C frontend against:

- the pinned libvfio-user headers/library;
- `include/xdna_emu.h`; and
- `target/debug/libxdna_emu.so` by default.

The script reports missing `meson`, `ninja`, `libjson-c-dev`, or
`libcmocka-dev` directly. It does not install system packages itself.

The run harness starts the frontend first, then QEMU with:

- KVM and a Q35 machine;
- exactly 2 GiB of shared `memory-backend-memfd` guest RAM;
- no virtual IOMMU;
- the vfio-user PCI socket;
- the hypervisor CPUID feature hidden, because the pinned driver's
  `aie2_init` rejects a non-native hypervisor type; and
- guest kernel parameter `memmap=256M$1536M`.

The guest loads `amdxdna` with its default `force_iova=0`, waits for the primary
driver to expose the accelerator, writes `0x10000000@0x60000000` to the unique
`carveout` debugfs node, verifies the value by reading it back, and only then
launches XRT. A missing, duplicate, rejected, or mismatched carveout node is a
hard harness failure.

The exact QEMU command is checked into the integration harness once it has
passed the guest-physical direct-mapping smoke test. The guest uses the normal
XRT XDNA SHIM from the pinned driver stack, never `xdna-emu/xrt-plugin`.

Halo is unnecessary for the C frontend, Rust unit tests, or brief dependency
build. It remains available only if constructing the disposable guest kernel
or image becomes a genuinely heavy build.

## TDD and Integration Order

Implementation proceeds in this order, retaining each RED result:

1. `HostMemory` tests require live external read/write, adjacent-span access,
   union range validation, overlap rejection, exact unmap, and post-unmap
   fallback without changing named-region behavior.
2. FFI tests require null/zero/overflow rejection and prove a C-owned buffer is
   changed directly by emulator host-memory writes.
3. Firmware-bus tests require a zero-to-nonzero I2X status transition to
   produce exactly one channel bit, with no duplicate before host clear.
4. Service-ABI tests prove quiescent, bounded-progress, wait-mode, MSI-X drain,
   and fatal-stop mapping by reusing the existing runtime pump.
5. The C frontend's self-test covers BAR address translation, little-endian
   split access, PSP/SMU notify edges, PSP reads from live mapped physical
   memory, and cold-reset map replay without QEMU.
6. The QEMU direct-mapping smoke test proves live GPA mappings, including the
   reserved carveout, before PSP work is trusted.
7. The pinned primary driver reaches a clean probe, exposes
   `/dev/accel/accel0`, and accepts the reserved carveout before XRT opens it.
8. The frozen Chess `add_one_using_dma` artifact runs through XRT and returns
   `2..=65`.
9. The already-validated Peano form runs through the identical path.

## Acceptance Gates

### Gate A: Driver probe

Record:

- exact QEMU, libvfio-user, driver, firmware, guest-kernel, and XRT versions;
- guest `lspci -nnvv` for the virtual function;
- unmodified primary-driver probe log;
- default `force_iova=0`, no guest vIOMMU, and successful carveout readback;
- successful firmware alive/init/query traffic;
- sixteen allocated MSI-X vectors; and
- `/dev/accel/accel0`.

No locally patched driver and no emulator XRT plugin may be present in the
guest.

### Gate B: End-to-end completion

The frozen kernel must execute:

```text
test.exe
  -> normal XRT
  -> amdxdna.ko
  -> vfio-user
  -> unmodified management firmware
  -> shared array emulator
  -> genuine firmware completion
  -> amdxdna.ko
```

Acceptance requires:

- the expected `2..=65` output;
- successful XRT command completion;
- genuine X2I consumption and I2X publication;
- genuine context-vector MSI-X delivery;
- no frontend mailbox response synthesis;
- no `xdna_emu_assign_partition` call;
- no unresolved firmware poll, unknown instruction, engine stall, or engine
  error; and
- clean context teardown.

Probe success alone does not close this milestone.

## Explicitly Deferred

- IOVA mode, a guest vIOMMU, and a simultaneous PSP-physical/ordinary-IOVA
  aperture;
- SVA, PASID, and PCI PASID capability;
- any QEMU extension or driver compatibility shim for that dual-aperture path;
- exact PSP/SMU failure codes and controller latency;
- firmware/AIE clock-ratio calibration;
- BAR mmap acceleration, ioeventfd shortcuts, and multithreading;
- vfio-user migration;
- multiple simultaneous guests or devices;
- AIE2P/XDNA2 PCI identities;
- the complete older-Phoenix firmware matrix; and
- default posted-write timing validation after the synchronous first gate.

Each deferred item needs evidence or a new acceptance requirement. None is
scaffolded into the first implementation.

## Verification

Rust gates use the linked-worktree-safe environment:

```bash
source /home/triple/npu-work/toolchain-build/activate-npu-env.sh
MLIR_AIE_PATH=/home/triple/npu-work/mlir-aie \
LLVM_AIE_PATH=/home/triple/npu-work/llvm-aie \
AIE_RT_PATH=/home/triple/npu-work/aie-rt/driver/src \
cargo test --lib

cargo test -p xdna-emu-ffi
cargo fmt --all --check
git diff --check
```

The frontend self-test, guest-physical direct-map QEMU smoke test, pinned-driver
probe, and end-to-end frozen-kernel run are separate required gates. A
library-test pass cannot substitute for either driver-boundary acceptance gate.
