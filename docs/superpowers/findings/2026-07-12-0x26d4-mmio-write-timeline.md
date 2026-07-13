# Phoenix 0x26d4 MMIO-write timeline

Date: 2026-07-12

Target: Phoenix/NPU1 `1502_00/npu.dev.sbin`

Firmware SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

Branch: `feat/m2c-mapping-boot-to-idle`

Base commit: `108f0a76`

## Verdict

**VERIFIED (reconstructed execution): the firmware performs no MMIO write that
can select the low instruction view.** The raw non-local timeline is not empty:
91 stores retire between the early AT crossing and the later BASE entry. Every
one classifies as ordinary NPU bring-up, scheduler notification, SRAM data, or
a high-region data access. The actual service/view-transition chain executes
zero non-local stores.

The MMIO-selector candidate set is therefore **cleanly empty**. Combined with
commit `108f0a76`'s zero I-cache/page-root/ITLB selector result, the surviving
class is an **external or below-CPU instruction-bank agent with no firmware
MMIO trigger**. Nothing in this trace maps to the AMD-private `sram_alias` slot
model.

This is bounded to the executed interval and the firmware-visible stores. It
does not identify whether the external agent is a fixed hardware fetch bank, a
context-coupled bank below the Xtensa MMU, or a non-CPU agent. It does close the
specific blind spot posed by the brief: a plain vendor-register view-selector
write by this firmware is absent.

## Method

The committed env-gated
`m2c_probe_26d4_cache_pageroot_timeline` now also records every retired store
whose firmware-visible effective address is outside ordinary low
`local_data`. It reuses the same two `HARNESS_VIEW` transport events; those are
marked `not firmware` and excluded from store counts.

For each store the probe records:

- instruction count, full PC, width, effective address, and value;
- phase (`between-views` or `post-view`);
- target-region classification; and
- the dynamically maintained call-target chain.

`S32c1i` is counted only when its compare succeeds. Stores inside a FLIX bundle
are inspected individually. At the early `0x26d4` crossing the probe disables
the interpreter's fill-loop optimization so a collapsed loop cannot hide
individual stores; this changes only host execution speed, not architectural
state or firmware behavior.

The NPU1 device-region split comes from xdna-driver
`drivers/accel/amdxdna/npu1_regs.c:30-47`: BAR0 management starts at
`0x03000000`, BAR2 shared SRAM at `0x03080000`, and BAR4 mailbox registers at
`0x030c0000`. These ranges are checked before the emulator's provisional
64-MiB low-data predicate because the device addresses numerically fall below
`0x04000000`.

## Ordered evidence

The view epochs remain unchanged:

```text
n=47672 pc=0x0026d3 early AT Movi, len=3, spans byte 0x26d4
n=53784 pc=0x0026d4 later BASE Entry
```

The complete store log contains 91 lines. Its region summary is:

```text
STORE_SUMMARY count=91 regions={
  "aie-array-noc-mmio": 68,
  "device-bar0-management": 3,
  "device-bar2-shared-sram": 1,
  "device-mailbox": 18,
  "high-data-alias": 1
}
post-view stores=0
critical-transition-path stores=0
```

There are also zero stores to Segment-B RAM, the `0x2000xxxx` high code alias,
or the page-table aperture in this interval.

### The only BAR0 writes: task notifications, not bank selection

```text
n=50908 pc=0x08b04229 STORE4 EA=0x03010d7c value=0x00020405
  chain=0x08b041bc>0x08b0424c>0x000055f8>0x08b041f0
n=51762 pc=0x08b04229 STORE4 EA=0x03010d7c value=0x02040506
  chain=...>0x000055f8>0x00009704>0x08b041f0
n=52194 pc=0x08b04229 STORE4 EA=0x03010d7c value=0x04050607
  chain=...>0x000055f8>0x00007cf0>0x00007bd0>0x08b041f0
```

**VERIFIED:** all three are the same Segment-B helper at `0x08b041f0`.
The earlier instruction trace established its operation: read the old
`0x03010d7c` word, shift it by one byte, append the task index, and write it
back (`2026-07-08-boot-wake-breach-journey.md:1902-1911`). The rolling values
above are the direct dynamic signature of that queue/notification operation.
Hardware observation independently saw byte-shifting at BAR0 offset `0x10d7c`
under workload activity
(`2026-07-07-hw-mmio-observability-and-smu-crash.md:97-106`).

This cannot be mapped to the `sram_alias` model:

- it is a repeated queue RMW, not a one-time bank/slot selection;
- its value packs task-history bytes rather than an aligned size plus slot
  index; and
- no companion write carries a power-of-two size from 4 bytes through 4 KiB,
  the constraint recovered from `sram_alias.c` strings.

### The one BAR2 write: shared-SRAM data

```text
n=52551 pc=0x00008964 STORE4 EA=0x030b27c0 value=0x00000000
  region=device-bar2-shared-sram
  chain=...>0x00007c5c>0x00007d4c>0x00008934
```

**VERIFIED:** this is the already-traced shared-SRAM publish-path data store.
It writes zero into BAR2 SRAM; it is not a control-register transaction and
has no slot/size encoding.

### AIE/NOC and `0x272xxxxx` programming

The 68 AIE/NOC stores are confined to the `0x95ec..0x9775` config-helper
family. Representative groups are:

```text
pc=0x964e  EA=0x9c0fff20..0xa40fff20 value=1  (five columns)
pc=0x9668  EA=0x9c0fff28..0xa40fff28 value=1  (five columns)
pc=0x9677  EA=0x9c0fff28..0xa40fff28 value=0  (five columns)
pc=0x96a7  EA=0x9c036030..0xa4536030 value=0  (30 tiles)
pc=0x9704 family, EA=0xac000000..0xac000208      (NOC/config block)
```

These offsets are independently named by the open-source toolchain:

- `0x00036030` is `Tile_Control`
  (`mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json:8889-8893`,
  `aie-rt/driver/src/global/xaiemlgbl_params.h:3696`);
- `0x000fff20` is `Column_Clock_Control`
  (`aie_registers_aie2.json:88602-88606`,
  `xaiemlgbl_params.h:15878`); and
- `0x000fff28` is `AIE_Tile_Column_Reset`
  (`aie_registers_aie2.json:88621-88637`).

The 18 `0x272xxxxx` stores likewise come from publisher/config helpers:

| Store PC | Count | Targets | Dynamic caller |
|---:|---:|---|---|
| `0x8db0` | 12 | `0x27200904..0x27200920` | `goalive_runfn -> 0x50d4/0x5044 -> config helpers` |
| `0x871c` | 2 | `0x27200304` | `goalive_runfn -> 0x5044 -> 0x58dc -> 0x86f8` |
| `0x4a33` | 3 | `0x272100bc/f8/fc` | `goalive_runfn -> 0x5044 -> 0x4a0c` |
| `0x89ac` | 1 | `0x27220040` | publisher scheduler/lookup chain |

They are ordinary NPU-array/mailbox programming, not low-IRAM alias control.
None runs in the later service chain.

### The remaining high-region store

```text
n=52819 pc=0x0000c8a4 STORE4 EA=0x40000013 value=0
  region=high-data-alias chain=...>0x0000c6b0>0x0000c894
```

This isolated, unaligned high-data write is not a plausible MMIO register
selector: it is produced in the scheduler/event path, has value zero, has no
companion slot/size write, and is 748 retired instructions before the service
callback begins. It is retained in the inventory rather than silently
discarded because the exact silicon meaning of the `0x4000xxxx` data view is
not yet documented.

### The transition seam is store-free

The last non-local store is the `0x40000013` line at `n=52819`. The service
callback starts 748 instructions later:

```text
n=53567 pc=0x00283b Callx8 -> 0x8770
n=53578 pc=0x00878a Call8  -> 0xc530
n=53596 pc=0x00c55c Callx8 -> 0x08b0e710
n=53630 pc=0x00c56e Call8  -> 0x7fc4
n=53639 pc=0x007fe1 Call8  -> 0x8c6c
n=53673 pc=0x007fe4 Call8  -> 0xd7f0
n=53690 pc=0x00d836 Call8  -> 0xc938
n=53783 pc=0x007fe7 Call8  -> 0x26d4
n=53784 pc=0x0026d4 Entry  (later BASE view)
n=53813 pc=0x002734 Call8  -> 0xc530
n=53874 pc=0x007fec service sink
```

**VERIFIED:** no non-local store retires from `n=52820` through the sink.
The probe additionally guards zero stores whose PC lies in `0x26d4`,
`0x7fc4`, `0x8c6c`, `0xc530`, or the Segment-B `0x08b0e710` helper. Thus the
actual context/service seam contains no hidden plain-store selector.

## Mechanism implication

The firmware does execute normal MMIO during the broad early-to-late interval,
but each group has an independently observed purpose and none has the register
or value shape of the private alias-slot facility. More decisively, no store at
all occurs on the final dispatch chain where the instruction view changes.

The result therefore strengthens the architecture model from “no architected
Xtensa selector” to:

> **VERIFIED negative:** no firmware-executed cache, MMU, TLB, or plain MMIO
> operation selects the `0x26d4` view in the reconstructed transition.

The temporal view must be supplied without a firmware instruction-level
trigger: hardware instruction banking keyed below the CPU-visible state, or an
external agent. Firmware-image RE can still explain what each context expects,
but it cannot recover a selector instruction that is not present.

## Critical-path function worklist seed

These names are deliberately conservative. `VERIFIED` notes describe executed
edges or operations; semantic names not yet backed by a full function RE remain
worklist hypotheses.

| Function | Evidence-backed role | Next static-RE question |
|---:|---|---|
| `0x2630` | **VERIFIED:** early AT context-switch body; the `0x26d3` instruction spans byte `0x26d4` and the function calls `0xc48c`. | What context identity is saved/restored, and is any non-MMIO state available to a hardware bank selector? |
| `0xc48c` | **VERIFIED:** IPC/cache-maintenance bridge; calls the Segment-B `0x08b0e710` helper over local `0xfae0..0xfb60`. | Name its message object and cache-coherency contract. |
| `0x8770` | **VERIFIED:** registered service callback loaded from `[0x1187c]`; reaches `0xc530`. | Decode its argument transformation before the `0x878a` call. |
| `0xc530` | **VERIFIED:** service wrapper; calls `0x08b0e710`, then `0x7fc4`. It runs both before and after the later BASE `0x26d4` entry and performs no non-local store. | Identify the object fields passed to `0x7fc4` and why the second call carries `a7=6`. |
| `0x08b0e710` | **VERIFIED:** Segment-B data-cache helper; nine `Dhwbi` operations over `0xfae0..0xfb60`, `Dsync`, return. | Recover its exact API: buffer base, length, and direction. |
| `0x7fc4` | **VERIFIED:** service sequencer; calls `0x8c6c`, `task_dispatcher`, then `0x26d4`; later rejects `a7>=6` at `0x7fc7`. | Define the state machine and meanings of `a7=0` and `a7=6`. |
| `0x8c6c` | **VERIFIED:** genuine BASE-framed service subroutine, called at `0x7fe1`, returning at `0x8cba`; no non-local store. | Recover the branch at `0x8c8b` and the object update made before return. |
| `0xd7f0` | **VERIFIED name:** `task_dispatcher`; calls `0xc938` in this trace. | Reconcile this path with the later ready-list scheduler model. |
| `0xc938` | **OBSERVED:** scheduler helper entered from `0xd836`, returning at `0xc97e`; older findings call this the ready/popcount path. | Revalidate the full body under the now-coherent framing before assigning a final name. |
| `0x26d4` | **VERIFIED:** later BASE `Entry a1,0x50`; loads task state and calls `0xc530` at `0x2734`. Earlier AT execution crosses the same VMA byte from `0x26d3`. | Compare BASE and AT function contracts field-by-field; identify why two contexts share this VMA. |
| `0x08b041f0` | **VERIFIED:** Segment-B task-notify helper; the sole writer of BAR0 `0x03010d7c` in this interval. | Name the notification protocol and all call sites; no alias-selector RE is warranted here. |
| `0x95ec..0x9775` | **VERIFIED:** NOC/tile/clock/reset programming family responsible for all 68 AIE/NOC stores. | Split into functions using the toolchain register names and recover each argument contract. |

This worklist keeps the later function-by-function track on the trace-identified
critical path without reopening either closed investigation.

## Reproduction

```bash
mkdir -p build/experiments/firmware-re
XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_26d4_cache_pageroot_timeline -- --nocapture \
  > build/experiments/firmware-re/0x26d4-mmio-write-timeline.log 2>&1
```

The targeted probe reached `0x7fec` at `n=53874` and passed with the exact
91-store inventory above. Final `cargo test --lib` verification passed:
**4091 passed, 0 failed, 30 ignored** in 49.46 seconds. No production firmware
behavior was changed.
