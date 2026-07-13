# Phoenix alive-gate provenance: the `_NPU` builder is not gated off

Date: 2026-07-13
Branch: `feat/m2c-mapping-boot-to-idle`
Base commit: `12a99780`
Image: Phoenix `1502_00/npu.dev.sbin`
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Verdict

**VERIFIED: there is no defaulted SMU/PSP input that gates the clean emulator
away from the alive-struct builder. The clean Phoenix execution already runs
that builder.** The task brief's premise that the `_NPU` producer lies in an
unexecuted continuation behind the later `a7 < 6` service guard is false for the
pinned image and current `load_m2c` mapping.

The unique `_NPU` producer is the function at `0x5044..0x50d2`. On the clean
path it:

1. constructs every non-reserved field of the 0x40-byte
   `mgmt_mbox_chann_info` at management-local `EA/PA 0x14800..0x1482f`, including
   `_NPU` at `0x14820`; and
2. composes the device-absolute pointer `0x030bb000` bytewise at management-local
   `EA/PA 0..3`.

Both actions precede the first service guard and every `HARNESS_VIEW` action.
The later `a7=6` reject therefore cannot gate this builder. The scheduler state
that selects slot 6 is genuine firmware-local state and is the state that
*causes* the one go-alive dispatch; it does not steer execution away from it.

The new read-only input observer records **zero firmware loads** from every
driver-defined Phoenix SMU/PSP handoff register through the clean `0x8cb1`
frontier. The open-source command values (`SMU_POWER_ON=3`, `PSP_START=2`,
`PSP_START_COPY_FW=1`, and so on) are host-to-SMU/PSP commands. They are not
management-firmware data inputs. Writing them into firmware-visible memory
would be state injection, not derivation.

This finding does **not** claim that host-visible alive publication is complete.
The previously verified negative store inventory still stands: before the
current `0x8cb1` frontier, no retired store writes the staged descriptor to
device SRAM `0x030bb000..0x030bb03f`, and no retired store writes its pointer to
`FW_ALIVE_OFF` at `0x030bf000`. What is overturned is the narrower predecessor
claim that no struct was built and that its builder must be behind the service
guard. The remaining problem is a **downstream host-visibility/copy-out
provenance problem**, not an upstream builder-input gate.

## Fidelity boundary

- The image hash above was recomputed immediately before the probe run.
- No `17f1_10` byte or semantic was used.
- The new `m2c_probe_alive_handoff_inputs` test decodes through the same fetch
  translation used by the existing probe family, then reads post-retirement
  register values and the already resident DTLB entry. The pre-decode can fill
  the ITLB immediately before the same instruction would fill it; the asserted
  clean terminus guards architectural equivalence. Data observations never
  invoke translation. The probe does not write firmware data, registers,
  interrupts, branches, mappings, or MMIO.
- The run uses plain `FirmwareProcessor::load_m2c`. It adds no overlay, forced
  branch, scheduler value, interrupt, or host input.
- Production `load_m2c`, `mod.rs`, `mmio.rs`, `system.rs`, `sysstub.rs`, and the
  runtime MMIO behavior are unchanged.
- The clean observer stops at the established natural decode frontier:
  `Unknown pc=0x8cb1 word=0x61a800`, at step index 53,659.

## Builder location

### Static producer

An exact little-endian byte scan for `5f 4e 50 55` finds one occurrence in the
entire signed image:

```text
file 0x3388: 5f 4e 50 55 = 0x55504e5f
```

The live low-VMA section uses `file = VMA + 0x100`, so this is literal-pool VMA
`0x3288`. The builder instruction sequence is:

```text
0x5044  Entry
0x504a  L32r a2, [0x3274]  -> 0x00014800  (descriptor base)
...
0x507f  L32r a15,[0x3288]  -> 0x55504e5f
...
0x5092  S32iN a15,[a2+32]  -> descriptor.magic
...
0x50ba  L32r a2, [0x31bc]  -> 0
0x50c6  S8i [a2+0] <- 0x00
0x50c9  S8i [a2+3] <- 0x03
0x50cc  S8i [a2+2] <- 0x0b
0x50cf  S8i [a2+1] <- 0xb0
0x50d2  RetwN
```

The four byte stores compose little-endian `0x030bb000` at local word zero.
They do not themselves establish that local word zero is the driver's
`FW_ALIVE_OFF` slot; the resident DTLB resolves these accesses to PA 0..3.

### Dynamic construction

The clean trace retires these builder stores:

| n | PC | Local EA/PA | Value | Driver field |
|---:|---:|---:|---:|---|
| 51995 | `0x5058` | `0x14800` | `0x030ec000` | `x2i_tail` |
| 51996 | `0x505a` | `0x14804` | `0x030ec004` | `x2i_head` |
| 52011 | `0x5067` | `0x14808` | `0x030bc000` | `x2i_buf` |
| 52016 | `0x5072` | `0x1480c` | `0x00000400` | `x2i_buf_sz` |
| 52015 | `0x5070` | `0x14810` | `0x030ed000` | `i2x_tail` |
| 52017 | `0x5074` | `0x14814` | `0x030ed004` | `i2x_head` |
| 52033 | `0x5082` | `0x14818` | `0x030bd000` | `i2x_buf` |
| 52038 | `0x508c` | `0x1481c` | `0x00000400` | `i2x_buf_sz` |
| 52041 | `0x5092` | `0x14820` | `0x55504e5f` | `magic` |
| 52039 | `0x508e` | `0x14824` | `0x0000000e` | `msi_id` |
| 52031 | `0x507d` | `0x14828` | `0x00000005` | `prot_major` |
| 52040 | `0x5090` | `0x1482c` | `0x00000008` | `prot_minor` |

The reserved words at `0x14830..0x1483f` remain zero. This field order is an
exact match for the driver's `struct mgmt_mbox_chann_info` in
`aie2_pci.c:54-69`. It also matches the Phoenix hardware object previously read
at device `0x030bb000`, word for word: the same ring registers, 1 KiB buffers,
magic, MSI vector 14, and protocol 5.8. This is not merely a magic stamp; it is
the alive-channel descriptor builder.

The pointer-staging stores then retire at `n=52119..52122`, with local EA and PA
both equal to 0..3. This preserves the predecessor's address-space distinction:
the *value* is the device address `0x030bb000`; the executed destination is still
private low memory in the current model.

## Backward chain from builder to launch

All dynamic rows below are before the first `HARNESS_VIEW` at `n=53640`.
“Ordinary local” means management-core local data written by firmware, not a
host BAR value supplied by the harness.

| Status | n / PC | Backward edge | Memory class and consequence |
|---|---|---|---|
| **VERIFIED** | `52041 / 0x5092` | `_NPU` is stored into `[0x14800+0x20]`. | Firmware-local descriptor state; class **(b)**. |
| **VERIFIED** | `51765 / 0x560d` -> `51766 / 0x5044` | `goalive_runfn` makes a direct `Call8 0x5044`. Its top-level code from `0x55f8` through `0x560d` contains no conditional branch, and every intervening callee returns on the clean trace. | No top-level go-alive decision diverts the clean path before the builder. |
| **VERIFIED** | `49925 / 0x55f8` | The fixed-pool worker dispatches the one go-alive run function. | Firmware-local queued record; class **(b)**. |
| **VERIFIED** | `47985 / 0x285d` | Scheduler writes `[0x2278] <- 0x10dfc` after loading runnable slot 6 through `[0x22a0]`. | Firmware-local current-task state; class **(b)**. |
| **VERIFIED** | `47884..47886 / 0x27b7..0x27bc` | Two queued-record/task criteria match, so `Moveqz` selects loop index 6 and the scheduler stores selector 6. | Firmware arithmetic plus ordinary local data; class **(b)**. |
| **VERIFIED** | `47361..47383 / 0xd6e3..0xd785` | The mapped queue enqueuer writes record fields, the `0x55f8` run-function record, and count `[0x24c4] <- 1`. | Firmware-local queue state; class **(b)**. |
| **VERIFIED** | `39730..39852 / 0xd4ef..0xd60f` | The task initializer writes task slot `[0x10e04] <- 6`, match fields, and runnable link `[0x22a0] <- 0x10dfc`. | Firmware-local task construction; class **(b)**. |
| **VERIFIED source contract; CLAIMED emulator equivalence** | host boot | The driver completes SMU power cycling and PSP validate/start before polling alive; `load_m2c` directly begins the already-loaded CPU instead of replaying that host-controller protocol. | Launch side effect, not a management-firmware load. It has already happened if reset code is executing. |

The queue count later retires from 1 to 0 at `0xccae`, and the second scheduler
scan falls back to selector 0. The first service call reaches `Bgeui 0x7fc7`
with `a7=0` at `n=53632`, **after** `_NPU` was stored at `n=52041`. The later
`a7=6` check at `n=53873` occurs only after the probe's counterfactual view
transports. Neither service value can be an upstream builder gate.

This also answers why slot 6 exists: it is deliberately seeded by the firmware's
task initializer and queue enqueuer, selected once while the queue count is one,
and dispatched once. There is no missing host value in that producer cone.

## SMU/PSP handoff audit

The open-source driver defines these host-side transactions:

- `aie_smu_init` sends power-off command 4 and then power-on command 3, each
  with argument 0; it toggles `SMU_INTR` 0 then 1 and requires response 1
  (`aie_smu.c:15-23,33-84`).
- `aie_psp_start` first validates the firmware, then sends `PSP_START=2` with
  `arg0=PSP_START_COPY_FW=1`, `arg1=0`, and `arg2=0`; the driver toggles the
  PSP interrupt with notify value 1, waits for status bit 31, and requires
  response 0 (`aie_psp.c:16-25,37-45,59-98,142-173` and
  `aie2_pci.c:584-589`).
- Phoenix maps those registers to BAR0 device addresses
  `0x03010034`, `0x03010090`, `0x03010094`, and
  `0x030100a0..0x030100bc` (`npu1_regs.c:16-47,119-135`).

The observer checks every retired scalar and FP load by both effective address
and resident-DTLB physical address against all ten registers. Result:

```text
builder_entry=Some(51766)
builder _NPU store: n=52041 pc=0x5092 EA=PA=0x14820 value=0x55504e5f
handoff_loads=0
stop=Unknown pc=0x8cb1 word=0x61a800
```

Thus the apparently “specific values” in the driver are not values the
management firmware waits to read. They instruct two other agents to power and
launch it. Once the management CPU executes reset code, those commands have
already served their purpose.

The only low NPU-aperture load sites observed on the clean path are useful
negative controls:

| Site | Address and values | Classification |
|---|---|---|
| `0x08b0421f` | BAR0 `0x03010d7c`, five reads evolving `0 -> 2 -> 0x204 -> 0x20405 -> 0x2040506` | Firmware-owned task-notification shift register. The same helper writes the shifted value back. Its initial zero is not a driver handoff value and it does not gate the builder. Operationally class **(b)** despite the BAR0 address. |
| `0x8946` | BAR2 `0x030b27c0 -> 0`, once at `n=52547` | Post-builder read/modify/write data path. `0x8948` masks the value and `0x894d` unconditionally jumps to the store at `0x8964`; it does not feed a control branch and cannot explain builder reachability. Outside the gate cone. |

This distinction matters because address class alone does not establish input
provenance. A BAR address can hold firmware-owned state, and a driver-written
controller register is not automatically readable by the launched firmware.

## (a)/(b)/(c) classification and corrected discriminator

| Proposed input | Classification | Verdict |
|---|---|---|
| Driver SMU/PSP command/status registers | Would be class **(a)** if a firmware load consumed them, but **zero such loads execute**. | Not a gate. Do not inject command/status values into firmware memory. |
| Task slot, queue record/count, runnable table, current task, and direct `0x560d` call | Class **(b)**: genuine firmware logic over correctly modeled local state. | Complete and sufficient; it reaches the builder once. Nothing is missing. |
| Later `a7=0` / `a7=6` service checks | Class **(b)** values, but temporally downstream; `a7=6` is additionally reached only after counterfactual transport. | Not a builder gate. |
| First real host-visible writer PC/EA/PA and its natural control predecessor | Not present in the executed trace or specified by the consuming driver. Under the present evidence boundary this is the unresolved downstream hardware datum, not an upstream input value. | Open publication-provenance question; do not relabel it as a defaulted SMU/PSP read. |

There is therefore no honest class-(a) “input + driver value” fix for the task's
stated goal of making firmware run its builder: firmware already runs it. All
actual upstream gating state is class (b), and it is present. The requested
three-way discriminator was framed around a nonexistent divergence; forcing it
to return (a) would manufacture a causal edge the trace disproves.

The host contract remains exact and independently verified: after PSP start,
the driver polls BAR2 `FW_ALIVE_OFF`, reads the pointer, consumes the 0x40-byte
object, then clears the slot (`aie2_pci.c:71-131,375-405`; Phoenix offsets in
`npu1_regs.c:28-30,105-135`). That contract specifies the required output. It
does not specify which management-firmware instruction or internal transport
copies the already-built local object into host-visible SRAM.

## Ranked derive-only next step

**1. Capture the first real Phoenix store to either
`0x030bb000..0x030bb03f` or `0x030bf000`, including writer PC, EA/PA, current
task/`a7`, and the immediately preceding control edge; then byte-match that PC
against this exact `1502_00` image.**

This is the single observation that separates a later direct CPU copy, an
internal transport, and another host-visible alias without guessing. It should
be a read-only, single-shot kernel/IRQ-side capture or management-core trace;
never sustained host polling of BAR0. It requires no branch forcing, scheduler
injection, PSP-loader reopening, `0x8cae` mechanism work, below-CPU-bank hunt,
or `17f1_10` semantics.

Until that provenance exists, no SMU/PSP input model should be added on the
strength of this task. The shortest correct implementation change today is no
production change.

## Probe change and reproduction

Added one test-only, env-gated observer:

```text
src/firmware/boot_tests/coherence_mapper.rs
  m2c_probe_alive_handoff_inputs
```

It records the builder's local stores, every load whose EA or PA falls in the
Phoenix NPU apertures, and exact reads of all driver-defined SMU/PSP registers.
It uses the resident DTLB after retirement and makes no extra translation call
for data observations.

```bash
XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_alive_handoff_inputs -- --nocapture \
  > build/experiments/firmware-re/alive-handoff-inputs.log 2>&1

XDNA_FW_PROBE=1 XDNA_FW_DISASM=0x501c:0x5190 cargo test --lib \
  m2c_probe_disasm_range -- --nocapture \
  > build/experiments/firmware-re/alive-builder-disasm.log 2>&1
```

Fresh verification results are recorded after the final full-suite run below.

## Verification

Fresh results for this uncommitted finding/probe diff:

```text
targeted handoff-input probe:
  1 passed; 0 failed; 4121 filtered out

cargo test --lib:
  4092 passed; 0 failed; 30 ignored
  (4091 baseline tests plus this env-gated probe)

cargo fmt --all -- --check:
  exit 0

git diff --check:
  exit 0

untracked finding whitespace scan:
  no errors
```
