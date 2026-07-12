# Alive SRAM writer remains behind a runtime VMA-view collision

Date: 2026-07-11  
Target: Phoenix/NPU1 firmware `1502_00/npu.dev.sbin`  
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`  
Branch: `feat/m2c-mapping-boot-to-idle`

## Verdict

Within the current vaddr-keyed BASE/AT model, no static `load_m2c` overlay tuple
or boundary change can carry the natural boot to the device-SRAM alive writer.
The current `0x8cb1` stop is not a mis-sized instruction: it is the first
visible consequence of two naturally executed call graphs requiring different
file views at the same VMA.

The hardware oracle remains unmet. A natural emulator boot leaves all words at
device `0x030bb000..0x030bb03c` zero and leaves `FW_ALIVE_OFF` at
`0x030bf000` zero. The only observed store in the `0x030bxxxx` page remains
`n=52551 pc=0x8964: 0x030b27c0 <- 0`.

The exact missing mechanism is unresolved. The evidence requires a
time-dependent code view (or an equivalent mechanism outside the current
vaddr-keyed flat overlay model), but the signed image contains no selector or
scatter metadata that identifies it. A test-only counterfactual that supplies
the two immediately required view changes crosses both Unknowns, then reaches
an intentional service-reject sink without building the SRAM struct. Therefore
even those two oracle-selected changes are not a sufficient hardware model.

## 1. Reproduction and acceptance RED

The current natural boot reproduces deterministically:

```text
n=53660 stop=Unknown pc=0x8cb1 word=0x61a800
SRAM stores: n=52551 pc=0x8964 EA=PA=0x030b27c0 value=0
exact FW_ALIVE stores=0
```

`m2c_probe_alive_device_sram_struct` is the BAR2-dump acceptance oracle. With
`XDNA_FW_PROBE=1`, it records each SRAM-page store's PC, EA, translated PA,
width, and value, then asserts the exact 16-word hardware layout at
`0x030bb000` and the pointer `0x030bb000` at `0x030bf000`. On the current
natural mapping it fails for the intended reason:

```text
IdleReport: unknown_op=Some((0x8cb1, 0x61a800)), instrs_executed=53659
no store targets 0x030bb000..0x030bb03f or 0x030bf000
0x030bb000..0x030bb03c = sixteen zero words
0x030bf000 = 0
```

This is an env-gated probe; the normal suite does not opt into the failing
hardware acceptance assertion.

## 2. `0x8cb1` is a real BASE instruction boundary

The brief's statement that both framings are garbage at `0x8cb1` is
overturned. Linear decode and the executed predecessor establish a clean BASE
tail:

| VMA | File (BASE = VMA+0x5c) | Decode |
|---|---:|---|
| `0x8cae` | `0x8d0a` | `Addi a8,a8,0x60` (3 bytes) |
| `0x8cb1` | `0x8d0d` | `Addmi a4,a4,0x1000` (3 bytes) |
| `0x8cb4` | `0x8d10` | `Addmi a5,a5,0x1000` (3 bytes) |
| `0x8cb7` | `0x8d13` | `Wsr 0xe6,a3` (3 bytes) |
| `0x8cba` | `0x8d16` | `RetwN` (2 bytes) |

Natural execution reaches the cell from the BASE-framed service function:

```text
n=53640 pc=0x8c6c Entry
n=53657 pc=0x8c8b Bbci bit=3 target=0x8cae
n=53658 pc=0x8cae       # production AT overlay serves S8i, not BASE Addi
n=53659 pc=0x8cb1       # production AT overlay serves Unknown 0x61a800
```

Thus the predecessor is correctly sized and its branch target is real. The
over-broad AT interval `[0x8c98,0x8d52)` causes the bad bytes.

## 3. A static boundary fix regresses the earlier publisher

The same boot executes the AT-framed function rooted at `0x8c98` five times
before the service path. Its `Bgeu` at `0x8cac` is three bytes long, so its
third byte occupies VMA `0x8cae`; the taken path then executes the AT
`MoviN a11,-1` at `0x8cb4`. The later BASE service instruction must begin at
that same byte `0x8cae`. The collision is therefore byte-level even though the
two paths do not share every instruction PC.

`m2c_probe_execution_guided_framing_search` pins both backward cones and tests
all four assignments for the shared code cell `[0x8cae,0x8cbc)` and shared
literal cell `[0x354c,0x3550)`:

```text
code=BASE literal=BASE: publish fails at 0x8c32; magic remains 0
code=BASE literal=AT:   publish fails at 0x8c32; magic remains 0
code=AT   literal=BASE: publish reaches 0x5645; service Unknown at 0x8cb1
code=AT   literal=AT:   publish reaches 0x5645; service Unknown at 0x8cb1
free section variables: []
```

Therefore shrinking or splitting the production interval fixes the later
service decode only by breaking the earlier publisher. Extending the AT tuple
preserves the publisher only by breaking the service. There is no honest static
tuple to land.

## 4. Runtime-view counterfactual

`m2c_probe_runtime_view_discriminator` changes only test-time fetch overlays.
It does not write firmware data, inject registers, assert interrupts, or alter
production decode/MMU/scheduler behavior.

At the natural entry to `0x8c6c` (`n=53640`), the probe selects BASE only for
the service-only overlap and its literal. The BASE tail returns coherently to
`0x7fe4`, proving the `0x8cb1` wall itself is understood. Execution then exposes
a second shared-view call:

```text
n=53783 pc=0x7fe7 Call8 target=0x26d4
n=53784 pc=0x26d4
```

At `0x26d4`, BASE file `0x2730` begins `36 a1 00` (`Entry a1,0x50`), while
the production AT view reads file `0x27d4` as `a0 ff 39` (`Unknown`). Those
BASE bytes are the `+0xa4` alias of the same context-switch section already
required at AT VMA `0x2630`; changing that production interval statically would
regress the earlier context-switch path.

After test-time selection of the BASE `0x26d4` alias, execution remains
decodable but reaches a deliberate reject sink:

```text
n=53872 pc=0x7fc4 Entry
n=53873 pc=0x7fc7 Bgeui a7,6,target=0x7fec
n=53874 pc=0x7fec a7=6 EXCCAUSE=1 EPC1=0x08b0e713 INTERRUPT=0
0x7fec: J 0x7fec  [06 ff ff]
```

No additional `0x030bxxxx` store occurs before the sink. This refutes the
narrow hypothesis that supplying just the two evident runtime views is enough
to reach the hardware writer.

## 5. Consequence

The next faithful step is not another `add_rom_overlay` tuple. It requires
external evidence for the runtime code-view/bank state (or another mechanism
that explains both naturally executed aliases) and for the service state that
avoids the `a7=6` reject. The flat `$PS1` image and current execution trace do
not provide that discriminator.

No production overlay, MMU, scheduler, interrupt, or firmware state change was
made. The only code additions are env-gated acceptance/counterfactual probes
and a `#[cfg(test)]` overlay-removal helper used by the counterfactual.

## Reproduction

```text
XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib \
  m2c_probe_alive_publish_mechanism -- --nocapture

XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_execution_guided_framing_search -- --nocapture

XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib \
  m2c_probe_alive_device_sram_struct -- --nocapture

XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib \
  m2c_probe_runtime_view_discriminator -- --nocapture
```
