# Phoenix alive-publication reconciliation

Date: 2026-07-13  
Branch: `feat/m2c-mapping-boot-to-idle`  
Base commit: `d5824e21`  
Image: Phoenix `1502_00/npu.dev.sbin`  
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Verdict

**VERIFIED: the successor over-claimed alive publication. The predecessor's
service-reject wall stands.** The clean go-alive worker writes the *value*
`0x030bb000` into management-core local `VA/PA 0..3`; it does not write the
device-SRAM object at device `0x030bb000`, and it does not write the pointer
slot at `FW_ALIVE_OFF` (BAR2 offset `0x3f000`, device `0x030bf000`).

The local word is not terminal, but its clean-path consumers do not publish it
to the host alive destination. They use byte `0xb0` to derive a device-mailbox
control value, test the pointer value as a bitset, and set bit 0 back in the
same local word. The word is then wholly repurposed before the first
`HARNESS_VIEW`. Across every retired store through the `0x7fec` sink, including
the counterfactual service prefix, there are **zero** writes to either the
`0x030bb000..0x030bb03f` alive struct or `0x030bf000`.

The hardware oracle remains decisive about what real publication looks like:
Phoenix constructs the 0x40-byte `mgmt_mbox_chann_info` object directly in
device SRAM at `0x030bb000`, then transiently writes its device-absolute
pointer to `0x030bf000`. Since none of those stores occurs before the service
guard in this execution, that producer is in the unexecuted continuation
behind the service wall. Local `VA 0..3` is only pre-staged control state.

This does not retract the successor's verified selector result. Slot 6 is
selected once by mapped firmware, the queue retires, and `0x55f8` runs once.
What is retracted is only the interpretation of four local byte stores as host
publication.

The wall statement has one exact fidelity qualification: the current probe
reaches the later `a7=6` guard only after its labeled `HARNESS_VIEW`
counterfactuals. Therefore this finding does not claim that the clean mapped
run naturally reaches `0x7fc7`. It says that the counterfactual transport still
cannot reach the real publication continuation because genuine scheduler
state fails that continuation's `a7 < 6` contract. The reject is the remaining
modeled wall, not evidence supplied by the harness.

## Probe and fidelity boundary

The env-gated `m2c_probe_26d4_cache_pageroot_timeline` observer was extended to
record:

- every post-publisher load overlapping local `VA 0..3`, covering every
  implemented scalar load form, including `lsi` FP-register loads and fused
  `Flix1` slots;
- every retired scalar store, with its effective address and its physical
  address derived read-only from the already-resident DTLB entry;
- a complete translated BAR2 store inventory; and
- the ordered local-word store/load history with
  `clean` versus `post-forcing` phase tags.

The PA observer does not call the mutating translation/autorefill path. The
real instruction has already retired and populated any required DTLB entry;
the probe only reads that entry. The additions do not change firmware memory,
registers, MMU state, interrupts, MMIO, branches, mappings, or production
behavior. No new forcing was added. Existing `HARNESS_VIEW` actions remain
explicitly labeled as counterfactual transport.

The Phoenix image hash above was recomputed. No `17f1_10` byte or semantic was
used.

Path tags in this finding are exact:

```text
clean / un-forced:                 n < 53640
first HARNESS_VIEW counterfactual: n = 53640
BASE 0x26d4 counterfactual:        n = 53784
a7=6 reject:                       n = 53873
service sink / trace stop:         n = 53874
```

## Consumer chain for local `VA 0..3`

All rows are **VERIFIED** retired Phoenix instructions. `ordinary local` means
management-core local RAM. The DTLB-resolved physical addresses are the same
low addresses; they are not device `0x030bb000` and are outside every host BAR.

| n | PC | Executed edge | Address/value | Consequence |
|---:|---:|---|---|---|
| 52119-52122 | `0x50c6..0x50cf` | four `S8i` stores | local `EA/PA 0..3 <- 00 b0 0b 03` | Stages local word `0x030bb000`; no device-SRAM write occurs. |
| 52552 | `0x8966` | `L8ui a6,[a3]` | local `EA/PA 1 -> 0xb0` | First post-publisher consumer. |
| 52555 | `0x896f` | `AddN a3,a6,a6` | `0xb0 + 0xb0 -> 0x160` | Starts a mailbox control value from the staged byte. |
| 52561 | `0x897d` | `L32iN a7,[a4]` | local `EA/PA 0 -> 0x030bb000` | Loads the complete staged pointer value. |
| 52568 | `0x898f` | `Bbc a7,a10,...` | `a10=0`, bit 0 of `0x030bb000` is clear | Uses the word as a bitset/control predicate; it does not dereference `0x030bb000`. |
| 52569 | `0x899c` | `L32iN a10,[a6]` | device mailbox `0x27220040 -> 0` | Reads the mailbox control register selected by an image literal. |
| 52571-52572 | `0x89a1..0x89a4` | shift and add | `0x160 << 12 -> 0x00160000` | Completes the value derived from local byte `0xb0`. |
| 52575 | `0x89ac` | `S32iN a9,[a6]` | device mailbox `0x27220040 <- 0x00160000` | The immediate non-local write data-dependent on the staged byte. It is not BAR2, the alive struct, or `FW_ALIVE_OFF`; its payload is not `0x030bb000`. |
| 52577 | `0x89b1` | `L32iN a7,[a4]` | local `EA/PA 0 -> 0x030bb000` | Third and final load overlapping local `0..3` through trace stop. |
| 52585 | `0x89c5` | `Or a3,a7,a3` | `0x030bb000 | 1 -> 0x030bb001` | Sets the local control bit. |
| 52588 | `0x89ce` | `S32iN a3,[a4]` | local `EA/PA 0 <- 0x030bb001` | Writes the marked value back to local RAM only. |
| 52590 | `0x89d2` | `RetwN` | clean mapped-firmware return | Ends this direct consumer routine. Publication is decided below from the global store inventory, not from an assumption about register liveness across the return. |
| 53242-53247 | `0x7f2e..0x7f3c` | five `S8i` stores | local bytes become `6f 15 cd 70` | Replaces the complete local word with `0x70cd156f`, still before forcing. |

The exhaustive post-publisher load inventory contains exactly these three
overlapping loads:

```text
n=52552  pc=0x8966  LOAD1  EA=PA=0x00000001  value=0x000000b0
n=52561  pc=0x897d  LOAD4  EA=PA=0x00000000  value=0x030bb000
n=52577  pc=0x89b1  LOAD4  EA=PA=0x00000000  value=0x030bb000
```

There is no fourth load before the first counterfactual, after it, or before
the trace stops at `n=53874`. Thus no later instruction re-obtains the staged
pointer by dereferencing local `0..3`. The direct consumer slice performs a
mailbox control write and local-word mutation, followed by complete local-word
replacement. The separate global writer inventory below rules out a later
register-carried store to either host-visible destination.

## Host-visible-destination writer inventory

The inventory classifies both effective and resident-DTLB physical addresses,
so an ordinary-looking VA translated into BAR2 would appear here. It does not.

| Destination / control | Clean path (`n < 53640`) | Post-forcing service prefix (`n >= 53640`) | Result |
|---|---|---|---|
| Local `VA/PA 0..3` | publisher stores at `n=52119..52122`; mark at `n=52588`; replacement at `n=53242..53247` | none | Private local control state, not host-visible alive. |
| Any BAR2 shared SRAM, device `0x03080000..0x030bffff` | exactly one: `n=52551`, `PC=0x8964`, `STORE4 EA=PA=0x030b27c0 <- 0` | none | Positive control for BAR2 detection; outside both alive destinations. |
| Alive struct `0x030bb000..0x030bb03f` | **none** | **none** | No struct word is built anywhere in the executed trace. |
| `FW_ALIVE_OFF` (BAR2 `+0x3f000`, device `0x030bf000`) | **none** | **none** | No pointer publication occurs anywhere in the executed trace. |
| Device mailbox `0x27220040` | `n=52575`, `PC=0x89ac`, `<- 0x00160000` | none | Device control write, but neither host alive destination nor pointer payload. |

The existing non-local summary, active before the publisher and through the
sink, saw 91 stores:

```text
68  AIE-array/NoC MMIO
18  device-mailbox
 3  BAR0 management
 1  BAR2 shared SRAM: 0x030b27c0 <- 0
 1  high-data alias
```

None overlaps the alive struct or pointer slot. There are no BAR2 stores at all
after the first `HARNESS_VIEW`.

The executed-writer inventory is therefore empty for the two requested host
destinations on **both** traced phases. The real writer is not assigned a fake
`n` or PC: hardware observation proves the destination contents, while this
execution stops before their producer. The derived path classification is
unambiguous nonetheless:

- **OBSERVED on Phoenix hardware:** a complete 16-word channel object persists
  at device `0x030bb000`, including device-absolute ring pointers, sizes,
  `_NPU` magic, and protocol 5.8; the transient value captured at
  `FW_ALIVE_OFF` was `0x030bb000`.
- **VERIFIED in this trace:** neither destination is written before the
  `a7 < 6` service contract, and local `VA/PA 0` does not alias either one.
- **DERIVED:** the actual direct device-SRAM struct builder and pointer
  publisher are in the service continuation this reconstruction does not
  execute. An accepted `a7` service context must do what the clean worker does
  not: populate the host-read object and store its device address to
  `FW_ALIVE_OFF`.

## Reconciliation

The successor correctly established:

```text
one intended slot-6 selection
one 0x55f8 dispatch
one clean four-byte local pointer staging sequence
correct queue retirement and selector fallback
```

It incorrectly collapsed these two address spaces:

```text
management local VA/PA 0x00000000 = private control word
device address       0x030bb000 = host-readable BAR2 channel object
```

The numeric value stored in the first is a pointer to the second; it is not a
write to the second. The subsequent mapped-firmware consumers confirm that
distinction operationally. They read local `0..3`, derive a mailbox control
write, set a local bit, and then replace the local word. No mapped instruction
turns it into the 0x40-byte BAR2 object or the `FW_ALIVE_OFF` doorbell.

Therefore the successor's “clean path already publishes alive; only an address
map gap remains” conclusion is superseded by this finding. Adding a
`VA 0 -> FW_ALIVE_OFF` mapping would be calibration against a coincidental
pointer value and would erase the observed local-control behavior. It is not a
valid fix locus.

The predecessor's concrete wall remains the required frontier: the
reconstructed service continuation needs an in-range context (`a7 < 6`) before
it can reach the real device-SRAM builder/publisher. The successor adds useful
upstream context—slot-6 selection is intentional and the pointer is pre-staged
locally—but it does not remove that wall.

## Ranked single next step

**1. Obtain a read-only Phoenix management-core trace of the first real store
to `0x030bb000..0x030bb03f` or `0x030bf000`, and walk its mapped call chain
backward to the accepted service context.** Record the writer PC, EA/PA,
`a7`/current-task identity, and preceding control edge, then byte-match the PCs
against this exact `1502_00` image. The 16-word hardware descriptor and
transient alive pointer are already the output oracle; the missing evidence is
the natural producer chain.

This is the decisive derive-only observation of what the blocked path
publishes. It requires no branch forcing, task/state injection, fitted address
mapping, PSP-loader work, `0x8cae` mechanism work, below-CPU-bank hunt, or
`17f1_10` semantics. If captured on hardware, use a single-shot trace or
IRQ-side observation; do not sustained-poll BAR0.

## Verification

Fresh results for this uncommitted reconciliation diff:

```text
targeted reconciliation probe:
  1 passed; 0 failed; 4120 filtered out

cargo test --lib:
  4091 passed; 0 failed; 30 ignored

cargo fmt --all -- --check:
  exit 0

git diff --check:
  exit 0
```

## Reproduction

```text
XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_26d4_cache_pageroot_timeline -- --nocapture \
  > build/experiments/firmware-re/alive-publish-reconciliation.log 2>&1

cargo test --lib
```

The probe remains test-only and env-gated. Production `load_m2c`, `mod.rs`,
`mmio.rs`, and `system.rs` behavior is unchanged. Nothing was committed by
this pass.
