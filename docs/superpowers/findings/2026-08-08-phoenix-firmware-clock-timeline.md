# Phoenix Signed-Firmware Clock Timeline

## Verdict

The narrower firmware timeline produced a deterministic hardware law at the
Phoenix low-QoS clock identity, but it **falsified** treating the emulator's
interpreter attempts as management-processor timing. The current functional
one-array-cycle-per-firmware-boundary scheduler cannot be repaired with a
scalar multiplier. No scheduler change is authorized from this result.

On physical NPU1, the settled CDO-NOOP segments have an exact marginal cost of
66 shim timestamp cycles per authentic CDO `NOOP`. At the reported
MP-NPU/H-clock identity `400/800 MHz`, that is 33 nominal MP-NPU cycles per
`NOOP`, with no ratio rounding. The ordinary marker-only path costs 84 shim
cycles, or 42 nominal MP-NPU cycles.

The same unmodified signed firmware takes 14 interpreter instruction attempts
per CDO `NOOP` and 19 attempts through the marker `Write32` path. These two
independent ratios disagree:

```text
33 / 14 != 42 / 19
```

Therefore an average CPI, scalar step multiplier, fixed delay, or immediate
firmware/array scheduler patch would encode a known false model. The next
authorized boundary is to characterize the missing Xtensa timing classes,
starting with the loads, stores, and branches in these two already-isolated
paths.

## Pinned tuple and artifacts

- Phoenix/NPU1 firmware: `1.5.5.391`, payload SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`;
- kernel: `7.1.7-custom+`;
- loaded `amdxdna.ko` SHA-256:
  `21a46896fc9db2d7c3a41e7376fddeacdb70e6cbfc3c38d872462db49abecf35`;
- amdxdna source audit: `216cefececd74effcd7a88350c71b99f5ef9a215`;
- XCLBIN SHA-256:
  `d25ab5b8b45a0119c7a62efbe291599020adf86e27609fdc01a6346637ab51b3`;
- instrumented `insts.bin` SHA-256:
  `b23adbd7d37a90196fe560ee95f4269cff4ae42713a84d8d8522419f009efeed`;
- expected output SHA-256:
  `64ed86b909d6d0502b64b28db0ea1272ffb358e20e9b1d88b63ccb07fa900cf5`.

The complete physical receipts and raw traces are preserved under:

- `build/experiments/phoenix-pm-clock-characterization/20260808T053036Z-firmware-clock-timeline/`;
- `build/experiments/phoenix-pm-clock-characterization/20260808T055513Z-firmware-clock-high-qos/`.

Both campaigns restored an ordinary no-QoS context, reproduced the expected
output, and returned to the original reported `default`, `600/1028 MHz` state.

## Instrument

`tools/phoenix-pm-clock-characterize.py` inserts genuine CDO `NOOP` records
after the final TCT and brackets each block with a CDO `Write32` to the
toolchain-derived shim `Event_Generate` register. The event number comes from
the open aie-rt event definitions, and the register address comes from the
AM025 register database. No event ID, register address, or CDO field layout is
hand-encoded in the emulator guard.

The block sequence is:

```text
0, 1, 0, 2, 0, 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128, 0, 256, 0, 64
```

Twenty blocks produce 21 ordered `USER_EVENT_0` markers. The ordinary
`DMA_S2MM_0_START_TASK` and `DMA_S2MM_0_FINISHED_TASK` events qualify the trace
lifecycle and output equality qualifies kernel integrity. The source's former
trace-stop event is changed to a distinct, unused `PERF_CNT_0` event so the
final marker is observable before a real stop closes the trace.

Several earlier instrument revisions taught us that an open trace or a marker
used as its own stop is not a valid endpoint: the last marker may remain in a
partial packet or the stop event may be consumed before occupying a slot. The
final instrument requires all 21 markers and an explicit distinct stop.

## Physical observations

Each row below lists the 20 adjacent marker intervals in shim timestamp cycles,
in the same order as the block sequence above.

| Regime | Run | Reported MP/H MHz | Intervals |
|---|---:|---:|---|
| default | 1 | `600/1028` | `106,178,133,243,72,355,72,582,72,1033,73,1939,72,3752,72,7386,73,14647,117,3759` |
| default | 2 | `600/1028` | `107,165,132,245,72,356,72,583,72,1038,72,1946,72,3761,72,7382,72,14627,117,3770` |
| low QoS `(1,1000)` | 1 | `400/800` | `120,206,150,284,84,424,84,688,84,1216,84,2272,84,4384,84,8608,84,17056,136,4384` |
| low QoS `(1,1000)` | 2 | `400/800` | `120,206,150,284,84,424,84,688,84,1216,84,2272,84,4384,84,8608,84,17056,136,4384` |
| high QoS `(1,1800)` | 1 | `600/1028` | `107,164,133,243,72,355,73,581,72,1035,72,1943,72,3758,72,7394,72,14629,117,3762` |
| high QoS `(1,1800)` | 2 | `600/1028` | `132,192,132,244,72,355,72,581,72,1034,72,1940,72,3753,72,7389,72,14644,117,3759` |

The complete low-QoS timeline repeats digit-for-digit. Its settled nonzero
blocks from 4 through 256, plus the final repeated 64 block, obey:

```text
shim cycles = 160 + 66 * NOOPs
nominal MP cycles = 80 + 33 * NOOPs
```

Successive block differences independently establish the 33-cycle marginal;
it is not obtained by fitting a regression. The settled marker-only intervals
are 84 shim cycles, or 42 nominal MP cycles. The initial four intervals and the
late marker-only interval are preserved above but excluded from the steady
law: they contain repeatable timeline-boundary work that this experiment did
not localize. Nothing is averaged away.

The default and high-QoS runs show small interval variation even though both
report `600/1028`. The high-QoS control rules out the absence of a QoS request
as the sole cause. It does **not** prove that the two clocks were physically
phase-locked or held at an exact live ratio.

## Clock-metadata audit

On NPU1, `DRM_AMDXDNA_QUERY_CLOCK_METADATA` is not a live frequency sampler.
`aie2_query_clock_metadata()` invokes `aie2_update_counters()`, but the NPU1
`hw_ops` table supplies only `set_dpm`; it has no `update_counters` callback.
The query therefore returns `ndev->npuclk_freq` and `ndev->hclk_freq`, cached
from the SMU responses to the last NPU1 DPM-setting operation.

Consequently:

- the reported pairs are valid DPM identities and useful provenance;
- before/after equality does not prove instantaneous frequency or phase;
- variation at `600/1028` must not be fitted, averaged, or labeled clock drift
  from this evidence alone; and
- the exact low-QoS shim-cycle law remains directly observed, while its
  conversion to 33 MP cycles is explicitly relative to the reported exact
  `800/400` identity.

## Signed-firmware emulator reconciliation

The conditional guard
`m2c_firmware_clock_timeline_matches_physical_noop_work` reuses the exact
signed-firmware `CHAIN_EXEC_NPU` path and fails closed on all three pinned input
hashes. The general library suite skips it unless both external timeline inputs
are supplied.

The access probe now optionally records a monotonic CPU-step index and PC
history while armed. These are diagnostic attempted-step coordinates, not
cycles or a claim of retired-instruction accounting. The marker addresses and
events are decoded through the live device register model.

The emulator produces the exact interval sequence:

```text
19,33,19,47,19,75,19,131,19,243,19,467,19,915,19,1811,19,3603,19,915
```

That is structurally exact for every block:

```text
interpreter attempts = 19 + 14 * NOOPs
```

One CDO `NOOP` loop iteration executes one of each of these 14 instructions:

```text
S32iN L32iN AddiN Beqz L8ui MoviN S32iN Bgei Blti
J Bnei Beqi AddiN J
```

The marker `Write32` path executes 19 instructions: three marker-path
instructions, nine shared dispatch instructions, and seven write-handler
instructions. The exact PCs and decoded operations are printed by the
conditional guard so later timing probes stay tied to this pinned firmware.

The physical marginal and marker costs cannot both be generated by one scalar
cost per interpreter attempt. This is evidence for missing instruction-class,
memory-access, branch, and/or firmware-microarchitectural timing; it does not
yet distinguish among those causes.

## Licensed conclusions and next boundary

This finding licenses only the following:

1. Keep the present scheduler RED; do not tune it green.
2. Preserve the optional firmware PC/step witness as RE infrastructure.
3. Build the next physical discriminator from these two signed-firmware paths,
   varying one dynamic instruction class or dependency at a time where the
   authentic command format permits it.
4. Correct the known `WAITI`/already-halted work-accounting ambiguity before
   comparing a broader firmware path, but do not mistake that semantic cleanup
   for a timing model.
5. Design a rational firmware/array scheduler only after the measured timing
   classes predict held-out signed-firmware paths exactly.

This finding does **not** establish a general Xtensa CPI table, exact live
`600/1028` clock behavior, interrupt-entry cost, firmware cycle accuracy,
older-firmware timing, AIE2P timing, or a production scheduler cadence. No
production scheduler change was made.
