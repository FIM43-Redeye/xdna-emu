---
name: aietools control-path latency schema
description: AMD's proprietary AIE simulator exposes a JSON schema for IPU command-path latency that mirrors what we need to model. Schema extracted via string-table inspection of libaie2_cluster_msm_v1_0_0.osci.so.
type: project
---

# aietools control-path latency schema discovery

## Status

Schema discovered and replicated in `src/npu/cycle_cost.rs`. Numeric
defaults still pending (#322/#323).

## What we found

`aietools/lib/lnx64.o/libaie2_cluster_msm_v1_0_0.osci.so` reads a JSON
config keyed `AIE_CONTROL_PATH_LATENCY` with these subkeys:

```
AIE_CONTROL_PATH_LATENCY.AIE_TILE_BD     // BD config write to compute tile
AIE_CONTROL_PATH_LATENCY.MEM_TILE_BD     // BD config write to memtile
AIE_CONTROL_PATH_LATENCY.SHIM_TILE_BD    // BD config write to shim tile
AIE_CONTROL_PATH_LATENCY.WRITE_32        // any other Write32 (catch-all)
```

The library also references `AIE_LATENCY_OPTIONS` and
`AIE_MODEL_STREAM_PORT_LATENCY` and 41 other `AIE_*` env vars
(AIE_FAST_CM_REG_WRITE / AIE_FAST_DM_WRITE / AIE_DUMP_VCD_*, etc.).

## How we found it

```
nm -D libaie2_cluster_msm_v1_0_0.osci.so   # symbols are obfuscated
strings libaie2_cluster_msm_v1_0_0.osci.so   # but config keys are not
```

Specifically:

```
strings libaie2_cluster_msm_v1_0_0.osci.so | grep AIE_CONTROL_PATH_LATENCY
strings libaie2_cluster_msm_v1_0_0.osci.so | grep "Control Path Latency"
```

The accompanying human-readable strings:

```
"Control Path Latency for AIE Tile BD "
"Control Path Latency for Mem Tile BD "
"Control Path Latency for Shim Tile BD "
"Control Path Latency for all other Write Configurations"
"[INFO]: Reading AIE Control Path latency information from file"
```

confirmed the schema's intent.

## What's still hidden -- and why disassembly resolved it

Initial hypothesis: numeric defaults are hardcoded immediates that
disassembly could recover. We followed that path. Result: **the
defaults do not exist as hardcoded constants.**

Walking the `MathEngine` constructor disassembly:

```
e441ec:  cmpq   $0x0, -0x2e0(%rbp)        # is the property tree node empty?
e441f4:  je     e444f2                    # if empty -> skip parsing (default branch)
e441fa:  lea    0x1cb4727(%rip),%rdi  # 0x2af8928 = "[INFO]: Reading ..."
e44201:  call   c893f0                    # log the message + parse JSON

e444f2:  ... close ifstream ... jmp to merge point at e408f7
```

The "default branch" at `0xe444f2` contains zero immediate-value writes
to MathEngine member fields -- it just closes the file handle and
joins the merge point. There are no implicit defaults loaded earlier
in the constructor either.

Two `.bss`-located globals confirm the picture:

```
nm -D libaie2_cluster_msm_v1_0_0.osci.so | grep latency
  0000000003575f41 B g_disable_stream_switch_latency
  000000000357c6e0 B g_aie_func_latency_offset
```

Both are uninitialised globals (`B` = `.bss`), zero-initialised at
program start. Latency only takes a non-zero value if the JSON config
is supplied or environment-variable overrides are set.

## Implication

aietools' default behaviour is **no control-path latency modelling**.
This matches the project's prior assessment that aiesimulator is
"not cycle-accurate" (per `CLAUDE.md`): the timing model is opt-in.
AMD likely ships internal calibration JSONs for their own validation
work, but those aren't part of the distribution we have.

For us: the numerical defaults are **not** something we can extract
from any open or proprietary source. Calibration against real NPU
traces is the only path to truthful numbers. The framework now mirrors
the schema so calibrated values can ship as a sidecar JSON without
code churn.

## Why this is high-leverage even without the numbers

The schema *itself* is the architectural insight: AMD's own model
distinguishes per-tile-type BD writes from generic Write32. Our
emulator's category surface should match. Calibration of the four
constants is well-bounded -- four scalars, each measurable as a delta
between adjacent test points on real NPU traces.

Combined with what we already have from open sources:

| Component | Cycles | Source |
|-----------|--------|--------|
| Stream-switch hop, in-tile path | 3 | AM020 ch.2 (6-deep FIFO) |
| Stream-switch hop, boundary path | 4 | AM020 ch.2 (8-deep FIFO) |
| PLIO bridge AIE->PL baseline | 4 | aietools `aie_xtlm.cpp:202` |
| PLIO bridge PL->AIE baseline | 3 | aietools `aie_xtlm.cpp:232` |
| Register write completion | 1 | AM025 |
| AIE_CONTROL_PATH_LATENCY.AIE_TILE_BD | calibrate | aietools schema |
| AIE_CONTROL_PATH_LATENCY.MEM_TILE_BD | calibrate | aietools schema |
| AIE_CONTROL_PATH_LATENCY.SHIM_TILE_BD | calibrate | aietools schema |
| AIE_CONTROL_PATH_LATENCY.WRITE_32 | calibrate | aietools schema |
| CMP-to-shim NoC entry | calibrate | not in any open source |

The framework in `src/npu/cycle_cost.rs::CycleCostModel` carries all
these as named fields with cited sources. Default profile is
`legacy_one_per_packet` (preserves existing test calibration);
`with_known_constants` engages the derived structural pieces. A
calibrated profile (#322) will populate the four AMD-schema entries.

## Source policy

Per `CLAUDE.md` "Correctness Principle", aietools is read-only
reference. Extracting hardware facts via `strings`/`nm`/`objdump` is
observation, not copying -- the same pattern we already use for the
trace decoder library (`libxv_trace_decoder_opt.so`, "readable for
symbols, never linked or copied"). No aietools code or constants are
copied into this repository.
