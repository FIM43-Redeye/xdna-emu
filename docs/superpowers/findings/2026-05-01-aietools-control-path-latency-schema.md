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

## What's still hidden

The numeric defaults each subkey takes when no config file is provided
are hardcoded inside the `.so` and would need disassembly to recover.
The library logs `"Reading AIE Control Path latency information from
file"` only when a JSON path is supplied -- without one, defaults apply
silently.

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
