---
name: trace-config schema — design + implementation log
description: Single-source-of-truth schema for trace-injection configuration. Replaced the scattered arg_idx / trace_size / tile-list bookkeeping that was previously spread across mlir-trace-inject, trace-prepare, cpp_trace_patch, parse-trace, and trace-compare.
status: implemented (2026-05-05)
---

# trace_config.json — design + implementation log

**Status**: shipped 2026-05-05. `events.json` is gone; `trace_config.json` is the only input to downstream trace tools.

## Why

Today, the trace BO's kernel-arg slot is computed independently in three places:

- `tools/mlir-trace-inject.py:425` → `chosen_arg_idx = 3 + max_existing_memref_args`
- `tools/cpp_trace_patch.py:430` → fallback `max(group_ids) + 1`
- aiecc → emits `aiex.npu.address_patch arg_idx = K`, then xclbin's kernel signature gets some number of BO slots that may or may not match

Coordination happens via stdout (`trace-arg-idx: K`) and a partial manifest (`events.json`). Each step does some arithmetic to translate between coordinate systems (kernel-arg index, runtime_sequence memref index, xclbin BO id, host group_id). Drift between any two of those is silent — for the bridge run on 2026-05-05, BOs were full of zeros despite arg_idx being threaded through "correctly", and we can't easily tell which step's view of "correct" is wrong.

The fix is a single source of truth that every tool reads from, and that records *every* coordinate system in one place so mismatches are detectable rather than implicit.

## File location

`<traced_dir>/trace_config.json`, where `<traced_dir>` is the existing `build/test/npu-xrt/<test>/traced/` directory used by `trace-prepare.py`. The bridge script also mirrors a copy into each `<results>/<test>.<compiler>.{hw,emu}/` directory so the trace output dirs are self-contained.

`events.json` is gone. No derivation path, no shim, no legacy fields. This system has no public callers — it answers only to us — so we didn't owe anyone a deprecation window.

## Schema (JSON Schema draft, Draft 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://xdna-emu/schemas/trace_config-v1.json",
  "title": "TraceConfig",
  "type": "object",
  "required": [
    "schema_version", "test_name", "src_mlir",
    "buffer", "kernel_signature", "tracing", "tiles_traced", "routing"
  ],
  "additionalProperties": false,
  "properties": {

    "schema_version": {
      "const": 1,
      "description": "Bump on any breaking change. Tools refuse to load mismatched majors."
    },

    "test_name": {
      "type": "string",
      "description": "Stable identifier (matches mlir-aie test directory name)."
    },

    "src_mlir": {
      "type": "string",
      "description": "Absolute path to the original (pre-injection) aie.mlir. The injector reads this; everyone else uses it for cache invalidation."
    },

    "buffer": {
      "type": "object",
      "required": ["size_bytes", "kernel_arg_slot"],
      "additionalProperties": false,
      "properties": {
        "size_bytes": {
          "type": "integer",
          "minimum": 4096,
          "description": "Trace BO size. Multiple of 4 KiB. Both HW and EMU allocate exactly this much."
        },
        "kernel_arg_slot": {
          "type": "integer",
          "minimum": 0,
          "description": "Authoritative kernel-arg slot for the trace BO. This is the value used by `aie.trace.host_config(arg_idx=...)`, `aiex.npu.address_patch arg_idx=...`, and `kernel.group_id(...)` in the host code. ALL coordinate systems derive from this."
        },
        "embedded_in_memref_idx": {
          "type": ["integer", "null"],
          "description": "If non-null, the trace BO is NOT a separate kernel arg — it's appended to the existing memref at runtime_sequence index N (mlir-aie's `arg_idx=-1` semantics). Mutually exclusive with separate-BO mode (i.e., when this is set, `kernel_arg_slot` refers to that memref's slot)."
        }
      }
    },

    "kernel_signature": {
      "type": "object",
      "required": ["args"],
      "additionalProperties": false,
      "properties": {
        "args": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["slot", "kind"],
            "additionalProperties": false,
            "properties": {
              "slot": {
                "type": "integer",
                "minimum": 0,
                "description": "Kernel arg position as XRT sees it (matches `kernel.group_id(slot)`)."
              },
              "kind": {
                "enum": ["scalar", "bo"],
                "description": "scalar = pass-by-value, bo = pass-by-buffer-handle."
              },
              "name": {
                "type": "string",
                "description": "Human-readable name. For BOs, matches the host code's variable (e.g. `bo_inA`)."
              },
              "role": {
                "enum": ["instruction_buffer", "data", "trace", "ctrl_packet", "phantom"],
                "description": "What the BO is for. `phantom` = aiecc emitted a slot that the host doesn't bind; we record it so the mismatch is visible. Scalars use absent or 'data' role."
              },
              "ctype": {
                "type": "string",
                "description": "C type for scalars (`uint64_t`, `uint32_t`)."
              },
              "memref_idx": {
                "type": ["integer", "null"],
                "description": "For data BOs: the index in the runtime_sequence's memref args list (0..N-1). null otherwise."
              }
            }
          }
        }
      }
    },

    "tracing": {
      "type": "object",
      "required": ["mode"],
      "additionalProperties": false,
      "properties": {
        "mode": {
          "enum": ["event_time", "event_pc", "inst_exec"],
          "description": "Trace unit mode. Determines event vs. PC-anchored vs. instruction-execution semantics."
        },
        "core_grounding": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Event names always traced on core tiles (e.g. PERF_CNT_0)."
        },
        "core_sweep": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Per-batch sweep events (variable across runs in a sweep)."
        },
        "shim_grounding": {
          "type": "array",
          "items": {"type": "string"}
        },
        "shim_sweep": {
          "type": "array",
          "items": {"type": "string"}
        },
        "memtile_grounding": {
          "type": "array",
          "items": {"type": "string"}
        }
      }
    },

    "tiles_traced": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["col", "row", "kind", "events", "packet_id"],
        "additionalProperties": false,
        "properties": {
          "col": {"type": "integer", "minimum": 0},
          "row": {"type": "integer", "minimum": 0},
          "kind": {
            "enum": ["shim", "memtile", "core"],
            "description": "Derived from row: 0 = shim, 1 = memtile, ≥2 = core."
          },
          "events": {
            "type": "array",
            "minItems": 1, "maxItems": 8,
            "items": {"type": "string"},
            "description": "Up to 8 event slots in trace_unit. Decoders read these to label each 4B word."
          },
          "packet_id": {
            "type": "integer", "minimum": 0,
            "description": "Stream-switch packet ID for this tile's trace channel. Matches `aie.packet_flow(packet_id)`."
          },
          "module": {
            "enum": ["core", "mem"],
            "description": "Which module's trace_unit (only meaningful for core tiles, which have both)."
          }
        }
      }
    },

    "routing": {
      "type": "object",
      "required": ["shim_col", "shim_dma_channel", "trace_done"],
      "additionalProperties": false,
      "properties": {
        "shim_col": {
          "type": "integer",
          "description": "Column of the shim tile that drains all trace packets."
        },
        "shim_dma_channel": {
          "type": "integer",
          "minimum": 0, "maximum": 1,
          "description": "Shim DMA S2MM channel (0 or 1). Channel 1 is the conventional trace channel."
        },
        "shim_bd_id": {
          "type": "integer",
          "default": 15,
          "description": "Shim BD assigned to drive trace traffic. Defaults to 15."
        },
        "trace_done": {
          "type": "object",
          "required": ["broadcast", "user_event"],
          "additionalProperties": false,
          "properties": {
            "broadcast": {"type": "integer", "description": "Broadcast channel for trace_done finalizer."},
            "user_event": {
              "enum": ["USER_EVENT_0", "USER_EVENT_1", "USER_EVENT_2", "USER_EVENT_3"],
              "description": "User event that gen_trace_done_aie2 fires."
            }
          }
        }
      }
    },

    "diagnostics": {
      "type": "object",
      "additionalProperties": false,
      "description": "Optional cross-checks that consumers can validate against. Mismatches are warnings, not errors — they're meant to surface drift.",
      "properties": {
        "expected_xclbin_bo_count": {"type": "integer"},
        "expected_address_patch_arg_idx": {"type": "integer"},
        "expected_runtime_sequence_memref_count": {"type": "integer"}
      }
    }
  }
}
```

## Worked example: add_one_using_dma

The block below is illustrative — a sketch of the *target* shape including a `bo_phantom_4` entry to show how a phantom xclbin slot would surface. The actual fixture committed at `tools/trace_config_examples/add_one_using_dma.json` is a verbatim dump from the live injector; it has no `bo_phantom_4` (we don't yet parse the post-aiecc xclbin to discover phantom slots — see open question 1 below).


```json
{
  "schema_version": 1,
  "test_name": "add_one_using_dma",
  "src_mlir": "/home/triple/npu-work/mlir-aie/test/npu-xrt/add_one_using_dma/aie.mlir",
  "buffer": {
    "size_bytes": 1048576,
    "kernel_arg_slot": 6,
    "embedded_in_memref_idx": null
  },
  "kernel_signature": {
    "args": [
      {"slot": 0, "kind": "scalar", "name": "opcode", "ctype": "uint64_t"},
      {"slot": 1, "kind": "bo", "name": "instr", "role": "instruction_buffer"},
      {"slot": 2, "kind": "scalar", "name": "ninstr", "ctype": "uint32_t"},
      {"slot": 3, "kind": "bo", "name": "bo_inA",  "role": "data", "memref_idx": 0},
      {"slot": 4, "kind": "bo", "name": "bo_inB",  "role": "data", "memref_idx": 1},
      {"slot": 5, "kind": "bo", "name": "bo_out",  "role": "data", "memref_idx": 2},
      {"slot": 6, "kind": "bo", "name": "bo_trace","role": "trace"},
      {"slot": 7, "kind": "bo", "name": "bo_phantom_4", "role": "phantom"}
    ]
  },
  "tracing": {
    "mode": "inst_exec",
    "core_grounding": ["PERF_CNT_0", "INSTR_EVENT_0", "INSTR_EVENT_1"],
    "core_sweep": ["INSTR_VECTOR", "MEMORY_STALL", "STREAM_STALL", "LOCK_STALL", "INSTR_LOCK_ACQUIRE_REQ"],
    "shim_grounding": [],
    "shim_sweep": [],
    "memtile_grounding": []
  },
  "tiles_traced": [
    {
      "col": 0, "row": 2, "kind": "core", "module": "core",
      "events": [
        "PERF_CNT_0", "INSTR_EVENT_0", "INSTR_EVENT_1", "INSTR_VECTOR",
        "MEMORY_STALL", "STREAM_STALL", "LOCK_STALL", "INSTR_LOCK_ACQUIRE_REQ"
      ],
      "packet_id": 1
    }
  ],
  "routing": {
    "shim_col": 0,
    "shim_dma_channel": 1,
    "shim_bd_id": 15,
    "trace_done": {
      "broadcast": 14,
      "user_event": "USER_EVENT_2"
    }
  },
  "diagnostics": {
    "expected_xclbin_bo_count": 5,
    "expected_address_patch_arg_idx": 6,
    "expected_runtime_sequence_memref_count": 3
  }
}
```

The phantom `bo4` slot (`bo_phantom_4`) is now *named*: it's a known artifact of aiecc's BO emission that the host doesn't bind. If we eventually understand why it exists (control-packet reservation? alignment slot?), the role enum gets extended. Until then, naming it explicitly stops it from being a silent mismatch.

## Per-tool migration

| Tool | Before | After |
|---|---|---|
| `mlir-trace-inject.py` | computed arg_idx, printed `trace-arg-idx: N` on stdout | parses MLIR, calls `_build_trace_config(...)`, writes via `trace_config.dump(path)`. New flags `--trace-config-out`, `--config-test-name`, `--config-src-mlir`. Stdout marker line deleted. |
| `trace-prepare.py` | parsed inject's stdout, wrote `events.json` partial manifest | passes `--trace-config-out` to inject, reads it back via `trace_config.load(path)`, hands it to `cpp_trace_patch`. `events.json` writer deleted. |
| `cpp_trace_patch.py` | `trace_arg_index: int \| None`; fell back to `max(group_ids)+1` heuristic | `trace_arg_index: int` (required); fallback heuristic removed; `_extract_group_ids` helper deleted. |
| `parse-trace.py` | trace BO bytes + lowered MLIR | unchanged — already drives off the lowered MLIR; no events.json dependency to migrate. |
| `src/bin/trace_compare.rs` | events JSON files (HW + EMU) | unchanged — its `events.json` is parse-trace OUTPUT (decoded events), not our manifest. |
| `scripts/emu-bridge-test.sh` | copied `events.json` into result dirs (mostly dead — the `events_file` variable that depended on it was never referenced) | copies `trace_config.json` instead; dead `events_file` variable deleted. Cache invalidation already keyed on traced-MLIR mtime, naturally captures schema changes. |

`mlir-trace-inject.py` is the **only writer** of trace_config. Everyone else is a consumer or a passthrough.

Original design doc said `trace-prepare` should be the writer. In practice, the data (runtime_sequence arg count, compute tile coordinates, chosen arg slot) all comes from MLIR introspection that the injector already does — having `trace-prepare` re-parse the MLIR just to write a manifest would have been duplicate work. The architectural property that matters (single writer, every consumer reads from one file) is preserved either way.

## Migration order (as actually shipped)

Done in a single sweep on 2026-05-05 rather than the staged plan originally drafted, because the user explicitly asked for the whole thing in one go and the changes were tightly coupled enough that a partial state would have been awkward. Result:

1. **Schema + JSON Schema file** (`tools/trace_config_schema.json`) — landed alongside a shared loader (`tools/trace_config.py`) with `load`/`dump`/`validate` API + CLI validator. 7 negative tests cover field-level rejection.
2. **`mlir-trace-inject.py`** — became the writer. New `_build_trace_config(...)` helper assembles the dict from injection state; `--trace-config-out` flag emits to disk. Stdout `trace-arg-idx:` line deleted.
3. **`trace-prepare.py`** — passes `--trace-config-out` + `--config-test-name` + `--config-src-mlir` to the injector, reads the result back, hands it to `cpp_trace_patch`. `events.json` writer deleted; `_TRACE_DECL_RE` regex deleted (no longer parsing MLIR text for tile names).
4. **`cpp_trace_patch.py`** — `trace_arg_index` is now mandatory; fallback heuristic and `_extract_group_ids` helper removed. Tests refactored to use a `_patch(...)` wrapper that supplies the explicit slot; the two tests that exercised the fallback specifically were deleted.
5. **`parse-trace.py` / `trace_compare.rs`** — no migration needed. Their `events.json` references are to parse-trace's own decoded-events output (different file), not our manifest.
6. **Bridge script** — copies `trace_config.json` instead of `events.json`; the dead `events_file` local variable in Phase 5 was cleaned up. Cache invalidation: traced MLIR mtime already captures every relevant change, so no separate schema-hash check is needed.
7. **`events.json` deleted** — 67 cached files in `mlir-aie/build/test/npu-xrt/*/traced/` and 442 in `xdna-emu/build/bridge-test-results/` removed. No writer remains; no reader remains.

## Open questions

1. **Where does the phantom `bo4` slot come from?** Probably aiecc reserves one for ctrl-packet support. The schema models it explicitly via `role: "phantom"`. We should figure out the real role and either rename the role or remove the slot. Tracked separately, doesn't block schema.

2. **Multi-device / multi-segment xclbins.** Schema currently assumes one device, one runtime_sequence. Tests like multi-segment ctrl-pkt may need a top-level array of trace_configs (one per device) or a `device_index` field in the existing object. Defer until we have a multi-device test that needs tracing.

3. **`embedded_in_memref_idx` mode.** mlir-aie supports `arg_idx=-1` ("append trace to last memref"). We've never used this in xdna-emu; do we need to support it? If yes, `kernel_arg_slot` semantics need clarification (it then refers to the embedding memref's slot, with `embedded_in_memref_idx` and the offset implied by that memref's element count). Probably a v1 omission, add in v2 if needed.

4. **Quarantined tests.** Some tests are marked trace-quarantined in the bridge script. Should the schema record this, or stays a script-side concern? Argument for inclusion: makes the manifest self-describing. Argument against: schema becomes coupled to bridge-script policy. Lean toward exclusion for now.

5. **Sweep state.** When the bridge script runs a sweep across event sets (`--trace=sweep`), each batch has different events. Does each batch get its own trace_config, or does the schema represent a sweep with a list of batches? Probably "one config per batch" (simpler), with the bridge script orchestrating multiple invocations of trace-prepare.

## What this catches

The 2026-05-05 empty-trace-bug symptom would manifest in this schema as: `kernel_signature.args` has 5 BOs, host bound 4, `diagnostics.expected_xclbin_bo_count = 5`. A consumer that asserts `bound_bo_count == declared_bo_count` would error out instead of silently producing a zero-filled buffer. The actual bug (whatever it is — maybe trace_unit not starting, maybe BD enqueue not firing) is still ours to find, but at least we'd know whether the symptom is "didn't bind right BO" vs "bound right BO, no events generated".

## Cross-references

- `tools/mlir-trace-inject.py:411–425` — current arg_idx computation.
- `tools/trace-prepare.py:155–177` — events.json manifest writer.
- `tools/cpp_trace_patch.py:344–430` — test.cpp patcher.
- mlir-aie's `aie.trace.host_config` and `AIEInsertTraceFlows.cpp` — upstream lowering pipeline.
- `docs/coverage/peano-trace-window-gap.md` — prior empty-trace investigation (now resolved differently).
