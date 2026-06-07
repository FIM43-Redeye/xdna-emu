# In-process NPU1 VCD mapping tree -- design (#88)

Status: IMPLEMENTED (`src/vcd/inproc_mapping.rs`, `build_npu1_inproc_mapping_tree`).
Typed/raw split signed off. Coverage verified at **99.8%** (39575/39673) against
`build/experiments/inproc-vcd/add_one_using_dma.inproc.vcd` via
`vcd-compare --device npu1-inproc --coverage`. The 98 unmapped signals are
tile-less top-level oddments (`pm_adapt.*` power-management, `aie_ctrl_*`,
`shim_reset_n_*`, `record_timer_event_id`) that belong to no tile -- correctly
left unmapped. Subsystem split: core 450, dma 7120, lock 1760, stream 2495,
memory 6400, event 9040, perf 180 (all typed/comparable); other 12130 (raw tier).

**One implementation note not in the original design:** the MSM
`vcd_trace_file_writer` emits a single flat `$scope module SystemC` whose vars
carry the full dotted `aiesim_top.math_engine...` name, so wellen's resolved
hierarchy is `SystemC.aiesim_top.math_engine...`. The tree root is therefore
`SystemC` > `aiesim_top` > `math_engine`.

## What this is

`build_npu1_inproc_mapping_tree()` -- a third `MappingTree` (alongside
`build_aie2_mapping_tree` and `build_vc2802_mapping_tree`) that resolves the
**in-process NPU1 cluster's** VCD signal names to `StatePath` identities. This
is the tree the three-way timing leg (#84) and the future interp<->aiesim diff
(#86) consume via `vcd-compare --device npu1-inproc`.

## The discovery that shapes the design

The in-process MSM VCD is its **own layout** -- structurally different from both
existing trees, and far more detailed than either:

- Scope root is `aiesim_top.math_engine` (not `top` / not the vc2802
  `SystemC.tl.aie_logical.aie_xtlm` chain).
- Geometry: 5 cols (0-4), shim row 0, mem row 1, **array rows 2-7** (6 compute
  rows; NPU1 hardware has 4, rows 2-5 -- the extra two rows are inert but the
  model still emits them). Active tiles align with NPU1 at matching coords.
- **39,673 signals / 2,856 distinct templates.** For comparison, the existing
  aie2 tree enumerates ~6,080 paths.

The reason for the blow-up: the MSM model dumps its **full internal signal
set**, not just the architecturally-visible state. The compute core
(`array.tile.cm.proc`) alone exposes the entire ISS internals --
`iss.dme_ada_e_out`, `iss.dmo_pob_w_out`, `lock_adaptor_E.req_int`,
`processor_bus.cntReq`, hundreds of adaptor/bus signals -- none of which the
interpreter has any analog for, and none of which can ever participate in a
cross-source diff.

## The decision: typed vs raw

`StatePath` exists for **cross-source comparison identity** -- two signals with
the same `StatePath` get paired and diffed. A signal only one source emits never
pairs; giving it a typed variant adds permanent surface to the shared vocabulary
(which #86's interpreter side must also speak) for zero comparison value.

So the in-process signals fall into two tiers:

### Typed tier -- signals with a real interpreter analog (adapt to existing `StatePath`)

| Subsystem | In-process signal | Maps to |
|-----------|-------------------|---------|
| locks | `locks.value_{i}`, `locks.lock_op_{i}` | `LockValue` / `LockOp` (reuse) |
| dma | `{s2mm,mm2s}_state{ch}.cur_bd*`, `.address`, `.data`, `.status`, `.processed_{mem,stream}` | `Dma*` (adapt child names) |
| stream | `stream_switch.event_{idle,running,stalled,tlast}_{port}`, `.from_s*`/`.to_m*` data | `StreamPort*` (adapt) |
| core | `cm.proc.pc_E{n}`, `cm.proc.iss.{pm_rd_in,tm_rd_in,tm_wr_out,tm_ad_out,reset}`, `cm.proc.debug_status.pc_breakpoint_halted` | `CorePc`, `CorePm*`, `CoreTm*`, `CoreReset`, `CoreBreakpointHalted` (adapt) |
| memory | `dm.conflict_{i}`, `dm.conflict_addr_{i}`, `dm.port_*` | `MemBankConflict` / `MemConflictAddr` / `MemPortAccess` (adapt) |
| event | `event_trace.event{code}_{name}` | `EventTrace` (reuse -- exact pattern match; **these are the #84 timing anchors**) |
| perf | `performance_counter.counter_{i}` | `PerfCounter` (adapt) |

Note the DMA child set is *richer* than the current `Dma*` variants cover
(e.g. `cur_bd_lock_acq_ID`, `cur_bd_enable_compression`, `channel_running`,
`start_task`/`finished_task`, `memory_starvation`). The ones with a clear
existing variant map to it; the channel-internal flags drop to the raw tier
(below) rather than minting a dozen new typed DMA variants.

### Raw tier -- model-internal / aiesim-only (resolvable, never diffs)

Everything else: `cm.proc.iss.*` ISS internals, `cm.proc.dm_{Ae,Ao,Be,Bo,Se,So}.*`
banks, `lock_adaptor_*`, `*_adaptor_*`, `processor_bus.*`, `pl_interface.*`
(shim), `tile_control.*`, `event_broadcast*`, `first_level_interrupt_*` /
`second_level_interrupt`, `proccore_status`, `stream_switch.fifo0_used_size`,
`column_reset_n`, `shim_reset_n_*`, the top-level `pm_adapt.*`, and the DMA
channel-internal flags.

**Proposed mechanism:** add one variant

```rust
StatePath::Raw { col: u8, row: u8, subsystem: Subsystem, signal: String }
```

(plus a `Subsystem::Other` for signals outside the seven existing classes:
interrupts, tile_control, pl_interface, broadcast). `Raw` resolves for **coverage
accounting** (so `--coverage` reports ~100% "every signal has an identity") but
the comparison engine skips `Raw` pairs -- they carry no field semantics, so
there is nothing meaningful to diff, and the interpreter never emits them.

This realizes your "full signal coverage" -- every signal maps to *an* identity
and shows up in the coverage audit -- **without** bloating the typed comparison
vocabulary with ~20 variants that can never pair across sources.

## Why this is the right line (and the alternative I rejected)

The alternative -- mint a typed `StatePath` variant for every distinct signal --
would add ~20-25 variants (ISS bus lanes, adaptor handshakes, interrupt
controllers, broadcast nets) to the canonical type. Each would enumerate on the
aiesim side and *always* show as "missing on the interpreter side," because the
interpreter models architectural state, not the MSM model's internal SystemC
wiring. That is coverage theater: 100% typed, 0% comparable. The `Raw` tier gives
honest accounting (resolved vs comparable are distinct, visible numbers) and
keeps `StatePath`'s typed surface meaning exactly what it says -- signals worth
diffing.

## Build plan once the split is agreed

1. `state_path.rs`: add `Subsystem::Other` + `StatePath::Raw { .. }`; wire
   `subsystem()` / `tile()` / `field_name()` / `Display`.
2. `compare.rs`: skip `Raw` in pairing (or treat as always-equal -- TBD, pick the
   one that keeps the report honest).
3. New mapping infra for the in-process layout:
   - reuse `lock_mapping`, `event_mapping`, `PerfCounter` as-is;
   - adapt DMA (same `{s2mm,mm2s}_state{ch}` groups, in-process child names);
   - adapt stream (`event_*_{port}` at switch level), core (`cm.proc` nesting),
     memory (`dm.port_*`);
   - a `raw_mapping(subsystem)` that resolves any remaining leaf to
     `StatePath::Raw`.
4. `build_npu1_inproc_mapping_tree()`: 5 cols, shim row 0 / mem row 1 / array
   rows 2-7; `cm`+`mm` nested scopes per compute tile.
5. `vcd_compare.rs`: add `"npu1-inproc"` to `parse_device`; `cycles.rs` default.
6. Verify: `vcd-compare --coverage --device npu1-inproc
   build/experiments/inproc-vcd/add_one_using_dma.inproc.vcd` -> expect
   near-100% resolved, with the typed-vs-raw split reported.

## The one question for you

Sign off on the **typed/raw split + `StatePath::Raw`** before I touch the shared
vocabulary. The split is the load-bearing decision: it keeps the comparison type
honest and bounds the work to ~7 adapted typed mappings + one generic raw
fallback, instead of ~25 typed variants that can never diff. If you'd rather mint
typed variants for the aiesim-only subsystems anyway (e.g. you foresee the
interpreter eventually emitting interrupt/broadcast state), say so and I'll size
that instead.
