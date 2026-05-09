#!/usr/bin/env python3
"""
mlir-trace-inject.py -- Programmatically add declarative trace ops to an
AIE MLIR design.

Uses mlir-aie's Python bindings to parse an existing .mlir, walk its
aie.device body, insert aie.trace declarations for each compute tile, and
insert aie.trace.host_config + aie.trace.start_config ops at the top of
the aie.runtime_sequence body.

Rationale: mlir-aie's declarative trace IRON API (aie.utils.trace) is
designed to be called from within an IRON Python design while a context
is active. We instead need to inject into pre-existing MLIR produced by
an arbitrary upstream tool. Rather than text-munging, we parse into the
MLIR IR and construct ops through the aie dialect Python bindings.

Dependencies (verified Task 2, Step 1 of the Phase B plan):
  - aie.ir (MLIR Python bindings from mlir-aie install/python)
  - aie.dialects.aie -- exposes the AIE dialect (aie.*) including:
      RuntimeSequenceOp, TileOp, CoreOp, BufferOp, TraceOp, ...
  - aie.dialects.aiex -- exposes AIEX dialect (aiex.*) and imports
      RuntimeSequenceOp from _aie_ops_gen into its own namespace.
      Importing aiex triggers the module-level register_dialect() call
      that makes both "aie.*" and "aiex.*" ops parseable.

Trace-related symbols found in aie.dialects.aie (from API probe):
  TraceComboEventOp, TraceComboEventOpAdaptor,
  TraceConfigOp, TraceConfigOpAdaptor,
  TraceEdgeEventOp, TraceEdgeEventOpAdaptor,
  TraceEventOp, TraceEventOpAdaptor,
  TraceHostConfigOp, TraceHostConfigOpAdaptor,
  TraceMode, TraceModeOp, TraceModeOpAdaptor,
  TraceOp, TraceOpAdaptor,
  TracePacketOp, TracePacketOpAdaptor, TracePacketType,
  TracePortOp, TracePortOpAdaptor,
  TraceRegOp, TraceRegOpAdaptor,
  TraceShimRouting,
  TraceStartConfigOp, TraceStartConfigOpAdaptor,
  TraceStartEventOp, TraceStartEventOpAdaptor,
  TraceStopEventOp, TraceStopEventOpAdaptor,
  _trace_event_attr, trace, trace_combo_event, trace_config,
  trace_edge_event, trace_event, trace_host_config, trace_mode,
  trace_packet, trace_port, trace_reg, trace_start, trace_start_config,
  trace_stop

Dialect registration note: importing aie.dialects.aiex is sufficient to
register both the "aie" and "aiex" dialects for MLIR text parsing.  The
aiex module runs register_dialect(get_dialect_registry()) at import time.
Using Context() without any explicit register_dialect() call then works,
provided aiex was imported before entering the context.

Usage:
  mlir-trace-inject.py --input design.mlir --out design-traced.mlir
  mlir-trace-inject.py --no-op --input design.mlir --out copy.mlir

Exit codes:
  0 -- success (output written)
  1 -- unexpected error (e.g. parse failure, argparse rejection)
  2 -- input already contains aie.trace ops; refusing to double-inject
       (output file is NOT written on exit 2)
"""
import argparse
import io
import sys
from pathlib import Path

# Shared default for --perfcnt-period.  Lives in tools/perfcnt_defaults.py
# so trace-sweep.py and the Rust trace-compare can reference the same value.
sys.path.insert(0, str(Path(__file__).parent))
from perfcnt_defaults import DEFAULT_PERFCNT_PERIOD  # noqa: E402
from trace_config import dump as trace_config_dump  # noqa: E402

# MLIR op name used by mlir-aie for the declarative trace op. If the
# aie dialect is ever renamed upstream, only this constant needs updating.
TRACE_OP_NAME = "aie.trace"

# Default trace configuration constants.  These mirror mlir-aie's own
# defaults in python/utils/trace/setup.py (configure_trace + _get_default_events_for_tile).
# Keeping them as named constants documents the derivation and makes upstream
# drift easy to track.
_TRACE_PACKET_ID_START = 1          # AIEInsertTraceFlows reassigns during lowering
_TRACE_BROADCAST_START = 15         # hardware broadcast channel that starts trace
_TRACE_BROADCAST_STOP = 14          # hardware broadcast channel that stops trace
_TRACE_DEFAULT_CORE_EVENTS = (
    # Matches mlir-aie's own core-tile defaults in
    # python/utils/trace/setup.py::_get_default_events_for_tile, minus
    # the two port events (PORT_RUNNING_0/1) which require additional
    # PortEvent config plumbing we don't currently emit through the
    # declarative ops.
    #
    # The three INSTR_* events are software pins / vector markers; they
    # fire on vector paths but are silent on pure scalar kernels
    # (Phase B Limitation 1). The three stall events and the two lock
    # request events fire whenever the core waits on memory, a stream,
    # or a lock, which covers virtually every scalar kernel that uses
    # DMA + locks. The set is chosen to produce usable trace signals
    # on both vector and scalar workloads within the 8-event trace-unit
    # limit.
    "INSTR_EVENT_0",         # software event pin 0 (kernel boundary marker)
    "INSTR_EVENT_1",         # software event pin 1 (kernel boundary marker)
    "INSTR_VECTOR",          # vector-instruction issue
    "MEMORY_STALL",          # core stalled waiting on local memory
    "STREAM_STALL",          # core stalled waiting on a stream
    "LOCK_STALL",            # core stalled waiting on a lock
    "INSTR_LOCK_ACQUIRE_REQ",# lock acquire requested
    "INSTR_LOCK_RELEASE_REQ",# lock release requested
)
_TRACE_DEFAULT_SHIM_EVENTS = (
    # Mirrors mlir-aie's own shim-tile defaults in
    # python/utils/trace/setup.py::_get_default_events_for_tile.
    # All eight are GenericEvents (not BasePortEvents), so unlike the
    # core defaults there is no slot-port assignment to thread through;
    # the declarative emit only needs trace_packet + trace_event ops.
    #
    # The set covers every shim DMA channel transition our designs
    # actually use (S2MM 0/1 and MM2S 0; the second MM2S is rarely
    # populated on Phoenix shim cols). START/FINISHED pairs anchor
    # DMA pipeline-fill cycle measurement (#355a), and the two
    # STREAM_STARVATION events flag back-pressure on S2MM ingestion.
    "DMA_S2MM_0_START_TASK",
    "DMA_S2MM_1_START_TASK",
    "DMA_MM2S_0_START_TASK",
    "DMA_S2MM_0_FINISHED_TASK",
    "DMA_S2MM_1_FINISHED_TASK",
    "DMA_MM2S_0_FINISHED_TASK",
    "DMA_S2MM_0_STREAM_STARVATION",
    "DMA_S2MM_1_STREAM_STARVATION",
)
_TRACE_DEFAULT_MEMTILE_EVENTS = (
    # Mirrors mlir-aie's own mem-tile defaults in
    # python/utils/trace/setup.py::_get_default_events_for_tile.
    # All eight are MemTilePortEvents (BasePortEvent subclass) bound
    # to DMA channel ports: PORT_RUNNING_0..3 watch DMA channels 0-3
    # in S2MM (master) direction, and PORT_RUNNING_4..7 watch the same
    # channels in MM2S (slave) direction.
    #
    # Unlike shim/core, the memtile trace unit's slot-port mapping is
    # not implicit in the event name -- each PORT_RUNNING_X must be
    # paired with an aie.trace.port slot config. The default port
    # configs are emitted alongside the events when this default set
    # is in use; the per-slot (port, channel, direction) mapping is
    # encoded in _memtile_default_port_config below.
    #
    # Memtile DMA channels carry inter-tile data movement (compute
    # tiles below pulling from / pushing to the memtile's shared
    # SRAM), so PORT_RUNNING events here flag stream activity at the
    # memtile boundary -- complementing the shim DMA events at the
    # NoC boundary and the core trace at the compute side.
    "PORT_RUNNING_0",
    "PORT_RUNNING_1",
    "PORT_RUNNING_2",
    "PORT_RUNNING_3",
    "PORT_RUNNING_4",
    "PORT_RUNNING_5",
    "PORT_RUNNING_6",
    "PORT_RUNNING_7",
)


_TRACE_DEFAULT_MEMMOD_EVENTS = (
    # Mirrors mlir-aie's own core memory-module defaults in
    # python/utils/trace/setup.py::_get_default_events_for_tile (the
    # is_mem_trace=True branch). All eight are GenericEvents -- no
    # port-slot bindings to thread through, unlike memtile.
    #
    # The set splits into three semantic groups:
    #   - 2x DMA START_TASK events (S2MM_0, MM2S_0): mark the boundary
    #     where the compute tile's local DMA begins a transfer to/from
    #     the memory module's banks. Useful for pairing core trace
    #     stalls with the DMA activity that caused them.
    #   - 4x CONFLICT_DM_BANK_X events: fire when two requesters
    #     (core data port, DMA, etc.) collide on the same DM bank in
    #     the same cycle -- a primary cause of stalls visible in the
    #     core trace's MEMORY_STALL slot.
    #   - 2x EDGE_DETECTION_EVENT_X: software pins emitted by the
    #     compute kernel for boundary marking, mirroring the role of
    #     INSTR_EVENT_0/1 in the core trace.
    #
    # Memmod's trace unit lives at register offset 0x140D0 (vs the
    # core trace unit at 0x340D0 on the same compute tile); aie-rt
    # routes events emitted from the memory module of compute tiles
    # to this second trace unit. Trace_Control0 here has no Mode
    # bitfield per regdb, so memmod traces always run in event_time.
    "DMA_S2MM_0_START_TASK",
    "DMA_MM2S_0_START_TASK",
    "CONFLICT_DM_BANK_0",
    "CONFLICT_DM_BANK_1",
    "CONFLICT_DM_BANK_2",
    "CONFLICT_DM_BANK_3",
    "EDGE_DETECTION_EVENT_0",
    "EDGE_DETECTION_EVENT_1",
)


def _memtile_default_port_config(event_name: str):
    """Return (channel, master) for a memtile PORT_RUNNING_<N> event, else None.

    Mirrors the default MemTilePortEvent layout used by mlir-aie's
    setup.py::_get_default_events_for_tile for memtile:
        slot 0..3 -> DMA channel 0..3, master=True  (S2MM, into tile)
        slot 4..7 -> DMA channel 0..3, master=False (MM2S, out of tile)

    The port bundle is always WireBundle.DMA for memtile defaults; if
    a future caller wants to monitor non-DMA bundles, that needs a
    richer override path than --memtile-sweep-events alone.
    """
    if not event_name.startswith("PORT_RUNNING_"):
        return None
    try:
        slot = int(event_name.rsplit("_", 1)[1])
    except ValueError:
        return None
    if not (0 <= slot < 8):
        return None
    return slot % 4, slot < 4


def _trace_sym(col: int, row: int, kind: str = "core") -> str:
    """Symbol name for the aie.trace op attached to tile (col, row).

    Single source of truth so a future rename only touches one place.
    The TraceOp is constructed with this name; the matching
    trace_start_config in the runtime sequence must reference the same
    name -- drift between the two sites would silently produce an
    unreachable trace config.

    kind="core" preserves the legacy "trace_t{col}_{row}" format that
    test_mlir_trace_inject.py asserts on. Other kinds use the
    "trace_{kind}_{col}_{row}" convention so shim/memtile/memmod traces
    are unambiguous in the lowered MLIR even when colocated.
    """
    if kind == "core":
        return f"trace_t{col}_{row}"
    return f"trace_{kind}_{col}_{row}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--input", required=True, help="path to input .mlir")
    p.add_argument("--out", required=True, help="path to write traced .mlir")
    p.add_argument("--buffer-size", type=int, default=8192,
                   help="trace buffer size in bytes (default: 8192); "
                        "consumed by the host_config emission in Task 5")
    p.add_argument("--no-op", action="store_true",
                   help="parse and reserialize without injecting "
                        "(testing only -- skips injection)")

    # Trace mode.
    p.add_argument("--trace-mode",
                   choices=("event_time", "event_pc", "inst_exec"),
                   default="event_pc",
                   help="trace mode for compute-core trace units. "
                        "event_pc (mode 1, default) records PC alongside each "
                        "swept event -- easiest to ground in the disassembly; "
                        "event_time (mode 0) records cycle deltas; inst_exec "
                        "(mode 2) records the executed instruction stream as a "
                        "compressed bit-packed frame tree (LC/PC/atom/RLE). "
                        "All three modes are core-only -- memmod/memtile/shim "
                        "trace units always remain in mode 0 (their "
                        "Trace_Control0 has no Mode bitfield per regdb).")

    # Grounding events (fixed slots, always present, never overwritten by sweep).
    p.add_argument("--core-grounding",
                   default="PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1",
                   help="comma-separated event names reserved in fixed slots "
                        "of every compute-core trace unit. Default reserves "
                        "perfcnt cycle clock + two software pin events.")
    p.add_argument("--memmod-grounding", default="",
                   help="grounding events for compute-tile memmod trace unit. "
                        "Default is empty so all 8 default memmod sweep events "
                        "fit. PERF_CNT_0 here is NOT useful today: the inject "
                        "tool emits no memmod performance counter config, so "
                        "the slot would be reserved for an event that never "
                        "fires. Pass --memmod-grounding PERF_CNT_0 explicitly "
                        "once memmod perfcnt support lands.")
    p.add_argument("--memtile-grounding", default="",
                   help="grounding events for memtile trace unit. Default is "
                        "empty so all 8 default memtile sweep events fit. "
                        "PERF_CNT_0 is NOT a useful memtile grounding today: "
                        "the inject tool emits no memtile performance counter "
                        "config, so the slot would be reserved for an event "
                        "that never fires. Pass --memtile-grounding PERF_CNT_0 "
                        "explicitly once memtile perfcnt support lands.")
    p.add_argument("--shim-grounding", default="",
                   help="grounding events for shim PL trace unit. Default is "
                        "empty so all 8 default shim sweep events fit. "
                        "PERF_CNT_0 is NOT a useful shim grounding today: "
                        "the inject tool emits no shim performance counter "
                        "config, so the slot would be reserved for an event "
                        "that never fires. Pass --shim-grounding PERF_CNT_0 "
                        "explicitly once shim perfcnt support lands.")

    # Sweep events (rotated per batch; injection writes the initial pattern).
    # 'all' is reserved for future use (auto-enumeration from the event DB);
    # currently the resolver treats it the same as None (= use defaults).
    p.add_argument("--core-sweep-events", default=None,
                   help="comma-separated event names to sweep on compute cores. "
                        "Default uses the 5 hard-coded core sweep events (the "
                        "_TRACE_DEFAULT_CORE_EVENTS minus whatever appears in "
                        "--core-grounding). 'all' is reserved for future use; "
                        "currently behaves the same as the default.")
    p.add_argument("--memmod-sweep-events", default=None,
                   help="comma-separated event names to sweep on compute memmod. "
                        "Default: don't inject memmod trace. 'all' is reserved "
                        "for future use; currently behaves the same as the default.")
    p.add_argument("--memtile-sweep-events", default=None,
                   help="comma-separated event names to sweep on memtile. "
                        "Default: don't inject memtile trace. 'all' is reserved "
                        "for future use; currently behaves the same as the default.")
    p.add_argument("--shim-sweep-events", default=None,
                   help="comma-separated event names to sweep on shim. "
                        "Default (None): don't inject shim trace at all. "
                        "Any non-None value activates shim injection: 'all' "
                        "(or any unrecognized value) falls through to the "
                        "8-event shim-DMA default (S2MM_0/1 + MM2S_0 "
                        "START/FINISHED + S2MM_0/1 STREAM_STARVATION).")

    p.add_argument("--perfcnt-period", type=int, default=DEFAULT_PERFCNT_PERIOD,
                   help=f"cycles between PERF_CNT_0_EVENT fires when grounding "
                        f"includes PERF_CNT_0 (default: {DEFAULT_PERFCNT_PERIOD}).")

    p.add_argument("--trace-config-out", type=Path, default=None,
                   help="path to write trace_config.json (single source of "
                        "truth for downstream tools). Required when injecting; "
                        "schema at tools/trace_config_schema.json.")
    p.add_argument("--config-test-name", default=None,
                   help="value for trace_config.json's test_name field. "
                        "Defaults to the input MLIR's parent directory name; "
                        "callers that pipe through tempfiles (trace-prepare) "
                        "should override this with the real test name.")
    p.add_argument("--config-src-mlir", default=None,
                   help="value for trace_config.json's src_mlir field. "
                        "Defaults to the resolved input MLIR path; callers "
                        "using tempfiles should override this with the path "
                        "of the real (non-temp) source.")

    return p.parse_args()


def _has_trace_ops(operation) -> bool:
    """Return True if any op in the subtree has name TRACE_OP_NAME.

    The MLIR Python bindings expose operation.walk() but its callback
    signature expects a C++ WalkCallback type that is not easily wrapped from
    Python (calling it raises std::bad_cast).  Manual traversal through
    operation.regions is reliable and fast enough for design-scale MLIR.
    """
    if operation.name == TRACE_OP_NAME:
        return True
    for region in operation.regions:
        for block in region:
            for child_op in block:
                if _has_trace_ops(child_op):
                    return True
    return False


def _resolve_events(
    grounding: str,
    sweep: str | None,
    defaults: tuple[str, ...] | None,
) -> list[str]:
    """Combine grounding (fixed) + sweep (rotated) into up to 8 trace slots.

    Grounding events fill the first slots; sweep events fill the remainder.
    The returned list is de-duplicated (grounding wins) and capped at 8.

    Args:
        grounding: comma-separated event names that must always appear first.
        sweep: comma-separated event names to fill remaining slots, or None to
               use ``defaults``, or 'all' (reserved for future use -- treated
               the same as None here since enumeration requires the event DB).
        defaults: fallback sweep events when ``sweep`` is None; if also None,
                  the module gets no sweep events (grounding only).

    Returns:
        Ordered list of up to 8 event name strings (grounding + sweep).
    """
    g = [s.strip() for s in grounding.split(",") if s.strip()]
    if sweep is None or sweep == "all":
        if defaults is None:
            s_list = []
        else:
            s_list = list(defaults)
    else:
        s_list = [e.strip() for e in sweep.split(",") if e.strip()]

    seen: set[str] = set(g)
    final = list(g)
    for ev in s_list:
        if ev not in seen and len(final) < 8:
            final.append(ev)
            seen.add(ev)
    return final


def _emit_perfcnt_config(
    aied,
    tile_val,
    module_type: str,
    col: int,
    row: int,
    period: int,
) -> str:
    """Emit an aie.trace.config block that programs the performance counter.

    The config block contains three aie.trace.reg ops:
      Performance_Control0  raw = 28  (Cnt0_Start_Event = 28 = ACTIVE, bits[6:0])
      Performance_Control2  raw = 5   (Cnt0_Reset_Event = 5 = PERF_CNT_0, bits[6:0])
      Performance_Counter0_Event_Value = period (cycles between fires)

    Hardware derivation (aie-rt xaiemlgbl_params.h):
      - Performance_Control0 Cnt0_Start_Event at bits[6:0]: event 28 = ACTIVE,
        counts only while core is executing.
      - Performance_Control2 Cnt0_Reset_Event at bits[6:0]: event 5 = PERF_CNT_0,
        self-resets the counter on every fire.
      - Period = user-specified via --perfcnt-period (default
        DEFAULT_PERFCNT_PERIOD from tools/perfcnt_defaults.py).

    NOTE: Performance_Control{0,1,2} have no named fields in the aie-rt regdb
    (aie_registers_aie2.json), so field= cannot be used with mlir-aie's
    AIETraceRegPackWritesPass -- it would fail with "Field not found in register".
    We write raw integer values instead; the AIEXInlineTraceConfig pass generates
    a full-register write32 with the value placed at bits[6:0] (the pre-shifted
    raw encoding for these start/reset event fields).

    The correct register mapping per aie-rt (XAIEMLGBL_CORE_MODULE_PERFORMANCE_*):
      Performance_Control0: Cnt0_Start_Event [6:0], Cnt0_Stop_Event [15:8],
                            Cnt1_Start_Event [22:16], Cnt1_Stop_Event [31:24]
      Performance_Control2: Cnt0_Reset_Event [6:0], Cnt1_Reset_Event [15:8],
                            Cnt2_Reset_Event [22:16], Cnt3_Reset_Event [31:24]

    Lowering: mlir-aie's AIEXInlineTraceConfig pass translates each trace.reg
    into npu.write32 ops in the runtime sequence. No new dialect ops required.

    Returns:
        The symbol name of the emitted trace.config op.
    """
    from aie.ir import InsertionPoint, IntegerAttr, IntegerType

    sym = f"perf_{module_type}_{col}_{row}"
    i32 = IntegerType.get_signless(32)

    def mk_i32(v: int):
        return IntegerAttr.get(i32, v)

    cfg = aied.trace_config(tile_val, sym)
    cfg_region = cfg.operation.regions[0]
    cfg_block = cfg_region.blocks.append()
    with InsertionPoint.at_block_begin(cfg_block):
        # Cnt0_Start_Event = 28 (ACTIVE) at bits[6:0] of Performance_Control0.
        # Raw value 28 writes only the LSB 7 bits; stop_event=0, cnt1 fields=0.
        # field= is NOT used: the aie2 regdb has no named fields for these regs.
        aied.trace_reg("Performance_Control0", mk_i32(28))
        # Cnt0_Reset_Event = 5 (PERF_CNT_0) at bits[6:0] of Performance_Control2.
        # Counter resets itself every `period` active cycles (self-reset loop).
        aied.trace_reg("Performance_Control2", mk_i32(5))
        # Counter threshold: fires every `period` active cycles.
        aied.trace_reg("Performance_Counter0_Event_Value", mk_i32(period))
        # Explicit terminator required by SingleBlockImplicitTerminator<EndOp>.
        # The implicit-terminator mechanism only applies when the block is
        # created by the MLIR parser; we create it manually, so we must add
        # aie.end ourselves.
        aied.end()

    return sym


def _build_trace_config(
    *,
    test_name: str,
    src_mlir: str,
    buffer_size: int,
    kernel_arg_slot: int,
    address_patch_arg_idx: int,
    memref_arg_count: int,
    trace_mode: str,
    core_grounding: list[str],
    core_sweep: list[str],
    shim_grounding: list[str],
    shim_sweep: list[str],
    memtile_grounding: list[str],
    memtile_sweep: list[str],
    memmod_grounding: list[str],
    memmod_sweep: list[str],
    tiles: list[tuple[int, int, list[str], str | None]],
) -> dict:
    """Construct the trace_config.json payload for the schema.

    Schema lives at tools/trace_config_schema.json; spec at
    docs/superpowers/findings/2026-05-05-trace-config-schema.md.

    Args:
        tiles: list of (col, row, events, module) for each tile traced.
            ``kind`` is derived from row (0/1/>=2 -> shim/memtile/core).
            ``module`` is "core" or "mem" for compute tiles (their two
            distinct trace units), and None for shim/memtile (only one
            trace unit).  Two entries with the same (col, row) but
            different modules represent the core trace and memmod trace
            of the same compute tile.
    """
    # Kernel signature: opcode + instr + ninstr + N memref-data args + trace BO.
    # Slots 0/1/2 are bridge convention (opcode/instr/ninstr); slots 3..3+N-1
    # carry the runtime_sequence memrefs as host-bound BOs; trace BO lands at
    # `kernel_arg_slot` (== 3+N when not embedded).
    args_list: list[dict] = [
        {"slot": 0, "kind": "scalar", "name": "opcode", "ctype": "uint64_t"},
        {"slot": 1, "kind": "bo", "name": "instr",
         "role": "instruction_buffer"},
        {"slot": 2, "kind": "scalar", "name": "ninstr", "ctype": "uint32_t"},
    ]
    for i in range(memref_arg_count):
        args_list.append({
            "slot": 3 + i,
            "kind": "bo",
            "name": f"bo_data{i}",
            "role": "data",
            "memref_idx": i,
        })
    args_list.append({
        "slot": kernel_arg_slot,
        "kind": "bo",
        "name": "bo_trace",
        "role": "trace",
    })

    tiles_traced = []
    for col, row, events, module in tiles:
        kind = "shim" if row == 0 else "memtile" if row == 1 else "core"
        entry = {
            "col": col, "row": row, "kind": kind,
            "events": list(events),
            "packet_id": 1,  # AIEInsertTraceFlows reassigns; record the seed.
        }
        # `module` is only meaningful for compute tiles (they have two
        # trace units: one for the core, one for the memory module).
        # When the caller passes a module ("core" or "mem"), record it;
        # otherwise fall back to the legacy "core" default for compute
        # tiles. Shim and memtile entries omit the field per schema.
        if kind == "core":
            entry["module"] = module if module is not None else "core"
        tiles_traced.append(entry)

    # MLIR-declared placement origin -- the smallest (col, row) corner of
    # any instrumented tile. EMU honors this literally; HW may shift, in
    # which case the per-side events.json carries the runtime-observed
    # origin and trace-compare normalizes by it.
    placement = {
        "origin_col": min(t["col"] for t in tiles_traced),
        "origin_row": min(t["row"] for t in tiles_traced),
    }

    return {
        "schema_version": 1,
        "test_name": test_name,
        "src_mlir": src_mlir,
        "placement": placement,
        "buffer": {
            "size_bytes": buffer_size,
            "kernel_arg_slot": kernel_arg_slot,
            "embedded_in_memref_idx": None,
        },
        "kernel_signature": {"args": args_list},
        "tracing": {
            "mode": trace_mode,
            "core_grounding": core_grounding,
            "core_sweep": core_sweep,
            "shim_grounding": shim_grounding,
            "shim_sweep": shim_sweep,
            "memtile_grounding": memtile_grounding,
            "memtile_sweep": memtile_sweep,
            "memmod_grounding": memmod_grounding,
            "memmod_sweep": memmod_sweep,
        },
        "tiles_traced": tiles_traced,
        "routing": {
            "shim_col": 0,
            "shim_dma_channel": 1,
            "shim_bd_id": 15,
            "trace_done": {
                "broadcast": _TRACE_BROADCAST_STOP,
                "user_event": "USER_EVENT_2",
            },
        },
        "diagnostics": {
            "expected_address_patch_arg_idx": address_patch_arg_idx,
            "expected_runtime_sequence_memref_count": memref_arg_count,
        },
    }


def _inject_trace_ops(module, input_path: str, aied, args) -> int:
    """Walk module, find compute tiles (row >= 2), inject one aie.trace per tile,
    and prepend aie.trace.host_config + aie.trace.start_config ops to the
    aie.runtime_sequence body.

    Args:
        module: the parsed MLIR module
        input_path: source file path (for error messages)
        aied: aie.dialects.aie module (imported lazily)
        args: argparse.Namespace with CLI flags:
          - buffer_size: trace buffer size in bytes
          - trace_mode: "event_time" or "event_pc"
          - core_grounding, core_sweep_events
          - memmod_grounding, memmod_sweep_events (live as of #374)
          - memtile_grounding, memtile_sweep_events (live as of #373)
          - shim_grounding, shim_sweep_events (live as of #372)
          - perfcnt_period: PERF_CNT_0 firing period in cycles

    Compute tile classification for npu1 / npu1_1col:
      row 0  -- shim tile (NoC interface)
      row 1  -- memory tile (shared SRAM, no core)
      row >= 2 -- compute tile (VLIW core + vector unit)

    Per-tile trace injection:
    Construction uses the decorator-form lowercase builder aied.trace(tile, sym),
    which matches how mlir-aie's own configure_trace() (python/utils/trace/setup.py)
    constructs these ops.  The decorator creates the TraceOp with a proper region
    block, sets up the InsertionPoint for the body, calls the decorated function,
    and returns the completed op -- identical to how aied.core() and aied.device()
    work.  This is Path A (direct Python binding constructors).

    Perfcnt config blocks:
    When PERF_CNT_0 is in the core grounding set, one aie.trace.config block is
    emitted per compute tile alongside the aie.trace block.  The config block
    programs Performance_Control0/1 and Performance_Counter0_Event_Value so the
    hardware counter free-runs at args.perfcnt_period cycles.
    mlir-aie's AIEXInlineTraceConfig pass lowers each trace.reg to npu.write32.

    Runtime sequence injection:
    After the per-tile traces are inserted, we find aie.runtime_sequence and
    prepend two classes of ops at block-begin (before any existing runtime ops):
      aie.trace.host_config buffer_size = <N>
      aie.trace.start_config @trace_t{col}_{row}   (one per compute tile)
      aie.trace.start_config @perf_core_{col}_{row} (one per perfcnt block)
    This mirrors mlir-aie's own configure_trace() in python/utils/trace/setup.py
    (lines 534-542).  mlir-aie's AIEInsertTraceFlows pass then lowers these
    declarative ops to register-write sequences during aiecc.py compilation.

    Output note: the caller must use module.operation.print(print_generic_op_form=False)
    rather than str(module) because the aie.trace region body lacks an explicit
    block terminator in MLIR generic form, which breaks Module.parse() on reload.
    The custom text-form printer handles this implicitly via aie.trace.stop.

    Returns 0 on success, non-zero on error.
    """
    from aie.ir import InsertionPoint, IntegerAttr

    # Collect every aie.device op at the top level. Modules like
    # ctrl_packet_reconfig have more than one -- e.g. an @base overlay
    # containing only tiles plus an @main carrying the kernel and runtime
    # sequence -- and trace injection must land in the device that owns the
    # runtime sequence, not whatever device comes first in source order.
    device_ops = [
        op for op in module.body.operations
        if op.operation.name == "aie.device"
    ]
    if not device_ops:
        print(
            f"error: no aie.device op found in {input_path}",
            file=sys.stderr,
        )
        return 1

    # A target device = an aie.device whose body contains an
    # aie.runtime_sequence. Injecting trace decls into a device without a
    # runtime_sequence produces orphan ops that break aiecc's trace-lowering
    # pass ("aie.trace ops found but no runtime_sequence defined").
    targets = []  # list of (device_op, device_body, runtime_sequence_op)
    for device_op in device_ops:
        device_body = device_op.operation.regions[0].blocks[0]
        rs_op = next(
            (op for op in device_body.operations
             if op.operation.name == "aie.runtime_sequence"),
            None,
        )
        if rs_op is not None:
            targets.append((device_op, device_body, rs_op))

    if not targets:
        print(
            f"warning: no aie.device with an aie.runtime_sequence found in "
            f"{input_path}; trace decls not injected (would be orphans)",
            file=sys.stderr,
        )
        return 0

    # Pick the trace BO's BO arg index. Despite the name, address_patch's
    # arg_idx is NOT the XRT kernel slot -- it's the 0-based index into the
    # BO portion of the kernel regmap (firmware reads regmap[0x14 + N*8]
    # where 0x14 is the offset past opcode/instr-BO/ninstr). With N memref
    # args at BO arg_idx 0..N-1 (XRT slots 3..3+N-1), the next free slot
    # is N, which corresponds to XRT slot 3+N on the host side.
    #
    # Note on conventions: upstream IRON's start_trace defaults ddr_id=4
    # (XRT slot 7) and xrt_test_wrapper.h hardcodes group_id(7), assuming
    # a fixed 5-BO layout where slot 6 is a ctrlpkt placeholder. Our
    # injector instead places the trace BO directly after the existing
    # memrefs (matching what cpp_trace_patch will emit on the host side):
    # 3 memrefs + bo_trace = 4 BO slots total, no placeholder. This means
    # we land at arg_idx=N rather than 4; consumers need to derive the
    # trace slot from trace_config.json's buffer.kernel_arg_slot or by
    # scanning insts.bin's max DdrPatch arg_idx (the standalone runner
    # heuristic). The 5-BO kernels.json template is harmless extra slots.
    max_existing_memref_args = max(
        len(rs.operation.regions[0].blocks[0].arguments) for _, _, rs in targets
    )
    chosen_arg_idx = max_existing_memref_args

    # Resolve per-module-type event lists from CLI args.
    # _resolve_events(grounding, sweep, defaults) -> list of up to 8 event names.
    #
    # Compute the actual core grounding set from --core-grounding (NOT from a
    # hardcoded default), so that a user passing a custom grounding set still
    # gets a non-overlapping sweep default. The dedup in _resolve_events would
    # save us from a real bug today, but driving the sweep-default exclusion
    # off the hardcoded set would be silently misleading the moment someone
    # changes --core-grounding.
    core_grounding_names = {
        s.strip() for s in args.core_grounding.split(",") if s.strip()
    }
    _core_sweep_defaults = tuple(
        e for e in _TRACE_DEFAULT_CORE_EVENTS if e not in core_grounding_names
    )

    core_events = _resolve_events(
        args.core_grounding,
        args.core_sweep_events,
        _core_sweep_defaults,
    )
    # Determine trace mode attribute for core tiles.
    mode_attr = {
        "event_pc": aied.TraceMode.EventPC,
        "inst_exec": aied.TraceMode.Execution,
    }.get(args.trace_mode, aied.TraceMode.EventTime)
    # Whether to emit perf counter config blocks (only when PERF_CNT_0 is in grounding).
    emit_core_perfcnt = "PERF_CNT_0" in core_grounding_names

    # Shim trace injection is opt-in. When --shim-sweep-events is unset (None)
    # we leave row-0 tiles alone, matching pre-stage-1 behavior. Any non-None
    # value (including "all") activates the shim injection path: row-0 tiles
    # get a TraceOp with packet type ShimTile, no trace_mode (shim has no
    # Mode bitfield per regdb), and no perfcnt config (shim has no
    # Performance_Counter0 -- if --shim-grounding includes PERF_CNT_0 the
    # slot is reserved but the event will never fire until shim perfcnt
    # support lands as its own feature).
    inject_shim = args.shim_sweep_events is not None
    shim_events = _resolve_events(
        args.shim_grounding,
        args.shim_sweep_events,
        _TRACE_DEFAULT_SHIM_EVENTS,
    ) if inject_shim else []

    # Memtile trace injection is opt-in on the same pattern. Any non-None
    # --memtile-sweep-events value activates row-1 tile instrumentation:
    # TraceOp with packet type MemTile, no trace_mode (memtile has no Mode
    # bitfield per regdb), and one aie.trace.port slot config per
    # PORT_RUNNING_<N> event. Memtile *does* expose Performance_Counter0
    # in its register space, but the perfcnt config emit path is currently
    # core-only -- so PERF_CNT_0 in --memtile-grounding would reserve a
    # slot for an event that never fires today (same caveat the shim
    # branch documents).
    inject_memtile = args.memtile_sweep_events is not None
    memtile_events = _resolve_events(
        args.memtile_grounding,
        args.memtile_sweep_events,
        _TRACE_DEFAULT_MEMTILE_EVENTS,
    ) if inject_memtile else []

    # Memmod (compute mem-module) trace injection is opt-in. Stage 3 of #374.
    # The memmod trace unit lives on the SAME compute tile as the core trace
    # unit -- it's the second of two trace units that compute tiles expose
    # (core at 0x340D0, memmod at 0x140D0). Activating this path emits a
    # second aie.trace decl per compute tile alongside the existing core
    # decl, distinguished by:
    #   - sym name: trace_mem_<col>_<row> (vs. trace_t<col>_<row> for core).
    #   - TracePacketType.Mem (vs. .Core).
    #   - No trace_mode: memmod's Trace_Control0 has no Mode bitfield per
    #     regdb -- it always runs in event_time, regardless of --trace-mode.
    #   - No perfcnt config: memmod *does* expose its own Performance_Counter0
    #     in the regdb, but the perfcnt config emit path is core-only today.
    #     PERF_CNT_0 in --memmod-grounding reserves a slot for an event that
    #     never fires until that wiring is added.
    inject_memmod = args.memmod_sweep_events is not None
    memmod_events = _resolve_events(
        args.memmod_grounding,
        args.memmod_sweep_events,
        _TRACE_DEFAULT_MEMMOD_EVENTS,
    ) if inject_memmod else []

    # Track every traced tile across all target devices for trace_config.json.
    # Tuple shape: (col, row, events, module). `module` is "core" or "mem"
    # for compute tiles (which have two trace units), and None for shim
    # and memtile (single trace unit each).
    all_traced_tiles: list[tuple[int, int, list[str], str | None]] = []

    # Inject into every device that has a runtime_sequence. Each device gets
    # its own set of trace decls for its own compute tiles, and its own
    # runtime-sequence prologue (host_config + one start_config per tile).
    for device_op, device_body, rs_op in targets:
        # Collect tile SSA values from this device body, partitioned by row.
        # Keep them in declaration order so injected traces appear
        # deterministically. compute_tiles always populated; shim_tiles only
        # when shim injection is opted into.
        compute_tiles = []  # list of (col: int, row: int, tile_ssa_value)
        shim_tiles = []     # list of (col: int, row: int, tile_ssa_value)
        memtile_tiles = []  # list of (col: int, row: int, tile_ssa_value)
        for inner in device_body.operations:
            if inner.operation.name == "aie.tile":
                col = int(IntegerAttr(inner.operation.attributes["col"]).value)
                row = int(IntegerAttr(inner.operation.attributes["row"]).value)
                if row >= 2:
                    compute_tiles.append((col, row, inner.operation.result))
                elif row == 1 and inject_memtile:
                    memtile_tiles.append((col, row, inner.operation.result))
                elif row == 0 and inject_shim:
                    shim_tiles.append((col, row, inner.operation.result))

        if not compute_tiles:
            # No kernels live on this device -- skip silently. A device with
            # a runtime_sequence but no compute tiles is unusual but not an
            # error (e.g. a placeholder device). Skip shim/memtile injection
            # here too: tracing DMA boundaries without any compute tiles to
            # drive them would produce an empty trace anyway.
            continue

        # Record traced tiles for the trace_config.json payload. Core events
        # apply to compute tiles (module="core"); memmod events apply to the
        # *same* compute tiles when memmod injection is opted in (module="mem"
        # entry sits alongside the "core" entry); shim events apply to
        # row-0 tiles; memtile events apply to row-1 tiles.
        for col, row, _ in compute_tiles:
            all_traced_tiles.append((col, row, core_events, "core"))
        if inject_memmod:
            for col, row, _ in compute_tiles:
                all_traced_tiles.append((col, row, memmod_events, "mem"))
        for col, row, _ in memtile_tiles:
            all_traced_tiles.append((col, row, memtile_events, None))
        for col, row, _ in shim_tiles:
            all_traced_tiles.append((col, row, shim_events, None))

        # Insert trace ops before the device body's terminator.  The aie.device
        # block's last op is always its terminator (aie.end for well-formed input,
        # but using "last op" rather than matching name == 'aie.end' is resilient
        # to upstream renames and to blocks where the terminator differs).
        ops_list = list(device_body.operations)
        if not ops_list:
            print(
                f"error: aie.device body is empty in {input_path}; cannot inject",
                file=sys.stderr,
            )
            return 1
        terminator = ops_list[-1]

        # Track perfcnt config symbols so we can add trace_start_config for them.
        perf_syms: list[str] = []

        with InsertionPoint(terminator.operation):
            for col, row, tile_val in compute_tiles:
                # The @aied.trace(tile, sym) decorator-form builder:
                #   1. Constructs TraceOp(tile=tile_val, sym_name=sym_name)
                #   2. Appends a block to the op's single region
                #   3. Sets the InsertionPoint to at_block_begin of that block
                #   4. Calls the decorated function synchronously (populates body)
                #   5. Restores the previous InsertionPoint
                # This exactly mirrors how mlir-aie's configure_trace() works.
                # The decorator invokes _trace_body synchronously before the next
                # loop iteration reassigns the name, so redefinition is safe.
                # core_events / mode_attr are loop-invariant, so the closure
                # references them directly; when per-tile event sets are
                # introduced this becomes a per-iteration capture.
                @aied.trace(tile_val, _trace_sym(col, row, "core"))
                def _trace_body():  # noqa: F811 -- redefinition is intentional (one per tile)
                    aied.trace_mode(mode_attr)
                    # packet id is the conventional starting id; AIEInsertTraceFlows
                    # reassigns ids when it lowers the declarative ops.
                    aied.trace_packet(_TRACE_PACKET_ID_START, aied.TracePacketType.Core)
                    for event_name in core_events:
                        aied.trace_event(event_name)
                    aied.trace_start(broadcast=_TRACE_BROADCAST_START)
                    aied.trace_stop(broadcast=_TRACE_BROADCAST_STOP)

                # Emit perfcnt config block alongside the trace op, if grounding
                # includes PERF_CNT_0.  The block programs Performance_Control0/1
                # and Performance_Counter0_Event_Value so the counter free-runs at
                # the requested period.  mlir-aie's AIEXInlineTraceConfig pass
                # lowers each trace.reg into npu.write32 at runtime-sequence emit.
                if emit_core_perfcnt:
                    sym = _emit_perfcnt_config(
                        aied, tile_val, "core", col, row, args.perfcnt_period
                    )
                    perf_syms.append(sym)

            # Shim trace ops mirror configure_trace()'s shim-tile path in
            # mlir-aie's setup.py. Differences from the core block:
            #   - No trace_mode: shim Trace_Control0 has no Mode bitfield.
            #   - TracePacketType.ShimTile.
            #   - No perfcnt config: shim has no Performance_Counter0
            #     register exposed for trace use today.
            #   - No trace_port: the default shim events are GenericEvents
            #     (DMA_*_TASK / DMA_*_STREAM_STARVATION), not BasePortEvents,
            #     so no port-slot configuration is needed. If a future caller
            #     passes shim PORT_RUNNING events via --shim-sweep-events,
            #     we'd need to mirror configure_trace's port_configs handling
            #     here -- punt until there's an actual user.
            for col, row, tile_val in shim_tiles:
                @aied.trace(tile_val, _trace_sym(col, row, "shim"))
                def _shim_trace_body():  # noqa: F811 -- redefinition is intentional (one per tile)
                    aied.trace_packet(_TRACE_PACKET_ID_START, aied.TracePacketType.ShimTile)
                    for event_name in shim_events:
                        aied.trace_event(event_name)
                    aied.trace_start(broadcast=_TRACE_BROADCAST_START)
                    aied.trace_stop(broadcast=_TRACE_BROADCAST_STOP)

            # Memtile trace ops mirror configure_trace()'s mem-tile path.
            # Differences from the shim block:
            #   - TracePacketType.MemTile.
            #   - PORT_RUNNING_<N> events require a paired aie.trace.port
            #     slot config (the upstream MemTilePortEvent class binds
            #     each event to a (port, channel, direction) tuple before
            #     calling trace_port). We derive the default DMA bundle
            #     mapping from the slot index via _memtile_default_port_config.
            #   - Non-PORT_RUNNING events fall through as plain GenericEvents
            #     (no port slot needed) -- this lets a user pass a mixed
            #     event list via --memtile-sweep-events without breaking.
            #   - No perfcnt config emitted: memtile *does* expose
            #     Performance_Counter0 in its register space (per the
            #     AM025 regdb), but the perfcnt config emit path is
            #     core-only today; PERF_CNT_0 in --memtile-grounding
            #     would reserve a slot for an event that never fires
            #     until that wiring is added.
            for col, row, tile_val in memtile_tiles:
                # Pre-compute the slot-port configs for this tile's events.
                # Multiple events can share a slot (e.g. PORT_RUNNING_0 and
                # PORT_TLAST_0 both monitor slot 0); the upstream code
                # validates that they agree on (port, channel, direction).
                # We just build the dict here -- _memtile_default_port_config
                # returns the same answer for every PORT_*_X variant of the
                # same X, so no per-tile validation is needed today.
                memtile_port_cfgs: dict[int, tuple[int, bool]] = {}
                for ev_name in memtile_events:
                    cfg = _memtile_default_port_config(ev_name)
                    if cfg is not None:
                        slot = int(ev_name.rsplit("_", 1)[1])
                        memtile_port_cfgs[slot] = cfg

                @aied.trace(tile_val, _trace_sym(col, row, "memtile"))
                def _memtile_trace_body():  # noqa: F811 -- redefinition is intentional (one per tile)
                    aied.trace_packet(_TRACE_PACKET_ID_START, aied.TracePacketType.MemTile)
                    for event_name in memtile_events:
                        aied.trace_event(event_name)
                    # Emit one trace_port per unique slot. Mirrors mlir-aie's
                    # setup.py loop (line 499-502): "for slot, (config, _) in
                    # port_configs.items(): trace_port(slot, port, channel,
                    # direction)". master=True maps to S2MM, False to MM2S.
                    for slot, (channel, master) in memtile_port_cfgs.items():
                        direction = (
                            aied.DMAChannelDir.S2MM if master
                            else aied.DMAChannelDir.MM2S
                        )
                        aied.trace_port(
                            slot, aied.WireBundle.DMA, channel, direction,
                        )
                    aied.trace_start(broadcast=_TRACE_BROADCAST_START)
                    aied.trace_stop(broadcast=_TRACE_BROADCAST_STOP)

            # Memmod trace ops mirror configure_trace()'s is_mem_trace=True
            # branch in mlir-aie's setup.py. Emitted on the SAME compute
            # tiles as the core trace; the memmod trace unit lives at the
            # tile's secondary trace register block (0x140D0+ vs 0x340D0
            # for core). Differences from the core block:
            #   - sym name: trace_mem_<col>_<row> via _trace_sym(.., "mem").
            #   - TracePacketType.Mem (the dialect's enum value for the
            #     memory module of a compute tile, distinct from MemTile
            #     which is the row-1 shared SRAM tile).
            #   - No trace_mode: memmod's Trace_Control0 has no Mode
            #     bitfield per regdb -- always runs in event_time.
            #   - No trace_port: the upstream defaults are all
            #     GenericEvents (DMA START tasks, bank conflicts, edge
            #     detection pins), so no port-slot bindings to thread.
            #     If a future caller passes PORT_RUNNING_X events on the
            #     memmod, the upstream MemEvent / PortEvent layout needs
            #     to be threaded through here -- punt until a real user
            #     surfaces.
            #   - No perfcnt config: same caveat as memtile/shim above;
            #     memmod has its own Performance_Counter0 in the regdb,
            #     but the perfcnt config emit path is core-only today.
            if inject_memmod:
                for col, row, tile_val in compute_tiles:
                    @aied.trace(tile_val, _trace_sym(col, row, "mem"))
                    def _memmod_trace_body():  # noqa: F811 -- redefinition is intentional (one per tile)
                        aied.trace_packet(_TRACE_PACKET_ID_START, aied.TracePacketType.Mem)
                        for event_name in memmod_events:
                            aied.trace_event(event_name)
                        aied.trace_start(broadcast=_TRACE_BROADCAST_START)
                        aied.trace_stop(broadcast=_TRACE_BROADCAST_STOP)

        rs_block = rs_op.operation.regions[0].blocks[0]
        with InsertionPoint.at_block_begin(rs_block):
            # trace_host_config(buffer_size=N, arg_idx=K) emits:
            #   aie.trace.host_config buffer_size = N, arg_idx = K
            # Signature: trace_host_config(buffer_size, *, arg_idx=4,
            #                              routing=TraceShimRouting.Single, ...)
            # Mirror of mlir-aie python/utils/trace/setup.py line 534.
            # We override arg_idx (default 4) with chosen_arg_idx so the
            # trace BD's address_patch lands on a free kernel arg slot rather
            # than colliding with an existing memref arg.
            aied.trace_host_config(
                buffer_size=args.buffer_size, arg_idx=chosen_arg_idx,
            )
            # trace_start_config(name) emits:
            #   aie.trace.start_config @<name>
            # One per compute tile core decl, then memmod decls (when
            # injected -- same tile coords as core), then memtile decls,
            # then shim decls, then one per perfcnt config block. Order
            # only matters for human-readable diffs; aiecc's trace
            # lowering passes don't care about emission order.
            # Mirror of mlir-aie python/utils/trace/setup.py line 542.
            for col, row, _ in compute_tiles:
                aied.trace_start_config(_trace_sym(col, row, "core"))
            if inject_memmod:
                for col, row, _ in compute_tiles:
                    aied.trace_start_config(_trace_sym(col, row, "mem"))
            for col, row, _ in memtile_tiles:
                aied.trace_start_config(_trace_sym(col, row, "memtile"))
            for col, row, _ in shim_tiles:
                aied.trace_start_config(_trace_sym(col, row, "shim"))
            for sym in perf_syms:
                aied.trace_start_config(sym)

    # Write trace_config.json -- the single source of truth consumed by
    # cpp_trace_patch, parse-trace, trace_compare, and the bridge script.
    # Spec: docs/superpowers/findings/2026-05-05-trace-config-schema.md.
    # Required when injecting; without it downstream tools have nothing
    # to read.
    if args.trace_config_out is None:
        print(
            "error: --trace-config-out is required when injecting "
            "(downstream tools read this file as the single source of truth)",
            file=sys.stderr,
        )
        return 1

    def _split(s: str | None) -> list[str]:
        return [tok.strip() for tok in (s or "").split(",") if tok.strip()]

    cfg = _build_trace_config(
        test_name=(args.config_test_name
                   if args.config_test_name
                   else Path(input_path).parent.name),
        src_mlir=(args.config_src_mlir
                  if args.config_src_mlir
                  else str(Path(input_path).resolve())),
        buffer_size=args.buffer_size,
        # XRT kernel slot for the trace BO = 3 (opcode/instr/ninstr) + BO arg_idx.
        kernel_arg_slot=3 + chosen_arg_idx,
        address_patch_arg_idx=chosen_arg_idx,
        memref_arg_count=max_existing_memref_args,
        trace_mode=args.trace_mode,
        core_grounding=_split(args.core_grounding),
        core_sweep=_split(args.core_sweep_events),
        shim_grounding=_split(args.shim_grounding),
        shim_sweep=_split(args.shim_sweep_events),
        memtile_grounding=_split(args.memtile_grounding),
        memtile_sweep=_split(args.memtile_sweep_events),
        memmod_grounding=_split(args.memmod_grounding),
        memmod_sweep=_split(args.memmod_sweep_events),
        tiles=all_traced_tiles,
    )
    trace_config_dump(cfg, args.trace_config_out)

    return 0


def main():
    args = parse_args()

    # Warn when --trace-mode event_pc is combined with non-core sweep events.
    # The Mode bitfield exists only in the core tile's Trace_Control0 register
    # (per the AM025 register database); memmod, memtile, and shim trace units
    # have no Mode field and always operate in event_time (mode 0). This is a
    # portability warning: the non-core events will be recorded, just in mode 0.
    #
    # All non-core tile types are now driven by their --*-sweep-events
    # flags: shim (#372), memtile (#373), memmod (#374). The Mode
    # constraint applies regardless of which trigger fires; non-core
    # trace units always run in event_time per regdb.
    if args.trace_mode in ("event_pc", "inst_exec") and any(
        s for s in [
            args.memmod_sweep_events,
            args.memtile_sweep_events,
            args.shim_sweep_events,
        ]
        if s is not None
    ):
        print(
            f"warning: --trace-mode {args.trace_mode} applies to compute-core "
            "trace units only; memmod/memtile/shim trace units stay in "
            "event_time per regdb (Mode bitfield exists only in core's "
            "Trace_Control0). Non-core sweep events will be recorded in mode 0.",
            file=sys.stderr,
        )

    text = Path(args.input).read_text()
    # Import here so --help works without the mlir-aie env activated.
    # Importing aiex (before creating a Context) triggers the module-level
    # register_dialect() that makes both aie.* and aiex.* ops parseable.
    import aie.dialects.aie as aied   # noqa: F401 -- side-effect: registers aie dialect
    import aie.dialects.aiex as aiex  # noqa: F401 -- side-effect: registers aiex dialect
    from aie.ir import Context, Location, Module
    with Context(), Location.unknown():
        module = Module.parse(text)
        if not args.no_op:
            # Idempotency guard: refuse to inject if trace ops already exist.
            # --no-op is an identity round-trip and is always safe to re-run.
            if _has_trace_ops(module.operation):
                print(
                    f"error: {args.input} already contains aie.trace ops; "
                    "refusing to double-inject (exit 2)",
                    file=sys.stderr,
                )
                return 2
            rc = _inject_trace_ops(module, args.input, aied, args)
            if rc != 0:
                return rc
        buf = io.StringIO()
        # Use print_generic_op_form=False so the aie dialect's custom printers
        # emit text-form syntax (e.g. "aie.trace @sym(tile) { ... }") rather
        # than generic form ("aie.trace"(%0) <{sym_name = ...}> ({...})).
        # The generic form lacks explicit block terminators for aie.trace
        # regions and cannot be round-tripped through Module.parse().
        module.operation.print(print_generic_op_form=False, file=buf)
        Path(args.out).write_text(buf.getvalue())


if __name__ == "__main__":
    sys.exit(main() or 0)
