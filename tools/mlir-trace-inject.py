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


def _trace_sym(col: int, row: int) -> str:
    """Symbol name for the aie.trace op attached to tile (col, row).

    Single source of truth so a future rename only touches one place.
    Task 4 constructs the TraceOp with this name; Task 5's
    trace_start_config must reference the same name -- drift between the
    two sites would silently produce an unreachable trace config.
    """
    return f"trace_t{col}_{row}"


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
    p.add_argument("--trace-mode", choices=("event_time", "event_pc"),
                   default="event_time",
                   help="trace mode for compute-core trace units. "
                        "event_time (mode 0, default) records cycle deltas; "
                        "event_pc (mode 1) records PCs. Mode 1 is core-only -- "
                        "memmod/memtile/shim trace units always remain in mode 0 "
                        "(their Trace_Control0 has no Mode bitfield per regdb).")

    # Grounding events (fixed slots, always present, never overwritten by sweep).
    p.add_argument("--core-grounding",
                   default="PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1",
                   help="comma-separated event names reserved in fixed slots "
                        "of every compute-core trace unit. Default reserves "
                        "perfcnt cycle clock + two software pin events.")
    p.add_argument("--memmod-grounding", default="PERF_CNT_0",
                   help="grounding events for compute-tile memmod trace unit.")
    p.add_argument("--memtile-grounding", default="PERF_CNT_0",
                   help="grounding events for memtile trace unit.")
    p.add_argument("--shim-grounding", default="PERF_CNT_0",
                   help="grounding events for shim PL trace unit.")

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
                        "Default: don't inject shim trace. 'all' is reserved "
                        "for future use; currently behaves the same as the default.")

    p.add_argument("--perfcnt-period", type=int, default=1024,
                   help="cycles between PERF_CNT_0_EVENT fires when grounding "
                        "includes PERF_CNT_0 (default: 1024).")

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
      - Period = user-specified via --perfcnt-period (default 1024).

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
          - memmod_grounding, memmod_sweep_events (not yet injected)
          - memtile_grounding, memtile_sweep_events (not yet injected)
          - shim_grounding, shim_sweep_events (not yet injected)
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
    mode_attr = (
        aied.TraceMode.EventPC
        if args.trace_mode == "event_pc"
        else aied.TraceMode.EventTime
    )
    # Whether to emit perf counter config blocks (only when PERF_CNT_0 is in grounding).
    emit_core_perfcnt = "PERF_CNT_0" in core_grounding_names

    # Inject into every device that has a runtime_sequence. Each device gets
    # its own set of trace decls for its own compute tiles, and its own
    # runtime-sequence prologue (host_config + one start_config per tile).
    for device_op, device_body, rs_op in targets:
        # Collect compute tile SSA values from this device body.
        # Keep them in declaration order so injected traces appear deterministically.
        compute_tiles = []  # list of (col: int, row: int, tile_ssa_value)
        for inner in device_body.operations:
            if inner.operation.name == "aie.tile":
                col = int(IntegerAttr(inner.operation.attributes["col"]).value)
                row = int(IntegerAttr(inner.operation.attributes["row"]).value)
                if row >= 2:
                    compute_tiles.append((col, row, inner.operation.result))

        if not compute_tiles:
            # No kernels live on this device -- skip silently. A device with
            # a runtime_sequence but no compute tiles is unusual but not an
            # error (e.g. a placeholder device).
            continue

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
                @aied.trace(tile_val, _trace_sym(col, row))
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

        rs_block = rs_op.operation.regions[0].blocks[0]
        with InsertionPoint.at_block_begin(rs_block):
            # trace_host_config(buffer_size=N) emits:
            #   aie.trace.host_config buffer_size = N
            # Signature: trace_host_config(buffer_size, *, arg_idx=4,
            #                              routing=TraceShimRouting.Single, ...)
            # Mirror of mlir-aie python/utils/trace/setup.py line 534.
            aied.trace_host_config(buffer_size=args.buffer_size)
            # trace_start_config(name) emits:
            #   aie.trace.start_config @<name>
            # One per compute tile trace decl, then one per perfcnt config block.
            # Mirror of mlir-aie python/utils/trace/setup.py line 542.
            for col, row, _ in compute_tiles:
                aied.trace_start_config(_trace_sym(col, row))
            for sym in perf_syms:
                aied.trace_start_config(sym)

    return 0


def main():
    args = parse_args()

    # Warn when --trace-mode event_pc is combined with non-core sweep events.
    # The Mode bitfield exists only in the core tile's Trace_Control0 register
    # (per the AM025 register database); memmod, memtile, and shim trace units
    # have no Mode field and always operate in event_time (mode 0). This is a
    # portability warning: the non-core events will be recorded, just in mode 0.
    #
    # TODO: when memmod/memtile/shim injection actually lands, extend this
    # trigger to also check whether the user passed a non-default value for
    # --memmod-grounding / --memtile-grounding / --shim-grounding.  The Mode
    # constraint applies to grounding too, but today the grounding flags are
    # inert (only sweep events drive injection), so warning on sweep alone is
    # the only condition that produces a misleading outcome.
    if args.trace_mode == "event_pc" and any(
        s for s in [
            args.memmod_sweep_events,
            args.memtile_sweep_events,
            args.shim_sweep_events,
        ]
        if s
    ):
        print(
            "warning: --trace-mode event_pc applies to compute-core trace units "
            "only; memmod/memtile/shim trace units stay in event_time per regdb "
            "(Mode bitfield exists only in core's Trace_Control0). Non-core "
            "sweep events will be recorded in mode 0.",
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
