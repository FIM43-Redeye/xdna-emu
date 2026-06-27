# Finding: LOCK_<N>_REL trace events are selection-block indexed, not physical-lock indexed

**Date:** 2026-06-27
**Status:** characterized (HW-confirmed). Informs any future lock-event tracing.

## What

On AIE2 the memory-module lock trace events `LOCK_0_REL..LOCK_7_REL` (and the
`LOCK_SEL<N>_ACQ_*` family) are indexed by one of **8 lock-event-generation
blocks**, NOT by physical lock id. Each block selects which physical lock it
watches via a separate register `Locks_Event_Selection_<S>` (0x1F100 + S*4),
field `Lock_Select` (bits 19:16), which **resets to 0x0**
(`mlir-aie/.../aie_registers_aie2.json`, `Locks_Event_Selection_0/1/...`).

So at reset, *every* block watches physical lock 0: `LOCK_1_REL` (block 1) does
**not** observe physical lock 1 — it observes whatever
`Locks_Event_Selection_1.Lock_Select` points at, which is lock 0 unless
programmed. To trace physical lock R you must program some block S's
`Lock_Select = R` and then trace block S's event.

## HW proof (decisive)

In a two_col capture watching `LOCK_0_REL..LOCK_3_REL` on a compute memmod with
the selection registers at reset (the patcher writes only `Trace_Event*` /
`Trace_Control`, never `Locks_Event_Selection`), all four events fired at
**identical timestamps** — 64/64 `soc` overlap:

```
LOCK_0_REL first 5 soc: [757507, 757689, 757871, 758053, 758235]
LOCK_1_REL first 5 soc: [757507, 757689, 757871, 758053, 758235]
LOCK_2_REL first 5 soc: [757507, 757689, 757871, 758053, 758235]
```

Identical timestamps ⇒ all four blocks watched the same physical lock (the reset
default, lock 0). They are not distinct physical locks.

## Why it matters

It is the trap that sank the lock-substitution cross-column grounding design
(v2): keying on `LOCK_<lock_rel_id>_REL` by id assumed `LOCK_1_REL` watched
physical lock 1. It does not. An objectfifo input BD (`lock_acq_id=0,
lock_rel_id=1`) cycles *both* locks once per iteration, so "LOCK_1_REL fires 64×"
could not distinguish lock 1 from the default lock 0.

This is the **same failure class** as the `_resolve_tile_dma` S2MM/MM2S inversion
(`2026-06-27-event-map-tile-dma-direction-inversion.md`): an indexed AIE2 mapping
(port direction; lock-event selection) assumed without verifying the indirection.
The AIE2 trace/lock/port event space is dense with selection-register
indirection — verify the mapping against the register database, never assume the
name's number is the physical resource's number.

## To trace a specific physical lock (the correct recipe)

1. Pick a free selection block S; program `Locks_Event_Selection_S.Lock_Select =
   R` (the physical lock).
2. Trace block S's event (`LOCK_<S>_REL` for release, the `LOCK_SEL<S>_ACQ_*`
   family for acquire).
3. The patcher (`tools/trace-patch-events.py`) would need a new write target for
   `Locks_Event_Selection_*` (it currently writes only `Trace_Event*` /
   `Trace_Control*`).

## Relationship to connectivity (#140) and timer-sync

The cross-column lock-grounding is parked: even with the selection programmed,
the grounded cross-column edge is existence-only / non-falsifiable (both
endpoints always-on), because a falsifiable cross-timer-domain witness needs
timer-sync (BROADCAST_15). Faithful cross-column *orientation* was the bankable
win (the inversion fix). A meaningful cross-column *grounding* awaits the
timer-sync sub-project, which will also need correct lock/event selection — hence
this finding is recorded for that work.
