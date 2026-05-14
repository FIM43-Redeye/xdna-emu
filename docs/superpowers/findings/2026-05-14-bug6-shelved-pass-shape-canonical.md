---
name: bug #6 shelved 2026-05-14, canonical pass shape confirmed across 4 captures
description: Bug #6 (memtile_dmas/writebd hang at run.wait) has not naturally recurred in 4 trace-capture attempts spanning 2026-05-12 through 2026-05-14, across multiple reboots and both pre- and post-leak-fix module versions. The pass-state shape captured by tools/bug6-trace.sh is byte-stable in structure -- 5 mailbox round-trips (op 0x02 -> 0x106 -> 0x11 -> 0x18 -> 0x03), only timestamps/PIDs/ctx-IDs vary across captures. Today's "post-reboot-1" capture under the freshest possible conditions (16-min uptime, no prior NPU activity) also passed, eliminating "fresh boot" as a reliable trigger. Investigation is shelved -- if the hang recurs naturally, run bug6-trace.sh and diff against the canonical shape, walking the discrimination ladder in the 2026-05-12 finding doc.
type: finding
---

# Bug #6 -- shelved 2026-05-14, canonical pass shape confirmed

Picks up from
[2026-05-12-bug6-tracepoint-instrumentation-ready.md](2026-05-12-bug6-tracepoint-instrumentation-ready.md)
which paused with "investigation gated on natural recurrence." After
two more attempts and a confirmed-stable canonical pass shape across
all four captures, we're shelving the investigation rather than
continuing to wait.

## TL;DR

- Four trace captures spanning 2026-05-12 through 2026-05-14 all
  produced the **same canonical pass shape** (5 mailbox round-trips,
  identical opcode order). Only timestamps, PIDs, ctx-IDs vary.
- Today's `post-reboot-1` capture under the freshest possible
  conditions (16-min uptime, no prior NPU activity since reboot) also
  passed cleanly. **"Fresh boot" is not a reliable trigger.**
- The bug has not fired naturally for ~48 hours; we have no hang
  trace to diff against.
- **Action**: shelve. If `memtile_dmas/writebd` hangs again, run
  `tools/bug6-trace.sh` and diff against the shape below. The
  discrimination ladder in the prior finding doc maps the first
  missing event to the failing boundary.

## Canonical pass-state shape

Verified identical (modulo PID/timestamp/ctx-ID) across:

| Capture | Date | Uptime | Module sv | Notes |
|---|---|---|---|---|
| `pass-baseline-v2` | 2026-05-12 21:09 | warm | 5CA2BD72 | Initial baseline |
| `first-attempt` | 2026-05-12 23:27 | warm | 5CA2BD72 | 2h after baseline |
| `post-leak-fix-1` | 2026-05-13 17:07 | warm | FBCB2D05 | After leak fix |
| `post-reboot-1` | 2026-05-14 16:52 | **fresh (16 min)** | FBCB2D05 | Freshest possible |

The 5-step canonical sequence:

```
1. mbox_set_tail mgmt  op=0x02   -> irq -> rx -> set_head 0x02
2. mbox_set_tail mgmt  op=0x106  -> irq -> rx -> set_head 0x106
3. mbox_set_tail USER  op=0x11   -> job_run -> irq -> rx -> set_head 0x11   (CONFIG_CU)
4. mbox_set_tail USER  op=0x18   -> irq -> rx -> signaling_fence -> job_free -> set_head 0x18  (CHAIN_EXEC_NPU)
5. mbox_set_tail mgmt  op=0x03   -> irq -> rx -> set_head 0x03   (DESTROY_CONTEXT)
```

Total wall-clock: 1.33-3 ms host-side, with rc=0 in every capture.

## What's eliminated

- **"Fresh boot" as a reliable trigger.** `post-reboot-1` was the
  freshest-state capture we can mechanically produce (16-min uptime,
  no NPU activity since boot, first writebd of the session) and
  passed in 3s.
- **Module-version dependence.** Pass shape is identical across the
  pre- and post-leak-fix module srcversions (5CA2BD72 vs FBCB2D05).
- **Trace-instrumentation overhead as a Heisenbug masker.** Trace
  events fire on every capture; pass shape is unchanged whether or
  not the test happens to reach failing state, so the instrumentation
  itself is not preventing the bug.

## What we still don't know

- The actual trigger condition. The 2026-05-12 hypothesis ("first
  writebd after sufficient device idle / fresh fd state") is not
  falsified by today's data, but it's also not predictive enough to
  reliably reproduce on demand.
- Whether the bug is genuinely fixed by some intervening change
  between the morning of 2026-05-12 (when it last fired) and now.
  Could be the leak fix, could be a kernel/firmware/microcode update
  loaded on one of the reboots in between, could be coincidence.
- Whether the bug is `writebd`-specific. We never tried other
  `memtile_dmas/*` tests under capture.

## If it fires again

```bash
./tools/bug6-trace.sh hang ~/npu-work/mlir-aie/build/test/npu-xrt/memtile_dmas/writebd/chess 30
diff build/experiments/bug6/pass-baseline-v2.trace build/experiments/bug6/hang.trace
```

Walk the discrimination ladder in
[2026-05-12-bug6-tracepoint-instrumentation-ready.md](2026-05-12-bug6-tracepoint-instrumentation-ready.md)
to map the first missing event to the failing boundary. The first
event in the canonical sequence that is **absent** from the hang
capture identifies the failing layer (host-side, mailbox post,
firmware response, IRQ delivery, RX worker dispatch, msg_id
handler, fence signal, or fence-to-XRT propagation).

## Cross-references

- [2026-05-12-bug6-tracepoint-instrumentation-ready.md](2026-05-12-bug6-tracepoint-instrumentation-ready.md) -- discrimination ladder + canonical shape + pickup-pointer details
- [2026-05-12-bug6-state-dependent-post-num_rqs-fix.md](2026-05-12-bug6-state-dependent-post-num_rqs-fix.md) -- prior hypothesis list, num_rqs=0 fix
- `tools/bug6-trace.sh` -- single-pkexec trace+test+snapshot
- `build/experiments/bug6/*.{trace,dmesg,meta,rc}` -- all four pass captures preserved on disk
