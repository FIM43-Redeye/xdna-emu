---
name: 'EMU gap: trace routing through widened devices and distributed channels'
description: The three Tier-6 TRACE=ERROR tests (add_one_ctrl_packet, dmabd_task_queue, packet_flow_fanout) all share a single root cause with vec_mul_trace_distribute_lateral -- the emulator does not route trace traffic when the trace planner places it on a widened device column or splits it across distributed shim DMA slots.
type: project
---

# EMU gap: trace routing through widened / distributed shim DMA

## TL;DR

Four tests with HW PASS and EMU PASS produce **zero** bytes of trace
data on the emulator side, all for the same reason. The trace planner
finds the application already occupies the default shim DMA channels,
and rather than failing, it either:

1. **Widens the device** -- adds a spare column so trace traffic can use
   that column's shim DMA, OR
2. **Distributes the trace BD** -- splits trace across multiple shim DMA
   slots (e.g., channel 0 and channel 1) when a single channel can't
   carry it.

Both code paths produce kernels that run correctly on HW (real silicon
handles the alternative routing transparently) and on EMU (kernel
output is identical), but the emulator's trace event capture path
**only handles the default single-channel, origin-column layout**. When
trace events land on a widened column's shim DMA or are routed to a
distributed BD slot, the EMU drops them silently.

The trace summary classifier sees HW with hundreds of events and EMU
with zero, classifies the comparison as "count mismatch", and reports
TRACE=ERROR.

## Evidence -- the failing trio share the same `trace-prepare.log` signature

| Test | trace-prepare verdict |
|------|----------------------|
| `add_one_ctrl_packet` | occupied=[(0,0),(0,1)] → widened to **npu1_2col** |
| `packet_flow_fanout`  | occupied=[(0,0),(0,1)] → widened to **npu1_2col** |
| `dmabd_task_queue`    | occupied=[(2,0)] → **distributed plan across 4 shim DMA slots** |
| `vec_mul_trace_distribute_lateral` | test explicitly opts into distribute-channels + lateral-routing via `-aie-insert-trace-flows="distribute-channels=true lateral-routing=true"` (no occupied-channel pressure -- test design) |

For comparison, CLEAN tests:
- `add_one_cpp_aiecc`: no occupancy → default routing → CLEAN
- `packet_flow_fanin`: occupied=[(0,0)] but **fits** in remaining default channels → CLEAN
- `add_one_ctrl_packet_4_cores`: 4 application packet_flows but fits without widening → CLEAN
- `packet_flow`: 4 packet_flows, fits → CLEAN

Pattern: **widening or distribution → TRACE=ERROR. Default-channel routing → CLEAN.**

## Per-test trace_raw.bin

```
$ md5sum *.emu/trace_raw.bin *.hw/trace_raw.bin
b6d81b3...  add_one_ctrl_packet.chess.emu/trace_raw.bin   # all zeros
7dab1b5...  add_one_ctrl_packet.chess.hw/trace_raw.bin    # HW events
```

EMU trace buffers are 1 MB of zeros; HW buffers contain hundreds to
thousands of trace events.

## Why HW passes

Real silicon's shim DMA + stream switch don't care which column owns the
trace BD -- the packet flow's destination address is encoded in the BD,
and the firmware sets up the BD correctly regardless of which column we
chose. EMU's trace dispatch path checks the *origin column* and the
*default channel slot* and silently skips events that aren't routed
that way.

## What to fix in EMU

Three layers, in increasing order of effort:

1. **Trace BD instantiation on widened columns.** When trace-prepare
   widens the device to npu1_2col and the trace BD lives in column 1
   (the spare column), EMU must instantiate that column's shim tile
   and accept its DMA writeback. Today the emulator's shim DMA model
   may not be reaching the spare column for trace traffic.

2. **Distributed-channel BD writeback.** When the trace planner splits
   trace across two shim DMA channels (channel 0 + channel 1), each
   half-buffer needs the corresponding BD's writeback to land at its
   offset in the trace BO. This is what `vec_mul_trace_distribute_lateral`
   tests directly: channel 0 → bytes 0..N/2-1, channel 1 → bytes N/2..N-1.

3. **Lateral-routing trace ingress.** Trace events generated in column 0
   that get routed laterally (through the stream switch fabric to column 1's
   shim) must propagate through the EMU's stream switch model the same way
   they do on HW. This is essentially "stream switch routing already works
   for data flows -- make sure trace packet flows take the same path."

The first two are local to the shim DMA / DDR patch path. The third is
about stream switch trace traffic.

## Connection to prior work

This unifies what the 2026-05-10 bridge coverage classification doc
flagged as two separate gaps:

- "Forward gap #1: Trace-decoder ERROR on three otherwise-passing
  tests (Tier 6)" -- those three tests, fixed.
- "Forward gap #2: EMU support for distribute-channels + lateral-routing
  trace" -- the underlying mechanism, fixed.

They're the same root cause. Fixing the EMU's widened/distributed trace
routing recovers all four tests.

## Root cause confirmed (2026-05-11 night)

The drop point is upstream of the stream switch -- the trace units on the
application column never **arm**. Trace start uses BROADCAST_15: on a
widened device the trace planner places `Event_Generate` on the spare
column's shim (e.g., (2,0)), generating BROADCAST_15 there, and expects
every tile's trace unit to see the corresponding per-module hw_id
(compute = 122, shim = 125, memtile = 157). EMU's `propagate_broadcasts`
in `src/device/state/effects.rs` only propagates within the source
column. Bridge log:

```
Tile(2,0) Event_Generate: event 127 -> BROADCAST channel 15
Propagating BROADCAST channel 15 from tile (2,0) to column 2 at cycle 5099
```

Column 2 only. Column 1 (where the application + trace units live)
never receives the event, so `(1,2)` / `(1,3)` / `(1,1)` / `(1,0)`
trace units stay in `Idle` and emit no packets. The stream-switch
routing in the original breadcrumb was correct -- there's nothing to
route because the trace units are silent.

`TraceUnit (1,2) Control0: mode=EventTime start=122 stop=121 -> Idle`
is what the debug log shows; the trace unit is configured but parked.

## Why the naive "broadcast to all columns" fix breaks other tests

Attempted 2026-05-11: extended `propagate_broadcasts` to iterate every
column, on the theory that real HW's broadcast network spans the array
and default block masks are clear. Results:

| Test                        | TRACE before | TRACE after | Kernel before | Kernel after |
| --------------------------- | -----------: | ----------: | ------------: | -----------: |
| add_one_ctrl_packet         |        ERROR |   **CLEAN** |          PASS |     **PASS** |
| dmabd_task_queue            |        ERROR |   **CLEAN** |          PASS |     **PASS** |
| packet_flow_fanout          |        ERROR |       CLEAN |          PASS |     **FAIL** |
| vec_mul_trace_distribute_*  |         NONE |        NONE |          FAIL |          FAIL |

`packet_flow_fanout` regressed: kernel output came back as zeros where
HW returns 8. The `chess.trace.log` shows `EDGE_DETECTION_EVENT_0` on
tile `(0,0)` firing **963 times on EMU vs 7 on HW**, with a regular
2-cyc interval -- a classic sign of a signal toggling continuously
where it shouldn't. Some tile's edge detector input is matching the
flooded broadcast hw_id and triggering every cycle, propagating into
the kernel's data path.

The fix needs to do one of:

1. **Honor per-tile per-channel broadcast block masks** (registers
   `EVENT_BROADCAST_BLOCK_{S,W,N,E}_{SET,CLR,VALUE}` at `0x34050+stride*16`).
   On HW, the absence of trace-relevant block configuration means the
   broadcast IS supposed to flood the array, but apparently HW handles
   the flood in a way EMU doesn't model -- need to investigate whether
   trace-prepare actually writes block masks (CDO inspection) or whether
   EMU's edge detector / trace-unit notify paths have spurious activation.
2. **Verify EMU's edge-detector signal modeling.** A 2-cyc-period toggle
   pattern suggests `curr_active` is being set every step by some
   mechanism not present in HW. Possibly the broadcast notify is
   re-firing the same channel each cycle, or the edge detector polarity
   is inverted from HW.
3. **Possibly trace-prepare CDO inspection.** Check whether the trace
   planner emits block-mask register writes that EMU doesn't recognize
   (silently dropped), in which case the planner-intended scoping is
   being lost.

Tracked as task #27. The revert preserves the previous "all four tests
TRACE=ERROR, all but vec_mul_trace_distribute_lateral PASS kernel-side"
state.

## Task #27 investigation (2026-05-11 late)

Three drop-candidate audits ran in parallel: EMU broadcast/mask state,
trace-prepare CDO output, and edge-detector signal modeling. Findings:

- **EMU already parses all block-mask registers and stores `block_mask`
  per direction** in `BroadcastConfig` (`src/device/events/broadcast.rs`),
  with `is_blocked()` and `allowed_directions()` helpers. The bug is
  upstream: `propagate_broadcasts` never consults them.
- **trace-prepare emits zero block-mask writes.** AIEInsertTraceFlows
  programs only `Event_Broadcast_N_A` source-select, never
  `EVENT_BROADCAST_BLOCK_*`. So on HW the masks stay at reset (0 = no
  blocking) and the broadcast does flood the array. The flood is
  benign on HW.
- **EMU has two unrelated bugs that the flood exposes.** Both fixed
  on this iteration as defensive hardening (see "Defensive fixes
  landed" below):
  1. `propagate_broadcasts` calls `notify_mem_trace_event(hw_id=0)` on
     shim tiles and `notify_core_trace_event(hw_id=0)` on memtiles
     (there's no matching module side; hw_id 0 is the EVENT_NONE
     sentinel). The receiving `notify_*_trace_event` didn't guard
     against hw_id=0 and iterated edge detectors, where disabled
     detectors default to `input_event=0`, so the comparison
     `det.input_event == hw_id` (0 == 0) accidentally activated them.
  2. Even with hw_id != 0, the edge-detector match was vulnerable
     because it didn't skip disabled detectors. Added the
     `det.input_event != 0` guard for defense in depth.

### What still blocks #23

Tried the BFS again with the defensive fixes in place. `packet_flow_fanout`
still regresses with a different failure mode: **kernel deadlocks at
~58k cycles** rather than producing zeros. The deadlock signature:

- One broadcast event (channel 15) fires at cycle 5088 from tile (2,0).
- BFS reaches application column 1; trace units on (1,2) and (1,3)
  ARM and start capturing trace events (LOCK_STALL = hw_id 26, etc.).
  Bridge log shows `TraceUnit (1,2): EventPc mode received event hw_id=26`
  warnings beginning shortly after the broadcast.
- DMA(2,0) ch1 (trace destination, 1 MB total) enters Transferring at
  cycle 6782 and never advances.
- Application DMAs continue making progress for ~50k cycles, then
  the last forward DMA activity is `DMA(1,1) ch9` entering
  Transferring at cycle 58049. After 100k cycles of no progress the
  stall detector halts the run.

So the bug surface has *moved* from edge detectors to the trace
plumbing itself: once the application-column trace units start
emitting words, something between trace_unit -> tile stream switch
-> col 2 shim DMA backpressures and eventually deadlocks the
application DMA chain. Candidates:

- Trace stream contention on the application column's stream switch
  arbiter (slave[23]/[24] in packet mode) starving the data flow.
- A timer-reset side effect: trace-prepare configures timer reset on
  BROADCAST events, and when the broadcast arrives on the application
  column the timer reset interacts with the kernel in a way that HW
  doesn't model the same.
- Stream switch packet header / pkt_id mismatch on the cross-column
  routing -- words leave (1,2) but never arrive at (2,0).

This is **task #27's true root cause for #23**: cross-column
propagation is necessary but not sufficient. The downstream trace
flow plumbing has to work end-to-end too.

## Defensive fixes landed (2026-05-11)

These two changes are safe on their own and address the spurious
edge-detector activation surfaced during this investigation:

- `notify_core_trace_event` / `notify_mem_trace_event` in
  `src/device/tile/mod.rs` now early-return when `hw_id == 0`
  (the EVENT_NONE sentinel). This prevents disabled trace units,
  unmatched edge detectors, and timer reset_event=0 sentinels from
  spuriously activating.
- Edge-detector match loop now also skips `input_event == 0`
  detectors explicitly, as defense in depth.

Bridge sweep across the four affected tests plus seven CLEAN controls
(add_one_using_dma, packet_flow_fanin, vec_vec_add_*, etc.) shows
11/11 PASS kernel-side. The pre-existing TRACE=ERROR on the four
victims is preserved -- not fixed -- by this iteration.

The `propagate_broadcasts` BFS attempt was reverted to column-only;
git history preserves the BFS code if/when the trace plumbing bug is
fixed and we can land cross-column propagation.

## Task #28 investigation (2026-05-11 night)

Re-ran BFS with instrumented stall-time stream-switch state dump.
At the stall (cycle 158051, after 100k cycles of no progress) the
state of `packet_flow_fanout` is:

```
STALL-DUMP TileSwitch(1,0) arbiter_locks=[_, Some(14), Some(15), _, _, _, _, _]
                          slave[14]:has_data slave[15]:has_data slave[22]:has_data
STALL-DUMP TileSwitch(1,1) arbiter_locks=[Some(13), _, _, Some(17), _, _, _, _]
                          slave[3]:has_data slave[13]:has_data slave[17]:has_data
STALL-DUMP TileSwitch(1,2) arbiter_locks=[_, _, _, _, Some(23), _, _, _]
                          slave[17]:has_data slave[23]:has_data
STALL-DUMP TileSwitch(1,3) arbiter_locks=[_, _, Some(23), _, _, _, _, _]
                          slave[23]:has_data
STALL-DUMP TileSwitch(2,0) arbiter_locks=[Some(12), _, _, _, _, _, _, _]
                          slave[10]:has_data slave[11]:has_data
```

Every tile along the trace flow path has at least one arbiter locked
with the holding slave **mid-packet** (`active_packets[holder] = Some`,
but slave FIFO either drained or stalled). The chain of holds traces
the trace stream path:

```
(1,2)/(1,3) compute trace slave[23]
        --> (1,1) slave[17] arb=3 (LOCKED, mid-packet)
        --> (1,0) slave[15] arb=2 (LOCKED)
        --> (2,0) slave[12] arb=0 (LOCKED)
        --> shim_mux S2MM ch1 --> DMA(2,0) ch1 (Transferring, never advances)
```

### The contention: arb=3 on memtile (1,1)

Slot config on memtile (1,1) (decoded with the AIE2 slot layout
`id[28:24] mask[20:16] enable[8] msel[5:4] arb[2:0]`):

| Slave | Slot raw  | pkt_id | mask | msel | arb |
|-------|-----------|--------|------|------|-----|
| `[3]` | `0x061F0103` | 6 | 0x1F | 0 | 3 |  ← data path (DMA ch9, pkt_id=6)
| `[17]`| `0x011F0113` | 1 | 0x1F | 1 | 3 |  ← trace path (pkt_id=1)

**Both slaves share `arb=3`.** Trace flow arrives first (~cycle 5088 +
propagation latency) and locks `arb=3`. Data flow arrives later (cycle
58049 when DMA(1,1) ch9 enters Transferring) but **never** acquires
`arb=3` because slave[17] holds it indefinitely (mid-packet, downstream
backpressured).

### Why the trace packet never TLASTs

The EMU's stream switch locks the arbiter for the **whole packet**
(from header to TLAST), and only releases on TLAST (`packet_switch.rs`
line ~879). Slave[17]'s packet pushes a few words into master[8] FIFO
and then... waits, because master[8] feeds shim (1,0) which feeds shim
(2,0) which feeds DMA(2,0) ch1. Somewhere in that chain a FIFO fills
and back-pressures all the way back to slave[17]. Slave[17] stays
"mid-packet, FIFO drained" forever and `arb=3` is locked.

Data flow `pkt_id=6` from DMA(1,1) ch9 pushes its 4-word header window
into slave[3], slave[3] fills (4-word FIFO), DMA can't push further and
stalls in Transferring forever. The application's MemTile MM2S
deadlocks behind it, then all the application DMAs deadlock waiting
on locks held by the stuck channels, and the kernel cores end up
waiting on locks that never release. End state: 50+ DMAs all in
"AcquiringLock granted=false", cores stalled.

### Cross-check: id=1 routings on (1,1) arb=3 today vs yesterday

|                  | Today (BFS, FAIL) | Yesterday (column-only, PASS) |
|------------------|------------------:|------------------------------:|
| `id=1` total     | 381 (mostly trace)| 4 (just the broadcast hits)   |
| `id=1` on (1,1) arb=3 | **92**       | 0                             |
| `id=6` routings  | 0                 | 5                             |

When trace doesn't fire (column-only), nothing contends `arb=3` and
data routes freely. When trace fires (BFS), `slave[17]` locks `arb=3`
and `slave[3]` never gets a chance.

### Where to fix

Three layered options, in increasing order of effort:

1. **Make the arbiter lock packet-bounded with a backpressure timeout
   or rotate fairness** -- e.g., if slave A holds an arbiter and can't
   push for N cycles, temporarily release. Not HW-accurate but breaks
   the EMU-specific deadlock. (Cheap, but a hack.)
2. **Trace EMU's downstream backpressure to its actual source** --
   probably the shim_mux/DMA(2,0) ch1 interface where the stream_in
   FIFO meets the DMA consumer rate. The DMA's words-per-cycle is 4
   and stream supply is 1/cycle, so on paper this should balance, but
   something is preventing forward progress. Likely candidates:
   `transfer_s2mm`'s "all words for this beat must be available"
   atomic-beat check, or a master FIFO sizing bug. (Medium effort,
   most informative.)
3. **Restructure trace lowering / slot allocation so trace doesn't
   share an arbiter with kernel data** -- changes the trace-prepare
   pipeline, not the EMU. Would also help HW determinism (HW handles
   this today only because its arbiter grant is fairer than ours).
   (Largest scope, out of EMU's purview.)

Option 2 is the right investment because the same backpressure
mechanism applies anywhere trace and data share an arbiter, not just
this one test. Task #28 captures it.

### Defense for #23

Cross-column broadcast propagation is correct on its own, but it must
not land in `propagate_broadcasts` until #28 is resolved -- otherwise
every widened-trace test deadlocks. The defensive `hw_id=0` /
`input_event=0` fixes from earlier today landed safely without #28
because column-only propagation never arms the application column's
trace units to begin with.

## Task #28: the true root cause is multicast deadlock

Instrumented the EMU with a deeper stall dump (master FIFO depths +
per-slave `active_packets` state) and re-ran with BFS. State at
deadlock:

```
TileSwitch(1,0) arb_locks=[None, Some(14), Some(15), None, None, None, None, None]
                s[14]:has=true active=true depth=4
                s[15]:has=true active=true depth=4
                s[22]:has=true active=false depth=4
                | m[18]:depth=2 m[19]:depth=2
TileSwitch(1,1) arb_locks=[Some(13), None, None, Some(17), None, None, None, None]
                s[3]:has=true active=false depth=4
                s[13]:has=true active=true depth=4
                s[17]:has=true active=true depth=4
                | m[7]:depth=2 m[8]:depth=2
TileSwitch(2,0) arb_locks=[Some(12), None, None, None, None, None, None, None]
                s[10]:has=true active=false depth=4
                s[11]:has=true active=false depth=4
                s[12]:has=false active=true depth=0
                |
```

Three observations make the picture crisp:

1. (2,0) `s[12]` holds `arb=0` mid-packet but its FIFO is empty
   (`has=false depth=0`). It's waiting for upstream to deliver more
   words.
2. `s[10]` and `s[11]` on (2,0) have full FIFOs (`depth=4`) but are
   idle (`active=false`) -- they're starved contenders for the same
   `arb=0`.
3. (1,0) `m[18]` is FULL (`depth=2`) and (1,0) `m[20]` is empty
   (not in the dump → `depth=0`).

### The multicast

Inspecting the routing log: `(1,0) slave[14]` routes pkt_id=1 to
**TWO** masters in the same step:

```
TileSwitch(1,0): pkt header 0x00220001 (id=1) slave[14] -> masters [18, 20]
```

Decoding the AIE2 master packet-config layout
(`enable[31] pkt[30] drop_hdr[7] msel_enable[6:3] arb[2:0]`):

- `m[18] = 0xC0000019` -> arb=1, **msel_enable=0b0011** (accepts msel=0 OR 1)
- `m[20] = 0xC0000009` -> arb=1, msel_enable=0b0001 (accepts msel=0)

`slave[14]` slot is arb=1, msel=0. Both masters accept it, so
`resolve_packet_route` returns `[18, 20]` -- an intentional multicast.
This is the *only* multicast in the entire `packet_flow_fanout` test
(verified by grepping all `masters [..]` lines).

### The deadlock cycle

The two multicast branches both go east to (2,0):

- `m[18] (1,0) -> s[10] (2,0)` (east master 18 / west slave 10)
- `m[20] (1,0) -> s[12] (2,0)` (east master 20 / west slave 12)

Both `s[10]` and `s[12]` on (2,0) have the same slot
(`pkt_id=1, arb=0, msel=0`) and both target the same master `m[5]`
through `arb=0`. **Only one slave can hold `arb=0` at a time.**

- `s[12]` wins, locks `arb=0`, starts forwarding.
- `s[10]` waits, FIFO fills (4 words), can't accept any more.
- `m[18] (1,0)` can't push to `s[10]` → fills (2 words).
- `slave[14] (1,0)` mid-packet multicast stalls because *one* of its
  target masters (`m[18]`) is full. EMU's mid-packet rule is
  all-or-nothing: pop only when *every* master can accept.
- `m[20] (1,0)` therefore gets nothing new.
- `s[12] (2,0)` drains its FIFO and waits for more data via `m[20]`.
- More data never arrives.
- **Circular deadlock:** `s[12]` can't release `arb=0` until it sees
  TLAST. TLAST is upstream of `slave[14]`. `slave[14]` won't forward
  until `m[18]` drains. `m[18]` won't drain until `s[10]` releases.
  `s[10]` won't get `arb=0` until `s[12]` releases.

In yesterday's column-only run, trace units on the application column
never arm so `slave[14]` never receives this trace data, the multicast
never happens, the deadlock never forms. In the BFS run, trace fires,
multicast fires, deadlock forms.

### Why this is structural

The EMU's `step_packet_routes` (see `src/device/stream_switch/mod.rs`)
locks an arbiter for an *entire packet* and enforces "if ANY target
master is full, don't pop." Multicast through diverging paths that
re-converge at a shared downstream arbiter is the classic deadlock
recipe under this rule.

Larger master FIFOs only delay the deadlock; they don't break the
cycle. The fix has to either:

1. **Per-target backpressure with independent forwarding.** Track
   per-master delivery progress within an active packet. Pop the word
   from the slave FIFO when *the lagging* master accepts it, but
   forward to faster masters immediately (with internal per-master
   buffering). Most HW-accurate; biggest refactor.
2. **Per-target buffered multicast.** Each multicast target gets its
   own pending-output FIFO. Slave pops a word once and pushes a copy
   into each target's pending FIFO. Each FIFO drains independently.
   Less invasive than #1 but introduces per-(slave x target) state.
3. **De-multicast at trace-prepare.** Recognize that trace doesn't
   need multicast to two east masters and program only one. Loses any
   intentional fanout (e.g., distribute-channels patterns), so this
   only helps if trace-prepare's planner is over-conservative.

The right move is #1 or #2 in the EMU because the real HW evidently
handles this multicast pattern (HW passes `packet_flow_fanout`
including trace). The structural property to preserve is: a slave
mid-packet that has at least one ready downstream must be able to
forward to that downstream regardless of the slowest one.

### Defense for the rest of the bridge

Multicast routings happen in real tests beyond just trace flow. A grep
across the bridge log catches them via `masters [.., ..]`. Any test
that hits a packet flow where two diverging branches reconverge at a
contended arbiter on the receiving end will tickle the same bug. So
far this only shows up under #23-widened-trace, which is why the
column-only mode hides it. Once #28 is fixed, cross-column
propagation can land safely (revert the column-only fallback in
`propagate_broadcasts`).

## Resolution (2026-05-11 late)

Both #28 (per-master backpressure for multicast packet routes) and the
cross-column broadcast propagation landed together:

### Stream switch refactor

`ActivePacket` now holds one `TargetState` per destination master, each
with its own `pending: VecDeque<(u32, bool)>` queue. The mid-packet
phase replaces the old "if ANY target master is full, don't pop" rule
with:

1. **Fill phase**: pop a word from the slave FIFO when every target's
   `pending` is below `MAX_PENDING_PER_TARGET` (16). Pushes a copy into
   every target's `pending`.
2. **Drain phase**: each target independently pushes as many words from
   its `pending` into its master FIFO as the master can accept this
   cycle. A slow target does not block a fast one.
3. **Completion**: release the arbiter only after the slave has popped
   TLAST AND every target has drained TLAST into its master -- this
   preserves the "no interleaving on the master FIFO" invariant for
   other slaves contending the same arbiter.

The arbiter is held from header arrival through full drain (not just
until the slave's TLAST pop), so another slave cannot start pushing
into the same masters mid-drain.

Two unit tests guard the fix: `test_multicast_slow_path_does_not_block_fast_path`
sets up a multicast where one target is never drained while the other is
drained every cycle; the fast target must still receive the full 12-word
packet. `test_multicast_reconverging_arbiter_no_deadlock` drains targets
at unequal rates (every cycle vs every 4th cycle); both targets must
eventually receive every word.

### BFS broadcast propagation

`propagate_broadcasts` in `src/device/state/effects.rs` now BFS-floods
across the array honoring each tile's per-channel per-direction block
masks (`BroadcastConfig::allowed_directions`). Trace-prepare emits no
block-mask writes today so the masks stay at reset (= no blocking) and
the broadcast effectively reaches every tile, but the BFS structure
keeps the model HW-accurate for any CDO that programs them.

### Bridge sweep result

Full EMU-only Peano sweep with both fixes:

```
56 compiled, 52 bridge pass, 3 bridge fail (1 BUDGET, 1 XFAIL)
```

The two true failures are pre-existing and orthogonal:
- `vec_mul_event_trace` -- `test.exe: No such file or directory`
  (compile-side issue; per task #20 this is NPU2-only).
- `vec_mul_trace_distribute_lateral` -- known FAIL kernel-side (the
  distribute-channels lateral-routing plumbing is still incomplete).
  Now produces 1 INSTR_EVENT_0 trace event instead of zero, so the
  trace ingress side has moved partially.

The trio of deadlock victims from the findings -- `packet_flow_fanout`,
`add_one_ctrl_packet`, `dmabd_task_queue` -- all PASS.

Task #23 still depends on full HW-vs-EMU trace-count comparison to
confirm widened-column and distributed-channel trace traffic decodes
end-to-end. The deadlock that blocked it is gone.

## Investigation breadcrumb (2026-05-11 evening)

Partial investigation on `add_one_ctrl_packet` (chess, widened to
npu1_2col, physical start_col=1):

- **EMU does configure the col-2 trace BD.** Bridge log:
  `DMA(2,0) ch1 BD15 start: total_bytes=1048576 base_addr=0x800000005000
  dir=S2MM`, channel reaches `Transferring` state at cycle 4545.
- **EMU does configure tile (1,2)'s trace slave ports.** Bridge log:
  `Tile (1,2) stream switch: slave[23] packet mode enabled`,
  `slave[23] slot[0] = 0x011F0102` (and slave[24] similarly). The
  compute trace slave port range is 23-24 (verified against
  generated `gen_stream_ranges.rs`).
- **Tiles (2,1) and (2,2) have no stream switch config.** Only tile
  (2,0) is configured in column 2 (8 stream-switch log lines vs 46 in
  column 1). This is plausibly expected if the pathfinder routes
  trace traffic vertically down column 1 first (1,2)→(1,1)→(1,0)
  and then crosses east into (2,0). Needs confirmation.
- **The DMA starves.** Channel (2,0) ch1 transitions Idle → BdSetup →
  MemoryLatency → HostPipelineLatency → Transferring, then nothing.
  No stream bytes arrive at master[5] (the source that feeds the
  shim mux's S2MM ch1).

Next step is to rerun with debug-level logging on `trace_unit` and
`stream_switch` modules to localize the drop:

```
RUST_LOG=xdna_emu::device::trace_unit=debug,\
xdna_emu::device::stream_switch=debug,\
xdna_emu::device::dma::engine=debug \
./test.exe ...
```

Drop candidates, in order of likelihood:
1. Stream switch packet header mismatch -- the events leaving (1,2)
   slave[23] don't match any master port's slot configuration on
   the route to (2,0).
2. Cross-column propagation -- the per-tile stream switch's master
   port pushes packets, but the East/West neighbor's slave doesn't
   pick them up (column boundary logic missing).
3. Trace unit not actually firing events on this kernel pattern --
   `Tile (1,2) Event_Generate` count is zero in INFO-level log; could
   be the tile's trace events never trigger because the kernel
   structure doesn't hit the configured event triggers (this would
   not be a routing bug, just an empty-trace artifact).

## See also

- `tools/trace-prepare.py` -- the planner that decides to widen vs distribute.
- `tools/mlir-trace-inject.py` -- the MLIR side that emits the post-decision routing.
- `scripts/emu-bridge-test.sh` (Phase 5 trace.summary classifier at line ~3140).
- `build/bridge-test-results/20260510/{add_one_ctrl_packet,dmabd_task_queue,packet_flow_fanout,vec_mul_trace_distribute_lateral}.*.trace.log`
  for per-test trace event count breakdowns (HW has data, EMU has zero).
- `docs/superpowers/findings/2026-05-10-bridge-coverage-classification.md`
  -- the classification doc that originally listed these as separate gaps.
