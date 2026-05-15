# NPU1 (AIE2/Phoenix) architecture coverage index

**Purpose**: a living catalogue of every AIE2 hardware subsystem — what it
is, where the authoritative source lives, and our emulator's coverage
state. Created after we belatedly realised control packets had been
missing from the model for months. The index is meant to make any *next*
oversight of that magnitude a one-search-away discovery.

**Update protocol**: when you touch a subsystem listed here, refresh the
**Coverage** column and the **Notes** line. When you discover a subsystem
not yet listed, add a row. The matrix is the source of truth — the
roadmap and component docs link into this, not vice versa.

**Audit checklist**: see [audit-checklist.md](audit-checklist.md) for the
questions to ask when grading a row. Item 3 (subsystem-integration
check) is the catch for "looks complete but no per-cycle consumer
exists" — the failure shape of both the missed control packets and
the timer-sync reset path.

**Cycle-accuracy mission**: this index tracks *what* is modeled.
For *how cycle-accurate* each subsystem is — open gaps, in-progress
work, deferred items, and the broadcast-latency / instruction-cost
calibration efforts — see [cycle-accuracy-mission.md](cycle-accuracy-mission.md).
Add new cycle-accuracy gaps there, not here.

**Last full sweep**: 2026-05-04, parallel Explore agents against
aie-rt, AM025 register database, xdna-driver, aietools event types, and
this repo's own `src/` tree.

## Coverage legend

- **MODELED** — emulator implements the behavior; tests exercise it.
- **PARTIAL** — register layouts decoded or status bits exposed, but at
  least one sub-behavior absent.
- **STUBBED** — placeholder exists, returns plausible defaults; no real
  state machine.
- **MISSING** — emulator has nothing matching this subsystem.
- **OUT_OF_SCOPE** — intentionally not modeling (see notes).

## Authoritative sources

| Source | Lives at | What it contributes |
|---|---|---|
| aie-rt | `../aie-rt/driver/src/` | Subsystem partition (every subdir = one HW concern), reference DMA / lock / SS programming sequences |
| AM025 register DB | `../mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json` | 1,806 regs / 6,412 fields / 4 tile types — exhaustive register surface |
| xdna-driver | `../xdna-driver/drivers/accel/amdxdna/` | What the kernel actually pokes at boot / partition / context |
| aietools events | `../amd-unified-software/aietools/data/eventanalyze/event_type_table.txt` | 95 event types — every observable signal source |
| mlir-aie device model | `tools/aie-device-models.json` | Topology + per-tile-type capabilities |

## Subsystem matrix

Grouped by tile type / scope. Compute tile = AIE-Tile (rows ≥ 2),
MemTile = mem tile (row 1 on NPU1), Shim = row 0.

### Compute tile (AIE-Tile, rows ≥ 2)

| Subsystem | Source | Coverage | Our location | Notes / known gaps |
|---|---|---|---|---|
| VLIW core | aie-rt `core/`, llvm-aie TableGen | MODELED | `src/interpreter/` | 100% ISA decode; SemanticOp coverage ~33%, rest in legacy handlers (gap). |
| Core control (enable / done / reset) | AM025 `Core_Control`, `Core_Status` | MODELED | `src/device/core_debug/` | enable, halt, done, reset paths |
| Core debug (halt / step / breakpoint) | AM025 `Debug_*` | PARTIAL | `src/device/core_debug/` | Halt + status bits modeled; programmable breakpoints / single-step PC trap not wired through interpreter. |
| Core error halt | AM025 `Error_Halt_Control` / `Error_Halt_Event` | MODELED | `src/device/core_debug/mod.rs:75`, `src/interpreter/core/interpreter.rs::raise_instr_error` | Generic `error_halt` path fires `INSTR_ERROR` (event 69) into core_trace + core_perf_counters at every CoreStatus::Error transition (decode failure, missing program memory, executor Error). Sets Core_Status bit 19. ECC errors still fire ECC_ERROR_STALL (bit 17) via `set_ecc_error`. Other error sources (saturation, watchdog) not yet detected. |
| Watchpoint hardware (memory-address triggers) | AM025 Compute mem `WatchPoint0/1` (2) | MODELED | `src/interpreter/execute/cycle_accurate.rs::matching_watchpoint_events` | Compute slots at 0x14100/4 with `WriteStrobes==0xF` gate, direction filter (Read/Write bits 31/30), 16-byte-aligned 12-bit address comparator [15:4]. Fires `WATCHPOINT_0/1` (mem events 16/17) into mem_trace + mem_perf_counters on every matching load/store (scalar AND vector -- the comparator sits at the bank interface and doesn't care which engine issued the access). **Wildcard filters**: AXI_Access [29], DMA_Access [28], and quadrant bits [27:24] are not consulted, so any matching direction+address fires. **Effective address**: `record_memory_access` dispatches through `MemoryUnit::get_address` / `get_store_address`, so indexed addressing through modifier registers (`[pN, mK]`) lands the watchpoint on the address the access actually hits, not the bare pointer. Post-modify (`op.post_modify`) is applied after the access and is correctly excluded from the recorded address. Distinct from `XDNA_EMU_WATCH` env-var debug aid. |
| Data memory (64KB, banked) | aie-rt `memory/`, AM025 | MODELED | `src/device/banking.rs`, `src/interpreter/timing/memory.rs` | 8 banks × 128-bit, conflict detection done. |
| Bank conflict events (intercore / intracore) | aietools events `MEM_CONFLICT_INTERCORE`, `_INTRACORE`, `DM_BANK_CONFLICT` | MODELED | `src/interpreter/execute/cycle_accurate.rs` | Per-bank `MEM_CONFLICT_DM_BANK_N` (events 77..84 compute / 112..120 memtile) fired into mem_trace + mem_perf_counters when scalar load/store conflict detected. INTERCORE/INTRACORE not modeled separately. |
| ECC (data memory) | aie-rt `pm/xaie_ecc.c` | OUT_OF_SCOPE | — | Status bit readable; no scrubber, no fault injection. Document as OOS unless workloads require. |
| Program memory (16KB) | AM025 | MODELED | `src/parser/elf.rs` + tile state | ELF load → run. |
| DMA engine | aie-rt `dma/`, AM025 (112 reg) | MODELED | `src/device/dma/` | BDs 16, channels 4, n-d addressing, padding, lock coupling, packet header. Compression module present. Repeat / Out-of-order BD execution: verify. |
| Locks (16 per compute tile) | aie-rt `locks/`, AM025 | MODELED | `src/device/tile/locks.rs` | acquire/release/get/set, semaphore semantics, round-robin arbiter. |
| Stream switch | aie-rt `stream_switch/`, AM025 (160 reg) | MODELED | `src/device/stream_switch/` | Circuit + packet, FIFOs, port events, packet header matching. |
| Cascade ports (tile↔tile) | aie-rt, aietools events | MODELED | `src/interpreter/execute/cascade.rs` | Read/write, deadlock not detected (`deadlock.rs` placeholder). |
| Events (128 per module) | aie-rt `events/`, AM025 | MODELED | `src/device/events/` | broadcast (16 channels), combo, group, port events. Combo/edge generators may have boundary cases — needs targeted tests. |
| Performance counters | aie-rt `perfcnt/`, AM025 | MODELED | `src/device/perf_counters/` | 4 counters; threshold events. |
| Trace unit | aie-rt `trace/` | MODELED | `src/device/trace_unit/` | Mode-0 / mode-1 / mode-2 supported. Pipelined start/stop and multi-tile timer sync modeled (2026-05-04). Residual 2-PC mode-2 divergence on `add_one_using_dma` is broadcast-event delivery latency, not a state-machine issue; deferred per [trace-start-stop-latency-gap.md](trace-start-stop-latency-gap.md). |
| Timer (per-module 64-bit) | aie-rt `timer/`, AM025 (5 reg) | MODELED | `src/device/timer.rs` | Free-running; trig_event_low/high used? — verify. |
| Tile_Control register (clock + isolation bits) | AM025 (compute) | PARTIAL | `src/device/registers.rs:237`, `src/device/state/effects.rs::apply_tile_local_effects` | Layout parsed; isolation bits S/W/N/E (low 4 bits) are now interpreted (see Tile isolation gates row). Clock-gating bits still pass through unmodeled. |
| Module clock control | AM025 `Module_Clock_Control` | MISSING | — | Clock-gating writes accepted but no effect on cycle counts / power model. Probably OK to stay OUT_OF_SCOPE for emulation. |
| Tile isolation gates (N/S/E/W) | aie-rt `pm/xaie_tilectrl.c`, AM025 | MODELED | `src/device/tile/mod.rs::isolation` (bit constants), `src/device/state/effects.rs` (Tile_Control snapshot), `src/device/array/routing.rs::propagate_inter_tile` (stream-switch gate), `src/interpreter/execute/memory/neighbor.rs` (NeighborMemory gate), `src/interpreter/engine/coordinator.rs` (NeighborLocks gate) | Tile_Control low 4 bits (S/W/N/E) snapshotted onto `tile.isolation` on register write. Inter-tile stream transfers, cross-tile NeighborMemory snapshots/reads/buffered writes, and cross-tile NeighborLocks slices all consult the destination/own isolation byte and short-circuit when blocked. Memtile uses 0x96030; compute uses 0x36030. Shim Tile_Control isolation is set up by privileged path in HW; writes pass through unmodeled here today. |

### Memory tile (MemTile, row 1)

| Subsystem | Source | Coverage | Our location | Notes / known gaps |
|---|---|---|---|---|
| Data memory (512KB) | AM025 | MODELED | `src/device/state/memtile.rs` | Larger banking. |
| DMA engine (433 reg, 64 BDs) | aie-rt, AM025 | MODELED | `src/device/dma/` | Largest DMA on the array. Verify BD count = 64 (vs 16 elsewhere). |
| Locks (64) | AM025 | MODELED | `src/device/tile/locks.rs` | 4× compute count — `params.rs` should reflect this. Verify. |
| Stream switch (119 reg) | AM025 | MODELED | `src/device/stream_switch/` | Per mlir-aie device model. |
| Events (161) | AM025 (highest count) | MODELED | `src/device/events/` | More than compute tile. |
| Performance counters (11 reg) | AM025 | MODELED | `src/device/perf_counters/` | |
| Trace unit | AM025 | MODELED | `src/device/trace_unit/` | |
| Timer | AM025 (5 reg) | MODELED | `src/device/timer.rs` | |
| Watchpoint hardware (4 slots) | AM025 `WatchPoint0..3` | MODELED | `src/interpreter/execute/cycle_accurate.rs::matching_watchpoint_events` | 4 memtile slots at 0x94100..0x9410C with `WriteStrobes==0xF` gate, direction filter (Read/Write bits 29/28), 16-byte-aligned 15-bit address comparator [18:4] covering the full 512KB span. Fires `WATCHPOINT_0..3` (memtile events 16..19) into mem_trace + mem_perf_counters on every matching scalar load/store. Same scalar-only / wildcard-filter / approximate-address constraints as the compute watchpoint row; AXI [27], DMA [26], and East/West neighbour-DMA bits [25:24] not consulted. |
| Packet handler status | AM025 `Control_Packet_Handler_Status` | MODELED | `src/device/tile/mod.rs::pkt_handler_status` + `src/device/tile/registers.rs` reads + `src/device/state/effects.rs` write-1-to-clear | Sticky bits at 0x3FF30 (compute) / 0xB0F30 (memtile). Reassembler header-parse error sets `Second_Header_Parity_Error` (bit 1). `Tlast_Error` / `SLVERR_On_Access` / `ID_Parity_Error` not yet wired (no current path detects them). |
| Memory_Control / Tile_Control / Module_Clock_Control | AM025 | PARTIAL | `src/device/registers.rs` | Layout parsed; behavior not interpreted. |

### Shim tile (NoC interface, row 0)

| Subsystem | Source | Coverage | Our location | Notes / known gaps |
|---|---|---|---|---|
| Shim DMA | aie-rt, AM025 (144 reg) | MODELED | `src/device/dma/` | DDR ↔ AIE transfers via host_memory. |
| Stream switch (149 reg) + Mux/Demux (2) | aie-rt, AM025 | MODELED | `src/device/stream_switch/` | Master/slave NoC-facing. |
| Locks (16) | AM025 | MODELED | `src/device/tile/locks.rs` | |
| Events (51) | AM025 | MODELED | `src/device/events/` | |
| Performance counters (6 reg) | AM025 | MODELED | `src/device/perf_counters/` | |
| Trace unit | AM025 | MODELED | `src/device/trace_unit/` | |
| Timer | AM025 | MODELED | `src/device/timer.rs` | |
| **L1 Interrupt controller** (per-tile) | aie-rt `interrupt/`, AM025 | MODELED | `src/device/interrupts/l1.rs` | 20 IRQs, mask/enable/status. |
| **L2 Interrupt controller** (NoC aggregator, 23 reg) | aie-rt, AM025 | PARTIAL | `src/device/interrupts/l2.rs` | Module exists; not sure if 23-reg shim_intc_l2 register surface is exhaustive. Privilege gating not modeled. |
| Direct NoC control (4 `NoC_Interface_AIE_to_NoC_SouthN` regs) | AM025 | MISSING | — | Memory-mapped NoC packet injection. We don't model NoC at all; impacts cycle accuracy more than functional. |
| AIE_AXIMM_Config | AM025 | MISSING | — | AXI-MM config for shim. Bus-width / endian bits — verify if they affect shim DMA correctness. |
| Tile column reset | AM025 `AIE_Tile_Column_Reset` | MISSING | — | Partition reset on context teardown. We don't simulate partition lifecycle; relevant once multi-context support is built. |
| Reset_Control_1, Module_Clock_Control_0/1 | AM025 | PARTIAL | `src/device/registers.rs` | Layout parsed. Multi-clock-domain semantics absent. |
| Column_Clock_Control | AM025 | MISSING | — | Per-column clock gating. |
| Packet_Handler_Status (shim) | AM025 | MISSING | — | Same as MemTile entry. |
| PL Interface (Upsizer / Downsizer) | aie-rt `pl/`, AM025 (3 reg) | OUT_OF_SCOPE | — | Versal-FPGA stream-width adaptation; NPU1 doesn't expose programmable PL. |
| NPI (privileged register access) | aie-rt `npi/` | OUT_OF_SCOPE | — | Driver-side privilege; emulator gives unrestricted access. |

### Array / global

| Subsystem | Source | Coverage | Our location | Notes / known gaps |
|---|---|---|---|---|
| Tile array topology | mlir-aie device model | MODELED | `src/device/array/` | NPU1 5×6 / NPU2 5×6 / NPU3 8×6 layouts. |
| CDO loading | aie-rt indirectly | MODELED | `src/parser/cdo/` | Framing, syntax, semantics → DeviceOps. |
| ELF loading | — | MODELED | `src/parser/elf.rs` | Per-core executables. |
| XCLBIN parsing | — | MODELED | `src/parser/xclbin.rs` | Container format. |
| Stream switch routing reconstruction | — | MODELED | `src/parser/stream_switch_topology.rs` | From CDO writes. |
| Control packet handling | — | MODELED | `src/device/control_packets/` | Headers, reassembly, register read/write effects, response packets. Recently added — keystone subsystem we previously missed. |
| NPU instruction stream | XRT host protocol | MODELED | `src/npu/` | WRITE32, BLOCKWRITE, BLOCKSET, MASKWRITE, MASKPOLL, CONFIG_SHIMDMA_*, DDR_PATCH. |
| NoC fabric (latency / arbitration) | hardware spec | STUBBED | scattered | Stream switch has tile-internal latency; NoC inter-tile latency is fudged. Cycle-accuracy impact catalogued in `docs/archive/findings/2026-05-04-control-path-cycle-calibration.md`. |
| Multi-tile timer sync (broadcast) | aie-rt `timer/` | MISSING | — | Cross-tile timer alignment via broadcast event. Used by trace correlation. |
| Cross-tile event broadcast network | aie-rt `events/`, AM025 | MODELED | `src/device/events/broadcast.rs` | 16 channels with directional masking. Verify L2 propagation. |

### Driver-side surfaces (mostly orchestration, not silicon)

These are kernel-driver / firmware concerns rather than hardware-state
surfaces. The emulator answers IOCTLs but doesn't model the driver
state machines themselves. Listed for completeness so we don't
re-discover them as "missing hardware."

| Subsystem | Source | Coverage | Notes |
|---|---|---|---|
| PSP firmware load | xdna-driver `aie_psp.c` | OUT_OF_SCOPE | Pre-NPU-live firmware validation — emulator skips. |
| SMU power / clock | xdna-driver `aie_smu.c` | OUT_OF_SCOPE | Power gating / DPM levels — irrelevant in functional sim. |
| Mailbox infra | xdna-driver `amdxdna_mailbox.c` | OUT_OF_SCOPE | Host↔FW messaging is what the emulator *replaces*. |
| Partition / context mgmt | xdna-driver `aie2_ctx.c` | PARTIAL | We accept context create/destroy via XRT; lifecycle state machine simplified. |
| Async event / error reporting | xdna-driver `aie2_error.c` | MISSING | Emulator doesn't surface ECC / DMA / saturation / stream / lock / instr errors as async events. Worth doing if we ever model error halts properly. |
| Telemetry / app-health | xdna-driver | OUT_OF_SCOPE | Driver-side instrumentation. |
| TDR (timeout detection / recovery) | xdna-driver `aie2_tdr.c` | OUT_OF_SCOPE | Emulator can't deadlock the same way real HW does. |
| Preemption / QoS | xdna-driver | OUT_OF_SCOPE | Driver-side scheduling. |
| Debug BO (HWCTX_ASSIGN_DBG_BUF) | xdna-driver | MISSING | Debug buffer attachment IOCTL — not modeled, would be useful for matching xrt-side debug flows. |

## Gaps summary by triage

### Likely-impactful gaps (model-correctness affecting)

0. ~~**Multi-tile timer sync**~~ **FIXED 2026-05-04**. `Timer_Control.Reset_Event` is now consumed via a `pending_reset` latch on `TileTimer`. Both `notify_core_trace_event` and `notify_mem_trace_event` route through it. See [timer-sync-gap.md](timer-sync-gap.md) for full implementation notes; bridge re-run pending to confirm mode-2 divergence drops.
0a. ~~**Trace controller pipelined start/stop**~~ **FIXED 2026-05-04**. After the timer-sync fix exposed it, the residual mode-2 PC divergence on `add_one_using_dma.chess` was exactly 2 frames (1 at start, 1 at end). Modeled HW's 1-cycle pipelined Idle→Running by deferring state transition until cycle advances past the arm cycle; same-cycle stop-window frames are now discarded. See [trace-start-stop-latency-gap.md](trace-start-stop-latency-gap.md). Bridge re-run pending.
1. ~~**Watchpoint hardware** (compute mem 2 / mem tile 4)~~ **FIXED 2026-05-14**. Per-tile register storage already persisted writes; this commit adds the access-checking path. Every scalar load/store calls `matching_watchpoint_events` which gates on `WriteStrobes==0xF`, decodes the direction filter (Read/Write bits), masks the address comparator, and returns the matching slots' event IDs. `fire_watchpoint_events` notifies mem_trace + mem_perf_counters with `WATCHPOINT_N` (compute mem 16/17, memtile 16-19). DMA-engine path and AXI/quadrant filters are follow-up work; core-halt wiring on watchpoint hit is also deferred.
2. ~~**Error halt path**~~ **FIXED 2026-05-14**. Generic `error_halt` flag set + `INSTR_ERROR` event fired at every CoreStatus::Error transition (decode failure, missing program memory, executor Error). ECC errors continue to fire `ECC_ERROR_STALL`. Saturation/watchdog/other error sources are still untracked. See `src/interpreter/core/interpreter.rs::raise_instr_error`.
3. ~~**Bank conflict event-fire**~~ **FIXED 2026-05-14**. Per-bank `MEM_CONFLICT_DM_BANK_N` events (compute 77..84, memtile 112..120) now fire into mem_trace + mem_perf_counters when scalar load/store conflict detected. See `src/interpreter/execute/cycle_accurate.rs::fire_bank_conflict_events`.
4. ~~**Tile isolation gates**~~ **FIXED 2026-05-14**. Tile_Control writes (compute 0x36030, memtile 0x96030) now snapshot the low 4 bits onto `tile.isolation`. Three gate sites consult that byte: stream-switch inter-tile routing drops cross-boundary transfers, NeighborMemory short-circuits cross-tile snapshots/reads/buffered writes, NeighborLocks hides the slice for blocked directions. Shim Tile_Control isolation (privileged-path setup in HW) still passes through unmodeled.
5. ~~**Packet handler status register**~~ **FIXED 2026-05-14**. `Control_Packet_Handler_Status` (compute 0x3FF30, memtile 0xB0F30) now backed by `tile.pkt_handler_status` with sticky bits + write-1-to-clear semantics. Reassembler header-parse failure sets bit 1 (`Second_Header_Parity_Error`). Other sticky bits (Tlast / SLVERR / ID_Parity) not yet wired -- no current code path detects those conditions.

### Cycle-accuracy gaps (functional-OK, timing-off)

7. **NoC latency / arbitration** — fudged. Documented in `docs/archive/findings/2026-05-04-control-path-cycle-calibration.md`.
8. **DMA FIFO size events** — not emitted; affects performance-counter output.
9. **Stream switch FIFO size events** — same.
10. **Module / column / tile clock control** — clock-gating writes are silent. Real HW gates state.

### Driver-state gaps (mostly OOS, but worth flagging)

11. **Async error reporting** — error injection from emulator side would let host-side tests exercise error paths.
12. **Debug BO IOCTL** — would help bridge tests match real-HW debug flows.

### Verifications needed (assumed MODELED, but worth confirming)

Items 13-14 cleared on 2026-05-04 against aie-rt
`xaiemlgbl_reginit.c`. Per-tile-type AIE-ML constants:

| | Compute | MemTile | Shim |
|---|---|---|---|
| Locks | 16 ✓ | 64 ✓ | 16 ✓ |
| DMA BDs | 16 ✓ | **48** ✓ (not 64 — earlier draft was wrong) | 16 ✓ |
| DMA channels | 2 ✓ | 6 ✓ | 2 ✓ |
| Address dims | 3 ✓ | 4 ✓ | 3 ✓ |

The `192` value in aie-rt's `AieMlMemTileDmaMod.NumLocks` is the
*reference-range* the MemTile DMA can address (cross-tile lock space),
not the count of lock slots in the MemTile itself.

Remaining verifications:

13. Timer `Trig_Event_Low/High_Value` registers — write effect plumbed?
14. Cascade `deadlock.rs` placeholder — promote to real detection or remove.
15. Combo / edge event generators (all tiles) — boundary-case tests?
16. Repeat / out-of-order BD execution — verify in `dma/`.
17. Event broadcast L2 propagation through interrupts/l2.rs.

## Pass 2: deep-dive priorities

Pass 1 quick wins are all closed. Outstanding follow-ups inherited
from those passes are queued below; promote any of them when impact
warrants.

Watchpoint follow-ups (status confirmed via 2026-05-14 deep-validation
pass; field positions and event IDs cross-checked against AM025 +
aie-rt and locked in by 17 unit tests):
- ~~**Vector loads/stores skip watchpoint**~~ **FIXED** (task #62).
  The `if !op.is_vector` guard now scopes only the bank-conflict block;
  `fire_watchpoint_events` runs unconditionally so VLD/VST fire
  `WATCHPOINT_N` exactly like scalar loads/stores. Locked in by
  `test_watchpoint_vector_load_fires_event`.
- ~~**DMA-engine path doesn't fire watchpoints**~~ **FIXED 2026-05-14**
  (task #68 + #69). `transfer_mm2s` and `transfer_s2mm` (incl. compressed
  and decompressed variants) now call `fire_watchpoint_events_with_origin`
  per word actually moved, after the data loop ends so the borrow on
  `target_tile.data_memory[_mut]` is released. Resolvers were widened to
  return `(MemTileTarget, &mut Tile, usize)` so the firing path can branch
  on own-vs-neighbour; the helper `dma_access_origin` then maps Own to
  `AccessOrigin::Dma` and West/East cross-tile targets to
  `Neighbour(East)`/`Neighbour(West)` (the direction the DMA *appears to
  arrive from* on the target tile, matching the AM025 memtile E/W filter
  bits at [25:24]). Locked in by 16 unit tests covering S2MM/MM2S firing,
  AXI-only filter exclusion, DMA-only filter inclusion, multi-word, wrong
  direction, address mismatch, compressed MM2S, memtile own window,
  source-tile silent on cross-tile DMA, West/East neighbour-side firing
  for both S2MM and MM2S, neighbour DMA-only filter exclusion, neighbour
  East-quadrant filter inclusion, and missing-neighbour fallback to Own
  with `Dma` origin.
- ~~**Approximate address calculation**~~ **FIXED 2026-05-15** (task
  #66). `record_memory_access` now routes through
  `MemoryUnit::get_address` / `get_store_address` (made `pub(crate)` for
  this), the same helpers the actual load/store path uses. Indexed
  addressing through modifier registers (`[pN, mK]`) is now reflected in
  both bank-conflict tracking and watchpoint matching, where it was
  previously dropped (the resolver only looked at the first PointerReg
  / Memory operand, ignoring the trailing modifier). Post-modify
  (`op.post_modify`) is intentionally still excluded -- the modifier
  updates the base register *after* the access, so the address the
  access lands on this cycle is the pre-modify value. Locked in by
  4 unit tests (modifier-register effective address fires watchpoint;
  same address does NOT fire at the bare-base watchpoint; post-modify
  fires at base and not at base+imm; store path mirrors load path).
- ~~**AXI_Access / DMA_Access filter bits unmodeled**~~ **FIXED
  2026-05-14** -- `matching_watchpoint_events_with_origin` now consults
  AXI_Access [29 compute / 27 memtile], DMA_Access [28 / 26], and the
  quadrant bits. Semantics: when any origin filter bit is set, the
  access origin must match one of the enabled bits; when all bits are
  zero, the slot is wildcard (any origin including Core fires). Core
  has no AM025 enable bit, so it never matches a non-zero filter --
  consistent with HW. Tasks #68 (own-tile DMA) and #69 (cross-tile
  MemTile-to-MemTile DMA) are now done -- the DMA engine passes
  `AccessOrigin::Dma` for own-tile traffic and `Neighbour(East/West)`
  for cross-tile traffic on the resolved target, exercising both the
  DMA filter and the E/W quadrant filter from real callers. AXI is the
  only remaining origin without a wired caller.
- ~~**East/North/West/South_Access quadrant bits unmodeled**~~ **FIXED
  2026-05-14** along with the above. Compute supports all four
  quadrants; memtile only has E/W per AM025, so Neighbour(North) and
  Neighbour(South) on memtile silently never match (equivalent to no
  enable bit existing), which is what HW would do.
- ~~**Core-halt-on-hit not wired**~~ **FIXED 2026-05-14** -- AIE2 has
  a *general* event-driven debug halt mechanism (Debug_Control1/2 +
  Debug_Status, per AM025), not a watchpoint-specific halt bit. We
  now decode Debug_Control1 (Event0/Event1/Resume event IDs) and
  Debug_Control2 (mem/lock/stream stall-halt enables); when an event
  matches a configured halt trigger, we request_halt and latch the
  matching Debug_Status cause bit (bits 5/6 for Event0/1; bits 2/3/4
  for the three stall categories; bit 0 for any-cause aggregate).
  `tile.notify_core_trace_event` and `tile.notify_mem_trace_event`
  (compute tiles only -- memtile/shim have no core to halt) call
  `core_debug.check_event_halt(event_id)` so every event source flows
  through the halt selector. Watchpoint events specifically were
  rerouted from `mem_trace.notify_event` (bypass) to the dispatcher,
  so a watchpoint hit configured as Debug_Halt_Core_Event0 now halts
  the core end-to-end. PC_Event_Halt (Debug_Control2 bit 0) **FIXED
  2026-05-14**: PC_Event0..3 registers (offsets 0x32020/4/8/C, layout
  bit 31 VALID + bits [13:0] PC_ADDRESS per aie-rt xaiemlgbl_params.h)
  are now modeled. `update_pc` drives `check_pc_events`, which
  broadcasts Core_PC_0..3 events (IDs 16-19 per xaie_events_aieml.h)
  on single-slot matches and Core_PC_Range_0_1 / Core_PC_Range_2_3
  (IDs 20/21) when PC is within a valid pair (both endpoints VALID;
  endpoint order is normalized). Each fired event flows through
  `check_event_halt` so HaltEvent0/1 wiring works; independently,
  Debug_Control2.PC_Event_Halt (bit 0) gates a halt that latches
  halt_cause_pc_event (Debug_Status bit 1). Reset clears all four
  PC_Event registers. Single-step-on-event **FIXED 2026-05-14**:
  Debug_Control1.SSTEP_EVENT (bits [14:8]) is now consumed.
  `check_event_halt` arms a `pending_single_step` latch on a matching
  event ID; the coordinator drains it after each core step via
  `consume_pending_single_step`, which calls `request_halt` so the
  triggering bundle is the last to commit before halt
  (interpretation (a) per AM025; the spec is ambiguous between (a)
  and "one more bundle after"). A resume event between arming and
  consume cancels the pending step. There is no dedicated
  Debug_Status cause bit for single-step halts per AM025 -- the
  aggregate `halted` bit is the only signal.

Tile isolation follow-ups (status confirmed via 2026-05-14 deep-
validation pass; gate sites cross-checked against aie-rt
`pm/xaie_tilectrl.c` semantics and locked in by 14 unit tests covering
all 4 cardinal stream directions, all 4 NeighborMemory quadrants
individually, partial-isolation mixing, local-not-blocked, and stale-
snapshot eviction):
- ~~**Shim Tile_Control isolation unmodeled**~~ **FIXED 2026-05-14**.
  Snapshot now extends to row-0 shim tiles at the same offset (0x36030)
  with the same SWNE bit layout. Of the routing directions, the gate
  that actually fires for shim is memtile->shim south-bound (gates on
  shim's NORTH bit per the inbound-direction rule); other shim
  isolation bits are snapshotted but no current routing path consults
  them. NeighborMemory and NeighborLocks don't apply (shim has no
  executing core that does cross-tile quadrant ops).
- ~~**No coordinator-level NeighborLocks integration test**~~
  **FIXED 2026-05-14**. Extracted the gate construction into
  `build_neighbor_locks_with_isolation(isolation, south, west, north)`
  in `coordinator.rs` and pinned the mapping with 4 unit tests covering
  no-isolation, each-bit-individual, all-directions, and pass-through-
  None-inputs. A separate end-to-end test brings up the engine, sets
  ALL_DIRECTIONS on a compute tile, and verifies the step loop still
  advances cleanly.

Items 7+ in the gaps list above are deliberately deferred until pass 1
deep-dives surface unforeseen interactions.
