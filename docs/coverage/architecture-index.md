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
| Watchpoint hardware (memory-address triggers) | AM025 Compute mem `WatchPoint0/1` (2) | MISSING | — | Register slots reserved in regdb but no emulator behavior. Distinct from `XDNA_EMU_WATCH` env-var debug aid. |
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
| Tile_Control register (clock + isolation bits) | AM025 (compute) | PARTIAL | `src/device/registers.rs:237` | layout parsed; field semantics (clock-gating, isolation gates) not interpreted. |
| Module clock control | AM025 `Module_Clock_Control` | MISSING | — | Clock-gating writes accepted but no effect on cycle counts / power model. Probably OK to stay OUT_OF_SCOPE for emulation. |
| Tile isolation gates (N/S/E/W) | aie-rt `pm/xaie_tilectrl.c`, AM025 | MISSING | — | Direction-gating of stream switch / DMA. If a kernel relies on isolation, our routing model may pass packets that real HW would block. |

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
| Watchpoint hardware (4 slots) | AM025 `WatchPoint0..3` | MISSING | — | Same as compute-tile watchpoints. |
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
1. **Watchpoint hardware** (compute mem 2 / mem tile 4) — kernels using debug breakpoints would silently no-op on emulator.
2. ~~**Error halt path**~~ **FIXED 2026-05-14**. Generic `error_halt` flag set + `INSTR_ERROR` event fired at every CoreStatus::Error transition (decode failure, missing program memory, executor Error). ECC errors continue to fire `ECC_ERROR_STALL`. Saturation/watchdog/other error sources are still untracked. See `src/interpreter/core/interpreter.rs::raise_instr_error`.
3. ~~**Bank conflict event-fire**~~ **FIXED 2026-05-14**. Per-bank `MEM_CONFLICT_DM_BANK_N` events (compute 77..84, memtile 112..120) now fire into mem_trace + mem_perf_counters when scalar load/store conflict detected. See `src/interpreter/execute/cycle_accurate.rs::fire_bank_conflict_events`.
4. **Tile isolation gates** — directional N/S/E/W gating. If a kernel relies on isolation, packets we route would be blocked on real HW.
6. ~~**Packet handler status register**~~ **FIXED 2026-05-14**. `Control_Packet_Handler_Status` (compute 0x3FF30, memtile 0xB0F30) now backed by `tile.pkt_handler_status` with sticky bits + write-1-to-clear semantics. Reassembler header-parse failure sets bit 1 (`Second_Header_Parity_Error`). Other sticky bits (Tlast / SLVERR / ID_Parity) not yet wired -- no current code path detects those conditions.

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

Order suggested by likely impact on current mode-2 divergences and
upcoming Option 1 cycle-validation:

1. **Watchpoint hardware** — small register-state machine; needed for debugger work.
2. **Tile isolation gates** — gate the routing layer on isolation bits.

Items 7+ in the gaps list above are deliberately deferred until pass 1
deep-dives surface unforeseen interactions.
