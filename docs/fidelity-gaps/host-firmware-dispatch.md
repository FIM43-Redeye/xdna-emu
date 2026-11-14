---
class: host-firmware-dispatch
subsystem: off-array host + NPU management-firmware launch / dispatch path
posture: off-array -- the wall-clock gap between core-start and the first runtime-sequence DMA dispatch, which the emulator deliberately does not model
status: launch timing pinned 2026-11-08 (cores start at their CDO enable writes, the runtime sequence ~28.5k cycles after the last); modelled in the real-firmware backend with a 9% residual; management-processor platform gaps are tracked in the private ledger named below
---

# Host / Firmware Dispatch Gaps

The wall-clock gap between core-start and the first runtime-sequence-driven DMA
dispatch. Most of this is genuinely **off-array**: the (jittery) time the host +
NPU launch path spend launching and configuring before the array does anything,
which the emulator deliberately does not model (modeling it would inject host
jitter without improving array cycle-accuracy). But the #140 SP-4a investigation
found that *part* of the EMU-side window was an on-array bug: the emulator ran
compute cores at CDO/config time because it ignored the `CORE_CONTROL` reset bit.

**This class is the seam between the array model and the host launch path** --
the reset deassert, the runtime sequence's own pacing (operation costs, token
service, sync waits) and the core warm-up are all observed at the host/array
boundary; the interpreter backend models the sequence's pacing from silicon
measurements (rows below), the rest is not modeled inside the array. See finding
`xdna-emu-priv: research/findings/2026-07-03-sp4a-core-reset-mechanism-part1.md`.

The management-processor platform that sits between the host driver and the
array is not part of this repository; its gap ledger lives in
`xdna-emu-priv: research/firmware/host-firmware-dispatch.md`. The rows below
are the gaps that belong to the public emulator.

| Gap | Model vs hardware | Where | Status / rationale | Obs. |
|-----|-------------------|-------|--------------------|------|
| Dispatch latency (core start -> first runtime-sequence op) | Silicon (launch probe, 2026-11-08): the last core of a design starts **28.5k cycles** before the runtime sequence's first op lands (28478-28610 in 10 of 12 single-core runs without padding ahead; the other two sit about 4.7k lower, 23836 and 23880: a second mode, not spread, which runs with padding ahead show too), and each op ahead of the first adds about 107-109 cycles (32 and 512 padding `write32`s). Post-reset padding changes nothing, so the start is anchored to the beginning of the sequence, not to its first transfer. The earlier "~25-32k core-start -> first drain dispatch" (w8-drainpace) is this gap plus the ops before the drain. | Real-firmware backend (`XDNA_BACKEND=phoenix-firmware`, private platform): the firmware applies the PDI and starts the sequence itself, clocked, and the gap emerges: **25860** cycles, 102-104 per op ahead (`build/experiments/launch-timing/probe2/runs/f3`). Interpreter backend: no firmware; the plugin runs the cores until every core is blocked (at most 100k cycles, `run_interpreter` warm-up in `crates/xdna-emu-ffi/src/backend.rs`) and then starts the sequence, so a core that blocks early sees the sequence a few hundred cycles later and a core that never blocks sees it 100k later (probe: 99956). Evidence: `build/experiments/launch-timing/probe2` (`tools/gen_launch_probe.py`, HW tag `r1`, 3 repeats, 0 TDR). | **MECHANISM PINNED, MODELLED in the real-firmware backend (2026-11-08)**; residual there: 2.6k short of the 28.5k mode (9%), inside silicon's range but not at its mode; the cost lies in the private firmware timing model (ledger: `xdna-emu-priv: research/firmware/host-firmware-dispatch.md`). The interpreter backend's warm-up is a stand-in, not a model: it has no firmware clock to derive the gap from, and fitting 28.5k there was rejected. | confirmed |
| Core reset-deassert at launch (`CORE_CONTROL` reset bit) | Current mlir-aie CDOs clear RESET in their init section and set ENABLE in their enable section, one write per core, so a core starts at its enable write, as the launch path applies the PDI: silicon starts the cores one gap apart and then begins the runtime sequence ~28.5k cycles after the last start (row above). The 2026-07-03 observation (cores held `CORE_CONTROL=0x03` until launch, "nothing in the CDO clears the reset bit") was made on CDOs of that time; for current CDOs no reset release by the launch path is needed. A later audit found that the device model retained RESET but published only derived runnable-state changes to the interpreter, collapsing a repeated RESET/unreset into an ordinary pause/resume with stale PC, registers, and executor pipeline. | `src/device/state/compute.rs` (run-state `enabled = ENABLE & !RESET`, ordered reset/enable publication, `stagger_core_starts`, `release_core_resets` for a CDO that leaves a core held); `src/interpreter/engine/coordinator.rs` and `src/interpreter/core/interpreter.rs` (architectural and executor reset); findings `xdna-emu-priv: research/findings/2026-07-03-sp4a-core-reset-mechanism-part1.md`, `2026-08-01-phoenix-context-repartition-proof-correction.md` | **FIXED (reset state, engine edge) and LAUNCH TIMING PINNED (2026-11-08)**: where a core starts against the runtime sequence is the dispatch row above; the start itself stays at the CDO enable write. `release_core_resets()` at config completion only matters for a CDO that leaves a core held (none of the current corpus); its timing for such a CDO is unobserved. | confirmed (current CDOs); held-reset CDO timing unobserved |
| Core register state across launches (dive node `core-reg-persistence`) | Silicon keeps a core's data registers (r, p, m/dn/dj/dc, s, ls, lc, dp, wl/wh, the accumulator quarters, q), both load-port rings, the sparse-load unit and the stream read latch from one launch to the next: in one process (several hw contexts, xclbins, launches on other tiles in between) and across processes and idle gaps up to 300 s; every launch resets the control registers (0, `crSCDEn` = `crMCDEn` = 1), the status registers (0) and `le` (0xFFFFF). EMU now does the same within one emulator instance; across processes only with `XDNA_EMU_STATE_DIR`. | `src/interpreter/state/carried.rs` (`CarriedCoreState`: what a core reset keeps; `ExecutionContext::after_reset`), `CoreState::reset` in `src/interpreter/engine/coordinator.rs` (every core reset: `Core_Control`, new hw context); power-on = `ExecutionContext::new_for_tile` (+ `XDNA_EMU_CORE_INIT`, the scrub pattern). Cross-process: `src/interpreter/engine/device_state.rs`, `XDNA_EMU_STATE_DIR=<dir>`: the FFI loads `<dir>/aie2-npu1-phoenix-hawkpoint-5x6.json` on `xdna_emu_create` and saves it on `xdna_emu_destroy`; a file of another format version or device is refused (fresh device, logged); delete it to reset the device. Evidence: `build/experiments/harness/scrub/step1` (17 launches, separate processes); `harness/core-persist/` (one process through XRT, `tools/harness/multi_launch.py`: HW r1 == r2, EMU == HW except r17-r23, which nothing in the process writes; two EMU processes with the state file == silicon's second process; `sparse/`, `latch/` probes). | **MODELLED (2026-11-06).** The harness keeps its scrub: each phase is its own process, i.e. a fresh emulated device. `lr`, `sp` and `r16` are written by crt0 / the aiecc main before any kernel code (r16 = 1 at kernel entry on both, even on a fresh device), so whether they carry is unobservable to such kernels; the model carries them as data registers. Open: the sparse unit's fill level over a launch and its reads in the underflow regime (next paragraph); a context switch between two live contexts on one column is unprobed. | confirmed |
| Firmware clock domain (dive node `fw-in-loop`, 2026-11-05) | Before: the management firmware ran up to an instruction budget per array cycle, i.e. infinitely fast against the array. Now: [`src/timebase.rs`](../../src/timebase.rs) couples the two clocks by an exact rational accumulator (`Coupler`, no floats); the firmware spends whole cycles per array cycle at the price of each step (the cost model lives in `xdna-emu-priv`, `firmware-phoenix/src/timing.rs`). The pair is one named configuration (`ClockPair`: driver DPM table entries and the SMU-reported values; `XDNA_EMU_DPM=0|1`), so DPM 0 is a setting, not a code change. Simulated wall time is `array cycles / array MHz` (`ClockPair::array_cycles_to_ps`, `xdna_emu_array_cycles_to_wall_ps`, the `run:` log line). | `src/timebase.rs`; `xdna-emu-priv: firmware-phoenix/src/{timing,runtime}.rs`; evidence `build/experiments/fw-clock/fw-timing/` and `xdna-emu-priv: research/findings/2026-11-05-phoenix-firmware-op-work-and-cpi-hypothesis.md` | **MODELLED against the runtime-sequence op probes (within a few percent on every probe case, both DPM points).** Open: the once-per-run stall on silicon (about 2-3 us after the first op arrival) now lands where silicon puts it but is about half its silicon size, and costs no probe measures yet (firmware accesses outside the array window); the itemised list is in `xdna-emu-priv: research/firmware/host-firmware-dispatch.md`. Interrupt inputs a platform drives as levels follow their line every step (`XtensaBus::interrupt_levels`, `src/firmware/xtensa/bus.rs`). | edge-observable; off-array costs unobservable (needs management-core register access) |
| Shim DMA address space against host memory (DDR aperture, 2026-11-13) | A shim BD holds an AIE address; host address = AIE address - `0x80000000`. Sources: XRT `core/common/api/elf_patcher.cpp` (`get_ddr_aie_addr_offset()`), XRT `xdp/profile/device/aie_trace/client/aie_trace_offload_client.cpp` (a trace BD set to `bo.address() + DDR_AIE_ADDR_OFFSET`), mlir-aie `include/aie/Runtime/TxnEncoding.h` (`kDDRAIEAddrOffset`, added by firmware for the first `kNumFirmwareTranslatedArgs` = 5 host arguments of a DDR patch). A BD base below the offset (for example a trace BD written by `npu.writebd` with no address patch) has no host address. Before: the emulator read such a base as the host address itself and reported an IOMMU fault for it. | `src/device/dma/engine/stepping.rs` (`resolve_shim_host_address`), `src/device/host_memory.rs` (`record_outside_aperture`) | **fixed** 2026-11-13: under the IOMMU model an unmapped shim address below the offset reaches no host memory (a write is dropped, a read gives zeros), is not an IOMMU fault, and is reported on a warn line of its own. The executor's DDR patch now writes the AIE address the firmware does (slot + arg_plus, plus the offset for the first 5 arguments; an unbound argument's slot is 0, so its BD points at host address 0 upward and faults), every shim address resolves as AIE - offset, and the in-process harness binds a buffer to each patched argument its test spec lacks instead of the executor inventing one. Host aperture confirmed on silicon; the destination of a sub-aperture access is not measured. | confirmable |
| Runtime-sequence pacing in the interpreter backend | Silicon paces a runtime sequence by what it executes. On the bridge traces and dedicated runtime-sequence probes the gap between two task pushes is the cost of the operations between them: about 105 cycles per register write or push, 179 per masked write, 584 per shim BD write with its address patch, about 450 per tile BD write and 215 per satisfied mask-poll; linear in the number of writes, no plateau, and a second task pushed on a busy channel does not wait for the first (its START follows the push by the same few tens of cycles as on an idle channel). A task-complete token that arrives while the sequence is executing costs about 2.3k cycles once. A sync costs a discrete level that depends on how late its token arrives after the sync is issued (x): on the first wait of a command that follows no other command (every bridge test), about 3.0k in total when the task had finished before the sync, x + 3.05k up to x = 0.6k, then plateaus fixed relative to the issue (5.85k to x = 0.95k, 9.54k to 5k, 14.1k to 7.2k), and beyond that the token plus about 8.75k, the resume falling on a grid of 219.8 cycles (128 controller cycles; 256 at the lower power level, where the levels and edges grow with the clock ratio too). Later waits of long runs, and waits of a command that follows another command, sit at other levels. One further 2.0..2.6k delay occurs once per command at a run-dependent point (it makes many leads and early gaps bimodal on silicon), and the first push lands 0.02..0.38k later than the operation costs predict, in levels that depend on the stream before the trace start. | `src/npu/cycle_cost.rs` (class costs, token service), `src/npu/sync_wait.rs` (`SyncWaitModel`), `src/npu/executor.rs` (`sync_charge`, token service), `src/device/dma/engine/stepping.rs` (`maybe_prefetch_next_task`) | **MODELLED, open residuals (2026-11-14).** Each operation retires at its class cost, stated in controller cycles and converted by the clock pair (`src/timebase.rs`), serially and with no gate on task pushes (the former `DispatchGate`, fixed 1086 / 3050-cycle push gaps, is removed); a token arriving during execution is serviced once at the next operation boundary; a task pushed behind a shim task still in its read warm-up starts at once. A sync costs the wait structure of `SyncWaitModel`, every constant in controller cycles with its measurement: token there at the issue, a fast path of 0.7k (the 3.0k level less the token's service, which the operation boundary before the sync already charged, so no token is charged twice); a short wait returns 3.05k after the token; once the wait passes the engage threshold (0.6k) the controller resumes at the measured level of the range the token falls in; beyond the last edge it wakes on the first tick after the token and pays an exit cost (8.64k, 8.75k on average with the tick). This replaces the former 1 cycle per sync, the token's row transit and the one-time 8000 cycles on the first sync. Open: the later-wait and warm-command levels (the model gives every wait the first-wait levels; on silicon mid-run waits at x = 0.5..2k resume at 3.5..5.5k or 7.5..8.7k, e.g. `sync_task_complete_token*`, `add_one_ctrl_packet*`, `dynamic_*.8`); the once-per-command delay; the lead levels; two tokens close together, which cost less than two services on silicon (3.6k for a pair 72 cycles apart, 4.3k for 1068 apart; the model charges one service per token, a dedicated probe is prepared); the fast path of a warm command (a dedicated probe reads 1.2k for the sync alone); final waits of some object-FIFO-shaped tests that return 1.5..5k before the plateau (`add_one_objFifo`, `add_maskwrite`, `core_dmas/dma_configure_task_token`; and a first wait of `add_one_ctrl_packet_col_overlay` that resumes about 4.9k after its issue with the token 1.0k late, where the model gives the 9.54k plateau); syncs on memtile channels or other columns, whose levels the shim trace cannot observe (the same structure is applied). Provisional: the tile BD write and satisfied mask-poll costs are fitted on the bridge corpus only (a dedicated probe is queued), and the split of the shim pair between block write and patch is not measured. Drives most `PERF_CNT_2` and trace-carrier starvation count rows, which measure the traced window. | confirmed |

**Sparse-load unit over a launch (2026-11-06, `harness/core-persist/sparse`).** The first sparse read
of a launch returns the previous launch's line (v1, v2: EMU == HW once the unit is carried), but
silicon then reports `srCompr_uf` = 1 and refetches as if its fill level were low, and its reads in
that regime do not match the emulator's ring model even within one launch: right after `reset`, a
`peek` returns a stale line from the previous launch and the next one the reset's line (`sp3_c`);
after a few pops silicon reads lines a line behind what the emulator predicts. HW r1 == r2 on every
probe. Belongs with the sparse open row of `addressing.md` ("bytes a sparse peek reads past the
first fetched line"); not modelled.

**A second run in the same hardware context (2026-11-06, `harness/core-persist/runs/hw-same-r*`).**
A run-once core (the aiecc main ends in `done`) is not restarted by a second command on the same
hw context: silicon never completes it (XRT state 8, one TDR each, r1 == r2). The emulator
completes it, because the plugin's interpreter path resets the partition and replays the PDI on
every submit (`xrt-plugin/src/platform_emu.cpp`, `submit_cmd`). Open; not a persistence issue
(registers still carry over that replay).

**Core start order and spacing (2026-11-04, cascade work).** Silicon starts the cores of a design
one at a time in column-major order, about 1448 cycles apart with run-to-run jitter (1103-2018);
see "Core start order" in [`core-compute-timing.md`](core-compute-timing.md). The emulator now
staggers the starts at the same seam (`DeviceState::stagger_core_starts`, then
`release_core_resets` for cores still held).

**Core launch against the runtime sequence (2026-11-08, launch probe).** `tools/gen_launch_probe.py`:
each core stamps its timer (`MOV_CNTR`, which reads the core module timer) at kernel entry and spins
until the value runs backwards; the runtime sequence resets the timers with a host `Timer_Control`
write (bit 31) behind N padding `write32`s or ahead of them, before its only transfer. On silicon
(`build/experiments/launch-timing/probe2/runs/r1`, 10 cases x 3, no TDR) every core had started
before the reset write: the last core 28.5k cycles before a reset at the head of the sequence
(bimodal, a second mode ~4.7k lower), the others one gap earlier each (946-1188), and padding
ahead of the reset adds ~107-109 cycles per op while padding after it adds nothing. So the cores
start as the PDI is applied and the sequence begins a firmware-sized gap later; nothing in the
sequence releases them. The real-firmware backend reproduces the structure (25860, 102-104 per op,
gaps 1100-1144); the interpreter backend starts the sequence when every core blocks (row above).

**What the trace gate's core PERF_CNT_2 metric measured.** The 2026-11-07 gate reported the core's
first PERF_CNT_2 EMU-early by a median 6028 cycles. That counter is started by ACTIVE_CORE
(`Performance_Control1` = 0x1C from the runtime sequence) and self-resets every 1024: its first
trace edge gives the arm time modulo the period, against an anchor (the first lock request) that
both sides reach when data arrives. On silicon the counter arms while the core sits in the lock
stall it entered ~28k cycles earlier; the EMU asserted ACTIVE_CORE only in cycles the core
executed, so it armed at the grant. ACTIVE_CORE is now asserted while the core is enabled, stalled
or not (`active_core_arms_a_counter_while_the_core_is_lock_stalled`). The metric was never a
measure of where the core starts. Gate, full bridge suite on master `88c99a8` vs this change
(`build/experiments/launch-timing/bridge-{base,branch}`, `core-triage-*.txt`): PERF_CNT_2 first
divergence median -5807 (p10 -15432, p90 -3194; 144 of 149 EMU-early, 5 OK) -> +1875 (p10 -7338,
p90 +10739; 40 early, 106 late, 15 OK). What is left is the anchor, not the counter: the first lock
request is reached when data arrives, and the data path's timing differs on its own. Against the
trace start instead (first shim event, modulo the period; `tools/pc2_phase.py` there) the add_*
family sits at a constant -410 cycles (HW 947-967, EMU 346): the counter now arms at a
runtime-sequence op on both sides, a few ops apart; which op arms it on silicon (and whether the
interpreter backend's per-op costs or the cross-tile timer offset make up the rest) is open,
confirmable by reading Performance_Counter2 back over control packets. Bridge verdicts identical
(chess 90 / peano 116 pass, 1 XFAIL each) and trace verdicts identical per test.

**ACTIVE_CORE on an ended core (closed 2026-11-14).** ACTIVE_CORE is low on a core that has run to
`done`, and on a core held in reset. Silicon: PERF_CNT_2 (started by ACTIVE_CORE) is 0 in every
sample on the run-once cores that end before the trace is configured (multi_source_packet_flow
(0,2) (0,3), packet_flow_same_id_merge_3way (0,3) (0,4), static_L1_init (0,3), both compilers),
and on local_reset_core it first ticks 987 cycles after the re-run core's first lock stall, i.e. it
arms at the reset release and re-enable, not at the configuration. aie-rt measures core utilisation
with the same pair (start ACTIVE_CORE, stop DISABLED_CORE). The emulator asserted it for every
enabled core, ended or not, and ticked every 1025 cycles on those cores. It now requires the core
enabled, out of reset and not halted (`active_core_is_low_once_the_core_has_ended`,
`active_core_is_low_while_the_core_is_held_in_reset`,
`active_core_rises_when_an_ended_core_is_reset_and_reenabled`). Trace gate on the four tests: the
10 run-once rows 11..14 -> 0 against silicon's 0; local_reset_core 12 -> 1 against 4, its first tick
995 cycles after the re-run's lock stall (silicon 987); the remaining count is the traced window
after the re-run (runtime-sequence timing). No other row of the four tests moved. MODELLED. Not
measured: whether an error-halted or debug-halted core asserts it (the emulator keeps it high).

**Management-core access ring (2026-11-07).** The firmware interpreter's MMU used to translate every
access at ring 0. It now follows the Xtensa ISA: the access ring is CRING (`PS.EXCM ? 0 : PS.RING`),
L32E/S32E use PS.RING, instruction fetch and the fill fast path included, and the privileged
instructions (the RM's set; RSR/WSR of special registers 64 and up) raise PrivilegedCause at
CRING != 0 (`src/firmware/xtensa/interp/ring_tests.rs`). A runtime-sequence read of a page the task's
ring may not reach now faults in the emulator as it does on silicon ("Kernel did not complete");
every other run is unchanged. MODELLED.

**Management-firmware state across processes (2026-11-07).** Silicon's management firmware boots
once, at driver load, and keeps running while host processes come and go: a process exit or a TDR
destroys and recreates hardware contexts, and only a driver reload or a runtime suspend boots the
firmware again (the open driver's `aie2_hw_stop` / `aie2_hw_start`; with runtime PM allowed, the
device suspends after an idle autosuspend delay, so firmware state outlives a process gap only
when the gap is shorter than that delay or runtime PM is held off). Lazily built firmware state
(software-filled TLB entries, cache contents) therefore stays built for later processes. With
`XDNA_EMU_STATE_DIR` set, a backend that runs a management processor saves its whole platform when
the device closes, after destroying the live hardware context as the driver does on process exit,
and the next process restores it instead of booting: the generic Xtensa core state
(`src/firmware/xtensa/state.rs`: registers, both TLBs, MMU special registers) plus the platform's
own memories and devices, in a file beside the core-state file
(`NpuBackend::save_platform_state`). A platform that is not back at its idle when the device
closes (a hang whose recovery the emulator does not model) is not saved, and the older file is
removed, so the next process boots fresh. Without the variable every process boots the firmware
fresh, as before. MODELLED; the platform-specific parts and their evidence are in the private
ledger.

**Management-core instruction cache (2026-11-07).** The private per-step cost model gains an
instruction cache (hit/miss per fetch line); its geometry and refill price live with the private
cost model. Open: cold-code cost after the largest firmware paths still exceeds what the cache
explains (private ledger).
