# Debug Halt Timing and Single-Step Count

**Source**: Phase A `debug_halt_probe`, Task 5/5b (G1) / Task 7 (G2 -- derived).
**Ground truth**: Real NPU1 (Phoenix, XDNA/AIE2, `0000:c6:00.1`), 2026-05-18.
**Observation method**: A firmware `XAIE_IO_MASKPOLL` (opcode 4) post-compile-
injected into the probe `insts.bin` blocks the instruction stream until
`Core_Status & (1<<16)` (`DEBUG_HALT`) before the control-packet OP_READ push.
The OP_READ therefore reads `output_buffer[0..3]` + `Core_Status` (0x32004)
*provably after* the core has halted (synchronization-ordered, not
latency-ordered). Five OP_READ packets; responses via shim DMA S2MM ch0.
**Verdict signal**: `Core_Status` bit 16 = `DEBUG_HALT` (mask `0x00010000`,
`XAIEMLGBL_CORE_MODULE_CORE_STATUS_DEBUG_HALT_MASK`, aie-rt
`driver/src/global/xaiemlgbl_params.h`). `HALTED = ((cs & (1<<16)) != 0)`.
Verdict computed from the disassembled store *schedule*, not source order.

Raw logs: `/tmp/claude-1000/probe-exp1-hw.log`,
`build/bridge-test-results/20260518/debug_halt_probe.chess.hw.log` (HW),
`.chess.bridge.log` (EMU). Transcribed below; the `/tmp` log is ephemeral.
Probe source: `mlir-aie/test/npu-xrt/debug_halt_probe/` (branch
`xdna-emu-cycle-budget`): `aie.mlir`/`README` at `9a12651d99`, `test.cpp`
verdict at `8546397987`; MASKPOLL injector + emulator graceful poll-
termination at xdna-emu `8be784a` (+ post-review fixups `4cd02a1`).

---

## G1 -- Breakpoint / single-step halt timing

### Probe configuration (final, Exp1)

Straight-line core: acquire a blocking objectfifo `@gate`, write four sentinel
markers to `output_buffer[0..3]`, `aie.end`. No lock-gated DMA path;
observation is solely control-packet OP_READ. Compiler: Chess (xchesscc).

Three grounded redesigns were required before a valid G1 could be observed --
each a real hardware/toolchain fact surfaced by grounding, recorded here so the
derivation is durable:

1. **Arming race** (attempt 1): the core is CDO-enabled before `@seq` runs, so
   it completed before the `@seq` breakpoint-arming `write32`s landed
   (`Core_Status=0x100000` CORE_DONE, all markers written). Fix: a host->core
   **blocking objectfifo `@gate`** (`llvm.aie2.acquire`, a HW pipeline stall
   immune to load-elimination); `@seq` arms *then* feeds the gate (in-order
   guarantee, `AIEToConfiguration`).
2. **Shim-channel collision** (attempt 2): `@gate` defaulted its shim feed to
   shim MM2S ch0, which the hand-rolled ctrl-in OP_READ push also used, so the
   pathfinder compiled a single circuit broadcast
   (`switchbox(0,2): South:1 -> {TileControl:0, DMA:0}`) -- HW-unsafe. The
   objectfifo channel is not toolchain-pinnable; fix: leave `@gate` on default
   ch0 and **repoint the ctrl-in push to shim MM2S ch1** (stride `0x8`).
3. **No happens-after** (attempt 3, P2/P3): the static-`@seq` OP_READ had no
   synchronization ordering "core halted at trap" before "OP_READ reads" --
   pure relative latency. No on-device halt->lock/event actuation exists and
   `@seq` is strictly static, so fix: post-compile-inject a firmware
   **`XAIE_IO_MASKPOLL`** on `Core_Status[16]` before the OP_READ push.

Disassembly (`llvm-objdump-aie` of `main_core_0_2.elf`, gated chess build;
re-confirmed post-injector -- the injector patches only `insts.bin`, never the
core ELF):

```
0x17a  mova dj0, #0xbb ; movxm p0, #0x70400      (load constants; p0=output_buffer)
0x184  st   dj0, [p0, #4] ; mov m0, #0xaa        (TRAP bundle: store 0xBB=187 -> output_buffer[1] = s[1])
0x18c  st   m0,  [p0], #8 ; movx r1, #0x1        (strictly-later: store 0xAA=170 -> output_buffer[0] = s[0])
 ...   later bundles: store 1 -> s[3]; store 0xCC=204 -> s[2]
```

`TRAP_PC = 0x184` (the slot-1 `0xBB` store). Arming (in `@seq`, before the
gate-feed and OP_READ):
- `npu.write32` `0x32020 = 0x80000184` -- PC_Event0: VALID(31) | PC_ADDRESS=0x184
- `npu.write32` `0x32018 = 0x1` -- Debug_Control2[0] = PC_Event_Halt
- injected `MASKPOLL` `0x00232004`, mask=value=`0x00010000` -- block until DEBUG_HALT

Verdict map (schedule-derived; spec §4.2): trap slot = `s[1]`, strictly-later =
`s[0]`. `halted ∧ s[1] committed ∧ no strictly-later` -> AFTER_COMMIT;
`halted ∧ all slots zero` -> BEFORE_COMMIT (DEBUG_HALT=1 disambiguates from
"core never ran" -- a core that never ran is not debug-halted).

### Hardware observed (NPU1, 2026-05-18)

```
SLOTS: s0=0 s1=0 s2=0 s3=0 CORE_STATUS=0x10001 HALTED=1
TRAP_VERDICT:BEFORE_COMMIT
```

HW run: 1.9 s, bridge PASS, no wedge. `Core_Status = 0x10001` =
`DEBUG_HALT`(bit 16) | `ENABLE`(bit 0): the core ran and is debug-halted. All
four marker slots -- including the trap slot `s[1]` (the `0xBB`/187 store at
`TRAP_PC=0x184`) -- are zero. The MASKPOLL satisfied (DEBUG_HALT became set),
so the OP_READ provably read a *stable, halted* state.

### EMU observed (Phase B Unit 1b, same byte-identical injected binary)

```
SLOTS: s0=0 s1=0 s2=0 s3=0 CORE_STATUS=0x10003 HALTED=1
TRAP_VERDICT:BEFORE_COMMIT
PASS
```

EMU run: 4.1 s, bridge PASS, no hang. `Core_Status = 0x10003` =
`DEBUG_HALT`(bit 16) | `RESET`(bit 1) | `ENABLE`(bit 0). All four marker slots
zero; `TRAP_VERDICT:BEFORE_COMMIT` -- matching HW.

**RESOLVED (2026-05-19, §8 close-out):** the EMU `Core_Status = 0x10003`
RESET-bit divergence noted above is fixed. `Coordinator::enable_core`
now routes the core-debug enable through `write_control` (the same
register semantics as a CDO `Core_Control=0x1` write), clearing `reset`.
EMU now reports `Core_Status = 0x10001`, matching HW. See
`docs/superpowers/specs/2026-05-19-debug-halt-section8-closeout-design.md`.

**Phase B Unit 1b (xdna-emu `dev`, 2026-05-18) closes the earlier EMU
limitation:** the mutable `tile.read_register` path (used by the injected
MASKPOLL to poll `Core_Status` at `0x32004`) was previously falling back to the
raw register HashMap, which never reflects the dynamically-computed
`Core_Status[16]=DEBUG_HALT` from `core_debug.read_status()`. The fix dispatches
`Core_Status` and debug-register offsets (`0x32010`--`0x3202C`) into
`core_debug.read_register()`, mirroring exactly what Phase B Unit 1 (commit
`e0ec922`) did for `read_register_pure`. With the mutable path reconciled:

- The injected MASKPOLL now observes `DEBUG_HALT=1` and satisfies (no longer
  unsatisfiable on EMU).
- The OP_READ push runs, returning all marker slots zero (trap bundle did not
  commit) + `DEBUG_HALT` set.
- The schedule-derived verdict is `BEFORE_COMMIT` -- reproducing HW exactly.

The earlier `MASKPOLL_UNSATISFIED_EMU` EMU baseline (observed with Phase A /
the pre-Unit-1b emulator) is retired. It was caused solely by the divergent
read paths, not by any write-side gap or structural EMU limitation. The
write-side gap ("arming writes dropped into `write_core_register` catch-all")
was separately established as false in spec §4.2 "Mechanism correction": both
the control-packet path and `@seq npu.write32` arming writes always reached
`core_debug` via `apply_tile_local_effects`. The breakpoint was armed on EMU all
along.

The emulator graceful-poll-termination contract is retained as independent
hardening: an unsatisfiable MASKPOLL with a quiescent engine still terminates
deterministically with `MaskPollUnsatisfied` reason, no register fakery, no
pretend-halt. Its three unit tests remain green. This contract is not this
probe's path anymore; it guards any genuinely-unsatisfiable poll in other
contexts.

**The probe is now a self-checking EMU+HW regression of the G1 before-commit
fidelity fix.** An EMU run now validates: gate fed, MASKPOLL satisfies, OP_READ
issues, all slots zero + DEBUG_HALT => `TRAP_VERDICT:BEFORE_COMMIT`. A
regression in the Unit-1 seam or the Unit-1b read reconciliation fails the EMU
bridge run, not only HW. HW remains ground truth; EMU is now a faithful
reproduction of the silicon result.

### Conclusion

**On NPU1 (Phoenix/AIE2) silicon, a synchronous PC-event breakpoint halts the
core BEFORE the trap bundle commits.** Evidence: `DEBUG_HALT=1` proves the core
executed and is debug-halted; every marker slot is zero, including the trap
slot `s[1]` whose store *is* the trap bundle at `0x184` -- so that store had
not committed when the core halted. `DEBUG_HALT=1` rules out the "core never
ran" reading (a non-running core is not debug-halted). Incidental
corroboration: `ENABLE`(bit 0)=1 alongside `DEBUG_HALT`, consistent with
ENABLE-stays-1-while-halted -- though the verdict deliberately relies on
`DEBUG_HALT` alone (spec §4.2; ENABLE-stays-1 was an unverified assumption,
here merely observed, not depended on).

**Phase B implication (real, scoped fidelity fix -- spec §5.1 -- SHIPPED).**
Phase B Unit 1 (commit `e0ec922`) implemented the pre-execute seam in
`coordinator.rs`: for PC_Event/breakpoint-origin halts, the coordinator checks
`has_sync_pc_trap_at(pc)` *before* executing the bundle; on a match, the bundle
does not execute (`BEFORE_COMMIT`), `consume_sync_pc_trap()` latches
`halt_cause_pc_event` and requests halt. Phase B Unit 1b (this commit) closes
the read-path reconciliation that makes the probe self-checking on EMU.

The probe stays checked in as a permanent re-runnable regression: on both EMU
and HW it produces `TRAP_VERDICT:BEFORE_COMMIT` and bridge `PASS`. A
defensive `MASKPOLL_UNSATISFIED_EMU` branch in `test.cpp` remains to catch
regressions (returns exit code 1 rather than 0 -- a bridge failure, not a
pass). `test.cpp` commit `8546397987` updated for Unit 1b on
`xdna-emu-cycle-budget`.

---

## G2 -- Single_Step_Count (Debug_Control0[5:2])

**Verdict: `Debug_Control0[5:2]` `Single_Step_Count` is LIVE silicon on
NPU1 (Phoenix/AIE2), NOT dead/reserved state.** Writing a non-zero
count (halt bit `[0]` clear) halts the core via `DEBUG_HALT` after
single-stepping that many instructions. This disproves the pre-probe
audit hypothesis (spec §1/§2.A/§7: "dead state ... may be inert on
silicon (effectively reserved)").

### Probe configuration (final, Exp2)

8 distinct sequential marker stores (`output_buffer[k]=101+k`, k=0..7)
behind the Exp1 blocking objectfifo `@gate` (arming-race immunity:
`@seq` writes `Debug_Control0` *then* feeds `@gate`, so the count-step
arm provably lands before the core proceeds); ctrl-in OP_READ on shim
MM2S ch1 (collision fix, inherited). `@seq` arms a single
`aiex.npu.write32 Debug_Control0 (0x32010) = value`, feeds `@gate`,
then a **double** ctrl-in OP_READ readback (pass A then verbatim pass
B, 8 marker slots + `Core_Status` each). Verdict (`test.cpp`):
`SETTLED = (A == B)` (both snapshots identical -> provably stable
terminal state); `LANDED` = committed marker count. **No injected
poll** -- `DEBUG_HALT_PROBE_WITNESS=none` skips MASKPOLL injection.
Minimal set: `0x00` (count-step off, control) vs `0x10`
(`Single_Step_Count`=4, halt bit clear).

Raw logs: `/tmp/claude-1000/probe-exp2-hw-0x00.log`,
`probe-exp2-hw-0x10.log` (ephemeral); durable copy
`build/bridge-test-results/.../debug_halt_probe.chess.{hw,bridge}.log`.
Probe source: `mlir-aie/test/npu-xrt/debug_halt_probe/` branch
`xdna-emu-cycle-budget`, Exp2 no-poll double-read at `ce86439a70`;
bridge no-inject wiring xdna-emu `dev` `b701e61`.

### The Task-7 detour (durable -- a no-timeout-poll hazard + a real driver bug)

The *first* Task-7 HW attempt used the earlier conditional-`CORE_DONE`-
MASKPOLL design (a no-timeout firmware `XAIE_IO_MASKPOLL`). On point
`0x10` the poll's predicate never became true, the poll blocked the
mailbox forever, and `modprobe -r` recovery hit a **kernel NULL-deref /
use-after-free in `amdxdna`**: timed-out mailbox messages were left in
`chan_xa` and flushed at channel teardown via
`notify_cb(handle, NULL, 0)` with the callback dereferencing a
NULL/dangling handle (`xdna_msg_cb`/`aie4_xdna_msg_cb` ->
`complete(&cb_arg->comp)`, `do_raw_spin_lock` oops; SMU-wedged the NPU,
reboot-only).

Fixed and verified this session (xdna-driver branch
`fix/mailbox-teardown-stale-msg`): `1114562` (PR-candidate) adds
`xdna_mailbox_cancel_msg()` -- reclaim the timed-out message on
`-ETIME` via the existing xa_erase ownership model -- plus NULL-safe
teardown callbacks; `7d85f25` adopts the pre-existing local
keep-channel-alive-on-`-ETIME` change (LOCAL-ONLY, do not upstream).
Reproduced the exact pre-fix crash, then with the fix: `modprobe -r`
on a firmware-wedged device unloads cleanly, zero oops. The probe was
then redesigned to the **no-poll double-read** above (an unbounded
firmware poll whose predicate cannot be known in advance is an
unacceptable probe mechanism, and `CORE_DONE`-via-OP_READ was an
unproven happens-after). The no-poll probe **cannot** wedge the
device; both Task-7 HW points below ran clean (no TDR, no wedge).

### Hardware observed (NPU1, 2026-05-19, fixed driver, no-poll probe)

```
Debug_Control0 = 0x00  (Single_Step_Count = 0; count-step disabled):
  SLOTS: 101 102 103 104 105 106 107 108
  CORE_STATUS=0x100000 HALTED=0 DONE=1 SETTLED=1   LANDED:8
  bridge PASS, CLEAN, no wedge

Debug_Control0 = 0x10  (Single_Step_Count = 4, halt bit [0] = 0):
  SLOTS: 0 0 0 0 0 0 0 0
  CORE_STATUS=0x10001 HALTED=1 DONE=0 SETTLED=1    LANDED:0
  bridge PASS, CLEAN, no wedge
```

`0x00`: `Core_Status=0x100000` = `CORE_DONE`(bit 20); core ran all 8
stores to completion. `0x10`: `Core_Status=0x10001` =
`DEBUG_HALT`(bit 16) | `ENABLE`(bit 0); core halted with **zero**
markers committed. `SETTLED=1` on both (pass A == pass B) -- stable
terminal states, not read-too-early artifacts. The two runs are the
*same binary* save the one `Debug_Control0` value; `@gate` guarantees
the write landed before the core ran.

### EMU observed (same byte-identical binaries)

```
0x00:  SLOTS: 101..108  CORE_STATUS=0x100000  HALTED=0 DONE=1 SETTLED=1  LANDED:8
0x10:  SLOTS: 101..108  CORE_STATUS=0x100003  HALTED=0 DONE=1 SETTLED=1  LANDED:8
```

EMU is **count-step inert**: `0x10` behaves identically to `0x00`
(`LANDED:8`, not halted). This is the expected, spec-predicted baseline
-- *not* a probe defect and *not* a write-side routing gap: the
`Debug_Control0` write reaches `core_debug` via
`apply_tile_local_effects` -> `write_debug_control0` (mod.rs:787) and
is stored, but **nothing reads/acts on the stored count** -- the
count-step state machine is the unimplemented Phase B §5.2 work. (The
`0x100003` vs HW `0x100000` is the tracked, benign §8 `RESET`-bit
EMU/HW divergence; the verdict keys on bits 16/20, which agree.)

### Conclusion

**On NPU1 (Phoenix/AIE2) silicon, `Debug_Control0[5:2]`
`Single_Step_Count` is a functional single-step instruction-count
register.** With the halt bit `[0]` clear, `count=4` single-steps the
core and halts it (via `DEBUG_HALT`, `Core_Status[16]`) within the
first 4 instructions -- before the first marker store commits
(`LANDED:0`); `count=0` leaves the core running free (`LANDED:8`,
`CORE_DONE`). The halt is reached *without* setting the documented halt
bit `[0]`, so bits `[5:2]` drive the halt on their own. Count-step is
**not** the binary-unreachable dead state the audit assumed; it is live
hardware the emulator must model.

### Phase B implication (§5.2 -- now HW-anchored)

Phase B §5.2 (count-step state machine) was specified as
"behavior follows Experiment 2 findings." Anchored behavior: a non-zero
`Debug_Control0[5:2]=N` arms an N-instruction single-step budget;
on expiry the core halts with `Core_Status` `DEBUG_HALT` set (and
`ENABLE` stays 1, `CORE_DONE` 0). `N=0` = disabled. The emulator
currently stores the field (mod.rs:787) but no consumer decrements/
expires it -- §5.2 implements that consumer (one call adjacent to
`consume_pending_single_step`, per spec §5.2).

### Scope / §8 forward-commitment (honest bounds)

Derived: count-step is **active**, halts via `DEBUG_HALT`, independent
of the halt bit, `N=4` halts in the prologue. **Not** derived (single
`N=4` datapoint; minimal set by design after the no-timeout-poll
hazard): the exact decrement cadence / which instruction boundary `N`
counts to, expiry-vs-re-arm on resume, the `count + halt-bit` (`0x11`)
interaction, and larger-`N` behavior. Per spec §8 these remain a
**tracked count-step silicon-fidelity forward-commitment**: Phase B
§5.2 ships the most natural reading of the above (documented inline as
explicit modeling decisions citing this finding), and finer
characterization is revisited when a dedicated hardware-observation
budget / better register-poke tooling is available. The probe stays
checked in (Exp2 form re-runnable via `DEBUG_HALT_PROBE_WITNESS=none`;
Task 8 restores the Exp1 G1 regression as the default).

---

## Phase B inputs (closing -- no placeholders; Phase A complete)

Phase A is closed. Phase B is written against these derived answers, not
guesses:

- **(a) G1 -- synchronous-trap halt boundary: DERIVED and SHIPPED.**
  Silicon halts a PC-event/breakpoint *before* the trap bundle commits
  (`TRAP_VERDICT:BEFORE_COMMIT`, HW + EMU). Phase B Unit 1 (pre-execute
  PC_Event seam, coordinator) + Unit 1b (mutable `read_register`
  reconciliation) implemented it; the probe is a self-checking EMU+HW
  regression (restored Exp1 default reproduces
  `SLOTS: s0=0..s3=0 CORE_STATUS=0x10003 HALTED=1 BEFORE_COMMIT`, bridge
  PASS). Not an open Phase B item -- done; listed for the audit trail.

- **(b) G2 -- count-step semantics: DERIVED.** `Debug_Control0[5:2]`
  `Single_Step_Count` is **live** NPU1 silicon: `count=N` (halt bit
  `[0]` clear) single-steps `N` instructions then halts the core via
  `Core_Status` `DEBUG_HALT`; `count=0` disables (core runs free).
  Independent of the halt bit. Phase B §5.2 implements the count-step
  state machine to this: `write_debug_control0` arms an N-instruction
  budget; a coordinator consumer (adjacent to
  `consume_pending_single_step`) decrements per committed bundle and
  requests halt at expiry, setting `DEBUG_HALT` (ENABLE stays 1,
  CORE_DONE 0). The EMU is currently inert (field stored mod.rs:787,
  no consumer) -- that *is* the §5.2 gap, now HW-anchored.
  **Documented modeling decisions** (only `N=4` observed; the minimal
  set was deliberate after the no-timeout-poll hazard -- implement to
  the most natural reading, comment inline citing this finding):
  the instruction boundary `N` counts to (we observed `N=4` halts in
  the prologue, before the first store); decrement cadence;
  expiry-vs-re-arm on resume; `count + halt-bit` (`0x11`) interaction;
  larger-`N`. These are the **§8 count-step forward-commitment** --
  ship the natural reading now, revisit silicon-faithful
  characterization when a dedicated hardware-observation budget exists.

- **(c) No write-side routing gap (corrected).** The earlier "EMU drops
  control-packet debug-reg writes via a `write_core_register`
  catch-all" framing is **false** (spec §4.2 "Mechanism correction").
  Control-packet *and* `@seq npu.write32` debug-reg writes always
  reached `core_debug` via `apply_tile_local_effects`. Units 1/1b
  closed the only real defect -- the divergent *read* paths
  (`read_register_pure` and the mutable `tile.read_register`). The
  remaining Phase B work is therefore **not** a routing fix: it is
  (i) §5.2 the count-step state machine (the consumer that (b)
  anchors), and (ii) §5.1 the single-step halt boundary -- deferred as
  G2-coupled, now G2-unblocked (single-step is the count=1 special
  case of the same mechanism).

- **(d) §8 forward-commitments (tracked, surfaced in the `debug_halt`
  coverage narrative).** Count-step finer-characterization (above) --
  triggered, recorded. Resume HW-verification -- still open (a runtime
  sequence cannot deassert mid-run). `OUTBUF_ADDR` robustness -- still a
  derived magic constant; make non-fragile later. `Core_Status`
  `RESET`-bit EMU/HW divergence (`0x10003`/`0x100003` vs HW
  `0x10001`/`0x100000`) -- benign (verdict keys on bits 16/20),
  reconcile `enable_core()` vs `write_control()` later.

- **(e) Incidental, high-leverage: a real `amdxdna` kernel bug, fixed.**
  The retired no-timeout `CORE_DONE` MASKPOLL (Exp2 first attempt)
  SMU-wedged the NPU and exposed a **NULL-deref / use-after-free in
  `amdxdna` mailbox channel teardown**: timed-out messages were left in
  `chan_xa` and flushed at `xdna_mailbox_release_channel` via
  `notify_cb(handle, NULL, 0)` with the callback dereferencing the
  NULL/dangling handle -- a `do_raw_spin_lock` oops on `modprobe -r` of
  a firmware-wedged device (the recovery path we lean on constantly).
  Root-caused from source, fixed, and **verified against the exact
  reproduction** (clean unload, zero oops). xdna-driver branch
  `fix/mailbox-teardown-stale-msg`: `1114562` (PR-candidate --
  `xdna_mailbox_cancel_msg()` reclaim-on-`-ETIME` + NULL-safe
  callbacks) and `7d85f25` (LOCAL-ONLY keep-channel-alive adopt, do not
  upstream). This is the single highest-leverage outcome of Phase A:
  it removes a recurring reboot-class landmine. Upstream PR is a
  separate, Maya-gated step (reformat to AMD conventions; only
  `1114562` is PR-eligible).

**Phase B remainder -- IMPLEMENTED (2026-05-19).** Unit 2 shipped the
§5.2 count-step state machine (`Debug_Control0[5:2]` arms a live
N-committed-bundle budget; `tick_count_step` decrements per committed
bundle; expiry latches `halted` so the `is_halted` gate blocks bundle
N+1 -- before-commit, G2-anchored; review fix added
`halt_cause_count_step` to `clear_halt_causes`). Unit 3 shipped the
§5.1 single-step halt boundary as the principled split (PC-wired
SSTEP_EVENT -> before-commit via the Unit-1 seam
`has_sync_sstep_pc_trap_at`/`consume_sync_sstep_pc_trap`;
watchpoint/mem/lock/range -> documented after-commit). `debug_halt`
coverage is now `Modeled { completeness: Full }`. The §8 count-step
finer-characterization forward-commitment remains OPEN (decrement
cadence / larger-N / `0x11`-on-silicon -- only `N=4` observed); the
natural reading is shipped and documented inline citing this finding.
