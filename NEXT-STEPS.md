# NEXT-STEPS

Session handoff. Read this first in a new conversation. Updated 2026-05-23 (late evening 2026-05-22 work).

The repo (commits, finding docs, component docs, ROADMAP, MEMORY) is
canonical for *anything that's done*. This file is for *anything that's
not done yet* — active threads, parked items, the next concrete action
on each one. If something here gets done, delete or update it.

---

## Active threads

### A. Emu `dma_wait` / empty-ctrlpkt gap  (RESOLVED 2026-05-22 evening)

**Status.** Closed — the premise didn't reproduce on today's code.
The original NEXT-STEPS entry (written earlier today) claimed the
emu reported `emu_ok=True` for `add_one_ctrl_packet` with empty
`bo_ctrlIn`.  Empirical re-check found both runtime paths refuse to
declare success on this scenario:

- **`xclbin_suite` path:** engine reaches `EngineStatus::Stalled`
  at ~366k cycles, suite returns `Fail{Stall detected at cycle N}`.
- **`bridge-trace-runner` path:** TDR classifier fires
  `Wedged{Stalled}` at ~363k cycles; plugin translates
  `WedgeRecovered -> ERT_CMD_STATE_ABORT`; runner exits 1 with
  `error: kernel did not complete (state=6)`.

Diagnosis surfaces originally suspected (the fast-completion path
in `executor.rs:332-339`, the channel FSM in `channel.rs:80`) are
behaving as designed for this scenario — the executor stays
`BlockedOnSync` on the unsatisfied `dma_wait @out0`, `is_done`
remains false, so the TDR `NaturalCompletion` path is unreachable.

**Regression gate landed.**  See
`src/testing/xclbin_suite.rs::add_one_ctrl_packet_empty_ctrlpkt_must_not_report_success`.
It runs the chess xclbin with the suite's default (zero-filled)
input and asserts the outcome is NOT a pass-shaped variant; passes
today.

**Empirical evidence preserved at**
`build/experiments/2026-05-22-emu-empty-ctrlpkt/` (runner stderr +
JSON status line).

**Why this entry is preserved instead of deleted:**  The mismatch
between the earlier observation and today's behavior is unexplained
("something weird happened" -- 2026-05-22 evening session).  If a
similar symptom resurfaces, this entry plus the experiment dir
gives the next session enough context to triage quickly without
re-walking the same diagnosis surface.

Pointers: `2026-05-22-chain-exec-npu-silent-drop-captured.md`
(finding doc — still authoritative for the HW/firmware side).

---

### B. Upstream xdna-driver issue post  (RE done, drafting deferred — verify-first)

**Status.** The full firmware-side RE is complete and committed in
`2026-05-22-chain-exec-npu-silent-drop-captured.md`.  We have:

- Working standalone repro (Bug A: silent-drop, recoverable)
- Working bridge-test repro (Bug B: column-leak cascade,
  catastrophic)
- Decoded cascade with named status codes (MGMT_ERT_BUSY / NOAVAIL)
- Caller-side proof that the op-0x18 firmware path has no timeout
- Variant-immunity observation (low-level rt-seq required)

**Not done.** The actual github issue text, addressed to AMD's
`xdna-driver` repo.

**Caveat (added 2026-05-22 evening): VERIFIED 2026-05-22 evening.**
Thread A's "emu reports emu_ok=True" claim turned out to be stale on
re-check the same day, so the reproducers were re-verified before
investing in a draft.  Both Bug A and Bug B reproduce on today's
code with signatures matching the finding doc exactly:

| Bug | Recipe | Delta this run | Finding doc |
|---|---|---|---|
| A (silent-drop) | `trace-sweep.py --test add_one_ctrl_packet --core-sweep all --no-emu` | TDR +1, BUSY +1, NOAVAIL +0 | TDR≥1, BUSY≥1, NOAVAIL=0 ✓ |
| B (cascade) | `./scripts/emu-bridge-test.sh --no-emu --sweep ctrl_packet` | NOAVAIL +104, ret-22 +417 | NOAVAIL 100+, ret-22 400+ ✓ |

NPU recovered cleanly via `pkexec sh -c 'modprobe -r amdxdna &&
modprobe amdxdna'` (no synchronize_srcu hang); `xrt-smi validate`
PASSED both tests post-recovery with no new NOAVAIL.  Evidence at
`build/experiments/2026-05-22-evening-thread-b-verify/{bug-a,bug-b}/`
(stdout/stderr + counts.before/counts.after).

**Validity check (added 2026-05-22 evening).** Before drafting we
swept driver/XRT/upstream sources for correctness of our claims.
Three significant corrections to the framing recorded above:

1. **"Driver relies on TDR as sole backstop" is WRONG.**  The
   driver has a 5s `RX_TIMEOUT` (`wait_for_completion_timeout` in
   `amdxdna_mailbox_helper.h:9-10`) on every mailbox message
   including op-0x18.  TDR is a *separate* device-level watchdog
   (`tdr_timeout_ms = 2000` default × 2 stalls ≈ 4s effective);
   it fires first in practice.  Empirically (the reproducer below)
   each silent-drop times out at ~4080ms, well before the 5s
   RX_TIMEOUT — TDR is the actual mechanism that fires here.
   Both exist; neither is "sole."
2. **"FW has no timeout" is technically true but misleading.**  The
   firmware's internal APP-ERT event-wait has no timeout, so the
   wedged task never self-recovers.  But the driver doesn't depend
   on FW recovery — its own RX_TIMEOUT and TDR catch the drop and
   abort the command.  Userspace observes
   `ERT_CMD_STATE_TIMEOUT` (state=8) at ~4s.  Conflating "FW has
   no internal timeout" with "no recovery mechanism exists" was
   the original framing error.
3. **Three already-merged upstream commits would be reviewer
   reach-fors and need to be pre-emptively ruled out**:
   `3d32eb7a` (cu_idx memset fix, 2025-12-09),
   `cd77d5a4` (mailbox tail-pointer polling, 2025-12-04),
   `343f5683` (send-ring race, 2025-12-11).  All verified present
   in our installed amdxdna 2.23.0.  Also relevant: issue #906
   (NPU5 deterministic timeout — different platform, different
   reproducibility profile, not us) and our own PR #1347
   (mailbox UAF on cleanup of timed-out messages — downstream
   consequence of this bug, not duplicate).

**Verified deterministic reproducer** at
`tools/repros/op0x18_silent_drop/op0x18_repro.cpp`.  Self-contained
C++ using stock XRT API only (no xdna-emu deps).  Loads
`add_one_ctrl_packet`'s chess artifacts, leaves `bo_ctrlIn`
zero-filled, runs N=10 iterations with fresh `xrt::hw_context` per
iter.  Verified output (`tools/repros/op0x18_silent_drop/run.log`):
**10/10 non-COMPLETED, state=8 (TIMEOUT), ~4080ms per drop, dmesg
TDR=+10/NOAVAIL=+0/ret-22=+10**.  Deterministic when `bo_ctrlIn`
is empty — this is a stronger repro than the probabilistic ~6%
historical observation.

**Newer firmware exists in staging, not released.**
`npu.sbin.1.5.6.399` in `kernel-firmware/drm-firmware:amdnpu/1502_00/`
(committed 2025-05-19, ~1 year ago).  xdna-driver's `tools/info.json`
still pins NPU1 to 1.5.5.391.  Worth asking maintainers whether
1.5.6.399 fixes this before posting (or in the post itself).

**Approved post design (revised 2026-05-22 evening).**

- Title: *Phoenix FW 1.5.5.391: `MSG_OP_CHAIN_EXEC_NPU` silent-drops
  at ~6%; driver correctly aborts — is this drop-rate known?*
- Opening: inquiry-shaped ("Is this drop rate known? Staging FW
  status? Mitigation guidance?"). Acknowledges driver handles each
  drop correctly. *Removes* the false "no timeout / sole backstop"
  claim.
- Behavior summary: TX → no RX, no IRQ → driver RX_TIMEOUT and/or
  TDR returns -ETIME → ERT_CMD_STATE_TIMEOUT to userspace.
- *What we ruled out*: terse list of `3d32eb7a` / `cd77d5a4` /
  `343f5683` (verified present), issue #906 (NPU5, different),
  PR #1347 (our own downstream filing).
- Evidence collapsibles: (1) driver verbose mailbox log,
  (2) kernel tracepoint quartet, (3) frequency + reproducer
  output, (4) variant immunity (one structural sentence,
  no MLIR lowering claims).
- *No "recovery is clean" claim* — that was about Bug B's cascade
  and would conflate severity bounds.  One-line placeholder:
  "cascade behavior under sustained load is a separate follow-up."
- Inline reproducer: ~80 LoC C++ from `tools/repros/op0x18_silent_drop/`
  with verbatim run.log output.
- Environment: HW Phoenix NPU1, FW 1.5.5.391 (1.5.6.399 in staging
  noted), amdxdna 2.23.0 (`drivers/accel`), `add_one_ctrl_packet`
  from mlir-aie chess-compiled.
- Open questions: drop rate known? FW 1.5.6.399 expected to fix?
  Application-layer retry guidance for `-ETIME`/`ABORT` on op-0x18?

**Next concrete steps (start a new session here for the prose
draft):**

1. Draft the issue post per the approved design above.  Prose
   deserves a fresh context window; commit before posting.
2. Show Maya the full text before posting (CLAUDE.md global rule).
3. Post to `github.com/amd/xdna-driver` issues.  No firmware-vs-
   driver tag exists; file as a regular issue with
   `MSG_OP_CHAIN_EXEC_NPU` and `Phoenix NPU1 FW 1.5.5.391` in the
   title for self-routing.

---

### C. xdna-driver column-leak follow-on patch  (parked — "LATER")

Maya flagged during the wedge investigation: the column-leak cascade
(Bug B) might be addressable as a separate xdna-driver patch — the
driver could detect post-TDR `MGMT_ERT_BUSY` on DESTROY and take
additional reclaim action before re-attempting CREATE.

Pre-condition: B (issue post) lands first so the maintainers have
context.

---

### D. PR #1348 (mailbox UAF fix) status  (waiting on maintainer)

Posted to xdna-driver, no action required from us until maintainer
reviews.  Not blocking anything else here — the loaded driver is
already source-equivalent to the fix and the wedge investigation
confirmed PR #1348 is not implicated in the silent-drop.

---

## Parked / deferred (don't lose these)

These are real follow-ups but explicitly *not* the next thing.  Each
deserves its own pickup when its time comes.

- **`Get bo 4 failed` every-iteration debug-BO bug.**  Documented as
  "Side observation" in the silent-drop finding doc (line 287).
  Separate `drivers/accel` defect, fires on every sweep iteration
  (healthy too).  Should eventually become its own finding +
  upstream report.
- **Bridge-test A→B escalator mechanism.**  Standalone reproduces
  Bug A but never Bug B in ~30 min; bridge-test reproduces Bug B
  within one run.  Daemon was ruled out (`--no-amdxdna-trace` still
  cascades).  Candidates: Phase 3+4 healthy-execs warmup,
  back-to-back chess+peano on same kernel, accumulated NPU activity
  volume.  Not characterised.  Not blocking the upstream report
  since both bugs are independently reproducible.
- **APP-ERT in-loop syscall identification.**  Finding doc line 310
  notes the unknown park-point inside the wedged APP-ERT task.
  Requires `bridge-trace-runner --snapshot-on-timeout` path to
  capture register state at wedge time — code exists, hasn't been
  exercised against the fresh repro.

---

## Gap queue (emu coverage)

Per `docs/coverage/aie2/implementation-gaps.md` (generated, do not
hand-edit — `cargo run -p xdna-archspec --example
gen_coverage_artifacts`):

- `noc`: STUB  (forward-linked from control_packets SLVERR work —
  see `docs/superpowers/plans/2026-05-18-control-packets-slverr.md`
  Step 7 for the linkage)
- `clock_control`: STUB

Both are tracked coverage gaps; the durable rhythm
(`project_coverage_plan2_done_regroup_before_plan3` in MEMORY) is
*regroup before picking the next plan*.  Either could be the next
plan after thread A above stabilises.

---

## Long-running tasks (TaskList)

- `#201 in_progress`: RE Phoenix LX7 mgmt firmware — mailbox dispatch
  + SUSPEND/waiti quiesce path.  Substantial Ghidra work that
  predates today; touched today only via the op-0x18 handler
  decompilation cited in the silent-drop finding.  Not closed.

---

## Recent sessions (newest first)

### 2026-05-22 evening — Thread A closed, Thread B verified

Picked up Thread A (the emu `dma_wait` / empty-ctrlpkt gap).  Wrote
the failing regression test in `xclbin_suite` as NEXT-STEPS
prescribed; expected it to fail today by reporting `Pass`.  It did
fail — but with `Fail{Stall detected at cycle 366398}`, not `Pass`.
That contradicted Thread A's premise.  Cross-checked the FFI /
bridge-runner path empirically: ran `bridge-trace-runner` directly
against `add_one_ctrl_packet` chess xclbin with no `--ctrlpkt`; got
`halt_reason=wedge_recovered, state=6 (ABORT)` at ~363k cycles.  No
silent success in either path.  NEXT-STEPS Thread A's "emu reports
emu_ok=True" observation does not reproduce on today's code.  Cause
of the mismatch unexplained ("something weird happened").  Relaxed
the test assertion to `!is_pass()`, committed it as a regression
gate (`06a24ab`), marked Thread A RESOLVED.  Added a verify-first
caveat to Thread B since Thread A's premise turning out stale on the
same day suggested Bug A/B reproducers also deserved a re-check
before drafting an upstream report.

Then verified Thread B per that caveat.  Both reproducers fire with
signatures matching the finding doc exactly: Bug A standalone
(`trace-sweep.py`) produced TDR +1 / BUSY +1 / NOAVAIL +0 in ~5s;
Bug B bridge-test (`emu-bridge-test.sh --no-emu --sweep ctrl_packet`)
produced NOAVAIL +104 / ret-22 +417 over ~8m.  NPU recovered cleanly
post-Bug-B via `pkexec sh -c 'modprobe -r amdxdna && modprobe
amdxdna'`; `xrt-smi validate` PASSED both tests with no new NOAVAIL.
Evidence at `build/experiments/2026-05-22-evening-thread-b-verify/`.

Pivoted into Thread B drafting prep.  Maya flagged a credibility
concern: the cascade reproducing cleanly via modprobe (vs. historical
"requires reboot" framing) suggested our claims about Bug A's
mechanism might also be stale — extremely dangerous in an externally
visible upstream report.  Dispatched three parallel agents to
cross-check validity: (1) driver source for existing op-0x18 timeout
handling, (2) upstream github + linux history for related issues/PRs,
(3) XRT/SHIM timeout layer.  Result: **three significant corrections
to our framing**.  Driver does have per-message timeouts (5s RX_TIMEOUT
+ 2s × 2 TDR); "FW has no timeout, driver relies on TDR as sole
backstop" was wrong on both halves.  Three already-merged upstream
commits (`3d32eb7a` / `cd77d5a4` / `343f5683`) need pre-emptive
ruling-out.  FW 1.5.6.399 exists in staging branch, not released.

With corrections in hand, redesigned the post (inquiry-shape opening,
"what we ruled out" section, dropped the recovery claim that was
really about Bug B's cascade).  Built a deterministic reproducer at
`tools/repros/op0x18_silent_drop/op0x18_repro.cpp` — 80 LoC C++ using
only stock XRT API.  Verified: 10/10 non-COMPLETED, state=8 (TIMEOUT),
~4080ms each (the ~4s TDR, fires before the 5s RX_TIMEOUT), dmesg
TDR=+10/NOAVAIL=+0/ret-22=+10.  Empty `bo_ctrlIn` is a *deterministic*
trigger for the same firmware path that probabilistically affects
normal workloads at ~6%; stronger evidence than the historical
statistical observation.

Stopped before drafting the post prose — fresh context window
deserves the externally-visible writing.  See Thread B in this file
for the approved design + corrected framing + reproducer.

### 2026-05-22 daytime — silent-drop captured, Bugs A and B distinguished

Built a stubbed copy of `bridge-trace-runner` against stock XRT
(`build/experiments/2026-05-22-stock-repro/runner-min/`), hunted the
silent-drop repro through a long detour where I misread `hw_ok=False`
on every batch as "boring abort" instead of "wedge with TDR
recovering" (memory note added).  Eventually distinguished Bug A
(silent-drop, recoverable, fires per batch in standalone) from Bug B
(column-leak cascade, catastrophic, only reproduces via bridge-test
path).  Daemon ruled out as cascade ingredient.  Found the structural
diff between plain `add_one_ctrl_packet` and its variants — plain
uses 14 low-level register-poke ops in its rt-seq, variants use 0 —
which sharpens the upstream report.  Confirmed PR #1348-equivalent
driver is loaded, so the wedge is not the UAF fix masking anything.
Pivoted to emu: claimed empty-ctrlpkt against EMU reported success,
mapped the diagnosis surface for the gap (executor.rs sync logic,
channel FSM), did not write the failing test or fix.  Last commit
`e4b8290`.  *(Evening session subsequently could not reproduce the
"reports success" observation.)*
