# NEXT-STEPS

Session handoff. Read this first in a new conversation. Updated 2026-05-22 evening.

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

**Caveat (added 2026-05-22 evening).** Thread A's "emu reports
emu_ok=True" claim turned out to be stale on re-check the same day.
Before investing time drafting a public upstream report, **verify
Bugs A and B still reproduce today** with the exact recipes in the
finding doc.  If either doesn't reproduce, narrow the gap (different
machine state? different driver/firmware version? something
session-specific?) before posting -- a withdrawn or weakened
upstream report is worse than no report.

**Next concrete steps:**

1. **Verify the reproducers** before drafting.  Run each recipe
   from the finding doc once and confirm the dmesg signatures
   match (Bug A: TDR>0, BUSY≥1, NOAVAIL=0; Bug B: NOAVAIL flood).
2. Draft the issue post.  Frame as a *driver-interface gap*: the
   driver relies on TDR as the sole backstop for op-0x18 silent
   drops because the firmware has no timeout; the cascade-to-NOAVAIL
   means a single dropped exec can starve the column pool.  Reference
   the finding doc, the variant-immunity structural finding (low-
   level rt-seq triggers it), and link the standalone repro recipe.
3. Show Maya the text before posting (CLAUDE.md global rule for
   externally-visible writing).
4. Post.

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

### 2026-05-22 evening — Thread A reality-check, closed

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
gate, updated Thread A status to RESOLVED.  Added a verify-first
caveat to Thread B since Thread A's premise turning out stale on the
same day suggests Bug A/B reproducers also deserve a re-check before
drafting an upstream report.

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
