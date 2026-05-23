# NEXT-STEPS

Session handoff. Read this first in a new conversation. Updated 2026-05-22.

The repo (commits, finding docs, component docs, ROADMAP, MEMORY) is
canonical for *anything that's done*. This file is for *anything that's
not done yet* — active threads, parked items, the next concrete action
on each one. If something here gets done, delete or update it.

---

## Active threads

### A. Emu `dma_wait` / empty-ctrlpkt gap  (NEW 2026-05-22)

**The gap.** Running plain `add_one_ctrl_packet` against the emu with
`bo_ctrlIn` empty (no `--ctrlpkt`) reports `emu_ok=True`. On real
hardware the same setup hard-aborts (state=8) and probabilistically
triggers the firmware silent-drop. The emu should at minimum *not*
report success — the core blocks on a lock acquire that never gets
released, so the output S2MM channel never receives data, so
`aiex.npu.dma_wait` should never satisfy.

**Diagnosis surface (mapped today, not yet drilled in):**

```
NPU rt-seq:   aiex.npu.dma_wait                  @ MLIR
              |  aiecc compile
NPU instr:    Tct (opcode 128)                   @ wire format
              |  src/npu/parser.rs:280
              NpuInstruction::Sync
              |  src/npu/executor.rs:306
              is_sync_satisfied()
              |  checks
              ChannelFsm::is_active()             @ src/device/dma/channel.rs:41
```

The model is structurally right — `channel.rs:80` even says "S2MM
stalls transparently (stays in [Transferring], no advancement)".  So
the "where it goes wrong" is one of three candidate surfaces, listed in
descending likelihood:

1. **Plain test's low-level rt-seq isn't routed to the channel.**
   Plain uses `aiex.npu.blockwrite(0x1d000) + aiex.npu.write32(0x1d214)`
   (manual MM2S task-queue push) instead of `aiex.npu.dma_memcpy_nd`.
   The variants (`_4_cores`, `_col_overlay`) use *only* the high-level
   `dma_memcpy_nd` form and they pass cleanly both on EMU and HW.
   Bug-shape: the raw `write32(0x1d214)` likely hits a generic
   write-handler instead of being recognised as a BD-task-queue push,
   so no task ever lands on the shim S2MM/MM2S channel, channel stays
   `Idle`.  Then the `is_sync_satisfied` *fast-completion path*
   (`executor.rs:332-339`) trivially satisfies (`transfers_completed`
   probably nonzero from a prior batch + queue empty).
2. **`transfers_completed` not zeroed across batches.**  Sub-bug of
   #1, but also could bite independently — verify the channel stats
   reset on RESET / new hwctx.
3. **`ChannelFsm::Transferring` advances without core-side data.**
   The S2MM transfer would need an actual data-source gate, not just a
   length-driven byte counter.  Less likely given the explicit
   "transparent stall" comment, but worth confirming.

**Next concrete steps (start a new session here):**

1. Write a failing test under `src/testing/` (xclbin-suite runner is
   the right harness — see `.claude/components/testing.md`): load
   `add_one_ctrl_packet` chess xclbin + insts.bin, run it with empty
   `bo_ctrlIn`, assert the runtime sequence does **not** report
   success.  Should fail today; that pins the gap.
2. Trace one batch of plain through `npu/executor.rs` and
   `device/dma/` with `RUST_LOG=debug` (or the `XDNA_EMU_WATCH`
   memory-watch from `CLAUDE.md`) to identify which of the three
   surfaces above is at fault.
3. Fix.  Most likely: route `write32(0x1d214)` and related MM2S/S2MM
   task-queue offsets through the BD-enqueue path the way the
   high-level `dma_memcpy_nd` lowering does.  Possibly: gate
   `transfers_completed` so the fast-completion path doesn't trip on
   a never-started channel.

**Why this matters beyond "make a test pass":**

- The plain `add_one_ctrl_packet` rt-seq exercises a real lowering
  path some user kernels also use (low-level register pokes for
  control packets).  The emu silently passing kernels that would
  hang on real HW is the worst kind of correctness gap.
- This is the *exact* corner case that triggers the firmware
  silent-drop on HW (Bug A in the finding doc).  Once the emu
  reproduces the *symptom* (kernel never completes), we have a
  debuggable environment for the firmware bug — much better than
  the silicon, where firmware is opaque.

Pointers: `2026-05-22-chain-exec-npu-silent-drop-captured.md`
(finding doc, has the low-level-vs-high-level structural diff and
both repro recipes), MEMORY note
`feedback_distinguish_silent_drop_from_cascade.md`.

---

### B. Upstream xdna-driver issue post  (RE done, drafting deferred)

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

**Next concrete steps:**

1. Draft the issue post.  Frame as a *driver-interface gap*: the
   driver relies on TDR as the sole backstop for op-0x18 silent
   drops because the firmware has no timeout; the cascade-to-NOAVAIL
   means a single dropped exec can starve the column pool.  Reference
   the finding doc, the variant-immunity structural finding (low-
   level rt-seq triggers it), and link the standalone repro recipe.
2. Show Maya the text before posting (CLAUDE.md global rule for
   externally-visible writing).
3. Post.

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

## Today's session in one paragraph (so the next session knows the
shape)

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
Pivoted to emu: empty-ctrlpkt against EMU reports success, mapped
the diagnosis surface for the gap (executor.rs sync logic, channel
FSM), did not write the failing test or fix.  Last commit `e4b8290`.
