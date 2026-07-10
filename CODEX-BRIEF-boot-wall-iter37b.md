# Codex adversarial review brief: the 0x2450 boot wall (iter37b resolution)

**Your role.** You are a cross-model adversarial reviewer (different model family, max
effort). We (Claude + Maya) have spent many iterations reverse-engineering a boot wall
in AMD XDNA management firmware and have arrived at an **interpretive claim** we are
about to build a fix on. Before we invest a session writing that fix, we want you to
**try to break the claim**. Default to skepticism: your job is to find the hole, not to
confirm us. If the claim holds after a genuine attempt to refute it, say so and say why.

You have full read access to the repo (`/home/triple/npu-work/xdna-emu`) and can run the
diagnostic probes described below. Everything here is empirically checkable — please
re-derive the load-bearing facts yourself rather than trusting our summary.

---

## 0. Context in three sentences

`xdna-emu` runs the real AMD NPU management firmware (`npu.dev.sbin`, an Xtensa image
running Zephyr v3.7.1 + a run-to-completion dispatcher called MERT) on an in-tree
interpreter (`src/firmware/xtensa/`). Boot advances ~49,473 interpreted instructions and
then **walls**: it executes an `Unknown` opcode at `pc=0x2450`, which is an all-zeros
data region (calling a data pointer as if it were code). We want boot to reach idle;
this wall is the current frontier.

The wall has been chased across iters 18–37b. This brief covers only what you need to
audit the **iter37b resolution**. The full narrative is in
`docs/superpowers/findings/2026-07-08-boot-wake-unreached-breach.md` (search `iter35`,
`iter36`, `iter36b`, `iter37`, `iter37b`).

---

## 1. The claim we want you to attack

> **The 0x2450 wall is a scheduler-STATE divergence at the unique idle→first-task
> cooperative context switch. On that switch, `current` must be committed to the incoming
> task (`0x10dfc`) BEFORE the dispatch `FUN_00002730` reaches its head-seed at `+0x260`.
> Because the firmware unconditionally calls an UNGUARDED switch-hook (`Callx8 a7`) that
> walls for every task's `a7`, the state `head != tail` can never occur on real HW;
> therefore HW must enter the dispatch with `current` already equal to the incoming task.
> Our emulator instead enters with `current = init` and flips it too late (at the
> dispatch's own internal pick), producing `head != tail` → the unguarded hook → wall.
> The divergence is therefore UPSTREAM of `FUN_00002730`, in init's yield/swap path.**

If that is right, the fix is upstream (make init's yield commit `current` before
trapping). If it is wrong, the fix may be **local** (see §5, the alternative we most want
you to weigh) or the whole framing may be off.

---

## 2. The mechanism, with addresses (verify these)

Scheduler globals (derived, verify): `SCHED` base = `0x2250`. `current`-task pointer =
`[0x2278]` (= `SCHED+0x28`). Ready-slot = `[0x22a0]` (= `SCHED+0x50`). `0x2450` = `SCHED+0x200`.

Two tasks matter:
- **init**: task struct `0x10f10`, priority byte `[0x10f10+8] = 0xff`, frame `0x12048`.
- **0x10dfc** (the first real task): struct `0x10dfc`, priority `[0x10dfc+8] = 6`,
  entry `0x08b041bc` (real code, lives in the `0x08b04xxx` VMA overlay), frame `0x15f18`.

The dispatch is `FUN_00002730`. Relevant offsets (pc = 0x2730 + offset):
- `+0x12d` (pc `0x285d`): `[0x2278] := a2` — the **pick** commits `current`. Lives inside a
  nested windowed helper `+0x120..+0x148` (returns via `RetwN`), called from `+0x2c5` via
  `FUN_0000dab0`. The helper sets `current := ready_head` (`[ready_slot+56] = [0x22a0]`).
- `+0x256..+0x265`: the **head/tail seed**. `[lit 0x2888]=0x2b60` (head anchor),
  `[lit 0x288c]=0x2b64` (tail anchor). `+0x260`: `[0x2b60] := a3`; `+0x265`: `[0x2b64] := a3`;
  where `a3` = the **outgoing** (init) frame `0x12048`.
- `+0x301` (pc `0x2a31`): `[0x2b64] := [[0x2278]+0]` — **tail re-derived** from current
  (→ `0x15f18` once current = `0x10dfc`). NOTE: head is NOT re-derived here.
- `+0x306` (pc `0x2a36`): the register restore reads `a3 := [0x2b60]` (head) and restores
  a4..a15 from that frame, incl. `a7 := [head+0x1c]`.
- `+0x34f` (pc `0x2a7f`): `Beq a0,a1,0x2ae0` where `a0=[0x2b60]`(head), `a1=[0x2b64]`(tail).
  EQUAL → `0x2ae0` (no-hook: restore SRs, `rfe`). DIFFER → fall through to the hook.
- `+0x352`: `[0x2b60] := tail` (head advanced — but only here, AFTER the Beq).
- `+0x356` (pc `0x2a86`): `Call0 0xdf98` → the hook.

The hook function has two entry points:
- `0xdf8c` (GUARDED): `MovN a11,a3; L32iN a7,[a7+12]; ...; BeqzN a7,0xdf9f; Callx8 a7` —
  treats `a7` as a struct pointer, loads the real hook from `[a7+12]`, and null-skips.
- `0xdf98` (UNGUARDED, = `0xdf8c+0xc`): `Callx8 a7` directly — treats `a7` as a function.

`+0x356` targets the **unguarded** `0xdf98`. At the wall, `a7 = [head+0x1c] = [0x12048+0x1c]
= [0x12064] = 0x2450` (a data region) → `Callx8 0x2450` → wall.

**Runtime store history of `current [0x2278]`** (only two writes in the whole boot up to the
wall): `n=41463 → 0x10f10 (init)`, then `n=47985 → 0x10dfc` (the `+0x12d` pick). Nothing sets
it between. `0x10dfc` is created at `n=39730` and its pointer is stored to the ready slot
`[0x22a0]` at `n=39852`, then sits there untouched until the pick reads it at `n=47985`
(→ cooperative scheduling, no preemption).

---

## 3. The evidence chain (each step is reproducible)

Run probes with: `XDNA_FW_PROBE=1 [extra env] cargo test --lib <name> -- --nocapture`
(all probes self-skip unless `XDNA_FW_PROBE=1`). Firmware auto-resolves from
`../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`.

1. **Store-ordering of the anchors** — `m2c_probe_addr_store_watch`
   (`XDNA_FW_WATCH_ADDR=0x2278,0x2b60,0x2b64 XDNA_FW_MAX=50000`): head/tail seeded to init
   frame at `+0x260/+0x265` (n≈47519); current flips to `0x10dfc` at `+0x12d` (n=47985,
   AFTER the seed); tail re-derived at `+0x301` (n=49435); head advanced only at `+0x352`
   (n=49469, AFTER the Beq). **Head is never re-derived on the taken path before the Beq.**

2. **The pick is a legit priority dequeue** — disasm `+0x120..+0x148`
   (`XDNA_FW_DISASM=0x2850:0x2880`). `current := ready_head` when `current != 0`. `0x10dfc`
   (prio 6) genuinely outranks init (prio 0xff).

3. **The hook target is `0xdf98`, faithful (not an overlay/decode artifact)** — the
   epilogue executes at its natural VMA (runtime pc trace shows `0x2a7f`, `0x2ae0`, not
   `+0x100`-shifted), and static disasm (`XDNA_FW_DISASM=0x2a75:0x2a8a`) matches runtime
   (`m2c_probe_yield_callgraph` shows `CALL 0x00df98 [from FUN_00002730+0x356]`). Arithmetic:
   bytes `[05 51 0b]` at `0x2a86` = `Call0`, offset `0x2D44`, target `(0x2a84)+4+(0x2D44<<2)
   = 0xDF98`. **Please re-verify this decode independently — it is load-bearing.**

4. **NO task frame carries a valid hook at `[+0x1c]`** — `m2c_probe_head_advance_poke`
   (mode 1) dumps: init `0x12048→0x2450`, picked `0x15f18→0`, steady `0x15e78→0`. None is a
   function pointer → the unguarded hook walls for ANY frame.

5. **The poke experiments** — `m2c_probe_head_advance_poke`
   (`XDNA_FW_POKE=1|2 XDNA_FW_SW_MAX=300000`):
   - control (no poke): DIFFER → wall `0x2450` at n=49473.
   - **B1** (force `head := [[0x2278]] = 0x15f18` just before the restore, i.e. resume the
     incoming task): EQUAL → **548 dispatch cycles, ran to 300k budget, no wall**.
   - **B2** (force `current := init` before the tail-derive, i.e. resume init): EQUAL →
     **wall at `0xe035` in 38 instructions**.
   Interpretation: resuming the incoming task (B1) is healthy; resuming init (B2) dies fast.

6. **Steady state has no current-changing switch** — in the B1-poked run, all 548 passes
   seed `head=tail=0x15e78` (a CONSTANT), Beq always EQUAL, and `current` is never
   rewritten after the first switch. So the init→0x10dfc switch is the ONLY current-change
   in the entire trace, and there is no runtime example of a "correct" current-changing
   switch to compare against.

---

## 4. Why we concluded "upstream" (the reasoning to attack)

Given (3)+(4): the unguarded hook is unsurvivable for every frame, so `head != tail` can
never be a state HW passes through here. Given (1): head is structurally seeded from the
outgoing frame at `+0x260`, which is BEFORE the pick at `+0x2c5`, and there is no
head-re-derive on the taken path. The only way to have `head == tail` on a switch is for
`current` to already equal the incoming task at `+0x260` — i.e. committed BEFORE the
dispatch runs. Steady state (6) is consistent: there, `current` never changes, so head
seeds to the (unchanged) current frame and head==tail for free. Therefore the init→0x10dfc
switch must have `current` pre-committed upstream, and our emulator's failure to do so
(leaving `current=init` until the too-late internal pick) is the divergence.

We checked the prime upstream suspect `FUN_0000d6c0` (init's last scheduler call before its
yield-syscall, n≈47371–47404): it is the priority-indexed **ready-queue manager**
(enqueue/dequeue; `Addx4` indexing; per-priority head/tail bytes at struct+96/+117;
`a3+512 = SCHED`), and it does **not** write `current [0x2278]`. So if the claim holds, the
current-commit is elsewhere in init's yield path (the `0xb041f0` syscall wrapper →
`FUN_00003df8` → `0xb043cc` chain, n≈47407–47424).

---

## 5. The specific things to attack (ranked)

**A. The alternative we most want weighed: is the fix actually LOCAL, not upstream?**
Consider the hypothesis that `head` is *supposed* to be re-derived from `current` before
the restore/Beq (exactly as `tail` is at `+0x301`), and that our emulator **mis-executes a
branch** so that re-derive is skipped. The B1 poke ("set head := current-frame before the
restore") IS this local fix, and it works (548 clean cycles). Our argument against local is
"the only head-writes on the taken path are `+0x260` and `+0x352` (post-Beq), so there is no
on-path head-re-derive to mis-execute." Pressure that: is there a branch in `FUN_00002730`
(or its callees `FUN_0000dab0`/the pick helper/`schedule_next=FUN_00005958`) that writes the
head anchor `*0x2888` and that our run does not take? An address-watch on `[0x2b60]` shows no
executed store between `+0x260` and `+0x352` — but that only rules out *executed* stores;
an untaken branch would not show. Look for a static store to `*0x2888` on a path our trace
skips. If one exists, the fix is local and the "upstream" claim is wrong.

**B. Re-verify the load-bearing decode.** Independently confirm `+0x356` targets the
UNGUARDED `0xdf98` (not the guarded `0xdf8c`), and that `0x2a86` is not subject to a `+0x100`
(or other) VMA/LMA overlay in our loader (`src/firmware/image.rs`, `load_m2c`,
`psp_map.rs`). If HW actually calls `0xdf8c`, init's `a7=0x2450` null-skips cleanly and the
wall dissolves locally with no upstream change. We claim no overlay because the epilogue runs
at its natural VMA and static==runtime — but verify the overlay machinery for this region.

**C. Re-verify "no frame has a valid hook."** We dumped only three frames. Is there any
path where `a7` is reloaded from a hook table between the restore (`+0x333`) and the
`Callx8` (`+0x356`)? (We read the disasm and saw none, but check.) Could `[a7+12]` (the
guarded-entry semantics) point at a real function for some frame, implying the guarded
entry is intended?

**D. Challenge the head/tail semantics.** We interpret `head`=frame-to-restore (should be
incoming), `tail`=current's frame, Beq="did current change since entry." Is there a coherent
alternative reading of the `0x2884/0x2888/0x288c` anchor triple and the `+0x256..+0x265` /
`+0x2f0..+0x352` code that changes the conclusion? (`lit 0x2884 → Pcur` is a third anchor we
did not fully decode — disasm `0x2986:0x29a0` and `0x2a1c:0x2a40`.)

**E. Challenge "cooperative / current-changing switch is the ONLY one."** We infer
cooperative scheduling from `0x10dfc` sitting in `[0x22a0]` from creation to yield with no
preemption. Is there a reschedule/preemption point between n=39730 and n=47407 that SHOULD
have fired (making current flip earlier) and that we are missing? That would also be
"upstream" but at a different site (task-create/preempt, not init's yield).

**F. Is B2's fast wall (0xe035) a poke artifact?** We treat "resume init → wall in 38
instructions" as evidence that resuming init is wrong (hence the switch is real). But the B2
poke crudely forces `current := init`, possibly corrupting state the epilogue relies on. If
B2's wall is an artifact, one leg of "the switch is genuinely to 0x10dfc" weakens.

---

## 6. What we want back

A written verdict (a markdown reply or a file) covering:
1. Does the **iter37b resolution** (current must be pre-committed upstream) survive a real
   refutation attempt? If not, where exactly is the hole?
2. Specifically on **(5A)**: local vs upstream — is there evidence for an on-a-branch
   head-re-derive we mis-execute? This is the highest-value question; a local fix would be
   far cheaper than an upstream one.
3. Independent confirmation (or refutation) of the three load-bearing facts: `0xdf98` is the
   faithful unguarded target (5B); no frame's `a7`/`[a7+12]` is a valid hook (5C); head is
   seeded from the outgoing frame with no on-path re-derive (5A).
4. Any alternative hypothesis for why `head != tail` on the first switch that fits all of
   §3's observations better than ours.

Cite specific addresses/offsets and, where possible, a probe command or disasm range that
supports or refutes each point. Ground truth is the firmware image + the interpreter's
behavior; the real NPU is not available for this (the mgmt core can't be single-stepped).

## 7. Reproduction quick-reference

- Repo root: `/home/triple/npu-work/xdna-emu`. Build/test: `cargo test --lib`.
- Probes live in `src/firmware/mod.rs` under `mod boot_tests`, all gated on `XDNA_FW_PROBE=1`:
  - `m2c_probe_disasm_range` — `XDNA_FW_DISASM=<lo>:<hi>` (hex VMAs), static disasm.
  - `m2c_probe_addr_store_watch` — `XDNA_FW_WATCH_ADDR=hex,hex,...` `XDNA_FW_MAX=N`, store-by-address.
  - `m2c_probe_store_value_watch` — `XDNA_FW_WATCH_VAL=hex` `XDNA_FW_MAX=N`, store-by-value.
  - `m2c_probe_yield_callgraph` — `XDNA_FW_CG_WARMUP` / `XDNA_FW_CG_MAX` / `XDNA_FW_CG_LINES`, dynamic call tree.
  - `m2c_probe_head_advance_poke` — `XDNA_FW_POKE=1|2` `XDNA_FW_SW_MAX=N`, the poke experiment + frame hook-slot dump.
- Use translating loads (`data_read32`) not raw `peek8` when reading firmware data addresses
  (peek8 bypasses the MMU/overlay and returns garbage for VMA data/overlay regions).
- Full narrative + prior iterations: `docs/superpowers/findings/2026-07-08-boot-wake-unreached-breach.md`.
- Interpreter: `src/firmware/xtensa/interp/`. Loader/overlays: `src/firmware/image.rs`, `psp_map.rs`, `load_m2c`.

Thanks — genuinely try to break it. A confirmed hole here saves us a wasted session.
