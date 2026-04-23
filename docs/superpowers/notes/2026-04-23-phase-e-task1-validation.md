# Phase E Task 1 Validation — Handoff (2026-04-23)

Status at end of session: **Task 1 FAILED**, discovered a deeper emulator bug
than the plan anticipated. Picking this up later. WIP fixes are in the working
tree, uncommitted.

## tl;dr

EMU trace output is not HW-binary-compatible on the first representative test
(`vector_scalar_using_dma` / chess). HW trace = 15 real events across 41,181
cycles. EMU trace = 20 spurious events across 44 cycles. Two layers of bugs
underneath, one narrow and fixed (cycle threading), one broad and open
(event-layer classification).

The Phase E plan and spec are committed (`1e147f4`, `edad16a`). Task 1's
decision gate was explicit about this path: "If output is empty or unparseable,
the validation task opens a focused debugging sub-task." Output is parseable
but semantically wrong, which is a softer fail — we're treating it as the same
thing: stop Phase E, fix the emulator.

## What was measured

Invocation (after ensuring build is fresh):

```bash
cd /home/triple/npu-work/mlir-aie/build/test/npu-xrt/vector_scalar_using_dma/chess
XDNA_EMU=release \
  XDNA_EMU_DIR=/home/triple/npu-work/xdna-emu \
  XDNA_EMU_LOG_LEVEL=info \
  XRT_DEVICE_BDF=ffff:ff:1f.0 \
  /home/triple/npu-work/xdna-emu/bridge-runner/build/bridge-trace-runner \
    --xclbin ./aie.xclbin --instr ./insts.bin \
    --trace-out /tmp/claude-1000/phase-e-task1/trace_emu.chess.bin \
    --trace-size 8192
```

**Note the `XDNA_EMU_DIR` env var** — without it, the plugin falls back to
`/opt/xilinx/xrt/lib/libxdna_emu.so` which is a stale symlink to the **debug**
build. Setting `XDNA_EMU_DIR` + `XDNA_EMU=release` routes to
`$XDNA_EMU_DIR/target/release/libxdna_emu.so`. Bit me in this session. If not
setting it, `./scripts/rebuild-plugin.sh --release` updates the plugin via
pkexec so the symlink path works.

### HW reference (already Phase B baseline)
- Bin: `build/bridge-test-results/latest/vector_scalar_using_dma.hw-cycles/trace.chess.bin`
- Cycles: **41,181**
- Event mix: `{INSTR_EVENT_0: 8, INSTR_EVENT_1: 7}` (kernel-boundary markers; no `INSTR_VECTOR`)
- Span: `ts=3885 → 45066`

### EMU (after my WIP cycle fixes)
- Bin: `/tmp/claude-1000/phase-e-task1/trace_emu.chess.bin`
- Cycles: **44** (vs HW 41,181 — ratio 0.001)
- Event mix: `{INSTR_VECTOR: 39}` (19-20 events, each synthesized as B/E pair by parse_trace)
- Span: `ts=2 → 46`

### The kernel (for sanity)
`test/npu-xrt/vector_scalar_using_dma/scale.cc` is a **scalar** kernel:

```cpp
template <typename T_in, typename T_out, int N>
void scale(T_in *a, T_out *c, T_in factor) {
  event0();
  for (int i = 0; i < N; i++) {
    c[i] = factor * a[i];   // scalar multiply, no vector ops
  }
  event1();
}
```

Name is misleading. HW reports zero `INSTR_VECTOR`, consistent with the code.

## Bug 1 (narrow, FIXED in WIP — not committed)

**Symptom.** Trace-unit start/stop events time-stamped at cycle 0 regardless
of when they actually fired. `Event_Generate` register writes from the NPU
executor (firmware instruction stream) reached `notify_event` with a
hardcoded `0` cycle arg.

**Root cause.**
- `src/device/state/effects.rs:304-305`: `tile.core_trace.notify_event(event_id, 0)` and
  `tile.mem_trace.notify_event(event_id, 0)` — hardcoded 0.
- `src/device/state/effects.rs:367-368`: `tile.notify_core_trace_event(*hw_id, 0)` and
  `tile.notify_mem_trace_event(*hw_id, 0)` — same, inside `propagate_broadcasts`.
- `src/interpreter/engine/coordinator.rs:685`: drained event log events were
  notified with `evt.cycle` which is the per-core retire counter (`ctx.cycles`),
  not the global simulation cycle. Per-core counter stalls during stalls; the
  tile's trace clock doesn't.

**Fix (WIP, uncommitted).**
- `src/device/state/effects.rs`: snapshot `self.array.current_cycle` at the top
  of `apply_tile_local_effects` and `propagate_broadcasts`, pass it into
  `notify_event` calls.
- `src/interpreter/engine/coordinator.rs`: added
  `self.device.array.set_dma_cycle(self.total_cycles)` at the top of
  `step()` so `array.current_cycle` is fresh during Phase 2 (was previously
  only set at Phase 3).
- `src/interpreter/engine/coordinator.rs:685`: replaced `evt.cycle` with
  `self.total_cycles`.
- `crates/xdna-emu-ffi/src/execution.rs`: added
  `handle.engine.device_mut().array.set_dma_cycle(cycles)` before `try_advance`
  so NPU-executor register writes see the right cycle.

**Verified.** New info-level logs confirm:
- `Tile(0,0) Event_Generate: event_id=127 (offset=0x34008) cycle=45`
- `Propagating BROADCAST_15 (hw_id=122) from tile (0,0) to column 0 at cycle 45`
- `Propagating BROADCAST_14 (hw_id=121) from tile (0,0) to column 0 at cycle 43473`

The 45/43473 window matches the kernel duration (~43,474 cycles per
`XDNA_EMU_STATUS: halt_reason=completed cycles=43474`), so start/stop
timestamps are correct now.

This fix alone did **not** make the trace match HW, because of Bug 2.

## Bug 2 (broad, OPEN — event-layer classification)

**Symptom.** EMU emits 20 `INSTR_VECTOR` events for a scalar kernel that
should emit zero; EMU emits zero `INSTR_EVENT_0/1` events when HW emits 15.

**Root cause (hypothesized).** Two subsystems disagree:

1. `src/interpreter/execute/cycle_accurate.rs:466-470` records
   `EventType::InstrVector` unconditionally whenever any instruction executes
   in `SlotIndex::Vector` or `SlotIndex::Accumulator`. Hardware's
   `INSTR_VECTOR` (event 37) fires only for actual vector-pipeline ops, not
   for whatever scalar-ish thing happens to end up on those slots in this
   build. Slot-indexed classification ≠ ISA semantics.

2. `src/interpreter/execute/semantic.rs:164-183` *does* have a
   `SemanticOp::Event` handler that records `EventType::InstrEvent { id }`
   (which `trace/mod.rs` maps to HW IDs 33/34). But no `InstrEvent` records
   reach the trace unit in this run. Either the AIE2 `EVENT 0` /
   `EVENT 1` instructions (which `event0()`/`event1()` compile to) aren't
   classified as `SemanticOp::Event` by the decoder, or the semantic handler
   is being bypassed by the slot-index classification above, or there's some
   upstream filter.

**Scope of a real fix.** Touches the event-recording layer the whole emulator
uses for trace, VCD, and profiling. Needs:
- Decoder/TableGen audit: does the `EVENT` instruction get `SemanticOp::Event`? What
  slot does it occupy? (`llvm-aie` TableGen is authoritative.)
- `cycle_accurate.rs:463-475` rethink: HW fires class events (`INSTR_VECTOR`,
  `INSTR_LOAD`, `INSTR_STORE`, `INSTR_CALL`, `INSTR_RETURN`) based on
  instruction semantics, not VLIW slot. Slot is a coarse proxy and wrong for
  scalar-on-vector-slot. Move classification to semantic-op-driven.
- HW-observation validation across multiple kernels (add/add_one_objFifo
  scalar, anything actually-vector from the Phase B batch, cascade_flows
  multi-tile) to make sure the fix covers the catalog.

Estimated effort: **a day or more**, not a one-liner. I was wrong in my read
earlier when I told the user "might be a one-line fix."

**Why this mattered so much for Phase E.** Phase E's `trace-compare`
classification compares HW vs EMU traces by event-type counts and timing
deltas. With Bug 2, EMU would report `DRIFT` or `BUDGET` on essentially every
scalar kernel — the signal would be dominated by event mis-classification,
not the cycle-modeling drift Phase E was designed to surface. Landing Phase E
on top of Bug 2 wastes the tool.

## State of the world at end of session

### Committed
- `1e147f4` — Phase E implementation plan
- `edad16a` — Phase E design spec
- `4ad1ee8` — Phase B complete note on parent cycle-budget plan
- `26acca2` — Phase B validation results
- Plus prior Phase B commits.

### Uncommitted (WIP on `dev`)
- `src/device/state/effects.rs` — cycle threading, Bug 1 fix
- `src/interpreter/engine/coordinator.rs` — `set_dma_cycle` at top of step + use
  `self.total_cycles` at line 685
- `crates/xdna-emu-ffi/src/execution.rs` — `set_dma_cycle` before `try_advance`

Plus pre-existing (not mine, user's own subsys 7→8 work):
- `NEXT-STEPS.md`, `docs/arch/isa-execute-model.md`, `docs/arch/subsys7-audit.md`

### Artifacts (ephemeral, under `/tmp/claude-1000/phase-e-task1/`)
- `trace_emu.chess.bin` — the failing EMU trace
- `runner*.log` — various debug runs
- `cycles.EMU.txt` — extracted "44" cycles
- `/tmp/claude-1000/phase-e-task1/` will survive until reboot; these are
  useful as before-pictures if we re-run after the event-layer fix.

## Pick up next session — decision tree

The three-way fork I presented to the user before we stopped:

- **A. Go deep on the event-layer fix now.** Right per "fix bugs where they
  are." Multi-day open-ended work. Phase E stays on the shelf until event
  layer is right. I lean toward this path only if the event layer is also
  blocking other things (it probably is — VCD/profiling quality).

- **B. Commit my Bug 1 WIP, land Phase E with a known-quirk note that
  scalar kernels DRIFT until the event layer is fixed.** Phase E then becomes
  the tool that surfaces *exactly which* kernels hit event-classification
  bugs — nice forcing function. Bug 2 becomes a tracked follow-up, scoped
  from real data rather than guesswork. **My recommendation.**

- **C. Revert my Bug 1 WIP, write up Bug 2 in full, pause all Phase E work
  until the event layer is rewritten.** Most conservative; loses the
  cycle-threading groundwork and delays Phase E.

User hadn't decided between these when we stopped. They went to sleep. 

## Resume instructions (for the next session)

1. `git status` — confirm the WIP files listed above are still uncommitted.
2. Decide A/B/C with the user.
3. If B: commit the three WIP files as a focused "fix(trace): thread cycle
   through event path" commit, then start Phase E Task 2. Note the expected
   DRIFT outcome in the Phase E task 14 validation — it's a known issue, not
   a Phase E bug.
4. If A: start with `src/interpreter/execute/cycle_accurate.rs:463-475` and
   `src/interpreter/execute/semantic.rs:164-183` as the primary sites. Walk
   the decoder → SemanticOp → event-recording path for the `EVENT` and
   vector-pipeline opcodes. Cross-check against `llvm-aie` TableGen for the
   canonical semantic classification. Iterate against HW traces from the
   Phase B 7-test batch, not just `vector_scalar_using_dma`.
5. Either way, do **NOT** pretend Task 1 passed. It didn't. The plan's
   decision gate is explicit about this.

## Raw repro commands (preserve context for later)

```bash
# Full EMU capture run (release build, debug log):
cd /home/triple/npu-work/mlir-aie/build/test/npu-xrt/vector_scalar_using_dma/chess
XDNA_EMU=release XDNA_EMU_DIR=/home/triple/npu-work/xdna-emu \
  RUST_LOG=debug XDNA_EMU_LOG_LEVEL=debug \
  XRT_DEVICE_BDF=ffff:ff:1f.0 \
  /home/triple/npu-work/xdna-emu/bridge-runner/build/bridge-trace-runner \
    --xclbin ./aie.xclbin --instr ./insts.bin \
    --trace-out /tmp/claude-1000/phase-e-task1/trace_emu.chess.bin \
    --trace-size 8192 2>/tmp/claude-1000/phase-e-task1/runner.log

# Cycle extraction (needs ironenv):
source /home/triple/npu-work/mlir-aie/ironenv/bin/activate
cd /home/triple/npu-work/xdna-emu
PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
  python3 tools/trace-to-cycles.py \
    --trace-bin /tmp/claude-1000/phase-e-task1/trace_emu.chess.bin \
    --xclbin-mlir /home/triple/npu-work/mlir-aie/build/test/npu-xrt/vector_scalar_using_dma/chess/aie_arch.mlir.prj/input_with_addresses.mlir \
    --out /tmp/claude-1000/phase-e-task1/cycles.EMU.txt

# Event dump (for bug 2 investigation):
PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python python3 - <<'EOF'
import numpy as np, collections
from pathlib import Path
from aie.utils.trace.parse import parse_trace
mlir_text = Path("/home/triple/npu-work/mlir-aie/build/test/npu-xrt/vector_scalar_using_dma/chess/aie_arch.mlir.prj/input_with_addresses.mlir").read_text()
for label, path in [("HW", "/home/triple/npu-work/xdna-emu/build/bridge-test-results/latest/vector_scalar_using_dma.hw-cycles/trace.chess.bin"),
                    ("EMU", "/tmp/claude-1000/phase-e-task1/trace_emu.chess.bin")]:
    raw = np.fromfile(path, dtype=np.uint32)
    events = parse_trace(raw, mlir_text)
    non_meta = [e for e in events if e.get('ph') in ('B','E','X','i')]
    c = collections.Counter(e['name'] for e in non_meta)
    print(f"{label}: {dict(c)}")
EOF
```
