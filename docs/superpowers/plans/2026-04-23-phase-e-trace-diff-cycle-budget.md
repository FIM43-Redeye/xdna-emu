# Phase E: Trace-Diff-Based Cycle Budget — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn opaque EMU bridge-test `TIMEOUT` results into actionable signals (MATCH / BUDGET / EMPTY / EMU_TRACE_BUG / HW_TRACE_BUG) by running the emulator through the same `bridge-trace-runner` flow Phase B uses for HW, feeding both trace buffers into the existing `trace-compare` binary, and classifying the result. Also add dual-bound EMU timing (cycle budget + wall-clock scaled from HW cycles), distinct surfacing of trace-related compile failures, and a per-test drift overrides / reporter UX.

**Architecture:** Five components, all in `scripts/emu-bridge-test.sh` except for two small new files. (1) A validation gate that proves EMU's existing trace-unit emission is HW-binary-compatible. (2) A unified `_run_trace_cycles_pipeline <side>` helper that runs `bridge-trace-runner` once per side (HW with normal env; EMU with `XDNA_EMU=debug`) writing symmetrically-named bin files. (3) Reuse of Phase B's `trace-to-cycles.py` on the EMU output. (4) Invocation of `target/release/trace-compare --hw ... --emu ... --stalls --extended` between the two bins, parsing the summary to classify the run. (5) Classification result fed into a new cycle-drift column in `print_report`, plus a `scripts/show-cycle-drift.sh` script for manual triage. The EMU path reuses the existing in-process trace-unit → host-memory routing built in commit `28711df`; no new Rust code is expected (Component #1 will validate that assumption).

**Tech Stack:** Bash 5 (bridge script), existing `bridge-runner/bridge-trace-runner` (C++/XRT), existing `tools/trace-to-cycles.py` (Python 3.13 + mlir-aie bindings), existing `target/release/trace-compare` (Rust), existing `src/trace/compare.rs` (report format), `awk` for floating-point arithmetic in bash.

---

## Scope Check

This plan covers a single subsystem: integrating EMU-side trace capture and binary trace comparison into the bridge test flow, plus the surrounding UX (dual-bound timing, compile-fail surfacing, overrides, reporter). It builds directly on Phase B (2026-04-22) which is shipped. It does not touch emulator internals unless the Task 1 validation gate exposes a real bug — in which case the bug fix is tracked as a discovered sub-task at that point, not as a pre-planned task.

---

## File Structure

**New files:**

- `docs/superpowers/notes/2026-04-23-phase-e-task1-validation.md` — One-page empirical validation record for Task 1 (measured EMU cycles vs HW, decision gate pass/fail).
- `scripts/trace-incompat-tests.txt` — List of tests whose compile fails when trace injection is applied. Loader skips injection for these tests. Format: one `<test_name>` per line; `#` comments OK; same loader style as `trace-quarantine.txt`.
- `scripts/cycle-drift-overrides.txt` — Per-test ratio override. Format: `<test_name> <ratio_lower> <ratio_upper>` per line; `#` comments OK.
- `scripts/show-cycle-drift.sh` — Sort the latest `bridge-test-results/latest/` directory's `.cycles.compare.*.txt` files by |log(HW/EMU)| and print the top N.
- `docs/superpowers/plans/2026-04-23-phase-e-validation.md` — End-of-plan batch validation record (one table, like Phase B's validation doc).

**Modified files:**

- `scripts/emu-bridge-test.sh` — All bridge integration (most tasks touch this single file).
- `docs/superpowers/plans/2026-04-22-cycle-budget-testing.md` — Note that Phase C and Phase D.3 are superseded and point at this plan (one-line update at the end of the document).

**No new Rust/C++ code is expected.** If Task 1 discovers a defect in EMU trace emission, that defect's fix is a sub-task under Task 1 and its extent is decided at the point of discovery.

---

## Non-Obvious Facts the Implementer Needs

1. **EMU trace emission path already exists end-to-end.** The coordinator's `flush_trace_to_host()` (`src/interpreter/engine/coordinator.rs:225`) flushes all tile trace units, then runs additional `step_data_movement()` passes to route buffered words through the stream switch into host DDR memory. The FFI's execution loop calls it at run-end (`crates/xdna-emu-ffi/src/execution.rs:207`). That means when `bridge-trace-runner` (or any XRT client) syncs a trace BO from device at end of kernel execution under `XDNA_EMU=debug`, the bytes it reads come from the emulator's host_memory buffer, populated by the trace-routing machinery. **No new plumbing required** — it is the existing in-process path.

2. **EMU runs via `bridge-trace-runner`, not via `test.exe`.** The Phase E spec mentions "`./test.exe` emits trace", but the cleaner symmetric path is: run `bridge-trace-runner` with `XDNA_EMU=debug` + `XRT_DEVICE_BDF=ffff:ff:1f.0`, just like HW runs it without those env vars. Same binary, same allocation logic, same positional classifier. The bridge test's `./test.exe` run (in `run_one_bridge`) is what determines PASS/FAIL; the trace capture is a separate invocation of `bridge-trace-runner` that runs after the PASS/FAIL result is known, mirroring how Phase B already does HW side in `_run_hw_cycles_pipeline`.

3. **Phase B's HW trace bin is named `trace.<variant>.bin`**, not `trace_hw.<variant>.bin` as one might assume. Phase E renames it to `trace_hw.<variant>.bin` for symmetry with `trace_emu.<variant>.bin`. This is a Task 2 one-liner (change a single `local trace_bin=` assignment in `_run_hw_cycles_pipeline`).

4. **Bash has no floats.** Dual-bound timing requires multiplying HW cycles by a float (`SECONDS_PER_CYCLE = 1e-3`). Use `awk` inside a `$(...)` subshell — e.g., `local timeout=$(awk "BEGIN{t=$hw_cycles*2.0*0.001; if(t<600)t=600; printf \"%d\", int(t+0.5)}")`.

5. **`trace-compare` report format** (`src/trace/compare.rs:2049-2057`) includes two parseable summary lines: `Edge event types:    N clean, N diverged, N count mismatch` and `Level event types:   ...`. Phase E classification parses these. A "BUDGET" is triggered by any non-zero `diverged` OR `count mismatch` in either line, or by cycle ratio out of `[0.5, 2.0]` bounds.

6. **`trace-compare` exit code**: non-zero indicates an error (unreadable files, etc.), not divergence. A run with divergence returns 0 and reports it textually. Classification must not confuse the two.

7. **`HW_CYCLES_TRACED_MLIR` is exported by `compile_one()` pre-compile** (`scripts/emu-bridge-test.sh:1139`) and consumed by `compile_one_compiler()` (line 911). When it is set and the compile fails, the failure is on the injected MLIR. Phase E uses this signal to distinguish `COMPILE-FAIL(traced)` from a plain `COMPILE-FAIL`.

8. **`XDNA_TRACE_DIR` is already exported** for `run_one_bridge` (line 1389) and used by the patched `test.exe` to dump the legacy `trace_raw.bin`. Phase E does not reuse it — the Phase B pipeline dumps directly via `bridge-trace-runner --trace-out`. Keeping the two dump destinations separate avoids mixing the legacy trace (`--trace` path) and HW-cycles trace (`--with-hw-cycles` path).

9. **`trace-compare` uses `src/trace/compare.rs` internals, which decode per-tile events assuming `trace_raw.bin` is a concatenation of per-tile trace-unit buffers**, not a single packed stream. Task 1 validation must confirm `bridge-trace-runner`'s HW output and the emulator's host-memory dump both produce bin files with that layout. If they don't, the comparison does not work and Task 1 surfaces the defect.

10. **`TRACE_OK` and `WITH_HW_CYCLES` are independent.** `TRACE_OK=true` gates the old `trace-prepare.py`/`cpp_trace_patch.py` legacy path (separate file: `trace_raw.bin` via test.exe). `WITH_HW_CYCLES=true` gates the Phase B `mlir-trace-inject` path (separate file: `trace_hw.<variant>.bin` via bridge-trace-runner). They can coexist on the same run; Phase E uses only the latter.

---

## Task 1: Validation gate — EMU trace round-trip for vector_scalar_using_dma

**Files:**
- Create: `docs/superpowers/notes/2026-04-23-phase-e-task1-validation.md`

**Purpose:** Gate the rest of the plan on proof that EMU's existing trace emission is HW-binary-compatible. If the emulator produces empty or unparseable bytes, we debug the emulator (not the bridge) before any further Phase E work. The plan's task list stops here until this validation passes.

- [ ] **Step 1: Ensure a known-good traced `vector_scalar_using_dma` xclbin exists**

Run:
```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --with-hw-cycles --no-timeout --no-emu --chess-only -v '^vector_scalar_using_dma$'
```
Expected: produces an HW `cycles.HW.vector_scalar_using_dma.chess.txt` with an integer (Phase B reference: 41181 cycles).

Verify:
```bash
ls build/bridge-test-results/latest/vector_scalar_using_dma.*.cycles.HW.txt
cat build/bridge-test-results/latest/vector_scalar_using_dma.chess.cycles.HW.txt
ls mlir-aie/build/test/npu-xrt/vector_scalar_using_dma/chess/aie.xclbin
ls mlir-aie/build/test/npu-xrt/vector_scalar_using_dma/chess/insts.bin
ls mlir-aie/build/test/npu-xrt/vector_scalar_using_dma/chess/aie_arch.mlir.prj/input_with_addresses.mlir
```
Expected: every file exists; the cycles file contains a non-zero integer.

If missing, step 1 failed — run with `--compile` to force rebuild, or debug the Phase B pipeline before continuing.

- [ ] **Step 2: Ensure the latest emulator .so is installed**

Run:
```bash
cd /home/triple/npu-work/xdna-emu
cargo build --release --bin trace-compare
./scripts/rebuild-plugin.sh --release
```
Expected: build succeeds; `/opt/xilinx/xrt/lib/libxrt_driver_emu.so.2` is updated (pkexec prompt).

- [ ] **Step 3: Capture EMU trace via bridge-trace-runner under XDNA_EMU**

Run (from the xdna-emu repo root):
```bash
mkdir -p /tmp/claude-1000/phase-e-task1
cd mlir-aie/build/test/npu-xrt/vector_scalar_using_dma/chess
XDNA_EMU=debug \
  XDNA_EMU_LOG_LEVEL=info \
  XRT_DEVICE_BDF=ffff:ff:1f.0 \
  /home/triple/npu-work/xdna-emu/bridge-runner/build/bridge-trace-runner \
    --xclbin ./aie.xclbin \
    --instr ./insts.bin \
    --trace-out /tmp/claude-1000/phase-e-task1/trace_emu.chess.bin \
    --trace-size 8192 \
    2> /tmp/claude-1000/phase-e-task1/runner.log
```
Expected: exit code 0; `trace_emu.chess.bin` is 8192 bytes; runner.log mentions `XDNA emulator` or similar plugin-load marker.

If the runner exits non-zero or the plugin doesn't load, check `runner.log` — most likely causes are missing `.so`, `XRT_DEVICE_BDF` issue, or a kernarg classification regression in the runner.

- [ ] **Step 4: Extract EMU cycles**

Run:
```bash
cd /home/triple/npu-work/xdna-emu
PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
  python3 tools/trace-to-cycles.py \
    --trace-bin /tmp/claude-1000/phase-e-task1/trace_emu.chess.bin \
    --xclbin-mlir mlir-aie/build/test/npu-xrt/vector_scalar_using_dma/chess/aie_arch.mlir.prj/input_with_addresses.mlir \
    --out /tmp/claude-1000/phase-e-task1/cycles.EMU.txt \
    2> /tmp/claude-1000/phase-e-task1/extract.log
cat /tmp/claude-1000/phase-e-task1/cycles.EMU.txt
```
Expected: a non-zero integer on stdout. "Within the same order of magnitude as 41181" is a bonus; "non-zero, parseable, repeatable" is the gate.

- [ ] **Step 5: Decision gate**

**If step 4 produced a non-zero integer:** write the validation note (next step) and proceed to Task 2.

**If step 4 produced zero, "empty trace", or a parser error:** the validation has failed. STOP the plan here. File the finding as:

  - Empty trace → bug is in `src/device/trace_unit/` (trace unit not emitting) or in `src/interpreter/engine/coordinator.rs` `flush_trace_to_host()` (not routing to host). Start investigation there. Sub-task opens in this plan document under Task 1; continue with Phase E only after the sub-task closes.
  - Parseable bytes but zero cycles → trace unit state machine disabled or event IDs not firing. Check `src/trace/mod.rs` `event_to_hw_id` coverage and `src/interpreter/state/event_trace.rs` circular buffer semantics.
  - Parser error → layout mismatch between emulator's dumped bytes and `aie.utils.trace.parse_trace()` expectations. Compare byte-level with the HW bin (next step's diff is a useful starting point).

  ```bash
  xxd /tmp/claude-1000/phase-e-task1/trace_emu.chess.bin | head -20
  xxd build/bridge-test-results/latest/vector_scalar_using_dma.hw-cycles/trace.chess.bin | head -20
  ```

- [ ] **Step 6: Write the validation note**

Create `docs/superpowers/notes/2026-04-23-phase-e-task1-validation.md` with the following content, filling in the measured numbers:

```markdown
# Phase E Task 1 Validation (2026-04-23)

## Method
- Test: `vector_scalar_using_dma` / chess
- Traced xclbin: `mlir-aie/build/test/npu-xrt/vector_scalar_using_dma/chess/aie.xclbin`
- Runner: `bridge-runner/build/bridge-trace-runner`
- EMU env: `XDNA_EMU=debug XRT_DEVICE_BDF=ffff:ff:1f.0`
- Extractor: `tools/trace-to-cycles.py`

## Results

| Side | Bin size (bytes) | Parseable? | Cycles |
|------|------------------|------------|--------|
| HW   | <fill in>        | yes        | <fill in> |
| EMU  | <fill in>        | <yes/no>   | <fill in> |

## Verdict

<PASS — non-zero EMU cycles, parseable, proceed with Task 2 onward.>
or
<FAIL — <describe>; sub-task opened under Task 1 to fix <component>.>

## Raw artifact paths (for repro)
- `/tmp/claude-1000/phase-e-task1/trace_emu.chess.bin`
- `/tmp/claude-1000/phase-e-task1/runner.log`
- `/tmp/claude-1000/phase-e-task1/extract.log`
- `/tmp/claude-1000/phase-e-task1/cycles.EMU.txt`
```

- [ ] **Step 7: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/superpowers/notes/2026-04-23-phase-e-task1-validation.md
git commit -m "$(cat <<'EOF'
docs(phase-e): record task 1 validation — EMU trace round-trip on vector_scalar_using_dma

Confirms EMU's existing trace_unit + coordinator flush produces
HW-binary-compatible bytes that parse_trace() can read. Gates the rest
of Phase E on this result.

Generated using Claude Code.
EOF
)"
```

---

## Task 2: Rename HW trace bin to `trace_hw.<variant>.bin`

**Files:**
- Modify: `scripts/emu-bridge-test.sh:560`

**Purpose:** Symmetric naming for the HW and EMU bins going forward. One-line change with a re-run smoke test.

- [ ] **Step 1: Edit `_run_hw_cycles_pipeline`**

In `scripts/emu-bridge-test.sh`, locate:

```bash
    local trace_bin="$work_dir/trace.$variant.bin"
```
(around line 560). Replace with:

```bash
    local trace_bin="$work_dir/trace_hw.$variant.bin"
```

- [ ] **Step 2: Re-run to generate the renamed artifact**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --with-hw-cycles --no-timeout --no-emu --chess-only -v '^vector_scalar_using_dma$'
ls build/bridge-test-results/latest/vector_scalar_using_dma.hw-cycles/
```
Expected: listing contains `trace_hw.chess.bin` (no `trace.chess.bin`).

- [ ] **Step 3: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add scripts/emu-bridge-test.sh
git commit -m "$(cat <<'EOF'
refactor(bridge-test): rename HW trace bin to trace_hw.<variant>.bin

Symmetric with the forthcoming trace_emu.<variant>.bin from Phase E.
No behavior change beyond the filename.

Generated using Claude Code.
EOF
)"
```

---

## Task 3: Add `--with-cycle-diff` flag (scaffolding only)

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (flag parsing, help text, export)

**Purpose:** Introduce the flag surface now so downstream tasks can gate behavior on it. No runtime behavior change in this task — the flag merely sets and exports `WITH_CYCLE_DIFF`, and implies `WITH_HW_CYCLES=true`.

- [ ] **Step 1: Add default near the top of the flag block**

Find the `WITH_HW_CYCLES=${WITH_HW_CYCLES:-false}` line (around line 95). Add immediately after:

```bash
WITH_CYCLE_DIFF=${WITH_CYCLE_DIFF:-false}
```

- [ ] **Step 2: Add flag parsing**

Find the `--with-hw-cycles)` case (around line 119). Add immediately after:

```bash
    --with-cycle-diff)     WITH_CYCLE_DIFF=true; WITH_HW_CYCLES=true; shift ;;
```

- [ ] **Step 3: Extend the help text**

Find the `--with-hw-cycles` help line (around line 140). Add below it:

```bash
  --with-cycle-diff Additionally run EMU through the trace pipeline and
                    compare against HW via trace-compare. Implies
                    --with-hw-cycles.
```

- [ ] **Step 4: Add `WITH_CYCLE_DIFF` to the `export` list**

Find the `export RESULTS_DIR FORCE_COMPILE VERBOSE RUN_EMU NO_TRACE SWEEP RUN_AIESIM NO_TIMEOUT WITH_HW_CYCLES` line (around line 175). Append `WITH_CYCLE_DIFF`:

```bash
export RESULTS_DIR FORCE_COMPILE VERBOSE RUN_EMU NO_TRACE SWEEP RUN_AIESIM NO_TIMEOUT WITH_HW_CYCLES WITH_CYCLE_DIFF
```

- [ ] **Step 5: Smoke test**

Run:
```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --help 2>&1 | grep -A1 -- '--with-cycle-diff'
```
Expected: prints the help line. No other behavior change.

Run:
```bash
./scripts/emu-bridge-test.sh --with-cycle-diff -v '^__this_matches_nothing__$' 2>&1 | head -5
```
Expected: the command accepts the flag without error.

- [ ] **Step 6: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "$(cat <<'EOF'
feat(bridge-test): add --with-cycle-diff flag scaffolding

Implies --with-hw-cycles. No behavior change yet; downstream Phase E
tasks wire up EMU trace capture and comparison.

Generated using Claude Code.
EOF
)"
```

---

## Task 4: Unify HW/EMU trace pipeline into one side-parameterized helper

**Files:**
- Modify: `scripts/emu-bridge-test.sh:514-587` (`_run_hw_cycles_pipeline` → `_run_trace_cycles_pipeline`)

**Purpose:** Replace `_run_hw_cycles_pipeline <build_dir> <xclbin> <kernel> <instr> <variant>` with a `_run_trace_cycles_pipeline <side> <build_dir> <xclbin> <kernel> <instr> <variant>` that takes `side ∈ {HW, EMU}` and does the side-specific env/output selection internally. The HW caller is updated in this task; the EMU caller comes in Task 5.

- [ ] **Step 1: Rewrite the helper**

Replace the entire body of `_run_hw_cycles_pipeline` (between the opening `{` and the closing `}` exposed by `export -f` around line 815) with this side-parameterized version. Preserve surrounding comments:

```bash
# _run_trace_cycles_pipeline <side> <build_dir> <xclbin> <kernel> <instr> <variant>
#   side ∈ {HW, EMU}.
# Runs bridge-trace-runner against a traced xclbin on the requested side,
# then runs trace-to-cycles on the output. Writes:
#   RESULTS_DIR/<safe>.hw-cycles/trace_{hw,emu}.<variant>.bin
#   RESULTS_DIR/<safe>.<variant>.cycles.{HW,EMU}.txt
# Best-effort: logs and returns non-zero on failure; callers pass || true.
_run_trace_cycles_pipeline() {
    local side="$1"
    local build_dir="$2"
    local xclbin="$3"
    local kernel="$4"      # currently unused; runner auto-detects single kernel
    local instr="$5"
    local variant="$6"

    local test_name
    test_name="$(basename "$(dirname "$build_dir")")"

    local runner="$EMU_ROOT/bridge-runner/build/bridge-trace-runner"
    if [[ ! -x "$runner" ]]; then
        echo "[trace-cycles:$side] $test_name ($variant): runner not built at $runner; skipping" >&2
        return 0
    fi

    local src_mlir="$build_dir/aie_arch.mlir"
    if [[ ! -f "$src_mlir" ]]; then
        echo "[trace-cycles:$side] $test_name ($variant): no MLIR at $src_mlir; cannot extract cycles" >&2
        return 0
    fi
    if ! grep -q "aie.trace " "$src_mlir" 2>/dev/null; then
        echo "[trace-cycles:$side] $test_name ($variant): MLIR has no trace ops; skipping" >&2
        return 0
    fi
    local mlir_path="$build_dir/aie_arch.mlir.prj/input_with_addresses.mlir"
    if [[ ! -f "$mlir_path" ]]; then
        echo "[trace-cycles:$side] $test_name ($variant): no lowered MLIR at $mlir_path; cannot extract cycles" >&2
        return 0
    fi

    local safe
    safe="$(sanitize_name "$test_name")"
    local work_dir="$RESULTS_DIR/${safe}.hw-cycles"
    mkdir -p "$work_dir"

    local bin_label cycles_label
    case "$side" in
        HW)  bin_label="trace_hw";  cycles_label="HW" ;;
        EMU) bin_label="trace_emu"; cycles_label="EMU" ;;
        *)   echo "[trace-cycles] unknown side: $side" >&2; return 1 ;;
    esac

    local trace_bin="$work_dir/${bin_label}.${variant}.bin"
    local cycles_txt="$RESULTS_DIR/${safe}.${variant}.cycles.${cycles_label}.txt"
    local runner_log="$work_dir/runner.${bin_label}.${variant}.log"
    local extract_log="$work_dir/extract.${bin_label}.${variant}.log"

    # Side-specific env: EMU routes through the plugin; HW runs on real silicon.
    local -a env_prefix=()
    if [[ "$side" == "EMU" ]]; then
        env_prefix+=("XDNA_EMU=${XDNA_EMU:-debug}")
        env_prefix+=("XDNA_EMU_LOG_LEVEL=${XDNA_EMU_LOG_LEVEL:-info}")
        env_prefix+=("XRT_DEVICE_BDF=ffff:ff:1f.0")
        [[ -n "${XDNA_EMU_LIB:-}" ]] && env_prefix+=("XDNA_EMU_LIB=$XDNA_EMU_LIB")
    fi

    if ! env "${env_prefix[@]}" "$runner" \
        --xclbin "$xclbin" \
        --instr "$instr" \
        --trace-out "$trace_bin" \
        --trace-size 8192 \
        2>"$runner_log"; then
        echo "[trace-cycles:$side] $test_name ($variant): runner failed; see $runner_log" >&2
        return 1
    fi

    if ! PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
        python3 "$EMU_ROOT/tools/trace-to-cycles.py" \
        --trace-bin "$trace_bin" \
        --xclbin-mlir "$mlir_path" \
        --out "$cycles_txt" \
        2>"$extract_log"; then
        echo "[trace-cycles:$side] $test_name ($variant): extractor failed; see $extract_log" >&2
        return 1
    fi

    echo "[trace-cycles:$side] $test_name ($variant): cycles=$(cat "$cycles_txt")" >&2
    return 0
}
```

- [ ] **Step 2: Update the HW caller in `run_one_hardware`**

Find (around line 1304):

```bash
          _run_hw_cycles_pipeline "$build_dir" "$_hw_xclbin" "" "$_hw_instr" "${compiler}${vsuffix}" || true
```

Replace with:

```bash
          _run_trace_cycles_pipeline HW "$build_dir" "$_hw_xclbin" "" "$_hw_instr" "${compiler}${vsuffix}" || true
```

And update the adjacent log prefix (the line right below that reads `echo "[hw-cycles] ..."` — around line 1306) to match, so all output is tagged consistently:

```bash
          echo "[trace-cycles:HW] $name (${compiler}${vsuffix}): missing xclbin or insts.bin in $build_dir; skipping" >&2
```

- [ ] **Step 3: Update `export -f` line to match renamed function**

Find (around line 815):

```bash
export -f _strip_trace_flags _variant_from_cmd _run_hw_cycles_pipeline
```

Replace with:

```bash
export -f _strip_trace_flags _variant_from_cmd _run_trace_cycles_pipeline
```

- [ ] **Step 4: Smoke test — HW path still works**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --with-hw-cycles --no-timeout --no-emu --chess-only -v '^vector_scalar_using_dma$'
ls build/bridge-test-results/latest/vector_scalar_using_dma.hw-cycles/
cat build/bridge-test-results/latest/vector_scalar_using_dma.chess.cycles.HW.txt
```
Expected: `trace_hw.chess.bin` present; cycles file non-empty integer identical to Phase B's (~41181).

- [ ] **Step 5: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "$(cat <<'EOF'
refactor(bridge-test): unify HW/EMU trace-cycles helper (side param)

_run_hw_cycles_pipeline becomes _run_trace_cycles_pipeline <HW|EMU>.
HW caller updated; EMU caller arrives in the next task.

Generated using Claude Code.
EOF
)"
```

---

## Task 5: Wire EMU side into `run_one_bridge`

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (`run_one_bridge`, around line 1436)

**Purpose:** After the EMU bridge test produces a `result`, when `WITH_CYCLE_DIFF=true` and the test PASSed, invoke `_run_trace_cycles_pipeline EMU` on the same build dir. Mirrors Phase B's HW-side call in `run_one_hardware`.

- [ ] **Step 1: Add the EMU-side invocation**

In `run_one_bridge` (around line 1436 just after `echo "$result" > "$result_file"`), locate this region:

```bash
  echo "$result" > "$result_file"

  # Copy events.json for trace decoding.
```

Insert between those two statements (before the events.json copy):

```bash
  # Phase E: capture EMU cycle count via trace pipeline (best-effort).
  if [[ "$WITH_CYCLE_DIFF" == "true" && "$result" == "PASS" ]]; then
      local _emu_xclbin
      _emu_xclbin="$(find "$build_dir" -maxdepth 1 -name '*.xclbin' -print -quit 2>/dev/null || true)"
      local _emu_instr="$build_dir/insts.bin"
      [[ -f "$_emu_instr" ]] || _emu_instr=""
      if [[ -n "$_emu_xclbin" && -n "$_emu_instr" ]]; then
          _run_trace_cycles_pipeline EMU "$build_dir" "$_emu_xclbin" "" "$_emu_instr" "${compiler}${vsuffix}" || true
      else
          echo "[trace-cycles:EMU] $name (${compiler}${vsuffix}): missing xclbin or insts.bin in $build_dir; skipping" >&2
      fi
  fi
```

- [ ] **Step 2: End-to-end smoke test**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --with-cycle-diff --no-timeout --chess-only -v '^vector_scalar_using_dma$'
ls build/bridge-test-results/latest/vector_scalar_using_dma.hw-cycles/
```
Expected: both `trace_hw.chess.bin` and `trace_emu.chess.bin` exist. Both `vector_scalar_using_dma.chess.cycles.HW.txt` and `vector_scalar_using_dma.chess.cycles.EMU.txt` exist and contain integers.

Compare for sanity:
```bash
cat build/bridge-test-results/latest/vector_scalar_using_dma.chess.cycles.HW.txt
cat build/bridge-test-results/latest/vector_scalar_using_dma.chess.cycles.EMU.txt
```
Expected: both non-zero. Within an order of magnitude is plausible; identical is unlikely (emulator models, not replays).

- [ ] **Step 3: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "$(cat <<'EOF'
feat(bridge-test): capture EMU trace + cycles under --with-cycle-diff

Runs bridge-trace-runner against the traced xclbin under XDNA_EMU=debug
after a successful bridge test, producing trace_emu.<variant>.bin and
cycles.EMU.<test>.<variant>.txt for downstream comparison.

Generated using Claude Code.
EOF
)"
```

---

## Task 6: Invoke `trace-compare` and capture the report

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (new helper `_run_trace_compare`, caller in `run_one_bridge`)

**Purpose:** After both `trace_hw.<variant>.bin` and `trace_emu.<variant>.bin` exist for a given run, invoke `target/release/trace-compare` with `--stalls --extended` and write the report to `RESULTS_DIR/<safe>.<variant>.cycles.compare.txt`. Classification is the next task; this one produces the report only.

- [ ] **Step 1: Add the helper**

In `scripts/emu-bridge-test.sh`, immediately after `_run_trace_cycles_pipeline` (just before the `export -f` line), add:

```bash
# _run_trace_compare <test_name> <variant>
# Runs trace-compare on the HW and EMU trace bins produced earlier; writes
# the report to RESULTS_DIR/<safe>.<variant>.cycles.compare.txt.
# Returns 0 on success, 1 if either bin is missing, 2 if trace-compare errored.
_run_trace_compare() {
    local test_name="$1"
    local variant="$2"

    local safe
    safe="$(sanitize_name "$test_name")"
    local work_dir="$RESULTS_DIR/${safe}.hw-cycles"
    local hw_bin="$work_dir/trace_hw.${variant}.bin"
    local emu_bin="$work_dir/trace_emu.${variant}.bin"
    local report="$RESULTS_DIR/${safe}.${variant}.cycles.compare.txt"

    if [[ ! -f "$hw_bin" || ! -f "$emu_bin" ]]; then
        return 1
    fi

    local rust_bin="$EMU_ROOT/target/release/trace-compare"
    if [[ ! -x "$rust_bin" ]]; then
        echo "[trace-compare] trace-compare binary not built at $rust_bin" >&2
        return 2
    fi

    if ! "$rust_bin" \
        --hw "$hw_bin" \
        --emu "$emu_bin" \
        --stalls \
        --extended \
        -o "$report" 2>/dev/null; then
        return 2
    fi
    return 0
}
```

- [ ] **Step 2: Update the `export -f` line**

Find:

```bash
export -f _strip_trace_flags _variant_from_cmd _run_trace_cycles_pipeline
```

Replace with:

```bash
export -f _strip_trace_flags _variant_from_cmd _run_trace_cycles_pipeline _run_trace_compare
```

- [ ] **Step 3: Invoke from `run_one_bridge`**

In `run_one_bridge`, directly after the Task 5 EMU-side invocation block, add:

```bash
  # Phase E: compare HW vs EMU trace if both bins exist.
  if [[ "$WITH_CYCLE_DIFF" == "true" && "$result" == "PASS" ]]; then
      _run_trace_compare "$name" "${compiler}${vsuffix}" || true
  fi
```

- [ ] **Step 4: Smoke test**

```bash
cd /home/triple/npu-work/xdna-emu
cargo build --release --bin trace-compare   # ensure bin is fresh
./scripts/emu-bridge-test.sh --with-cycle-diff --no-timeout --chess-only -v '^vector_scalar_using_dma$'
cat build/bridge-test-results/latest/vector_scalar_using_dma.chess.cycles.compare.txt | head -30
```
Expected: a non-empty report with `Edge event types:` and `Level event types:` summary lines.

- [ ] **Step 5: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "$(cat <<'EOF'
feat(bridge-test): run trace-compare between HW and EMU bins

Writes cycles.compare.<test>.<variant>.txt after a successful bridge
test under --with-cycle-diff. Classification comes in the next task.

Generated using Claude Code.
EOF
)"
```

---

## Task 7: Classify the compare result

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (new helper `_classify_cycle_diff`, caller in `run_one_bridge`)

**Purpose:** Parse the `cycles.HW.*.txt`, `cycles.EMU.*.txt`, and `cycles.compare.*.txt` files to produce one of the classification strings the spec defines (`MATCH`, `DRIFT`, `EMPTY`, `EMU_TRACE_BUG`, `HW_TRACE_BUG`, `COMPARE-ERR`, `NO_DATA`). Writes to `RESULTS_DIR/<safe>.<variant>.cycle.result`. The column-rendering itself comes in Task 12.

The rules in bash (Phase E spec §Classification rules and §Error handling):

| Condition | Classification |
|-----------|----------------|
| Neither HW nor EMU bin exists | `NO_DATA` (cycle comparison not attempted) |
| HW bin exists, HW cycles = 0, EMU bin exists, EMU cycles = 0 | `EMPTY` |
| HW bin exists, HW cycles > 0, EMU bin missing | `EMU_TRACE_BUG` |
| EMU bin exists, EMU cycles > 0, HW bin missing | `HW_TRACE_BUG` |
| HW cycles > 0, EMU cycles = 0 | `EMU_TRACE_BUG` |
| EMU cycles > 0, HW cycles = 0 | `HW_TRACE_BUG` |
| `trace-compare` report missing or malformed | `COMPARE-ERR` |
| Report says 0 diverged AND 0 count mismatch on both Edge and Level lines AND ratio (EMU/HW) ∈ [lower, upper] | `MATCH(<ratio>)` |
| Otherwise | `DRIFT(ratio=<r>, diverge=<n>)` where `<n>` is the sum of Edge+Level diverged+count-mismatch |

`<ratio>` is formatted to 2 decimal places. Default bounds are `[0.5, 2.0]`; per-test overrides from `cycle-drift-overrides.txt` (Task 11) take precedence and load via a helper lookup function. In this task the override lookup is stubbed as a function that always returns the defaults; Task 11 replaces the stub with a real lookup.

- [ ] **Step 1: Add the override-lookup stub**

Add near the top of the script (after the existing `is_trace_quarantined` function around line 249):

```bash
# Drift-override lookup (stubbed in Task 7; real load in Task 11).
# Usage: read the lower and upper bounds for a given test into caller variables.
#   _lookup_drift_bounds <test_name> _out_lower _out_upper
_lookup_drift_bounds() {
    local _out_l="$2"
    local _out_u="$3"
    printf -v "$_out_l" '%s' "0.5"
    printf -v "$_out_u" '%s' "2.0"
}
export -f _lookup_drift_bounds
```

- [ ] **Step 2: Add the classifier**

Add near `_run_trace_compare` (just before the `export -f` line that follows it):

```bash
# _classify_cycle_diff <test_name> <variant>
# Reads HW/EMU cycle files and the compare report; writes one classification
# string to RESULTS_DIR/<safe>.<variant>.cycle.result. Does not set a return
# code based on classification (always returns 0 unless invoked incorrectly).
_classify_cycle_diff() {
    local test_name="$1"
    local variant="$2"

    local safe
    safe="$(sanitize_name "$test_name")"
    local work_dir="$RESULTS_DIR/${safe}.hw-cycles"
    local hw_bin="$work_dir/trace_hw.${variant}.bin"
    local emu_bin="$work_dir/trace_emu.${variant}.bin"
    local hw_cycles_txt="$RESULTS_DIR/${safe}.${variant}.cycles.HW.txt"
    local emu_cycles_txt="$RESULTS_DIR/${safe}.${variant}.cycles.EMU.txt"
    local report="$RESULTS_DIR/${safe}.${variant}.cycles.compare.txt"
    local out_file="$RESULTS_DIR/${safe}.${variant}.cycle.result"

    local hw_have_bin=false; [[ -f "$hw_bin"  ]] && hw_have_bin=true
    local emu_have_bin=false; [[ -f "$emu_bin" ]] && emu_have_bin=true

    if ! $hw_have_bin && ! $emu_have_bin; then
        echo "NO_DATA" > "$out_file"
        return 0
    fi

    local hw_cycles=0 emu_cycles=0
    [[ -f "$hw_cycles_txt"  ]] && hw_cycles="$(tr -d '[:space:]' < "$hw_cycles_txt")"
    [[ -f "$emu_cycles_txt" ]] && emu_cycles="$(tr -d '[:space:]' < "$emu_cycles_txt")"
    [[ -z "$hw_cycles"  ]] && hw_cycles=0
    [[ -z "$emu_cycles" ]] && emu_cycles=0

    # Asymmetry: one side traced, the other didn't.
    if $hw_have_bin && ! $emu_have_bin; then
        echo "EMU_TRACE_BUG" > "$out_file"
        return 0
    fi
    if $emu_have_bin && ! $hw_have_bin; then
        echo "HW_TRACE_BUG" > "$out_file"
        return 0
    fi
    if [[ "$hw_cycles" -gt 0 && "$emu_cycles" -eq 0 ]]; then
        echo "EMU_TRACE_BUG" > "$out_file"
        return 0
    fi
    if [[ "$emu_cycles" -gt 0 && "$hw_cycles" -eq 0 ]]; then
        echo "HW_TRACE_BUG" > "$out_file"
        return 0
    fi

    # Both zero: legitimate scalar-kernel case.
    if [[ "$hw_cycles" -eq 0 && "$emu_cycles" -eq 0 ]]; then
        echo "EMPTY" > "$out_file"
        return 0
    fi

    # Both non-zero: parse the compare report.
    if [[ ! -f "$report" ]]; then
        echo "COMPARE-ERR" > "$out_file"
        return 0
    fi

    local edge_line level_line
    edge_line="$(grep -E '^Edge event types:'  "$report" || true)"
    level_line="$(grep -E '^Level event types:' "$report" || true)"
    if [[ -z "$edge_line" || -z "$level_line" ]]; then
        echo "COMPARE-ERR" > "$out_file"
        return 0
    fi

    # Pattern: "Edge event types:    N clean, N diverged, N count mismatch"
    local e_div e_cmm l_div l_cmm
    e_div="$(echo "$edge_line"  | grep -oE '[0-9]+ diverged'       | awk '{print $1}')"
    e_cmm="$(echo "$edge_line"  | grep -oE '[0-9]+ count mismatch' | awk '{print $1}')"
    l_div="$(echo "$level_line" | grep -oE '[0-9]+ diverged'       | awk '{print $1}')"
    l_cmm="$(echo "$level_line" | grep -oE '[0-9]+ count mismatch' | awk '{print $1}')"
    [[ -z "$e_div" ]] && e_div=0
    [[ -z "$e_cmm" ]] && e_cmm=0
    [[ -z "$l_div" ]] && l_div=0
    [[ -z "$l_cmm" ]] && l_cmm=0
    local total_diverge=$(( e_div + e_cmm + l_div + l_cmm ))

    local lower upper
    _lookup_drift_bounds "$test_name" lower upper

    local ratio
    ratio="$(awk -v e="$emu_cycles" -v h="$hw_cycles" 'BEGIN{ if(h==0){print "0.00"} else{printf "%.2f", e/h} }')"
    local in_bounds
    in_bounds="$(awk -v r="$ratio" -v l="$lower" -v u="$upper" 'BEGIN{ if(r>=l && r<=u) print 1; else print 0 }')"

    if [[ "$total_diverge" -eq 0 && "$in_bounds" == "1" ]]; then
        echo "MATCH($ratio)" > "$out_file"
    else
        echo "DRIFT(ratio=$ratio,diverge=$total_diverge)" > "$out_file"
    fi
    return 0
}
```

- [ ] **Step 3: Update `export -f`**

Find the `export -f` line that currently lists `_strip_trace_flags _variant_from_cmd _run_trace_cycles_pipeline _run_trace_compare`. Replace with:

```bash
export -f _strip_trace_flags _variant_from_cmd _run_trace_cycles_pipeline _run_trace_compare _classify_cycle_diff
```

- [ ] **Step 4: Invoke the classifier from `run_one_bridge`**

In `run_one_bridge`, immediately after the Task 6 `_run_trace_compare` call, add:

```bash
  # Phase E: classify cycle diff into a persistent result file.
  if [[ "$WITH_CYCLE_DIFF" == "true" && "$result" == "PASS" ]]; then
      _classify_cycle_diff "$name" "${compiler}${vsuffix}" || true
  fi
```

- [ ] **Step 5: Smoke test**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --with-cycle-diff --no-timeout --chess-only -v '^vector_scalar_using_dma$'
cat build/bridge-test-results/latest/vector_scalar_using_dma.chess.cycle.result
```
Expected: one of `MATCH(<ratio>)`, `DRIFT(ratio=<r>,diverge=<n>)`, or `EMPTY`. Never `NO_DATA` or asymmetry tags for a PASSing test that produced both bins.

Verify EMPTY path manually by forcing HW and EMU cycles files to contain "0":
```bash
echo 0 > build/bridge-test-results/latest/vector_scalar_using_dma.chess.cycles.HW.txt
echo 0 > build/bridge-test-results/latest/vector_scalar_using_dma.chess.cycles.EMU.txt
# Re-invoke classifier manually via the script's exported function — use a subshell:
bash -c 'source scripts/emu-bridge-test.sh --help >/dev/null 2>&1; true'
# Easier: just spot-check the classifier logic with a small manual run:
RESULTS_DIR=build/bridge-test-results/latest bash -c '
  source <(grep -n . scripts/emu-bridge-test.sh | sed -n "/_classify_cycle_diff()/,/^}/p" | cut -d: -f2-)
  source <(grep -n . scripts/emu-bridge-test.sh | sed -n "/_lookup_drift_bounds()/,/^}/p" | cut -d: -f2-)
  sanitize_name() { echo "$1" | tr "/" "_"; }
  _classify_cycle_diff vector_scalar_using_dma chess
  cat "$RESULTS_DIR/vector_scalar_using_dma.chess.cycle.result"
'
```
Expected: prints `EMPTY`.

(If this manual test is awkward in practice, it's acceptable to verify the EMPTY path during the full-batch Task 14 run where scalar-kernel tests like `add_one_using_dma` naturally trigger it. The smoke test above is enough to move on.)

Restore the HW cycles file by re-running the bridge test:
```bash
./scripts/emu-bridge-test.sh --with-cycle-diff --no-timeout --chess-only -v '^vector_scalar_using_dma$'
```

- [ ] **Step 6: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "$(cat <<'EOF'
feat(bridge-test): classify cycle diff (MATCH / DRIFT / EMPTY / *_TRACE_BUG)

Parses trace-compare report + HW/EMU cycles files to produce a
persistent .cycle.result per test-variant. Override bounds are
defaulted ([0.5, 2.0]); per-test loader arrives in a later task.

Generated using Claude Code.
EOF
)"
```

---

## Task 8: Dual-bound EMU timing in `run_one_bridge`

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (`run_one_bridge` around line 1380)

**Purpose:** When a HW cycles file exists for the current test+variant, set `XDNA_EMU_MAX_CYCLES = ceil(HW_cycles * 2.0)` (Phase A cycle budget) AND scale the wall-clock timeout to `max(600, ceil(HW_cycles * 2.0 * SECONDS_PER_CYCLE))` with `SECONDS_PER_CYCLE = 1e-3`. Preserve `--no-timeout` semantics unchanged (it disables both bounds). Tests with no HW cycles file keep today's 600 s timeout and no cycle budget.

- [ ] **Step 1: Add the `SECONDS_PER_CYCLE` constant**

Near the top of the script (after the `WITH_CYCLE_DIFF=${WITH_CYCLE_DIFF:-false}` added in Task 3), add:

```bash
# Phase E dual-bound EMU timing constants.
# SECONDS_PER_CYCLE is a conservative starting value (~1000 sim-cycles/sec);
# calibrated empirically post-Task 14.
EMU_SECONDS_PER_CYCLE=${EMU_SECONDS_PER_CYCLE:-0.001}
EMU_CYCLE_BUDGET_MULTIPLIER=${EMU_CYCLE_BUDGET_MULTIPLIER:-2.0}
export EMU_SECONDS_PER_CYCLE EMU_CYCLE_BUDGET_MULTIPLIER
```

- [ ] **Step 2: Compute the bounds in `run_one_bridge`**

Locate in `run_one_bridge` (around line 1380) the existing block:

```bash
  local rc=0
  (
    cd "$build_dir"
    export XDNA_EMU="${XDNA_EMU:-debug}"
    export XDNA_EMU_LOG_LEVEL="${XDNA_EMU_LOG_LEVEL:-info}"
    # Pass through XDNA_EMU_LIB if set (explicit override).
    [[ -n "${XDNA_EMU_LIB:-}" ]] && export XDNA_EMU_LIB
    export XRT_DEVICE_BDF="ffff:ff:1f.0"
    export XDNA_TRACE_DIR="$trace_out_dir"
    if [[ "${NO_TIMEOUT:-false}" == "true" ]]; then
      bash -c "$run_cmd"
    else
      timeout 600 bash -c "$run_cmd"
    fi
  ) > "$log_file" 2>&1 || rc=$?
```

Replace the entire block (including the `local rc=0` line) with:

```bash
  # Phase E dual-bound timing: if a HW cycles file exists for this
  # test+variant+compiler, derive a tighter cycle budget (Phase A) and
  # a scaled wall-clock timeout. Otherwise fall back to today's 600 s.
  local _hw_cycles_file="$RESULTS_DIR/${safe}${vsuffix}.${compiler}.cycles.HW.txt"
  local _hw_cycles=0
  if [[ -f "$_hw_cycles_file" ]]; then
      _hw_cycles="$(tr -d '[:space:]' < "$_hw_cycles_file")"
      [[ -z "$_hw_cycles" ]] && _hw_cycles=0
  fi

  local _cycle_budget=""
  local _timeout_s=600
  if [[ "$_hw_cycles" -gt 0 ]]; then
      _cycle_budget="$(awk -v c="$_hw_cycles" -v m="$EMU_CYCLE_BUDGET_MULTIPLIER" \
          'BEGIN{ printf "%d", c*m + 0.5 }')"
      _timeout_s="$(awk -v c="$_hw_cycles" -v m="$EMU_CYCLE_BUDGET_MULTIPLIER" -v s="$EMU_SECONDS_PER_CYCLE" \
          'BEGIN{ t=c*m*s; if (t<600) t=600; printf "%d", t + 0.5 }')"
  fi

  local rc=0
  (
    cd "$build_dir"
    export XDNA_EMU="${XDNA_EMU:-debug}"
    export XDNA_EMU_LOG_LEVEL="${XDNA_EMU_LOG_LEVEL:-info}"
    # Pass through XDNA_EMU_LIB if set (explicit override).
    [[ -n "${XDNA_EMU_LIB:-}" ]] && export XDNA_EMU_LIB
    export XRT_DEVICE_BDF="ffff:ff:1f.0"
    export XDNA_TRACE_DIR="$trace_out_dir"
    if [[ -n "$_cycle_budget" ]]; then
      export XDNA_EMU_MAX_CYCLES="$_cycle_budget"
    fi
    if [[ "${NO_TIMEOUT:-false}" == "true" ]]; then
      bash -c "$run_cmd"
    else
      timeout "${_timeout_s}" bash -c "$run_cmd"
    fi
  ) > "$log_file" 2>&1 || rc=$?
```

- [ ] **Step 3: Smoke test — cycle budget path**

```bash
cd /home/triple/npu-work/xdna-emu
# Run Phase B first to create the HW cycles file:
./scripts/emu-bridge-test.sh --with-hw-cycles --no-emu --chess-only -v '^vector_scalar_using_dma$'
# Now run EMU with dual-bound active:
./scripts/emu-bridge-test.sh --with-cycle-diff --chess-only -v '^vector_scalar_using_dma$' 2>&1 | tee /tmp/claude-1000/phase-e-task8.log
grep -E 'XDNA_EMU_MAX_CYCLES|BUDGET|TIMEOUT|PASS' /tmp/claude-1000/phase-e-task8.log
grep 'XDNA_EMU_STATUS' build/bridge-test-results/latest/vector_scalar_using_dma.chess.bridge.log | tail -1
```
Expected: `PASS` for the test, and the `XDNA_EMU_STATUS:` line shows `halt_reason=completed` with cycles well under the budget (budget ≈ 82362 for this test).

- [ ] **Step 4: Smoke test — `--no-timeout` still overrides**

```bash
./scripts/emu-bridge-test.sh --with-cycle-diff --no-timeout --chess-only -v '^vector_scalar_using_dma$' 2>&1 | tee /tmp/claude-1000/phase-e-task8b.log
grep -E 'PASS|FAIL|TIMEOUT' /tmp/claude-1000/phase-e-task8b.log | head -5
```
Expected: PASS; no timeout invocation at all.

- [ ] **Step 5: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "$(cat <<'EOF'
feat(bridge-test): dual-bound EMU timing from HW cycle data

When a cycles.HW.<test>.<variant>.txt exists, set XDNA_EMU_MAX_CYCLES
to HW_cycles * 2.0 and scale the wall-clock timeout to
max(600s, HW_cycles * 2.0 * 1e-3). --no-timeout still overrides both.

Generated using Claude Code.
EOF
)"
```

---

## Task 9: Distinguish `COMPILE-FAIL(traced)` from generic `COMPILE-FAIL`

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (`compile_one_compiler`, `print_report`)

**Purpose:** When `HW_CYCLES_TRACED_MLIR` is set and the compile fails, it is specifically the trace-injected path that broke. The bridge already knows this locally; surface it so users can retry without `--with-hw-cycles`.

- [ ] **Step 1: Write `FAIL_TRACED` from `compile_one_compiler`**

Find in `compile_one_compiler` (around line 1041-1044):

```bash
  if $failed; then
    echo "FAIL" > "$result_file"
    echo "  COMPILE $name ($compiler): FAIL"
    return 0
  fi
```

Replace with:

```bash
  if $failed; then
    local fail_tag="FAIL"
    local label="FAIL"
    if [[ -n "$HW_CYCLES_TRACED_MLIR" ]] && [[ "$src_mlir" == "$HW_CYCLES_TRACED_MLIR" ]]; then
      fail_tag="FAIL_TRACED"
      label="FAIL(traced)"
    fi
    echo "$fail_tag" > "$result_file"
    echo "  COMPILE $name ($compiler): $label"
    return 0
  fi
```

(The xclbin-missing-after-compile case a few lines below — around line 1052 — keeps plain `FAIL`. That path indicates aiecc reported success but produced no xclbin, which is a different bug class than "trace injection broke the build." Only the `$failed` branch is tagged.)

- [ ] **Step 2: Render `COMPILE-FAIL(traced)` in `print_report`**

Find in `print_report` (around line 1603):

```bash
      if [[ "$cr" != "OK" ]]; then
```

The block that follows currently renders all non-OK non-SKIP_* compile results as `FAIL*`. Modify to distinguish `FAIL_TRACED`:

```bash
      if [[ "$cr" != "OK" ]]; then
        # Only count compile failure once per test:compiler pair.
        local ck="${name}:${compiler}"
        if [[ -z "${_compile_counted[$ck]+x}" ]]; then
          compile_fail[$compiler]=$(( ${compile_fail[$compiler]} + 1 ))
          _compile_counted["$ck"]=1
        fi
        has_compile_fail=true
        local _fail_label="FAIL*"
        [[ "$cr" == "FAIL_TRACED" ]] && _fail_label="FAILt*"
        if [[ "$run_hw" == "true" ]]; then
          printf "  %-${col_width}s" "$_fail_label"
        fi
        printf "  %-${col_width}s" "$_fail_label"
        continue
      fi
```

Update the footnote block a few lines later (around line 1721-1724). Find:

```bash
  if $has_compile_fail || $has_tdr; then
    echo ""
    $has_compile_fail && echo "* = compile failed"
    $has_tdr && echo "TDR = hardware timeout detection and recovery (NPU hung)"
  fi
```

Replace with:

```bash
  # Detect whether any FAIL_TRACED exists to tailor the footnote.
  local has_traced_fail=false
  for row in "${test_list[@]}"; do
    local _n="${row%%:*}"
    local _safe
    _safe="$(sanitize_name "$_n")"
    for compiler in "${compilers[@]}"; do
      if [[ -f "$RESULTS_DIR/${_safe}.${compiler}.compile.result" ]]; then
        local _cr
        _cr="$(< "$RESULTS_DIR/${_safe}.${compiler}.compile.result")"
        [[ "$_cr" == "FAIL_TRACED" ]] && has_traced_fail=true
      fi
    done
  done
  if $has_compile_fail || $has_tdr; then
    echo ""
    $has_compile_fail && echo "*  = compile failed"
    $has_traced_fail  && echo "t* = compile failed on trace-injected MLIR (retry without --with-hw-cycles)"
    $has_tdr          && echo "TDR = hardware timeout detection and recovery (NPU hung)"
  fi
```

- [ ] **Step 3: Smoke test**

The natural test is `ctrl_packet_reconfig`, which Phase B validation showed fails under trace injection:

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --with-hw-cycles --no-emu --no-hw --chess-only -v '^ctrl_packet_reconfig$' 2>&1 | tee /tmp/claude-1000/phase-e-task9.log
cat build/bridge-test-results/latest/ctrl_packet_reconfig.chess.compile.result
grep -E 'FAILt\*|compile failed on trace-injected' /tmp/claude-1000/phase-e-task9.log
```
Expected: `.compile.result` reads `FAIL_TRACED`; report footnote mentions trace-injected failures.

Also confirm a non-traced run still prints plain `FAIL*`:

```bash
./scripts/emu-bridge-test.sh --no-emu --no-hw --chess-only -v '^__matches_nothing_but_syntax_ok__$' 2>&1 | head -5
```
Expected: no crash; no `t*` footnote (because no tests matched).

- [ ] **Step 4: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "$(cat <<'EOF'
feat(bridge-test): distinguish COMPILE-FAIL(traced) from plain fail

When HW_CYCLES_TRACED_MLIR is the source of the failed compile, write
FAIL_TRACED to the compile.result file and render FAILt* in the
report with a footnote pointing users at the --with-hw-cycles flag.

Generated using Claude Code.
EOF
)"
```

---

## Task 10: Add `scripts/trace-incompat-tests.txt` skip list

**Files:**
- Create: `scripts/trace-incompat-tests.txt`
- Modify: `scripts/emu-bridge-test.sh` (loader near line 214, skip logic in `compile_one`)

**Purpose:** Known trace-incompatible tests (e.g., `ctrl_packet_reconfig` — Phase B Limitation 4) still produce a useful HW result path; the bridge should skip injection for them up-front rather than eating a noisy compile failure every run.

- [ ] **Step 1: Create the skip list file**

```bash
cd /home/triple/npu-work/xdna-emu
cat > scripts/trace-incompat-tests.txt <<'EOF'
# scripts/trace-incompat-tests.txt
# Tests whose compile fails when Phase B trace injection is applied.
# Skipped entirely under --with-hw-cycles / --with-cycle-diff.
# Format: one <test_name> per line; lines starting with # are comments;
# whitespace trimmed.
#
# Entries here should reference a Phase B limitation or a tracked issue.

# Phase B Limitation 4: AIEInsertTraceFlows conflicts with aie.packet_flow.
# Upstream mlir-aie pass-ordering issue; investigate separately.
ctrl_packet_reconfig
EOF
```

- [ ] **Step 2: Add a loader after the existing `TRACE_QUARANTINE` block**

Find the end of the `is_trace_quarantined` function (around line 249, right after `export -f is_trace_quarantined`). Add:

```bash
# ---------------------------------------------------------------------------
# Trace-injection incompatibility list (Phase E)
# ---------------------------------------------------------------------------

TRACE_INCOMPAT_FILE="${SCRIPT_DIR}/trace-incompat-tests.txt"
declare -A TRACE_INCOMPAT=()
if [[ -f "$TRACE_INCOMPAT_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line// /}"
    [[ -z "$line" ]] && continue
    TRACE_INCOMPAT["$line"]=1
  done < "$TRACE_INCOMPAT_FILE"
  if [[ ${#TRACE_INCOMPAT[@]} -gt 0 ]]; then
    echo ">>> Trace-injection incompat: ${#TRACE_INCOMPAT[@]} test(s) will have injection skipped"
  fi
fi
export TRACE_INCOMPAT_FILE

is_trace_incompat() {
  if [[ ${#TRACE_INCOMPAT[@]} -gt 0 ]] 2>/dev/null; then
    [[ -n "${TRACE_INCOMPAT[${1}]+x}" ]]
    return
  fi
  local name="$1"
  [[ -f "$TRACE_INCOMPAT_FILE" ]] || return 1
  while IFS= read -r line; do
    local entry="${line%%#*}"
    entry="${entry// /}"
    [[ -z "$entry" ]] && continue
    [[ "$entry" == "$name" ]] && return 0
  done < "$TRACE_INCOMPAT_FILE"
  return 1
}
export -f is_trace_incompat
```

- [ ] **Step 3: Skip injection for incompat tests in `compile_one`**

Find the existing pre-compile injection block in `compile_one` (around line 1118). Before the `if [[ "$WITH_HW_CYCLES" == "true" ]] && [[ -f "$src_dir/aie.mlir" ]]` guard, extend the conditional so incompat tests bypass injection:

Replace:

```bash
  if [[ "$WITH_HW_CYCLES" == "true" ]] && [[ -f "$src_dir/aie.mlir" ]]; then
```

with:

```bash
  if [[ "$WITH_HW_CYCLES" == "true" ]] && [[ -f "$src_dir/aie.mlir" ]] \
      && ! is_trace_incompat "$name"; then
```

And add a user-visible log just before that block to document the skip:

```bash
  if [[ "$WITH_HW_CYCLES" == "true" ]] && is_trace_incompat "$name"; then
      echo "  HW-CYCLES INJECT $name: SKIP (in trace-incompat-tests.txt)"
  fi
```

Place it immediately before the updated `if` guard.

- [ ] **Step 4: Smoke test**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --with-hw-cycles --no-emu --no-hw --chess-only -v '^ctrl_packet_reconfig$' 2>&1 | tee /tmp/claude-1000/phase-e-task10.log
grep -E 'HW-CYCLES INJECT|FAIL|PASS|SKIP' /tmp/claude-1000/phase-e-task10.log
cat build/bridge-test-results/latest/ctrl_packet_reconfig.chess.compile.result
```
Expected: log shows `HW-CYCLES INJECT ctrl_packet_reconfig: SKIP (in trace-incompat-tests.txt)`; compile result is `OK` (not `FAIL_TRACED`).

- [ ] **Step 5: Commit**

```bash
git add scripts/emu-bridge-test.sh scripts/trace-incompat-tests.txt
git commit -m "$(cat <<'EOF'
feat(bridge-test): skip trace injection for incompat tests

Adds scripts/trace-incompat-tests.txt with ctrl_packet_reconfig as
first entry (Phase B Limitation 4). Skipping these avoids a noisy
compile failure every run without giving up on their HW PASS/FAIL
signal.

Generated using Claude Code.
EOF
)"
```

---

## Task 11: Add `scripts/cycle-drift-overrides.txt` with real loader

**Files:**
- Create: `scripts/cycle-drift-overrides.txt`
- Modify: `scripts/emu-bridge-test.sh` (replace the Task 7 stub loader with the real one)

**Purpose:** Replace the stubbed `_lookup_drift_bounds` (always returns `[0.5, 2.0]`) with a real per-test override loader. Most tests will continue to use defaults; specific tests can tighten or loosen their bounds.

- [ ] **Step 1: Create the overrides file**

```bash
cd /home/triple/npu-work/xdna-emu
cat > scripts/cycle-drift-overrides.txt <<'EOF'
# scripts/cycle-drift-overrides.txt
# Per-test cycle-drift ratio overrides for --with-cycle-diff classification.
# Format: <test_name> <ratio_lower> <ratio_upper>
#   ratio = EMU_cycles / HW_cycles; a test with ratio in [lower, upper]
#   AND zero trace-compare divergence is classified MATCH; otherwise DRIFT.
# Lines starting with # are comments; blank lines are ignored.
#
# Example:
#   vector_scalar_using_dma 0.8 1.25
#
# Default bounds (applied when a test is not listed here): [0.5, 2.0].
EOF
```

- [ ] **Step 2: Replace the stubbed loader**

Find the Task 7 stub:

```bash
# Drift-override lookup (stubbed in Task 7; real load in Task 11).
# Usage: read the lower and upper bounds for a given test into caller variables.
#   _lookup_drift_bounds <test_name> _out_lower _out_upper
_lookup_drift_bounds() {
    local _out_l="$2"
    local _out_u="$3"
    printf -v "$_out_l" '%s' "0.5"
    printf -v "$_out_u" '%s' "2.0"
}
export -f _lookup_drift_bounds
```

Replace with:

```bash
# ---------------------------------------------------------------------------
# Cycle-drift overrides (Phase E)
# ---------------------------------------------------------------------------

DRIFT_OVERRIDES_FILE="${SCRIPT_DIR}/cycle-drift-overrides.txt"
declare -A DRIFT_LOWER=() DRIFT_UPPER=()
if [[ -f "$DRIFT_OVERRIDES_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | awk '{$1=$1; print}')"  # trim
    [[ -z "$line" ]] && continue
    read -r _t _l _u <<< "$line"
    [[ -z "$_t" || -z "$_l" || -z "$_u" ]] && continue
    DRIFT_LOWER["$_t"]="$_l"
    DRIFT_UPPER["$_t"]="$_u"
  done < "$DRIFT_OVERRIDES_FILE"
  if [[ ${#DRIFT_LOWER[@]} -gt 0 ]]; then
    echo ">>> Cycle-drift overrides: ${#DRIFT_LOWER[@]} test(s) with custom bounds"
  fi
fi
export DRIFT_OVERRIDES_FILE

# _lookup_drift_bounds <test_name> _out_lower _out_upper
# Returns per-test bounds if overridden, else [0.5, 2.0].
_lookup_drift_bounds() {
    local t="$1"
    local _out_l="$2"
    local _out_u="$3"
    local l="0.5" u="2.0"
    # Fast path: in-memory array (main process).
    if [[ ${#DRIFT_LOWER[@]} -gt 0 ]] 2>/dev/null; then
        if [[ -n "${DRIFT_LOWER[$t]+x}" ]]; then
            l="${DRIFT_LOWER[$t]}"
            u="${DRIFT_UPPER[$t]}"
        fi
    elif [[ -f "$DRIFT_OVERRIDES_FILE" ]]; then
        # Subshell path: file read.
        while IFS= read -r line; do
            line="${line%%#*}"
            line="$(echo "$line" | awk '{$1=$1; print}')"
            [[ -z "$line" ]] && continue
            read -r _t _l _u <<< "$line"
            if [[ "$_t" == "$t" ]]; then
                l="$_l"; u="$_u"; break
            fi
        done < "$DRIFT_OVERRIDES_FILE"
    fi
    printf -v "$_out_l" '%s' "$l"
    printf -v "$_out_u" '%s' "$u"
}
export -f _lookup_drift_bounds
```

- [ ] **Step 3: Smoke test**

Add a test override to confirm parsing:

```bash
cd /home/triple/npu-work/xdna-emu
printf '\n# Smoke test\nvector_scalar_using_dma 0.8 1.25\n' >> scripts/cycle-drift-overrides.txt
./scripts/emu-bridge-test.sh --with-cycle-diff --chess-only -v '^vector_scalar_using_dma$' 2>&1 | grep -E 'Cycle-drift overrides|DRIFT|MATCH'
cat build/bridge-test-results/latest/vector_scalar_using_dma.chess.cycle.result
```
Expected: loader announces one override; classification uses the tighter bounds — will likely flip `MATCH(1.00)`-ish to `DRIFT(...)` unless EMU happens to match HW within 20%, proving the override applied.

Revert the test override:

```bash
git checkout -- scripts/cycle-drift-overrides.txt
```

- [ ] **Step 4: Commit**

```bash
git add scripts/emu-bridge-test.sh scripts/cycle-drift-overrides.txt
git commit -m "$(cat <<'EOF'
feat(bridge-test): per-test cycle-drift ratio overrides

Replaces the Task 7 stub with a real loader that reads
scripts/cycle-drift-overrides.txt. Defaults to [0.5, 2.0] for tests
not listed. Ratio bounds tighten as empirical data accumulates.

Generated using Claude Code.
EOF
)"
```

---

## Task 12: Report column + summary for cycle drift

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (`print_report`)

**Purpose:** Render a per-compiler `CYCLES` column when `WITH_CYCLE_DIFF=true`, and add a summary block that counts `MATCH`/`DRIFT`/`EMPTY`/`*_TRACE_BUG`/`NO_DATA` per compiler. Tests not listed as DRIFT/BUG are healthy and don't warrant a detailed row — the block lists only outliers plus an EMPTY reminder.

- [ ] **Step 1: Thread `WITH_CYCLE_DIFF` through `print_report`**

At the top of `print_report` (around line 1473), after the existing `has_trace=false` block, add:

```bash
  local has_cycle=false
  [[ "$WITH_CYCLE_DIFF" == "true" ]] && has_cycle=true
```

- [ ] **Step 2: Extend the header print**

Find the `if $has_trace; then` block in the header section (around line 1504). After its closing `fi`, add:

```bash
  if $has_cycle; then
    for compiler in "${compilers[@]}"; do
      local label
      label="$(echo "$compiler" | sed 's/./\U&/')"
      printf "  %-24s" "${label}/CYCLES"
    done
  fi
```

And the corresponding separator (around line 1521):

```bash
  if $has_cycle; then
    for _ in "${compilers[@]}"; do
      printf "  %-24s" "$(printf '%0.s-' $(seq 1 24))"
    done
  fi
```

- [ ] **Step 3: Add per-compiler counters**

Around line 1553 (after the existing `trace_clean/diverge/...` declarations), add:

```bash
  declare -A cycle_match cycle_drift cycle_empty cycle_emu_bug cycle_hw_bug cycle_compare_err cycle_no_data
  for compiler in "${compilers[@]}"; do
    cycle_match[$compiler]=0
    cycle_drift[$compiler]=0
    cycle_empty[$compiler]=0
    cycle_emu_bug[$compiler]=0
    cycle_hw_bug[$compiler]=0
    cycle_compare_err[$compiler]=0
    cycle_no_data[$compiler]=0
  done
  declare -a cycle_offenders=()   # "name/variant compiler tag" lines for the summary block
  declare -a cycle_empty_list=()
```

- [ ] **Step 4: Render the column**

Find the `# Trace columns ...` loop (around line 1666). After its closing `fi` (end of the `$has_trace` block), add:

```bash
    # Cycle drift column (per-compiler, variant-aware).
    if $has_cycle; then
      for compiler in "${compilers[@]}"; do
        local cyc="-"
        local tag=""
        if [[ -f "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.cycle.result" ]]; then
          cyc="$(< "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.cycle.result")"
          case "$cyc" in
            MATCH*)          tag="match";           cycle_match[$compiler]=$(( ${cycle_match[$compiler]} + 1 )) ;;
            DRIFT*)          tag="drift";           cycle_drift[$compiler]=$(( ${cycle_drift[$compiler]} + 1 )) ;;
            EMPTY)           tag="empty";           cycle_empty[$compiler]=$(( ${cycle_empty[$compiler]} + 1 )) ;;
            EMU_TRACE_BUG)   tag="emu-trace-bug";   cycle_emu_bug[$compiler]=$(( ${cycle_emu_bug[$compiler]} + 1 )) ;;
            HW_TRACE_BUG)    tag="hw-trace-bug";    cycle_hw_bug[$compiler]=$(( ${cycle_hw_bug[$compiler]} + 1 )) ;;
            COMPARE-ERR)     tag="compare-err";     cycle_compare_err[$compiler]=$(( ${cycle_compare_err[$compiler]} + 1 )) ;;
            NO_DATA|-)       tag="no-data";         cycle_no_data[$compiler]=$(( ${cycle_no_data[$compiler]} + 1 )) ;;
          esac
          if [[ "$tag" == "drift" || "$tag" == "emu-trace-bug" || "$tag" == "hw-trace-bug" || "$tag" == "compare-err" ]]; then
            cycle_offenders+=("  $display_name ($compiler): $cyc")
          elif [[ "$tag" == "empty" ]]; then
            cycle_empty_list+=("  $display_name ($compiler)")
          fi
        else
          cyc="-"
          cycle_no_data[$compiler]=$(( ${cycle_no_data[$compiler]} + 1 ))
        fi
        printf "  %-24s" "$cyc"
      done
    fi
```

- [ ] **Step 5: Append summary blocks after the main table**

Find the overall-summary printing block near the end of `print_report`. After the per-compiler trace summaries (look for where `trace_clean`/`trace_diverge` are printed — around line 1800+), add a new summary block:

```bash
  if $has_cycle; then
    echo ""
    echo "==========================================================================="
    echo "  CYCLE DRIFT (Phase E)"
    echo "==========================================================================="
    for compiler in "${compilers[@]}"; do
      printf "  %-8s  %d MATCH  %d DRIFT  %d EMPTY  %d EMU_TRACE_BUG  %d HW_TRACE_BUG  %d COMPARE-ERR  %d skipped\n" \
        "$compiler" \
        "${cycle_match[$compiler]}" \
        "${cycle_drift[$compiler]}" \
        "${cycle_empty[$compiler]}" \
        "${cycle_emu_bug[$compiler]}" \
        "${cycle_hw_bug[$compiler]}" \
        "${cycle_compare_err[$compiler]}" \
        "${cycle_no_data[$compiler]}"
    done

    if [[ ${#cycle_offenders[@]} -gt 0 ]]; then
      echo ""
      echo "  Offenders (DRIFT / *_TRACE_BUG / COMPARE-ERR):"
      for line in "${cycle_offenders[@]}"; do
        echo "$line"
      done
    fi

    if [[ ${#cycle_empty_list[@]} -gt 0 ]]; then
      echo ""
      echo "  Empty-trace tests (default event set insufficient — see Phase B Limitation 1):"
      for line in "${cycle_empty_list[@]}"; do
        echo "$line"
      done
    fi
  fi
```

- [ ] **Step 6: Smoke test**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --with-cycle-diff --chess-only -v '^vector_scalar_using_dma$' 2>&1 | tee /tmp/claude-1000/phase-e-task12.log
grep -E 'CYCLES|CYCLE DRIFT|MATCH|DRIFT' /tmp/claude-1000/phase-e-task12.log
```
Expected: header includes `Chess/CYCLES` column; data row shows `MATCH(...)` or `DRIFT(...)`; summary block lists per-compiler counts; no offenders if it's a match.

- [ ] **Step 7: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "$(cat <<'EOF'
feat(bridge-test): CYCLES column + summary for --with-cycle-diff

Adds a per-compiler CYCLES column rendering the cycle.result tag, plus
an end-of-run block listing offenders (DRIFT / *_TRACE_BUG / COMPARE-ERR)
and EMPTY tests (default event set didn't fire events).

Generated using Claude Code.
EOF
)"
```

---

## Task 13: `scripts/show-cycle-drift.sh` triage reporter

**Files:**
- Create: `scripts/show-cycle-drift.sh`

**Purpose:** A small standalone script that reads the latest results directory and prints cycle-diff results sorted by |log(EMU/HW)| (i.e., "how far off is this one?"), for manual inspection.

- [ ] **Step 1: Write the script**

```bash
cat > /home/triple/npu-work/xdna-emu/scripts/show-cycle-drift.sh <<'SCRIPT'
#!/usr/bin/env bash
# show-cycle-drift.sh — sort bridge-test cycle-drift results by severity.
#
# Usage:
#   scripts/show-cycle-drift.sh [--results DIR] [--top N]
#
# Default: read build/bridge-test-results/latest/, print all results.
# --top N limits to the N highest |log(ratio)| entries.

set -euo pipefail

RESULTS="build/bridge-test-results/latest"
TOP=0  # 0 = unlimited

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RESULTS="$2"; shift 2 ;;
    --top)     TOP="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$RESULTS" ]]; then
  echo "Results dir not found: $RESULTS" >&2
  exit 1
fi

# Emit: |log(ratio)|  tag  test.variant.compiler
awk_script='
function abs(x) { return x < 0 ? -x : x }
function logr(r)  { if (r <= 0) return 999; return log(r)/log(10) }
BEGIN { }
{
  # line format: "<tag> <file>"
  tag=$1
  file=$2
  # Extract test+variant+compiler from filename:
  #   <safe>.<compiler>.cycle.result           (single-variant)
  #   <safe>.<variant>.<compiler>.cycle.result (multi-variant)
  name=file
  sub(/.*\//, "", name)
  sub(/\.cycle\.result$/, "", name)
  # Parse ratio if present
  ratio=0
  if (match(tag, /ratio=[0-9.]+/)) {
    r=substr(tag, RSTART+6, RLENGTH-6)
    ratio=r+0
  } else if (match(tag, /MATCH\(([0-9.]+)\)/, m)) {
    ratio=m[1]+0
  }
  sev=abs(logr(ratio))
  # Bugs/errors get an artificial high severity so they float up
  if (tag ~ /TRACE_BUG|COMPARE-ERR/) sev=10
  printf("%.4f  %-48s  %s\n", sev, name, tag)
}
'

mapfile -t lines < <(
  find "$RESULTS" -maxdepth 1 -name '*.cycle.result' -print 2>/dev/null |
  while IFS= read -r f; do
    printf "%s %s\n" "$(tr -d '[:space:]' < "$f")" "$f"
  done |
  awk "$awk_script" |
  sort -rn
)

if [[ ${#lines[@]} -eq 0 ]]; then
  echo "No .cycle.result files found under $RESULTS"
  exit 0
fi

printf "%-8s  %-48s  %s\n" "|log|" "TEST" "RESULT"
printf "%-8s  %-48s  %s\n" "--------" "------------------------------------------------" "------"

count=0
for line in "${lines[@]}"; do
  echo "$line"
  count=$(( count + 1 ))
  if [[ "$TOP" -gt 0 && "$count" -ge "$TOP" ]]; then
    break
  fi
done
SCRIPT
chmod +x /home/triple/npu-work/xdna-emu/scripts/show-cycle-drift.sh
```

- [ ] **Step 2: Smoke test**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/show-cycle-drift.sh --top 5
./scripts/show-cycle-drift.sh --help
```
Expected: first command prints a table with at most 5 rows, sorted with the worst (|log|) first; `--help` prints the usage header.

- [ ] **Step 3: Commit**

```bash
git add scripts/show-cycle-drift.sh
git commit -m "$(cat <<'EOF'
feat(scripts): show-cycle-drift.sh — sort cycle-diff results by severity

Small triage helper that reads the latest results dir, sorts by
|log(EMU/HW)|, and bumps *_TRACE_BUG / COMPARE-ERR to the top.

Generated using Claude Code.
EOF
)"
```

---

## Task 14: Batch validation + documentation

**Files:**
- Create: `docs/superpowers/plans/2026-04-23-phase-e-validation.md`
- Modify: `docs/superpowers/plans/2026-04-22-cycle-budget-testing.md`

**Purpose:** Run the full Phase E pipeline against the 7-test batch Phase B validated. Record the results as a validation doc (mirrors Phase B's validation record). Update the parent cycle-budget plan to note Phase C and Phase D.3 are superseded.

- [ ] **Step 1: Run the batch**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --with-cycle-diff --no-timeout \
  -v '^(vector_scalar_using_dma|add_one_using_dma|add_one_objFifo|cascade_flows|add_blockwrite|column_specific|ctrl_packet_reconfig)$' \
  2>&1 | tee /tmp/claude-1000/phase-e-batch.log
```
Expect ~15-30 minutes wall clock.

Collect results:
```bash
ls build/bridge-test-results/latest/*.cycle.result 2>/dev/null | sort
for f in build/bridge-test-results/latest/*.cycle.result; do
  echo "$(basename "$f" .cycle.result): $(< "$f")"
done
./scripts/show-cycle-drift.sh
```

- [ ] **Step 2: Write the validation doc**

Create `docs/superpowers/plans/2026-04-23-phase-e-validation.md` following the shape of `docs/superpowers/plans/2026-04-22-phase-b-trace-cycle-capture-validation.md`:

```markdown
# Phase E Validation Results (2026-04-23)

End-to-end validation of the trace-diff-based cycle budget pipeline
(`--with-cycle-diff`) on the Phase B 7-test batch.

## Setup

- Branch: `dev`
- Commits: <fill in the Phase E commit SHAs from `git log --oneline master..dev`>
- Invocation:
  ```
  ./scripts/emu-bridge-test.sh --with-cycle-diff --no-timeout \
    -v '^(vector_scalar_using_dma|add_one_using_dma|add_one_objFifo|cascade_flows|add_blockwrite|column_specific|ctrl_packet_reconfig)$'
  ```
- Classifier defaults: ratio bounds [0.5, 2.0], trace-compare DIVERGE_THRESHOLD=10 cycles
- SECONDS_PER_CYCLE: 1e-3 (un-calibrated starting constant)

## Results

| Test | Compiler | HW cycles | EMU cycles | Ratio | Compare diverge | Classification |
|------|----------|-----------|------------|-------|-----------------|----------------|
| <fill in> | chess/peano | ... | ... | ... | ... | MATCH/DRIFT/EMPTY/*_TRACE_BUG |

(One row per (test, compiler) that ran to PASS.)

## Aggregate

<fill in per-compiler counts: N MATCH, N DRIFT, N EMPTY, N *_TRACE_BUG, N COMPARE-ERR, N skipped>

## Surfacing sanity check

- `show-cycle-drift.sh --top 5` output: <paste>
- Worst offender's trace-compare report tail: <paste or reference path>

## Observations

<Notes on anything unexpected — e.g., cascade_flows EMU/HW ratio far from 1.0, or an EMU_TRACE_BUG that wasn't anticipated. If anything here is a real bug rather than an artifact, open a follow-up issue and link it.>

## Tuning opportunities surfaced

<Notes on whether the default [0.5, 2.0] is too loose for anything observed, whether SECONDS_PER_CYCLE = 1e-3 is obviously wrong given observed wall-clock times, whether any test needs a cycle-drift-overrides.txt entry.>

## Verdict

<MATCH/REGRESSION: Phase E pipeline is ready for normal bridge-test use.>
or
<REGRESSION: <describe>; follow-up tasks opened.>
```

- [ ] **Step 3: Update the parent cycle-budget plan**

Open `docs/superpowers/plans/2026-04-22-cycle-budget-testing.md`. Near the bottom (after any existing Phase B pivot note), append a one-line pointer:

```markdown

---

## Post-Phase B status update (2026-04-23)

Phase C ("cycle budget enforcement") and Phase D.3 ("HW integration
spot-check") are superseded by Phase E — see
[`2026-04-23-phase-e-trace-diff-cycle-budget.md`](2026-04-23-phase-e-trace-diff-cycle-budget.md)
for the replacement design and
[`2026-04-23-phase-e-validation.md`](2026-04-23-phase-e-validation.md)
for empirical results.
```

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/superpowers/plans/2026-04-23-phase-e-validation.md \
        docs/superpowers/plans/2026-04-22-cycle-budget-testing.md
git commit -m "$(cat <<'EOF'
docs(phase-e): validation results for --with-cycle-diff on Phase B batch

Records per-test classifications, aggregate counts, and any tuning
opportunities surfaced. Updates the parent cycle-budget plan to point
at Phase E as the replacement for Phase C + D.3.

Generated using Claude Code.
EOF
)"
```

---

## Self-Review

Skimmed against spec §Scope (items 1-9), §Architecture (five components), §Classification rules, §Bridge flag surface, §Naming conventions, §Dual-bound EMU timing, §Trace-related compile failures, §Error handling, §Testing approach. Gaps found during review:

- **Spec item 1 (validation)**: covered by Task 1. ✓
- **Spec item 2 (EMU capture)**: covered by Tasks 4-5. ✓
- **Spec item 3 (cycle extraction)**: folded into the same `_run_trace_cycles_pipeline` (Task 4) — trace-to-cycles runs on both sides by passing the same helper. ✓
- **Spec item 4 (comparison)**: Task 6. ✓
- **Spec item 5 (`--with-cycle-diff` flag)**: Task 3. ✓
- **Spec item 6 (dual-bound timing)**: Task 8. ✓
- **Spec item 7 (COMPILE-FAIL(traced))**: Task 9. ✓
- **Spec item 8 (show-cycle-drift.sh)**: Task 13. ✓
- **Spec item 9 (D.3 folded into Task 1)**: Task 1 is the first end-to-end run on `vector_scalar_using_dma`. ✓
- **Classification rules** (MATCH / DRIFT / EMPTY / EMU_TRACE_BUG / HW_TRACE_BUG): Task 7. ✓
- **Tolerance defaults + overrides**: Task 11. ✓
- **Compile-fail surfacing + incompat list**: Tasks 9 + 10. ✓
- **Summary counts**: Task 12. ✓

Placeholder scan: no "TBD" / "write tests for the above" / "similar to Task N" patterns. All steps include concrete code or exact commands. The Task 14 validation doc template has fill-in placeholders because the values are empirical — acceptable per plan style.

Type consistency: `_run_trace_cycles_pipeline` signature unchanged across Tasks 4 → 5; `_classify_cycle_diff` naming consistent; `.cycle.result` filename used identically in Tasks 7, 12, 13; `trace_hw.<variant>.bin` / `trace_emu.<variant>.bin` naming consistent across Tasks 2, 4, 5, 6, 7. `FAIL_TRACED` result file content consistent between Task 9 writer and reader.

One minor inconsistency caught and fixed inline: early draft had Task 7 logging `no-data` case for missing HW bin even when EMU bin was present — corrected to route to `HW_TRACE_BUG` per spec's asymmetry rule. Also corrected Task 7's edge case where the `--help` smoke test would invoke the classifier directly; replaced with a softer "acceptable to verify during Task 14" note since the exported-function manual harness is awkward in practice.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-23-phase-e-trace-diff-cycle-budget.md`. Two execution options:

1. **Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session with checkpoints.

Given we're in auto mode, I'll proceed with **Subagent-Driven execution** unless you redirect.
