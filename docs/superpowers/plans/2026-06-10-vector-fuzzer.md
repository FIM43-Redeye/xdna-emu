# Vector Differential Fuzzer (Phase D, #112) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the typed-pipeline vector differential fuzzer with coverage ledger per `docs/superpowers/specs/2026-06-10-vector-fuzzer-design.md`, then run the Phoenix harvest campaign.

**Architecture:** New `src/fuzzer/vector/` module set generates coverage-driven chains of 8-16 vector ops as `aie::` intrinsic C++ (per-stage stores into output slices), compiled via the existing Peano fuzz pipeline (`fuzz_template.py` unchanged), executed HW vs EMU through the existing runner. Ledger credit is execution-derived from a thread-local recorder in `VectorAlu::execute`, gated on whole-chain silicon match. Banked outputs/traces form the post-Phoenix replay corpus.

**Tech Stack:** Rust (xdna-emu, no new deps), aie_api C++ intrinsics, Peano (Chess RAM rule does NOT apply), existing fuzz template + bridge runner.

**Worker rules (load-bearing):** NEVER pipe builds/tests through head/tail/grep — bare or redirect-to-file + Read. One build at a time. `cargo test --lib` after each task; a regression blocks the task. Never put persistent work in /tmp. No emoji. Commit trailer exactly `Generated using Claude Code.` HW tasks: rebuild `cargo build --release --features tooling` first; never run two HW suites concurrently. Vector fuzzer compiles with Peano: jobs are unrestricted (Chess -j4 rule explicitly does not apply).

---

## File map

Create:
- `src/fuzzer/vector/mod.rs` — module decls
- `src/fuzzer/vector/table.rs` — typed op table (VecType, OpEntry, TABLE)
- `src/fuzzer/vector/chain.rs` — chain AST + coverage keys
- `src/fuzzer/vector/ledger.rs` — persistent ledger + report
- `src/fuzzer/vector/gen.rs` — coverage-first generation + edge input pool
- `src/fuzzer/vector/lower.rs` — C++ emission
- `src/fuzzer/vector/runner.rs` — compile/execute/compare/credit loop + replay
- `src/interpreter/execute/fuzz_recorder.rs` — thread-local executed-key recorder

Modify:
- `src/fuzzer/mod.rs` (+1 line), `src/fuzzer/cli.rs` (vector flags), `src/main.rs` (dispatch)
- `src/fuzzer/runner.rs` — make `ToolPaths`, `compile_one`-equivalents reusable (`pub(crate)`)
- `src/testing/test_cpp_parser.rs` — `InputPattern::Bytes`
- `src/interpreter/execute/vector_dispatch.rs` — record hook (2 lines)

---

### Task 1: Spike — Peano intrinsic reach probe

**Files:** Create `build/experiments/vector-fuzzer-spike/probe.cc` (throwaway), findings comment block in `src/fuzzer/vector/table.rs` later (Task 2 consumes the findings; save them to `build/experiments/vector-fuzzer-spike/FINDINGS.md`).

- [ ] **Step 1: Write a probe kernel exercising one intrinsic per target family**

```cpp
// probe.cc -- Peano reach probe. Each block stores its result so nothing folds.
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void fuzz_kernel(int32_t* __restrict in, int32_t* __restrict out) {
  aie::vector<int32_t,16> a = aie::load_v<16>(in);
  aie::vector<int32_t,16> b = aie::load_v<16>(in + 16);
  int s = 0;
  aie::store_v(out + 16*s++, aie::add(a, b));
  aie::store_v(out + 16*s++, aie::sub(a, b));
  aie::store_v(out + 16*s++, aie::min(a, b));
  aie::store_v(out + 16*s++, aie::max(a, b));
  aie::store_v(out + 16*s++, aie::neg(a));
  aie::store_v(out + 16*s++, aie::abs(a));
  aie::store_v(out + 16*s++, aie::band(a, b));
  aie::store_v(out + 16*s++, aie::bor(a, b));
  aie::store_v(out + 16*s++, aie::bxor(a, b));
  aie::store_v(out + 16*s++, aie::bneg(a));
  aie::store_v(out + 16*s++, aie::downshift(a, 3));
  aie::store_v(out + 16*s++, aie::upshift(a, 2));
  auto m = aie::lt(a, b);
  aie::store_v(out + 16*s++, aie::select(a, b, m));
  auto m2 = aie::ge(a, b); auto m3 = aie::eq(a, b);
  aie::store_v(out + 16*s++, aie::select(b, a, m2));
  aie::store_v(out + 16*s++, aie::select(a, b, m3));
  aie::store_v(out + 16*s++, aie::broadcast<int32_t,16>(in[0]));
  aie::store_v(out + 16*s++, aie::shuffle_up(a, 1));
  aie::store_v(out + 16*s++, aie::shuffle_down(b, 2));
  aie::store_v(out + 16*s++, ::shuffle(a, b, 28)); // raw VSHUFFLE mode 28 (T32_4x4)
  aie::store_v(out + 16*s++, aie::max_red(a) == 0 ? a : b); // reduction reach
  aie::vector<int16_t,32> p = aie::pack(aie::concat(a, b));
  aie::store_v((int16_t*)(out + 16*s), p); s++;
  aie::vector<int32_t,16> u = aie::unpack(p).extract<16>(0);
  aie::store_v(out + 16*s++, u);
}
```

- [ ] **Step 2: Compile through the actual fuzz pipeline** (binary debug build is fine):
`cargo build --features tooling` then run a 1-iteration scalar fuzz to create env, then directly:
```bash
cd /home/triple/npu-work/xdna-emu && mkdir -p build/experiments/vector-fuzzer-spike
$PEANO/bin/clang++ -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -I$MLIR_AIE/include -c probe.cc -o probe.o > compile.log 2>&1
```
where `$PEANO=/home/triple/npu-work/llvm-aie/install`, `$MLIR_AIE=/home/triple/npu-work/mlir-aie/install`. If any line fails, comment it out, note it in FINDINGS.md, retry until clean.

- [ ] **Step 3: Disassemble and enumerate emitted vector opcodes**:
`$MLIR_AIE/bin/llvm-objdump -d probe.o > probe.dis` then Read it; record per-block opcode (VADD/VSUB/VMIN/VMAX/VBAND/VSEL/VSHUFFLE/VBCST/VSHIFT...) in FINDINGS.md.

- [ ] **Step 4: Document.** FINDINGS.md lists reachable families + intrinsic spellings + unreachable ones with errors. This fixes Task 2's TABLE contents (table entries below marked `// spike-verified` must match findings — DROP entries the spike disproves, ADD spellings it corrects).

- [ ] **Step 5: Commit** FINDINGS.md only (probe.cc lives in experiments dir; do not commit).

### Task 2: Typed op table

**Files:** Create `src/fuzzer/vector/mod.rs`, `src/fuzzer/vector/table.rs`. Modify `src/fuzzer/mod.rs` (add `pub mod vector;` next to existing decls).

- [ ] **Step 1: Failing tests** (in `table.rs` `#[cfg(test)]`): table is non-empty; every entry's types are byte-width sane (vbytes==64 except `HalfW` 32); every (entry, mode) emit produces non-empty unique key; bf16 entries reference no int types.
- [ ] **Step 2: Implement** (adjust entries per Task 1 findings):

```rust
//! Typed vector-op table: every fuzz stage instantiates one entry.
//! Spike-verified intrinsics only (build/experiments/vector-fuzzer-spike/FINDINGS.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecType { I8x64, I16x32, I32x16, Bf16x32, I16x16, I32x8 } // last two = 256-bit half-width

impl VecType {
    pub fn ctype(self) -> &'static str { match self {
        VecType::I8x64 => "int8_t", VecType::I16x32 | VecType::I16x16 => "int16_t",
        VecType::I32x16 | VecType::I32x8 => "int32_t", VecType::Bf16x32 => "bfloat16" } }
    pub fn lanes(self) -> usize { match self {
        VecType::I8x64 => 64, VecType::I16x32 => 32, VecType::I32x16 => 16,
        VecType::Bf16x32 => 32, VecType::I16x16 => 16, VecType::I32x8 => 8 } }
    pub fn bytes(self) -> usize { match self { VecType::I16x16 | VecType::I32x8 => 32, _ => 64 } }
    pub fn is_float(self) -> bool { matches!(self, VecType::Bf16x32) }
}

pub struct OpEntry {
    pub name: &'static str,
    pub in_types: &'static [VecType],   // 1 or 2 operands
    pub out_type: VecType,
    pub modes: u8,                      // 1 = no mode dim; shuffle = 48
    /// Emit C++ expr. args: operand exprs, mode.
    pub emit: fn(&[String], u8) -> String,
}

macro_rules! bin { ($f:literal) => { |a, _| format!(concat!("aie::", $f, "({}, {})"), a[0], a[1]) } }
macro_rules! una { ($f:literal) => { |a, _| format!(concat!("aie::", $f, "({})"), a[0]) } }

fn for_int(name: &'static str, t: VecType, ar2: bool, e: fn(&[String], u8) -> String) -> OpEntry {
    OpEntry { name, in_types: if ar2 { Box::leak(vec![t, t].into_boxed_slice()) } else { Box::leak(vec![t].into_boxed_slice()) }, out_type: t, modes: 1, emit: e }
}

pub fn table() -> Vec<OpEntry> {
    let mut t = Vec::new();
    for vt in [VecType::I8x64, VecType::I16x32, VecType::I32x16] {
        for (n, e) in [("add", bin!("add") as fn(&[String], u8) -> String), ("sub", bin!("sub")),
                       ("min", bin!("min")), ("max", bin!("max")),
                       ("band", bin!("band")), ("bor", bin!("bor")), ("bxor", bin!("bxor"))] {
            t.push(for_int(n, vt, true, e));
        }
        for (n, e) in [("neg", una!("neg") as fn(&[String], u8) -> String), ("abs", una!("abs")), ("bneg", una!("bneg"))] {
            t.push(for_int(n, vt, false, e));
        }
        t.push(OpEntry { name: "downshift", in_types: Box::leak(vec![vt].into_boxed_slice()), out_type: vt, modes: 8,
            emit: |a, m| format!("aie::downshift({}, {})", a[0], m) });
        t.push(OpEntry { name: "sel_lt", in_types: Box::leak(vec![vt, vt].into_boxed_slice()), out_type: vt, modes: 1,
            emit: |a, _| format!("aie::select({}, {}, aie::lt({}, {}))", a[0], a[1], a[0], a[1]) });
        t.push(OpEntry { name: "sel_ge", in_types: Box::leak(vec![vt, vt].into_boxed_slice()), out_type: vt, modes: 1,
            emit: |a, _| format!("aie::select({}, {}, aie::ge({}, {}))", a[0], a[1], a[0], a[1]) });
        t.push(OpEntry { name: "sel_eq", in_types: Box::leak(vec![vt, vt].into_boxed_slice()), out_type: vt, modes: 1,
            emit: |a, _| format!("aie::select({}, {}, aie::eq({}, {}))", a[0], a[1], a[0], a[1]) });
        t.push(OpEntry { name: "bcast", in_types: Box::leak(vec![vt].into_boxed_slice()), out_type: vt, modes: 1,
            emit: |a, _| format!("aie::broadcast<{CT}, {L}>({}.get(0))", a[0]) }); // fix CT/L at build time per vt (see Step 2 note)
        t.push(OpEntry { name: "shup", in_types: Box::leak(vec![vt].into_boxed_slice()), out_type: vt, modes: 4,
            emit: |a, m| format!("aie::shuffle_up({}, {})", a[0], m as usize + 1) });
    }
    // Raw VSHUFFLE: 48 modes (verify exact valid range vs ShuffleMode::from_mode -- 0..=47), i32x16 both operands.
    t.push(OpEntry { name: "shuffle", in_types: Box::leak(vec![VecType::I32x16, VecType::I32x16].into_boxed_slice()),
        out_type: VecType::I32x16, modes: 48, emit: |a, m| format!("::shuffle({}, {}, {})", a[0], a[1], m) });
    // Couplers (count coverage too): pack 32->16, 16->8; unpack 8->16, 16->32 (half-width handled in chain gen).
    t.push(OpEntry { name: "pack16", in_types: Box::leak(vec![VecType::I32x16].into_boxed_slice()), out_type: VecType::I16x16, modes: 1, emit: una!("pack") });
    t.push(OpEntry { name: "pack8",  in_types: Box::leak(vec![VecType::I16x32].into_boxed_slice()), out_type: VecType::I8x64, modes: 1,
        emit: |a, _| format!("aie::concat(aie::pack({0}), aie::pack({0}))", a[0]) });
    t.push(OpEntry { name: "unpack16", in_types: Box::leak(vec![VecType::I8x64].into_boxed_slice()), out_type: VecType::I16x32,
        modes: 1, emit: |a, _| format!("aie::unpack({}.extract<32>(0))", a[0]) });
    // bf16 family (no int<->bf16 couplers; bf16 chains are bf16-only end to end):
    for (n, e) in [("add", bin!("add") as fn(&[String], u8) -> String), ("sub", bin!("sub")), ("min", bin!("min")), ("max", bin!("max"))] {
        t.push(OpEntry { name: n, in_types: Box::leak(vec![VecType::Bf16x32, VecType::Bf16x32].into_boxed_slice()), out_type: VecType::Bf16x32, modes: 1, emit: e });
    }
    t.push(OpEntry { name: "neg", in_types: Box::leak(vec![VecType::Bf16x32].into_boxed_slice()), out_type: VecType::Bf16x32, modes: 1, emit: una!("neg") });
    t
}
```
Note: where a template needs the concrete C type/lane count (`bcast`), generate per-vt closures with a small helper instead of placeholder `{CT}/{L}` — three explicit entries are fine. Apply Task 1 findings: drop unreached entries, fix spellings, adjust `shuffle` mode count to spike evidence.
- [ ] **Step 3:** `cargo test --lib` green. **Step 4: Commit** `vector fuzzer: typed op table (spike-verified intrinsics)`.

### Task 3: Coverage ledger

**Files:** Create `src/fuzzer/vector/ledger.rs`. Key = `name/vt/mode` string. Plain JSON map, no serde dep — match existing hand-rolled JSON style (see `tools/golden` writers; if serde already in Cargo.toml, use it).

- [ ] **Step 1: Failing tests:** new ledger empty; credit increments; persists round-trip; uncovered(target=10) lists keys below 10; full-key universe enumerates `sum(entry.modes)` keys; mark_divergent excludes from credit; `report()` string contains uncovered + divergent counts.
- [ ] **Step 2: Implement** `Ledger { hits: HashMap<String, u32>, divergent: HashSet<String>, timing_fail: HashSet<String>, unreachable: HashMap<String, String> }` with `universe()` from `table()`, `credit_keys(keys)`, `mark_divergent(key)`, `mark_timing(key)`, `mark_unreachable(key, why)`, `uncovered(n)`, `save(path)/load(path)`, `report(n)`.
- [ ] **Step 3:** `cargo test --lib`. **Step 4: Commit.**

### Task 4: Chain AST + coverage-first generation + edge input pool

**Files:** Create `src/fuzzer/vector/chain.rs`, `src/fuzzer/vector/gen.rs`. Use scalar fuzzer's xorshift64 pattern (copy struct).

- [ ] **Step 1: Failing tests:** 1000 seeds generate type-legal chains of 8-16 stages (each stage's in_types match preceding out_type or pool); chain targets a requested key (first stage uses key's entry+mode); pool bytes deterministic per seed; pool contains denormals (byte pattern 00 80 in bf16 slices) and NaN/extremes per class table; coupler insertion bridges I32x16 to I8x64; bf16 chains stay bf16-only.
- [ ] **Step 2: Implement** `Chain { seed, target_key, stages: Vec<Stage { entry_idx, mode, second_pool_slot }>, pool: Vec<u8> }`. Generation: pick target key, then random walk legal next-stages (coupler when types mismatch), 8-16 deep; every 2-op stage's second operand = next pool slot. Pool slot = 64B from class-weighted bytes: 25% denorm, 15% NaN/Inf, 20% sign extremes, 15% zero, 25% uniform.
- [ ] **Step 3:** test. **Step 4: Commit.**

### Task 5: C++ lowering

**Files:** Create `src/fuzzer/vector/lower.rs`.

- [ ] **Step 1: Failing tests:** balanced braces; contains `#include <aie_api/aie.hpp>`; per-stage `aie::store_v` count == stage count; half-width (32B) stages store correct cast; out slices stride 16 i32 words.
- [ ] **Step 2: Implement** signature `extern "C" void fuzz_kernel(int32_t* __restrict in, int32_t* __restrict out)`. Stage k: `auto vK = <entry.emit(operands, mode)>; aie::store_v(({ctype}*)(out + 16*K), vK);` first stage loads `aie::load_v<L>(({ctype})in)`, pool slots load at `in + 16*slot`. Pool bytes are NOT emitted into C++ — they come from `InputPattern::Bytes` (Task 6).
- [ ] **Step 3:** test. **Step 4: Commit.**

### Task 6: `InputPattern::Bytes`

**Files:** Modify `src/testing/test_cpp_parser.rs` (`InputPattern` enum at :133, `generate_input_data` at :735).

- [ ] **Step 1: Failing test:** Bytes pattern round-trips literally; truncates/zero-pads to buffer size.
- [ ] **Step 2:** add `Bytes(Vec<u8>)` variant + arm: copy `min(len)` bytes. Fix exhaustive matches (`cargo build` will list them).
- [ ] **Step 3:** `cargo test --lib`. **Step 4: Commit.**

### Task 7: Execution recorder + dispatch hook

**Files:** Create `src/interpreter/execute/fuzz_recorder.rs`; modify `vector_dispatch.rs:18` (`execute`).

- [ ] **Step 1: Failing test:** with recorder armed, running a unit vector op deposits its key; not armed = no cost; mode for shuffle = mode imm; for Srs/Pack/Convert = `(crSat<<4)|crRnd` from `ctx.srs_config`.
- [ ] **Step 2:** thread-local `RefCell<Option<Vec<String>>>`, `arm()/take()`, `record(semantic, et, mode)`. In `VectorAlu::execute` after the early-outs (`vector_dispatch.rs:49`), call `fuzz_recorder::record(semantic, et, mode_of(semantic, op, ctx))`. EMU runs single-threaded per case via `run_emulator`; arm/take around it. Recorder key must STRING-MATCH ledger keys (ledger maps table name -> semantic via map in table.rs; chain knows its key list, runner intersects).
- [ ] **Step 3:** `cargo test --lib` (3340+). **Step 4: Commit.**

### Task 8: Vector runner + CLI + banking + replay

**Files:** Create `src/fuzzer/vector/runner.rs`; modify `src/fuzzer/runner.rs` (make `ToolPaths`, compile env, `run_emulator`, NPU helpers `pub(crate)`), `cli.rs` (`--vector`, `--target-hits`, `--report`, `--replay <dir>`), `main.rs`.

- [ ] Reuse compile pipeline verbatim (dtype i32, size = pool words / out words via `--size`); buffer spec: input `InputPattern::Bytes(chain.pool)`, in/out sized to chain. Slice compare: first differing 64B slice -> stage -> key. PASS -> `credit_keys(recorder ∩ chain keys)`; FAIL -> bank dir `~/npu-work/experiments/phoenix-survival/vector/seed_N/` (kernel cc, chain JSON, npu_output.bin, npu_trace.bin) + mark divergent; ledger saved every batch. `--replay <dir>` = EMU vs banked npu_output.bin. Trace: always on; timing divergence -> `mark_timing` only (functional credit independent). Unit tests: slice localization vs synthetic mismatch; vacuous (zero pool) detection. HW smoke comes next task.
- [ ] `cargo test --lib`; compile-clean test 200 seeds `--no-hw` compile-only (debug, behind `#[ignore]`). Commit.

### Task 9: 50-case HW smoke + ledger verify

- [ ] `cargo build --release --features tooling` (also `cargo build -p xdna-emu-ffi` if EMU loads .so). `./target/release/xdna-emu fuzz --vector --hw --iterations 50 --jobs 16 --seed 1`, log to file. Expect pass>0, ledger gains, divergences banked not aborted. Commit ledger + findings note.

### Task 10: Campaign

- [ ] Loop batches of 500 until `--report` shows all reachable keys >= 10 or only DIVERGENT/UNREACHABLE remain; triage divergences as they bank. Operate per HW-suite serialization rules. Outside plan scope; coordinate with Maya.

---

## Self-review notes
- Spec coverage: kernel shape (T2/T4/T5), ledger (T3), execution credit (T7), runner/banking/replay/timing (T8), error handling (UNREACHABLE in T2/T8), edge pool (T4), smoke + campaign (T9/T10). f32/SRS-mode credit comes from couplers + DIVERGENT-aware credit.
- Box::leak slices: fine, table built once into `OnceLock`. Executors may instead construct per-vt static arrays.
- Stage stores serialize chains; acceptable per spec (timing leg compares like-for-like).
