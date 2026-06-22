# program_path: Through-Core Dataflow Edges — Implementation Plan (v2, post-spike)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **v2 (2026-06-22):** rewritten after the standalone P0 feasibility spike and the adversarial plan review. The spike proved static recovery TRACTABLE and pinned the exact technique; the lock half is idiom-recognition, the buffer half is a **stack-aware abstract-value pass**, ordering is a **delay-slot-aware linear-PC proxy**. v1's "read the immediate operand" model was wrong (Chess emits out-of-line lock calls); this version is built on the verified decoder output.

**Goal:** Derive the compute-tile relay `S2MM(in1) → core → MM2S(out1)` — which config-path Tier E structurally cannot reach — by statically parsing the Chess-compiled core ELF's lock-call idiom + buffer accesses, emit it as a sound `CoreLockRelay` route edge and a distinct `program_path` engine fact, and validate it against the emulator's runtime core-lock + buffer-touch enactment.

**Architecture:** A new Rust static-analysis module decodes the compute core's program (via the emulator's own `InstructionDecoder`, no execution) and recovers three things: (1) which locks the core acquires/releases, by recognizing the `JL → ACQ/REL`-helper idiom and resolving the lock-id register from the call's delay-slot window; (2) which buffers the core loads-from / stores-to, via a stack-aware abstract-value pass (buffer pointers flow `MOVXM → reg → stack spill → reload → p-reg`); (3) that acquire/load/store/release occur in the right order. A new `EdgeKind::CoreLockRelay` builder emits an edge only when lock-pairing ∩ buffer-contact ∩ ordering all hold. The edge rides the existing shared route graph; the Python engine gains a real `program_path` predicate. Validation is an E4-style superset check against two independent runtime witnesses (core lock handoffs + buffer touches). Ceiling X: trace/HW validation deferred to the separate experimenter-loop plan.

**Tech Stack:** Rust (route graph, decoder, interpreter), Python 3.13 (inference engine + config_extract generator), build-derived llvm-aie TableGen decoder.

**Design spec:** `docs/superpowers/specs/2026-06-21-program-path-through-core-design.md` (survived two adversarial design reviews). The §3 derivation rule and §1 ceiling/non-goals are authoritative; this plan implements them with the spike-verified technique.

## Global Constraints

- **Claim semantic:** the edge/fact claims *structural data-contact under producer/consumer lock ordering* ("the core had the opportunity to relay these bytes"), **NEVER value-dependence**. No cite, doc, or comment may say buffer-touch implies dataflow/value-flow. (Spec §1, §3.)
- **Soundness is intersection:** an edge requires lock-pairing AND buffer-contact AND ordering. Any criterion alone is insufficient. Unresolvable lock-id register, unresolvable buffer pointer, or unrecoverable ordering ⇒ **emit no edge (safe false-negative)** — never guess.
- **Orientation:** `src = S2MM master DMA port (writer/releaser)` → `dst = MM2S slave DMA port (reader/acquirer)`. Back-pressure (reverse) is never emitted. (Matches Tier E E2/E3.)
- **program_path is a real predicate**, separately queryable from `config_path`. (Spec §5.)
- **Coverage is honestly narrow** (objectFIFO-passthrough / simple-elementwise, **Chess-compiled**). Comments/plan must not claim general or compiler-agnostic through-core coverage. Peano coverage is documented follow-up.
- **DERIVE FROM THE TOOLCHAIN:** lock/MOVXM recognition is via the decoder's `SemanticOp`, never hardcoded opcodes. Helper addresses are found by **semantic scan** (the lone `LockAcquire`/`LockRelease` bundles), never hardcoded to 816/848.
- **Commit trailer** (no emoji), every commit:
  ```
  Generated using Claude Code.
  Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
  ```
- **Rust:** never pipe `cargo` through grep/head/tail — redirect to a file and Read it. Run `cargo test --lib` after Rust changes.
- **Python tests:** `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest <files> -v`.
- **add_one fixtures:** xclbin `/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin`; standalone core ELF `.../chess/aie_arch.mlir.prj/main_core_0_2.elf`. Guard tests to skip cleanly if absent.

## Spike-verified facts (the ground the code stands on)

Decoder output (`cargo run --bin spike_p0`, verified) — exact operand shapes to code against:
- `ACQ`/`REL` helpers: `SemanticOp::LockAcquire`/`LockRelease`, `sources=[ScalarReg(0), ScalarReg(1)]`, `dest=None`. add_one: ACQ@0x330, REL@0x350.
- Lock call: `SemanticOp::Call`, `sources=[Immediate(<target>)]` (target statically readable). (Indirect calls `Call sources=[PointerReg(n)]` exist — ignore; only `Immediate`-target calls to helper addrs are lock ops.)
- Lock-id set: `SemanticOp::Copy`, `dest=Some(ScalarReg(0))`, `sources=[Immediate(N)]`, N ∈ 48..=51. **local lock id = N − 48.**
- MOVXM: `SemanticOp::Copy`, `dest=Some(PointerReg(n) | ModifierReg(8) | ScalarReg(n))`, `sources=[Immediate(imm)]`. (`Immediate` is **i32**.)
- Buffer load (post-incr): `SemanticOp::Load`, `dest=Some(ScalarReg(n))`, `sources=[PointerReg(p)]`.
- Buffer store (post-incr): `SemanticOp::Store`, `dest=Some(PointerReg(p))`, `sources=[ScalarReg(val), PointerReg(p)]`.
- Stack spill/reload (NOT a buffer contact): `Store ... sources=[reg, Memory{base:255, offset}]` / `Load dest sources=[Memory{base:255, offset}]`. **`PointerReg(255)` = `sp`; `ModifierReg(8)` = `dn0`** — exclude both from buffer-pointer candidates.
- Recovered lock sequence (the test oracle): acquires `{1,2}`, releases `{0,3}` (body unrolled ×2). Buffer flow: in1 via `0x70400/0x70420`, out1 via `0x70440/0x70460`; reconcile `addr & 0xFFFF` (in1=0x400/0x420, out1=0x440/0x460). **Stack-end `_sp_end_DM_stack` aliases in1_buff_0 at 0x70400 → classify by EXACT masked base, never a range.**
- `SlotOp` fields: `semantic: Option<SemanticOp>`, `sources: SmallVec<[Operand;4]>`, **`dest: Option<Operand>`** (NOT `destinations`), `extra_dests`. `Operand::Memory{ base:u8, offset:i16 }`.
- `tile.program_memory() -> Option<&[u8; PROGRAM_MEMORY_SIZE]>` (NOT `program_bytes()`); `tile.is_compute()` exists. ELF `.text_section()/.text_address()` return `Option` (not `Result`).
- Main core function returns at `Ret` 0x31A; helpers (0x330/0x350) and init (0x370+) follow. Lock calls (0x134–0x2C0) + buffer accesses (0x130–0x2B2) all precede 0x31A.

## File Structure

- **Create** `src/device/stream_switch/core_relay.rs` — static core-program analysis (`analyze_core_program` → `CoreLockUsage`) + `core_lock_relay_edges` builder.
- **Modify** `src/device/stream_switch/route_graph.rs` — `EdgeKind::CoreLockRelay`; call the new builder in `resolve_route_graph`; a `load_state_with_core_elfs` test helper.
- **Modify** `src/device/stream_switch/mod.rs` — `pub mod core_relay;`.
- **Create/Modify** a shared ELF-loading helper (P0.5) reachable from tests + the dump example.
- **Modify** `examples/dump_config_json.rs` — load compute core ELFs before `resolve_route_graph`.
- **Modify** `src/interpreter/execute/control.rs` + `src/interpreter/execute/memory/mod.rs` + tile/array recorder plumbing — two runtime witnesses.
- **Modify** `tools/config_extract/dump_model.py`, `reachability.py`, `generator.py`; `tools/inference/ledger.py`, `rules.py`.
- **Tests:** inline Rust in `core_relay.rs`/`route_graph.rs`; Python in `tools/test_*.py`.

---

### Task P0: Feasibility spike — DONE (reference)

The spike (`cargo run --bin spike_p0`, throwaway under `src/bin/spike_p0.rs`) proved recovery TRACTABLE and produced the "Spike-verified facts" above. **No action except:** keep `spike_p0.rs` as a scratch reference during P1; **delete it before the final whole-branch review** (it is uncommitted/untracked — do not commit it). Verdict to honor: lock recovery is clean; the buffer pass must be stack-aware; ordering must be delay-slot-aware; this is Chess-specific.

---

### Task P0.5: Shared `load_state_with_core_elfs` helper

A DeviceState built from CDO alone has **empty core program memory** (`apply_cdo` is register/DMA-config only; ELFs load separately via `elf.load_into(tile)`). Every task that needs core lock ops (P2, P3, P6) must load the compute ELFs. Build this once.

**Files:**
- Modify: `src/device/stream_switch/route_graph.rs` (add the helper in the test module, or a shared `#[cfg(test)]`/`pub(crate)` location).
- Investigate: `src/parser/elf.rs` (does it expose a symbol table for function bounds? — report finding; P1c needs it).

**Interfaces:**
- Produces: `fn load_state_with_core_elfs(xclbin_path: &str) -> Option<DeviceState>` — `load_npu1_state` (CDO) + parse each compute-core ELF and `elf.load_into(tile)`.

- [ ] **Step 1: Find how core ELFs map to tiles.** Grep for `load_into` callers and how the interpreter/bridge path loads per-core ELFs from an xclbin partition (likely `src/testing/` or `src/interpreter/engine`). Determine: are the ELFs in the xclbin's `AiePartition`, named/indexed by (col,row)? Report the access path.

- [ ] **Step 2: Write the failing test.**
```rust
#[test]
fn loads_compute_core_program() {
    let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
    let Some(state) = load_state_with_core_elfs(path) else { println!("SKIP: fixture absent"); return; };
    let tile = state.array.get(0, 2).expect("compute tile (0,2)");
    let prog = tile.program_memory().expect("compute tile has program memory");
    assert!(prog.iter().any(|&b| b != 0), "compute (0,2) program memory must be non-empty");
}
```
This **guards every later edge test against the vacuous-empty-program pass.**

- [ ] **Step 3: Run — fails.** `cargo test --lib loads_compute_core_program 2>&1 | tee /tmp/p05.txt`; Read it.

- [ ] **Step 4: Implement** `load_state_with_core_elfs`: start from `load_npu1_state` (CDO), then for each compute tile, locate its ELF bytes in the partition (per Step 1), `crate::parser::elf::Elf::parse(bytes).ok()?.load_into(state.array.tile_mut(col,row))`. Return the state.

- [ ] **Step 5: Run — passes.** Re-run Step 3 cmd; Read; expect PASS (or clean SKIP if fixture absent).

- [ ] **Step 6: Commit.**
```bash
git add src/device/stream_switch/route_graph.rs
git commit -m "test(#140): load_state_with_core_elfs helper (CDO + compute ELFs into tiles)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P1a: Lock-call idiom recovery

**Files:**
- Create: `src/device/stream_switch/core_relay.rs`; Modify `mod.rs` (`pub mod core_relay;`)
- Test: inline in `core_relay.rs`

**Interfaces:**
- Consumes: `InstructionDecoder::load_cached()`, `.decode(bytes,pc)->Result<VliwBundle,_>`, `bundle.active_slots()`, `bundle.size()->u8`; `SlotOp { semantic, sources, dest }`; `SemanticOp::{LockAcquire,LockRelease,Call,Copy,Ret}`; `Operand::{ScalarReg,Immediate}`.
- Produces:
```rust
pub enum CoreLockKind { Acquire, Release }
pub struct CoreLockOp { pub lock_id: u8, pub kind: CoreLockKind, pub pc: u32 }
// (extended in P1b/P1c)
pub struct CoreLockUsage { pub locks: Vec<CoreLockOp>, /* accesses, fn_end added later */ }
pub fn recover_lock_ops(text: &[u8], text_base: u32, dec: &InstructionDecoder) -> Vec<CoreLockOp>;
```

- [ ] **Step 1: Test against the spike oracle.**
```rust
#[test]
fn recovers_add_one_lock_ops() {
    let Some((text, base)) = load_core_text("...main_core_0_2.elf") else { println!("SKIP"); return; };
    let ops = recover_lock_ops(&text, base, &InstructionDecoder::load_cached());
    let acq: BTreeSet<u8> = ops.iter().filter(|o| matches!(o.kind, CoreLockKind::Acquire)).map(|o| o.lock_id).collect();
    let rel: BTreeSet<u8> = ops.iter().filter(|o| matches!(o.kind, CoreLockKind::Release)).map(|o| o.lock_id).collect();
    assert_eq!(acq, BTreeSet::from([1,2]), "acquires");
    assert_eq!(rel, BTreeSet::from([0,3]), "releases");
}
```
(`load_core_text` reads the ELF, returns `(.text bytes, text_address)` via `Elf::parse` + `.text_section()`/`.text_address()` — both `Option`.)

- [ ] **Step 2: Run — fails.** `cargo test --lib core_relay::tests::recovers_add_one_lock_ops 2>&1 | tee /tmp/p1a.txt`; Read it.

- [ ] **Step 3: Implement.**
  1. Linear-decode the whole `.text` into a `Vec<(pc, VliwBundle)>` (walk: decode at offset, `pc += bundle.size()`, on `Err` advance by the smallest instruction size or 2 and continue).
  2. **Find helper addresses:** PCs of bundles whose any slot is `SemanticOp::LockAcquire` → `acq_addr`; `LockRelease` → `rel_addr`. (Semantic scan — no hardcoded address.)
  3. **For each call site** (slot `SemanticOp::Call` with `sources[0] == Immediate(t)` where `t == acq_addr` or `rel_addr`): resolve r0 by scanning the bundles from the call bundle through the next ≤6 bundles (delay-slot window), **stopping at the next `Call` or `Ret`**; take the last slot with `semantic==Copy && dest==Some(ScalarReg(0)) && sources[0]==Immediate(n)`. `lock_id = (n - 48) as u8`. If no such Copy found → **skip** (unresolved, no op). Push `CoreLockOp{lock_id, kind, pc}`.

- [ ] **Step 4: Run — passes.** Re-run Step 2 cmd; Read; expect PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/device/stream_switch/core_relay.rs src/device/stream_switch/mod.rs
git commit -m "feat(#140): recover core lock ops from JL->ACQ/REL idiom (delay-slot r0, -48)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P1b: Stack-aware buffer-contact recovery

The hard part. Buffer pointers flow `MOVXM imm → reg → ST reg,[sp,#off] → LDA p,[sp,#off] → load/store via p`. A naive p-reg+MOVXM tracker fails (proven by the spike). Build a forward abstract-value pass over regs **and stack slots**.

**Files:** Modify `core_relay.rs`; Test inline.

**Interfaces:**
- Consumes: P1a's decoded bundle stream; `Operand::{ScalarReg,PointerReg,ModifierReg,Immediate,Memory}`.
- Produces:
```rust
pub struct CoreBufAccess { pub local_off: u32, pub is_store: bool, pub pc: u32 } // local_off = resolved base & 0xFFFF
pub fn recover_buffer_accesses(bundles: &[(u32, VliwBundle)]) -> Vec<CoreBufAccess>;
```

- [ ] **Step 1: Test against the spike oracle (genuine contacts, no stack-spill false positives).**
```rust
#[test]
fn recovers_add_one_buffer_contacts() {
    let Some((text, base)) = load_core_text("...main_core_0_2.elf") else { println!("SKIP"); return; };
    let bundles = decode_all(&text, base, &InstructionDecoder::load_cached());
    let acc = recover_buffer_accesses(&bundles);
    // genuine in1 LOADs (0x400/0x420) and out1 STOREs (0x440/0x460)
    assert!(acc.iter().any(|a| !a.is_store && (a.local_off==0x400 || a.local_off==0x420)), "in1 load");
    assert!(acc.iter().any(|a|  a.is_store && (a.local_off==0x440 || a.local_off==0x460)), "out1 store");
    // NO stack-spill misclassified as a buffer store: there must be no "store" with local_off in a stack slot,
    // and the pointer-spill at 0xEA / 0x1B4 (ST pX,[sp,#off]) must NOT appear as a buffer store.
    assert!(!acc.iter().any(|a| a.pc == 0xEA || a.pc == 0x1B4), "pointer-spill must not be a buffer contact");
}
```

- [ ] **Step 2: Run — fails.** `cargo test --lib core_relay::tests::recovers_add_one_buffer_contacts 2>&1 | tee /tmp/p1b.txt`; Read it.

- [ ] **Step 3: Implement the abstract-value pass.** Lattice `Val = Known(i32) | Unknown`. State: `regs: HashMap<(RegClass,u8), Val>` (ScalarReg/PointerReg/ModifierReg) + `stack: HashMap<i16, Val>`. Single forward pass over `bundles`, per slot in order:
  - `Copy` `dest=reg`, `sources=[Immediate(v)]` → `regs[reg] = Known(v)`.
  - `Copy` `dest=reg`, `sources=[other_reg]` → `regs[reg] = regs[other_reg]` (copy-propagate; mov pX,regY).
  - `Store` with `sources=[val_reg, Memory{base:255, off}]` (base 255 = sp) → `stack[off] = regs[val_reg]` (spill; **not** a buffer contact).
  - `Load` `dest=reg`, `sources=[Memory{base:255, off}]` → `regs[reg] = stack[off]` (reload).
  - Any other write to a reg (`dest=reg`, non-immediate/non-copy, e.g. `Add`, `PointerAdd`) → `regs[reg] = Unknown`.
  - **Buffer access detection** (the contact): a `Load` with `sources=[PointerReg(p)]` (p ≠ 255) where `regs[PointerReg(p)] == Known(addr)` → push `CoreBufAccess{ local_off: (addr as u32) & 0xFFFF, is_store:false, pc }`. A `Store` with `dest=Some(PointerReg(p))` (p ≠ 255) where `regs[PointerReg(p)] == Known(addr)` → push `is_store:true`. (Post-increment leaves the base immediate as the contact address — sound for a superset; do not try to model the stride.) Exclude `Memory{base:255,...}` (stack) entirely from contact detection.
  - Skip `ModifierReg(8)`/`PointerReg(255)` as buffer-pointer candidates (dn0/sp).

- [ ] **Step 4: Run — passes.** Re-run Step 2; Read; expect PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/device/stream_switch/core_relay.rs
git commit -m "feat(#140): stack-aware buffer-contact recovery (reg+stack value lattice, sp excluded)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P1c: Ordering proxy + function scoping + `analyze_core_program`

**Files:** Modify `core_relay.rs`; Test inline.

**Interfaces:**
- Produces:
```rust
pub struct CoreLockUsage { pub locks: Vec<CoreLockOp>, pub accesses: Vec<CoreBufAccess>, pub fn_end: u32 }
pub fn analyze_core_program(text: &[u8], text_base: u32, dec: &InstructionDecoder) -> CoreLockUsage;
/// Sound proxy: within the core function, an acquire of L_in precedes an in1 LOAD, which precedes an
/// out1 STORE, which precedes/retires-at the release of L_out. Delay-slot-aware: the release CALL may
/// issue before trailing out1 stores in its delay slots, so "store precedes release" uses the release's
/// retire boundary (next bundle after the release's <=6 delay-slot window), not the release CALL pc.
pub fn relay_ordered(u: &CoreLockUsage, l_in: u8, l_out: u8, in_off: &[u32], out_off: &[u32]) -> bool;
```

- [ ] **Step 1: Function scoping.** `analyze_core_program` bounds the relay analysis to the **core function**: the instruction range from `text_base` to the first `Ret` reached at top level that returns from the entry function. Use the ELF symbol table if P0.5 Step 1 found one exposes function bounds; else use the first `Ret` at-or-after the last lock call (for add_one this is 0x31A; helpers/init follow). Set `fn_end`. Lock ops and buffer accesses past `fn_end` are dropped (helpers/init are not the core dataflow).

- [ ] **Step 2: Ordering test.**
```rust
#[test]
fn add_one_relay_ordered() {
    let Some((text, base)) = load_core_text("...") else { println!("SKIP"); return; };
    let u = analyze_core_program(&text, base, &InstructionDecoder::load_cached());
    assert!(relay_ordered(&u, 1, 3, &[0x400,0x420], &[0x440,0x460]), "add_one acq1..load..store..rel3 ordered");
    // a usage with the release retiring BEFORE the out store must be rejected:
    assert!(!relay_ordered(&bad_order_usage(), 1, 3, &[0x400], &[0x440]));
}
```

- [ ] **Step 3: Run — fails; implement; run — passes.** `cargo test --lib core_relay::tests::add_one_relay_ordered 2>&1 | tee /tmp/p1c.txt`. Implement `relay_ordered`: there exists an acquire(L_in) at pc_a, an in1 LOAD at pc_l, an out1 STORE at pc_s, and a release(L_out) at pc_r, with `pc_a < pc_l < pc_s` and `pc_s <= retire(pc_r)` where `retire(pc_r)` = pc of the bundle after the release's ≤6-bundle delay window. All within `[text_base, fn_end)`. Read; expect PASS.

- [ ] **Step 4: Full lib build + tests.** `cargo test --lib 2>&1 | tee /tmp/p1c_full.txt`; Read it. Expected all green.

- [ ] **Step 5: Commit.**
```bash
git add src/device/stream_switch/core_relay.rs
git commit -m "feat(#140): delay-slot-aware ordering proxy + function scoping (analyze_core_program)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P2: `EdgeKind::CoreLockRelay` + edge builder + wiring

**Files:** Modify `route_graph.rs` (enum + wiring + test); Modify `core_relay.rs` (`core_lock_relay_edges`).

**Interfaces:**
- Consumes: `analyze_core_program`, `CoreLockUsage`, `relay_ordered` (P1); `dma.get_bd`, `channel_bd_chain`, `start_bd_field_for`, `dma.resolve_lock_id→LockTarget::Own`, `ChannelType::from_channel_index`, `tile.stream_switch.dma_master/dma_slave`, `BdConfig{base_addr,length,acquire_lock,release_lock,release_value}`.
- Produces: `pub fn core_lock_relay_edges(tile: &Tile, dma: &DmaEngine, s2mm_count: usize, usage: &CoreLockUsage) -> Vec<RouteEdge>`.

- [ ] **Step 1: Add the enum variant** (after `LockPair`), with the data-contact doc-comment (no value-flow language):
```rust
    /// Intra-tile through-core relay: the compute CORE program bridges an S2MM channel
    /// (writes input buffer, RELEASES the data-ready lock the core ACQUIRES) to an MM2S
    /// channel (reads output buffer, ACQUIRES the data-ready lock the core RELEASES). Emitted
    /// only when lock-pairing INTERSECT buffer-contact INTERSECT delay-slot-aware ordering all
    /// hold (lock identity alone is the co-firing trap). Claims STRUCTURAL data-contact under
    /// producer/consumer lock ordering -- the core had the opportunity to relay these bytes --
    /// NOT value-dependence (the trace oracle cannot witness value flow). Oriented src = S2MM
    /// master DMA port (writer) -> dst = MM2S slave DMA port (reader); reverse is back-pressure,
    /// never emitted. Coverage is narrow + Chess-specific (objectFIFO passthrough); unresolvable
    /// lock/buffer/ordering -> no edge (safe false-negative).
    CoreLockRelay,
```

- [ ] **Step 2: Edge test (using P0.5's loader).**
```rust
#[test]
fn core_lock_relay_add_one_compute_tile() {
    let Some(state) = load_state_with_core_elfs(".../chess/aie.xclbin") else { println!("SKIP"); return; };
    // guard against vacuous empty-program pass:
    assert!(state.array.get(0,2).and_then(|t| t.program_memory()).map_or(false, |p| p.iter().any(|&b| b!=0)));
    let g = state.resolve_route_graph();
    let relays: Vec<_> = g.edges.iter().filter(|e| e.kind == EdgeKind::CoreLockRelay).collect();
    assert_eq!(relays.len(), 1, "expected 1 CoreLockRelay, got {:?}", relays);
    let e = relays[0];
    assert_eq!((e.src.col, e.src.row, e.src.dir), (0, 2, PortDir::Master));
    assert_eq!((e.dst.col, e.dst.row, e.dst.dir), (0, 2, PortDir::Slave));
    assert!(!g.edges.iter().any(|e| e.kind==EdgeKind::CoreLockRelay
        && e.src.dir==PortDir::Slave && e.dst.dir==PortDir::Master), "no reverse/back-pressure");
}
```

- [ ] **Step 3: Run — fails.** `cargo test --lib route_graph::tests::core_lock_relay_add_one_compute_tile 2>&1 | tee /tmp/p2.txt`; Read it.

- [ ] **Step 4: Implement `core_lock_relay_edges`.** For each S2MM channel: `l_in = own_local(release_lock of its first release_value>0 BD)`, `in_rng = (base_addr & 0xFFFF, +length)` of its BD chain; require `usage.locks` contains `Acquire(l_in)`. For each MM2S channel: `l_out = own_local(acquire_lock of its first BD)`, `out_rng`; require `usage.locks` contains `Release(l_out)`. Then require a buffer LOAD whose `local_off` ∈ in_rng (exact, masked) AND a STORE whose `local_off` ∈ out_rng, AND `relay_ordered(usage, l_in, l_out, in_offs, out_offs)`. Emit `src=dma_master(s2mm_ch)`, `dst=dma_slave(mm2s_ch)`, `EdgeKind::CoreLockRelay`; dedup by `(src.index,dst.index)`. (Structure mirrors `dma_lock_pair_edges`; reuse `channel_bd_chain`/`start_bd_field_for`/`resolve_lock_id`.)

- [ ] **Step 5: Wire into `resolve_route_graph`** — inside the `if let Some(dma) = self.array.dma_engine(...)` block, after `dma_lock_pair_edges`:
```rust
                if tile.is_compute() {
                    if let Some(prog) = tile.program_memory() {
                        let dec = crate::interpreter::decode::loader::InstructionDecoder::load_cached();
                        let usage = core_relay::analyze_core_program(&prog[..], 0, &dec);
                        for edge in core_relay::core_lock_relay_edges(tile, dma, s2mm_count, &usage) {
                            g.add_edge(edge);
                        }
                    }
                }
```
(text_base = 0; AIE core `.text` is loaded at program offset 0 — confirm against P0.5.)

- [ ] **Step 6: Run — passes.** Re-run Step 3 cmd; Read; expect PASS (or clean SKIP — then run A5/E4 to confirm no regression).

- [ ] **Step 7: Full lib tests.** `cargo test --lib 2>&1 | tee /tmp/p2_full.txt`; Read it. Expected: green; E2/E3/A5/E4 unaffected (additive).

- [ ] **Step 8: Commit.**
```bash
git add src/device/stream_switch/
git commit -m "feat(#140): CoreLockRelay edge (lock-pair INTERSECT buffer-contact INTERSECT ordering)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P3: Dump integration — load ELFs + serialize the edge

**Files:** Modify `examples/dump_config_json.rs`; Modify `tools/config_extract/dump_model.py`; regenerate fixture.

- [ ] **Step 1: Load compute ELFs in the dump.** In `dump_config_json.rs`, after CDO + insts.bin, load each compute-core ELF into its tile (reuse the P0.5 mechanism / the per-core ELF extraction from the xclbin partition) before `resolve_route_graph()`.

- [ ] **Step 2: Accept the kind in dump_model.** In `dump_model.py`, add `"core_lock_relay"` to the `RouteEdge.kind` comment enumeration (kind is a free `str`; `load_dump` doesn't validate a closed set — confirm). Add a round-trip test asserting a `core_lock_relay` edge loads. Put it in `test_config_extract_reachability.py` (has `load_dump` coverage) — do NOT invent a non-existent `test_config_extract_dump_model.py`.

- [ ] **Step 3: Regenerate the fixture.**
```bash
cd /home/triple/npu-work/xdna-emu
cargo run --example dump_config_json -- \
  /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin \
  /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/insts.bin \
  > tools/config_extract/fixtures/add_one_using_dma.config.json 2>/tmp/p3_dump.txt
grep -c core_lock_relay tools/config_extract/fixtures/add_one_using_dma.config.json
```
Read `/tmp/p3_dump.txt`; expect the grep ≥1.

- [ ] **Step 4: Run the round-trip test.** `cd tools && python -m pytest test_config_extract_reachability.py -v 2>&1 | tee /tmp/p3_py.txt`; Read it; expect PASS.

- [ ] **Step 5: Commit.**
```bash
git add examples/dump_config_json.rs tools/config_extract/dump_model.py tools/config_extract/fixtures/add_one_using_dma.config.json tools/test_config_extract_reachability.py
git commit -m "feat(#140): dump loads compute ELFs so CoreLockRelay edges enter the config dump

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P4: `program_path` predicate (ledger + rules)

**Files:** Modify `tools/inference/ledger.py`, `rules.py`; Test `test_inference_ledger.py`, `test_inference_rules.py`.

- [ ] **Step 1: Failing ledger test.**
```python
def test_program_kind_maps_to_program_path():
    from inference.ledger import ledger_facts
    led = {"program:p--via-core-->c": {"cite": "program:p--via-core-->c", "a": "p", "b": "c", "kind": "program"}}
    facts = ledger_facts(led)
    assert any(f.predicate == "program_path" and f.args[:2] == ("p","c") for f in facts)
    assert not any(f.predicate == "config_path" for f in facts)
```

- [ ] **Step 2: Run — fails.** `cd tools && python -m pytest test_inference_ledger.py::test_program_kind_maps_to_program_path -v 2>&1 | tee /tmp/p4a.txt`; Read it.

- [ ] **Step 3: Implement.** `ledger.py`: line 24 → `_KINDS = {"route","bd","lock","identity","program"}`. `ledger_facts` mapping:
```python
        if e["kind"] == "identity":
            pred = "identity"
        elif e["kind"] == "program":
            pred = "program_path"
        else:
            pred = "config_path"
```
Update the module docstring schema to add `program -> program_path(a,b,cite)`.

- [ ] **Step 4: Run — passes.** Re-run Step 2; Read; PASS.

- [ ] **Step 5: Failing rules test** (engine derives via `program_path`-only orientation). Mirror `test_engine_reconstructs_placement`'s `_runs` construction (child = parent+offset, parent stochastic); install a ledger with only a `kind:"program"` entry; assert `try_derives(run_dirs, kb, child, parent)` returns a `derives` Fact.

- [ ] **Step 6: Run — fails.** `cd tools && python -m pytest test_inference_rules.py::test_try_derives_consumes_program_path -v 2>&1 | tee /tmp/p4b.txt`; Read it.

- [ ] **Step 7: Implement.** `rules.py` `try_derives` orientation query (lines 40-41):
```python
    cp = next((f for f in (kb.by_predicate("config_path") + kb.by_predicate("program_path"))
               if f.args[0] == parent and f.args[1] == child), None)
```
(Distinct facts in the KB; only the orientation *query* unions them, so audits/reports still filter by predicate.)

- [ ] **Step 8: Run — passes.** Re-run Step 6; Read; PASS.

- [ ] **Step 9: Full inference suite.** `cd tools && python -m pytest test_inference_*.py -v 2>&1 | tee /tmp/p4_full.txt`; Read it; all green.

- [ ] **Step 10: Commit.**
```bash
git add tools/inference/ledger.py tools/inference/rules.py tools/test_inference_ledger.py tools/test_inference_rules.py
git commit -m "feat(#140): real program_path predicate (distinct from config_path, shared orientation rule)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P5: Generator — program-taint reachability + emission + audit

**Files:** Modify `tools/config_extract/reachability.py`, `generator.py`; Test `test_config_extract_generator.py`, `test_config_extract_reachability.py`.

- [ ] **Step 1: Reachability split test.**
```python
def test_program_only_reachability_split():
    from config_extract.reachability import Reachability
    edges = [mk_edge(A,B,"inter_tile"), mk_edge(B,C,"core_lock_relay")]
    full = Reachability(edges)
    cfg  = Reachability([e for e in edges if e.kind != "core_lock_relay"])
    assert full.reachable(A,C) and not cfg.reachable(A,C)
    assert cfg.reachable(A,B)
```

- [ ] **Step 2: Run.** `cd tools && python -m pytest test_config_extract_reachability.py::test_program_only_reachability_split -v 2>&1 | tee /tmp/p5a.txt`; Read it. (Likely passes if `Reachability` already takes an arbitrary edge list — confirms the split strategy; if it fails, make `Reachability` accept the filtered list.)

- [ ] **Step 3: Generator program-emission test.**
```python
def test_generates_program_path_for_through_core_pair():
    dump = load_dump(FIX)   # regenerated fixture (P3) has the core_lock_relay edge
    led = generate_ledger(dump, FIRED, start_col=START_COL)
    progs = [e for e in led["entries"] if e["kind"]=="program"]
    assert progs, "expected >=1 program entry"
    for e in progs:
        assert e["cite"].startswith("program:") and "--via-core-->" in e["cite"]
    assert any(e["kind"]=="route" for e in led["entries"]), "config pairs still route"
```

- [ ] **Step 4: Run — fails.** `cd tools && python -m pytest test_config_extract_generator.py::test_generates_program_path_for_through_core_pair -v 2>&1 | tee /tmp/p5b.txt`; Read it.

- [ ] **Step 5: Implement in `generator.py`.** Add `_make_program_cite(parent,child) -> f"program:{parent}--via-core-->{child}"` and `_RE_PROGRAM_CITE = re.compile(r"^program:(?P<parent>.+?)--via-core-->(?P<child>.+)$")`. In `generate_ledger`: build `full = Reachability(all_edges)`, `cfg = Reachability([e for e in all_edges if e.kind != "core_lock_relay"])`. For each fired pair (parent,child): if `cfg.reachable(parent,child)` → emit `kind:"route"` (existing path). Elif `full.reachable(parent,child)` → emit `kind:"program"` via `_make_program_cite`. Else decline. Both keep `a=parent,b=child`.

- [ ] **Step 6: Run — passes.** Re-run Step 4; Read; PASS.

- [ ] **Step 7: Extend `audit_ledger`.** It **appends a failure string and continues** (does NOT raise) — so the test asserts on the returned failure list. For `kind=="program"`: validate `_RE_PROGRAM_CITE` and `m.group("parent")==a`, `m.group("child")==b`. For `kind=="route"`: unchanged. Add `test_audit_accepts_program_and_catches_program_cite_mismatch` (empty failures on a good program entry; non-empty on a corrupted program cite).

- [ ] **Step 8: Run audit test + full config_extract suite.** `cd tools && python -m pytest test_config_extract_*.py -v 2>&1 | tee /tmp/p5_full.txt`; Read it; all green.

- [ ] **Step 9: Commit.**
```bash
git add tools/config_extract/reachability.py tools/config_extract/generator.py tools/test_config_extract_*.py
git commit -m "feat(#140): generator emits program_path for through-core-only-reachable pairs

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P6: Two-witness runtime validation (E4-style superset gate)

The runtime witness is *simpler* than the static pass: the effective address is concrete (no reaching-def). Classification by address: local buffer = `addr >> 16 == 7`, masked offset = `addr & 0xFFFF`; stack stores go elsewhere and are naturally excluded by the buffer-range check.

**Files:** Modify `control.rs`, `memory/mod.rs`, tile/array recorder plumbing; Test inline in `route_graph.rs`.

- [ ] **Step 1: Add gated recorders** (mirror E4's `Option<Vec<_>>` pattern, default `None`): `CoreLockEvent{cycle, lock_local_id, op:Acquire|Release, col, row}`, `CoreBufEvent{cycle, local_off, is_store, col, row}`; `array.enable_core_relay_recording()` / `take_core_relay_events() -> (Vec<CoreLockEvent>, Vec<CoreBufEvent>)`.

- [ ] **Step 2: Hook the two independent paths.**
  - `control.rs`: at the `LockResult::Success` arm of `SemanticOp::LockAcquire` (~L224) and the own-tile `defer_core_lock_release` site (~L308), if recording, push a `CoreLockEvent` (resolve raw lock id to local; record op, `ctx.cycles`, col, row).
  - `memory/mod.rs`: at the `get_address`/`get_store_address` hook sites, if recording AND `addr >> 16 == 7` (local), push `CoreBufEvent{ local_off: addr & 0xFFFF, is_store, cycle, col, row }`.

- [ ] **Step 3: Validation test** (static ⊇ enacted).
```rust
#[test]
fn static_graph_covers_enacted_core_relays_add_one() {
    let Some(state) = load_state_with_core_elfs(PATH) else { println!("SKIP"); return; };
    let static_set: HashSet<(PhysKey,PhysKey)> = state.resolve_route_graph().edges.iter()
        .filter(|e| e.kind==EdgeKind::CoreLockRelay).map(|e| (phys(&e.src), phys(&e.dst))).collect();
    let mut engine = InterpreterEngine::new_npu1();  // loads ELFs
    engine.device_mut().array.enable_core_relay_recording();
    /* run to completion (mirror E4 loop) */
    let (locks, bufs) = engine.device_mut().array.take_core_relay_events();
    // reconstruct enacted through-core handoffs on the compute tile by cycle order:
    //   core ACQUIRE(L_in released by an S2MM) ... in1-range LOAD ... out1-range STORE ... core RELEASE(L_out acq by MM2S)
    //   -> (S2MM master port -> MM2S slave port). Assert every enacted handoff is in static_set; assert non-empty.
    for h in &enacted { assert!(static_set.contains(&(h.src,h.dst)), "uncovered enacted relay {:?}", h); }
    assert!(!enacted.is_empty(), "add_one must enact >=1 through-core relay");
}
```

- [ ] **Step 4: Run — fails; implement recorders + reconstruction; run — passes.** `cargo test --lib route_graph::tests::static_graph_covers_enacted_core_relays_add_one 2>&1 | tee /tmp/p6.txt`; Read. A backwards static orientation would fail this — genuine oracle. Confirm A5/E4 still pass.

- [ ] **Step 5: Full lib tests.** `cargo test --lib 2>&1 | tee /tmp/p6_full.txt`; Read it; green.

- [ ] **Step 6: Commit.**
```bash
git add src/interpreter/ src/device/
git commit -m "test(#140): two-witness runtime validation of CoreLockRelay (static superset of enacted)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P7: End-to-end engine derive + full regression

**Files:** Test `test_config_extract_generator.py` (or new `test_program_path_e2e.py`).

- [ ] **Step 1: E2E derive test** (the C4/E6 lesson: derives ≠ loads). Using `build/experiments/infer-smoke` captures (guard skip-if-absent) + the regenerated fixture: `generate_ledger` → JSON → `run_engine` over captured runs; assert there is a `kind:"program"` entry, `provenance_ok`, `replication_violations==[]`, and the through-core pair appears in `rep["derives"]`.
```python
def test_engine_derives_through_core_relay_from_generated_ledger(tmp_path):
    dump = load_dump(FIX)
    fired = fired_keys_from_run(RUN_DIR, ANCHOR)
    led = generate_ledger(dump, fired, start_col=START_COL)
    assert any(e["kind"]=="program" for e in led["entries"])
    p = tmp_path/"gen.ledger.json"; p.write_text(json.dumps(led))
    rep = run_engine(RUN_DIRS, str(p), candidate_pairs_from(led))
    assert rep["provenance_ok"] and rep["replication_violations"]==[]
    # pin the exact derived tuple from the run output (resolve at impl):
    assert any(t for t in rep["derives"] if _is_through_core_pair(t))
```

- [ ] **Step 2: Run — reveals the real derive keys.** `cd tools && python -m pytest test_config_extract_generator.py::test_engine_derives_through_core_relay_from_generated_ledger -v 2>&1 | tee /tmp/p7.txt`; Read it; use the output to pin `_is_through_core_pair`.

- [ ] **Step 3: Finalize + run — passes.** Re-run; Read; PASS. **If `build/experiments/infer-smoke` is absent**, do NOT degrade to a provenance-only assertion — instead construct a synthetic run set (mirror `test_engine_reconstructs_placement`) where the through-core pair is the only stochastic-root-bridging derive, and assert that NAMED program-kind pair derives. (No vacuous fallback.)

- [ ] **Step 4: Full regression — both suites.**
```bash
cargo test --lib 2>&1 | tee /tmp/p7_rust.txt
cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_config_extract_*.py test_inference_*.py -v 2>&1 | tee /tmp/p7_py.txt
```
Read both; all green; config_path derivation unchanged.

- [ ] **Step 5: Delete the throwaway spike** (`rm src/bin/spike_p0.rs`) and commit the close-out.
```bash
git rm src/bin/spike_p0.rs 2>/dev/null; git add -A
git commit -m "test(#140): engine derives through-core relay from generated ledger (E2E); drop spike

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Self-Review (against the spec + spike + plan-review)

**Spec coverage:** §1 ceiling X → P0.5–P6 derive+validate, trace/HW deferred; §1 non-goals (no value-dependence, narrow+Chess coverage) → Global Constraints + EdgeKind doc + block-on-unresolved; §3 lock∩buffer∩ordering → P2 builder + P1a/b/c; §4 static analysis → P1 (idiom recovery + stack-aware lattice + delay-slot ordering, the spike-verified forms); §5 shared graph + real predicate → P2 + P4; §6 two-witness validation → P6 (control vs memory paths) + config-range(BD)∩ELF-access independence; §7 testing → throughout. ✓

**Plan-review fixes folded in:** `dest` not `destinations`; `program_memory()` not `program_bytes()`; `is_compute()` exists; `Immediate` is i32; shared P0.5 ELF-loader (no standalone-core-ELF assumption — uses the `.prj` ELF / xclbin extraction); vacuous-empty-program guard in P0.5+P2; `audit_ledger` appends-not-raises; no invented `test_config_extract_dump_model.py`; P7 has no vacuous fallback; `.text_section/.text_address` are `Option`. ✓

**Spike findings folded in:** lock idiom recovery (semantic-scan helpers + delay-slot r0 + −48); stack-aware buffer lattice (regs+stack, sp/dn0 excluded, exact-base classify for the stack-end alias, pointer-spills excluded); delay-slot-aware ordering; function scoping. ✓

**Type consistency:** `CoreLockUsage`/`CoreLockOp`/`CoreBufAccess` consumed identically P1→P2; `local_off` (`& 0xFFFF`) matches `BdConfig.base_addr & 0xFFFF`; `EdgeKind::CoreLockRelay`⇒serde `"core_lock_relay"` used in P3/P5; `program` kind⇒`program_path` predicate⇒`program:…--via-core-->…` cite consistent P4/P5/P7; `a=parent,b=child` throughout.

**Residual design note:** `RouteEdge` has no byte-range field (spec §5 mentions carrying it); dedup uses the port-index `seen` set instead, so the field is unnecessary for correctness — noted, not added (YAGNI).

**Carried risk:** P1b (stack-aware pass) is the main engineering cost; the spike validated the technique end-to-end on the real ELF, so it is de-risked but remains the task to watch. Chess-only by design; Peano is documented follow-up.
