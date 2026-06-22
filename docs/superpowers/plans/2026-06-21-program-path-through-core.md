# program_path: Through-Core Dataflow Edges — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive the compute-tile relay `S2MM(in1) → core → MM2S(out1)` — which config-path Tier E structurally cannot reach — by statically parsing the core ELF's `use_lock` ops + buffer accesses, emit it as a sound `CoreLockRelay` route edge and a distinct `program_path` engine fact, and validate it against the emulator's runtime core-lock + buffer-touch enactment.

**Architecture:** A new Rust static-analysis module decodes the compute core's program memory (no execution) to recover lock acquire/release ops and buffer load/store contacts. A new `EdgeKind::CoreLockRelay` builder in `resolve_route_graph` emits an edge only when lock-pairing ∩ buffer-contact ∩ straight-line ordering all hold (lock identity alone is the co-firing trap). The edge rides the existing shared route graph; the Python engine gains a real `program_path` predicate (not a cosmetic cite). Validation is an E4-style superset check using two independent runtime witnesses (lock handoffs + buffer touches). Ceiling X: trace/HW validation deferred to the separate experimenter-loop plan.

**Tech Stack:** Rust (route graph, decoder, interpreter), Python 3.13 (inference engine + config_extract generator), the build-derived llvm-aie TableGen decoder.

**Design spec:** `docs/superpowers/specs/2026-06-21-program-path-through-core-design.md` (survived two adversarial reviews). Read it first.

## Global Constraints

- **Claim semantic:** the edge/fact claims *structural data-contact under producer/consumer lock ordering* ("the core had the opportunity to relay these bytes"), **NEVER value-dependence**. No cite, doc, or comment may say buffer-touch implies dataflow/value-flow. (Spec §1, §3.)
- **Soundness is intersection:** an edge requires lock-pairing AND buffer-contact AND ordering. Any criterion alone is insufficient. Unresolvable operand (register-derived lock id or buffer pointer) or unrecoverable ordering ⇒ **emit no edge (safe false-negative) and, where relevant, record that the analysis blocked** — never guess.
- **Orientation:** `src = S2MM master DMA port (writer/releaser)` → `dst = MM2S slave DMA port (reader/acquirer)`. Back-pressure (reverse) is never emitted. (Matches Tier E E2/E3 exactly.)
- **program_path is a real predicate**, separately queryable from `config_path` (not a cite-string label on a config_path fact). (Spec §5.)
- **Coverage is honestly narrow** (objectFIFO-passthrough / simple-elementwise). Plan text and comments must not claim general through-core coverage. (Spec §1 non-goals.)
- **DERIVE FROM THE TOOLCHAIN:** lock + MOVXM instruction recognition is via the existing TableGen-driven decoder's `SemanticOp`, never hardcoded opcodes.
- **Commit trailer** (no emoji), every commit:
  ```
  Generated using Claude Code.
  Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
  ```
- **Rust:** never pipe `cargo` through grep/head/tail — redirect to a file and Read it. Run `cargo test --lib` after Rust changes.
- **Python tests:** `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest <files> -v` (NOT `pytest config_extract/` — collects 0).
- **add_one fixture:** `/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin`. Guard tests to skip cleanly if absent.

## Ground-truth values (add_one_using_dma, compute tile (0,2)) — assert against these

- Locks: `S2MM0` acq lock0 / **rel lock1**; core **acq lock1**, acq lock2, rel lock0, **rel lock3**; `MM2S0` **acq lock3** / rel lock2.
- Buffers (BdConfig `base_addr` is **tile-local**): in1 `[0x400, 0x440)`, out1 `[0x440, 0x480)`. Core-local addresses are `0x70000 + offset`; reconcile with `addr & 0xFFFF`.
- Core MOVXM immediates seen in the Chess ELF: `0x70400/0x70420` (in1 buffers), `0x70440/0x70460` (out1 buffers).
- The one expected edge: `(0,1)? ` NO — compute tile (0,2): `S2MM0(master DMA port) → MM2S0(slave DMA port)`. Reverse must NOT be emitted.

## File Structure

- **Create** `src/device/stream_switch/core_relay.rs` — static core-program analysis (`analyze_core_program`) + `core_lock_relay_edges` builder. One responsibility: turn a core's program bytes + DMA config into `CoreLockRelay` edges.
- **Modify** `src/device/stream_switch/route_graph.rs` — add `EdgeKind::CoreLockRelay`; call the new builder in `resolve_route_graph`; the new module is declared from here or `mod.rs`.
- **Modify** `src/device/stream_switch/mod.rs` — `mod core_relay;` (or `pub mod`), if route_graph doesn't already gate submodules.
- **Modify** `examples/dump_config_json.rs` — load the compute core ELF(s) into tile program memory before `resolve_route_graph`, so `CoreLockRelay` edges enter the JSON dump.
- **Modify** `src/interpreter/execute/control.rs` — runtime core-lock witness hooks (acquire-grant, release).
- **Modify** `src/interpreter/execute/memory/mod.rs` — runtime core buffer-touch witness hooks.
- **Modify** the tile/array recorder plumbing (mirror E4's `enable_lock_recording`/`take_lock_events_by_tile`) for the two new witnesses.
- **Modify** `tools/config_extract/dump_model.py` — accept `kind="core_lock_relay"` in `RouteEdge`.
- **Modify** `tools/config_extract/reachability.py` — config-only vs full reachability (program-taint).
- **Modify** `tools/config_extract/generator.py` — emit `program` entries (program cite) for program-only-reachable pairs; extend `audit_ledger`.
- **Modify** `tools/inference/ledger.py` — `_KINDS` + kind→predicate mapping (`program` → `program_path`).
- **Modify** `tools/inference/rules.py` — `try_derives` also consumes `program_path`.
- **Test files:** new Rust tests live inline in `core_relay.rs` / `route_graph.rs`; Python tests in `tools/test_config_extract_*.py`, `tools/test_inference_*.py`.

---

### Task P0: SPIKE — confirm static recoverability on the real add_one core ELF (GATE)

**No edge code until this passes.** This de-risks the two hardest assumptions (buffer-pointer reaching-def, straight-line ordering). The reviewer already found the MOVXM immediates exist; this formalizes it and pins the ordering shape.

**Files:**
- Create (throwaway): `src/bin/spike_core_decode.rs` (delete after, or keep as an example — your call at review).

**Interfaces:**
- Consumes: `crate::interpreter::decode::loader` (`InstructionDecoder::load_cached`), `decoder.decode(bytes, pc) -> Result<VliwBundle>`, `bundle.active_slots()`, `bundle.size()`, `SlotOp { semantic: Option<SemanticOp>, sources: Vec<Operand>, destinations: Vec<Operand> }`, `SemanticOp::{LockAcquire, LockRelease, Copy, Br, BrCond, Call, Ret}`, `Operand::{Lock, Immediate, PointerReg, ScalarReg, Memory}`.
- Produces: a findings note (no production API).

- [ ] **Step 1: Locate the compute core ELF.** Run:
```bash
ls -la /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/ > /tmp/p0_ls.txt 2>&1
```
Read `/tmp/p0_ls.txt`. Find the compute-core ELF (e.g. `core_0_2.elf`, `*_core_0_2.elf`, or extract from `aie.xclbin` if standalone ELFs are absent). Record the exact path. If only `aie.xclbin` exists, note that P3/P4 must extract the ELF from it (the xclbin carries per-core ELF sections).

- [ ] **Step 2: Write a spike binary that statically decodes the core ELF and prints lock + MOVXM + branch facts.** Load the ELF bytes, get its `.text` (use `crate::parser::elf::Elf::parse(data)` then `.text_section()` / `.text_address()`), and walk:
```rust
use xdna_emu::interpreter::decode::loader::InstructionDecoder;
use xdna_archspec::aie2::isa::types::SemanticOp;
// ... read elf bytes, get text + base ...
let decoder = InstructionDecoder::load_cached();
let mut pc = text_addr;
while (pc as usize) < text_addr as usize + text.len() {
    let off = (pc - text_addr) as usize;
    match decoder.decode(&text[off..], pc) {
        Ok(bundle) => {
            for op in bundle.active_slots() {
                match op.semantic {
                    Some(SemanticOp::LockAcquire) | Some(SemanticOp::LockRelease) => {
                        println!("pc={:#x} {:?} src0={:?} src1={:?}", pc, op.semantic, op.sources.first(), op.sources.get(1));
                    }
                    Some(SemanticOp::Copy) if matches!(op.destinations.first(), Some(crate::interpreter::bundle::Operand::PointerReg(_))) => {
                        println!("pc={:#x} MOVXM dst={:?} imm={:?}", pc, op.destinations.first(), op.sources.first());
                    }
                    Some(SemanticOp::Br) | Some(SemanticOp::BrCond) | Some(SemanticOp::Call) | Some(SemanticOp::Ret) => {
                        println!("pc={:#x} BRANCH {:?}", pc, op.semantic);
                    }
                    _ => {}
                }
            }
            pc += bundle.size() as u32;
        }
        Err(_) => { pc += 4; }
    }
}
```
(Adjust `Operand` import path to the actual one found in Step-0 grep.)

- [ ] **Step 3: Run the spike and capture output.**
```bash
cargo run --bin spike_core_decode -- <elf-path> > /tmp/p0_decode.txt 2>&1
```
Read `/tmp/p0_decode.txt`.

- [ ] **Step 4: Confirm the three feasibility claims against the output. Record findings in the task report:**
  1. **Lock ops resolvable:** lock-acquire of lock id 1 and lock-release of lock id 3 appear with `src0 = Lock(1)`/`Lock(3)` or `Immediate(1)`/`Immediate(3)` (NOT `ScalarReg`). Note all lock ops + their `src0` operand kinds.
  2. **Buffer pointers resolvable:** MOVXM ops load immediates in `0x70400..0x70480` into pointer registers (in1/out1 ranges). Note the immediates and dest registers.
  3. **Ordering shape:** between the lock1-acquire PC and the lock3-release PC, is the region straight-line (no `BRANCH` lines in that PC span), or is there control flow? Record the PC order of acquire(1), the loads, the stores, release(3), and any branches.

- [ ] **Step 5: DECIDE and report.**
  - If all three hold (lock ids static, buffer immediates in range, region analyzable) → **PROCEED**; the report pins the concrete ordering predicate for P1 (e.g. "straight-line: no branch between acq.pc and rel.pc").
  - If any operand is register-derived or the region is not analyzable by a straight-line scan → **STOP and report BLOCKED** with specifics; do not start P1. (This is the spec's "block and report, don't guess" path — a valuable finding, not a failure.)

- [ ] **Step 6: Commit** (spike binary + findings, or just findings if you delete the binary):
```bash
git add -A && git commit -m "spike(#140): confirm add_one core-ELF lock/buffer/ordering statically recoverable

<one-line verdict + the pinned ordering predicate>

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P1: Core static-analysis module (`core_relay.rs` — analysis half)

**Files:**
- Create: `src/device/stream_switch/core_relay.rs`
- Modify: `src/device/stream_switch/mod.rs` (add `pub mod core_relay;`)
- Test: inline `#[cfg(test)]` in `core_relay.rs`

**Interfaces:**
- Consumes: `InstructionDecoder` (load_cached), `VliwBundle`/`SlotOp`/`SemanticOp`/`Operand` (as P0).
- Produces:
```rust
pub enum CoreLockKind { Acquire, Release }
pub struct CoreLockOp { pub lock_id: u8, pub kind: CoreLockKind, pub pc: u32 }
pub struct CoreBufAccess { pub local_off: u32, pub is_store: bool, pub pc: u32 } // local_off = imm & 0xFFFF
pub struct CoreLockUsage { pub locks: Vec<CoreLockOp>, pub accesses: Vec<CoreBufAccess>, pub branch_pcs: Vec<u32> }
pub fn analyze_core_program(text: &[u8], text_base: u32, decoder: &InstructionDecoder) -> CoreLockUsage;
```
  (`local_off`, not full address, so it reconciles directly against `BdConfig.base_addr` tile-local ranges.)

- [ ] **Step 1: Write the failing test for lock-op extraction.** Use a tiny hand-built program OR (preferred) the real add_one core ELF text loaded via a test helper. For determinism, write a unit test over the real ELF guarded by skip-if-absent:
```rust
#[test]
fn extracts_add_one_core_lock_ops() {
    let Some((text, base)) = load_add_one_core_text() else { println!("SKIP: core ELF absent"); return; };
    let dec = InstructionDecoder::load_cached();
    let u = analyze_core_program(&text, base, &dec);
    let acq: Vec<u8> = u.locks.iter().filter(|l| matches!(l.kind, CoreLockKind::Acquire)).map(|l| l.lock_id).collect();
    let rel: Vec<u8> = u.locks.iter().filter(|l| matches!(l.kind, CoreLockKind::Release)).map(|l| l.lock_id).collect();
    assert!(acq.contains(&1), "core must acquire lock1, got {:?}", acq);
    assert!(rel.contains(&3), "core must release lock3, got {:?}", rel);
}
```
(`load_add_one_core_text` = a test helper that reads the ELF path from Step P0 and returns `(.text bytes, text_addr)`. Put it in the test module.)

- [ ] **Step 2: Run it — fails (module/fn missing).**
```bash
cargo test --lib core_relay::tests::extracts_add_one_core_lock_ops 2>&1 | tee /tmp/p1_a.txt
```
Read `/tmp/p1_a.txt`. Expected: compile error / not found.

- [ ] **Step 3: Implement lock-op extraction.** Walk the decoded stream (P0 pattern). For each `SemanticOp::LockAcquire`/`LockRelease`, resolve `lock_id` from `sources[0]`: `Operand::Lock(id) => id`, `Operand::Immediate(v) => v as u8`, else **skip** (register-derived → unresolvable, drop). Push `CoreLockOp { lock_id, kind, pc }`. Also collect `branch_pcs` for any `Br/BrCond/Call/Ret`.

- [ ] **Step 4: Run it — passes.**
```bash
cargo test --lib core_relay::tests::extracts_add_one_core_lock_ops 2>&1 | tee /tmp/p1_a.txt
```
Read it. Expected: PASS (or clean SKIP if ELF absent — then note you validated logic on a synthetic program instead, and add that synthetic test).

- [ ] **Step 5: Write the failing test for buffer-access extraction (reaching-def).**
```rust
#[test]
fn extracts_add_one_buffer_contacts() {
    let Some((text, base)) = load_add_one_core_text() else { println!("SKIP"); return; };
    let u = analyze_core_program(&text, base, &InstructionDecoder::load_cached());
    // in1 [0x400,0x440): at least one LOAD; out1 [0x440,0x480): at least one STORE
    let in1_load = u.accesses.iter().any(|a| !a.is_store && (0x400..0x440).contains(&a.local_off));
    let out1_store = u.accesses.iter().any(|a| a.is_store && (0x440..0x480).contains(&a.local_off));
    assert!(in1_load, "core must load from in1 range; accesses={:?}", u.accesses);
    assert!(out1_store, "core must store to out1 range; accesses={:?}", u.accesses);
}
```

- [ ] **Step 6: Run — fails.** `cargo test --lib core_relay::tests::extracts_add_one_buffer_contacts 2>&1 | tee /tmp/p1_b.txt`; Read it.

- [ ] **Step 7: Implement buffer-access extraction with reaching-def.** Maintain `ptr_imm: HashMap<u8, u32>` (pointer register → last MOVXM immediate). On each bundle, in slot order: if a slot is MOVXM (`SemanticOp::Copy`, `destinations[0] == PointerReg(r)`, `sources[0] == Immediate(v)`), set `ptr_imm[r] = v`. If a slot writes a pointer register by any OTHER means (a `PointerReg` destination that isn't an in-range MOVXM immediate — e.g. computed), **remove** `r` from `ptr_imm` (conservative: its value is now unknown). For each load/store slot, find its pointer register from `sources`/`destinations` (`Operand::PointerReg(r)` or `Operand::Memory{base,..}`); if `ptr_imm[r]` is known, push `CoreBufAccess { local_off: imm & 0xFFFF, is_store, pc }`. If unknown, **skip** (can't prove contact → safe false-negative). Distinguish load vs store by which side the `Memory`/`PointerReg` operand sits (memory in `sources` = load; in `destinations` = store), mirroring `get_address` vs `get_store_address`.

- [ ] **Step 8: Run — passes.** `cargo test --lib core_relay::tests::extracts_add_one_buffer_contacts 2>&1 | tee /tmp/p1_b.txt`; Read it. Expected PASS.

- [ ] **Step 9: Write the ordering helper + its test.** Add the function pinned by P0 (default: straight-line):
```rust
/// Sound, CFG-free ordering proxy: is there an acq(L_in) ... load(in1) ... store(out1) ... rel(L_out)
/// sequence in strict PC order with NO branch PC strictly between the acquire and release? If control
/// flow lands in the region, we cannot prove the contact happens between the locks -> return false (block).
pub fn relay_ordered(u: &CoreLockUsage, l_in: u8, l_out: u8, in_range: (u32,u32), out_range: (u32,u32)) -> bool { /* ... */ }
```
Test asserts `relay_ordered(usage, 1, 3, (0x400,0x440), (0x440,0x480)) == true` for add_one, and a constructed out-of-order `CoreLockUsage` returns `false`, and one with a branch PC inside the region returns `false`.

- [ ] **Step 10: Run the ordering test — fails, then implement, then passes.**
```bash
cargo test --lib core_relay::tests::relay_ordering 2>&1 | tee /tmp/p1_c.txt
```
Implement per P0's pinned predicate. Re-run; Read; expect PASS.

- [ ] **Step 11: Full lib build + tests.**
```bash
cargo test --lib 2>&1 | tee /tmp/p1_full.txt
```
Read `/tmp/p1_full.txt`. Expected: all pass (3533+ baseline, +your new tests).

- [ ] **Step 12: Commit.**
```bash
git add src/device/stream_switch/core_relay.rs src/device/stream_switch/mod.rs
git commit -m "feat(#140): static core-program analysis (lock ops + buffer contacts + ordering)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P2: `EdgeKind::CoreLockRelay` + edge builder + wiring

**Files:**
- Modify: `src/device/stream_switch/route_graph.rs` (add enum variant; call builder in `resolve_route_graph`)
- Modify: `src/device/stream_switch/core_relay.rs` (add `core_lock_relay_edges`)
- Test: inline in `route_graph.rs` (mirror the E2/E3 edge-count tests)

**Interfaces:**
- Consumes: `analyze_core_program`, `relay_ordered`, `CoreLockUsage` (P1); `dma.get_bd`, `channel_bd_chain`, `start_bd_field_for`, `dma.resolve_lock_id` → `LockTarget::Own(id)`, `ChannelType::from_channel_index`, `tile.stream_switch.dma_master/dma_slave`, `BdConfig { base_addr, length, acquire_lock, release_lock, release_value, .. }` (Section refs in extraction).
- Produces: `pub fn core_lock_relay_edges(tile: &Tile, dma: &DmaEngine, s2mm_count: usize, usage: &CoreLockUsage) -> Vec<RouteEdge>`.

- [ ] **Step 1: Add the enum variant.** In `route_graph.rs` `EdgeKind` (after `LockPair`):
```rust
    /// Intra-tile through-core relay: the compute CORE program bridges an S2MM channel
    /// (writes the input buffer, RELEASES the data-ready lock the core ACQUIRES) to an MM2S
    /// channel (reads the output buffer, ACQUIRES the data-ready lock the core RELEASES).
    /// Emitted only when lock-pairing INTERSECT buffer-contact INTERSECT straight-line
    /// ordering all hold (lock identity alone is the co-firing trap). Claims STRUCTURAL
    /// data-contact under producer/consumer lock ordering -- the core had the opportunity
    /// to relay these bytes -- NOT value-dependence (the trace oracle cannot witness value
    /// flow). Oriented src = S2MM master DMA port (writer) -> dst = MM2S slave DMA port
    /// (reader); reverse is back-pressure, never emitted. Coverage is narrow (objectFIFO
    /// passthrough); non-recoverable buffer handles / control flow -> no edge (false-negative).
    CoreLockRelay,
```
serde `snake_case` ⇒ `"core_lock_relay"` (free).

- [ ] **Step 2: Write the failing edge test.**
```rust
#[test]
fn core_lock_relay_add_one_compute_tile() {
    let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
    let Some(state) = load_npu1_state_with_core_elfs(path) else { println!("SKIP: fixture absent"); return; };
    let g = state.resolve_route_graph();
    let relays: Vec<_> = g.edges.iter().filter(|e| e.kind == EdgeKind::CoreLockRelay).collect();
    // exactly one: compute tile (0,2) S2MM0 master -> MM2S0 slave
    assert_eq!(relays.len(), 1, "expected 1 CoreLockRelay, got {:?}", relays);
    let e = relays[0];
    assert_eq!((e.src.col, e.src.row, e.src.dir), (0, 2, PortDir::Master));
    assert_eq!((e.dst.col, e.dst.row, e.dst.dir), (0, 2, PortDir::Slave));
    // reverse (MM2S->S2MM) must NOT exist
    assert!(!g.edges.iter().any(|e| e.kind == EdgeKind::CoreLockRelay
        && e.src.dir == PortDir::Slave && e.dst.dir == PortDir::Master));
}
```
`load_npu1_state_with_core_elfs` = `load_npu1_state` + loading the compute core ELF(s) into tiles (see P3 Step 1 for the ELF-load helper; if P3 lands the dump-side loader first, reuse it; otherwise add a small test helper here that parses the xclbin's core ELF sections and calls `elf.load_into(tile)`).

- [ ] **Step 3: Run — fails.** `cargo test --lib route_graph::tests::core_lock_relay_add_one_compute_tile 2>&1 | tee /tmp/p2.txt`; Read it.

- [ ] **Step 4: Implement `core_lock_relay_edges`.** Mirror `dma_lock_pair_edges` structure, but join with `usage`:
```rust
pub fn core_lock_relay_edges(tile: &Tile, dma: &DmaEngine, s2mm_count: usize, usage: &CoreLockUsage) -> Vec<RouteEdge> {
    use crate::device::dma::{ChannelType, engine::LockTarget};
    let (col, row) = (tile.col, tile.row);
    let start_bd_field = start_bd_field_for(tile);
    let own_local = |raw: u8| match dma.resolve_lock_id(raw) { Some(LockTarget::Own(id)) => Some(id), _ => None };

    // per-channel: the data-ready lock released (S2MM) / acquired (MM2S), and the buffer byte range
    let s2mm_release_lock = |flat| -> Option<u8> { /* first bd with release_value>0 -> own_local(release_lock) */ };
    let mm2s_acquire_lock = |flat| -> Option<u8> { /* first bd -> own_local(acquire_lock) */ };
    let channel_range = |flat| -> Option<(u32,u32)> { /* first valid bd in chain -> (base_addr as u32, base_addr+length) */ };

    let core_acquires = |id: u8| usage.locks.iter().any(|l| l.lock_id==id && matches!(l.kind, CoreLockKind::Acquire));
    let core_releases = |id: u8| usage.locks.iter().any(|l| l.lock_id==id && matches!(l.kind, CoreLockKind::Release));

    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for s2mm_flat in 0..dma.num_channels() {
        if ChannelType::from_channel_index(s2mm_flat, s2mm_count) != ChannelType::S2MM { continue; }
        let (Some(l_in), Some(in_rng)) = (s2mm_release_lock(s2mm_flat), channel_range(s2mm_flat)) else { continue; };
        if !core_acquires(l_in) { continue; }
        let Some(src_port) = tile.stream_switch.dma_master(s2mm_flat as u8) else { continue; };
        for mm2s_flat in 0..dma.num_channels() {
            if ChannelType::from_channel_index(mm2s_flat, s2mm_count) != ChannelType::MM2S { continue; }
            let (Some(l_out), Some(out_rng)) = (mm2s_acquire_lock(mm2s_flat), channel_range(mm2s_flat)) else { continue; };
            if !core_releases(l_out) { continue; }
            // buffer contact + ordering (the soundness intersection)
            let has_in_load  = usage.accesses.iter().any(|a| !a.is_store && (in_rng.0..in_rng.1).contains(&a.local_off));
            let has_out_store= usage.accesses.iter().any(|a|  a.is_store && (out_rng.0..out_rng.1).contains(&a.local_off));
            if !(has_in_load && has_out_store) { continue; }
            if !relay_ordered(usage, l_in, l_out, in_rng, out_rng) { continue; }
            let mm2s_ch = (mm2s_flat - s2mm_count) as u8;
            let Some(dst_port) = tile.stream_switch.dma_slave(mm2s_ch) else { continue; };
            if !seen.insert((src_port.index, dst_port.index)) { continue; }
            edges.push(RouteEdge {
                src: PortRef{col,row,port:src_port.index,dir:PortDir::Master,kind:src_port.port_type.as_kind_str().to_owned()},
                dst: PortRef{col,row,port:dst_port.index,dir:PortDir::Slave, kind:dst_port.port_type.as_kind_str().to_owned()},
                kind: EdgeKind::CoreLockRelay,
            });
        }
    }
    edges
}
```
Fill the three closures using `channel_bd_chain(tile, dma, start_bd_field, flat)` + `dma.get_bd`. Note `channel_range` uses `base_addr as u32` (tile-local) to match `local_off`.

- [ ] **Step 5: Wire into `resolve_route_graph`.** Inside the `if let Some(dma) = self.array.dma_engine(tile.col, tile.row)` block, after the `dma_lock_pair_edges` loop, add:
```rust
                // --- Through-core relay edges (program_path source) ---
                if tile.tile_kind.is_compute() {  // only compute tiles run a core program
                    let dec = crate::interpreter::decode::loader::InstructionDecoder::load_cached();
                    let usage = crate::device::stream_switch::core_relay::analyze_core_program(
                        tile.program_bytes(), tile.program_base(), &dec);
                    for edge in crate::device::stream_switch::core_relay::core_lock_relay_edges(tile, dma, s2mm_count, &usage) {
                        g.add_edge(edge);
                    }
                }
```
If `tile` lacks `program_bytes()/program_base()` accessors, add minimal ones returning the program-memory slice + base (mirror `data_memory()`); confirm the gate `tile.tile_kind.is_compute()` exists (else use the appropriate predicate). `load_cached()` is a cheap clone of a cached singleton.

- [ ] **Step 6: Run the edge test — passes.** `cargo test --lib route_graph::tests::core_lock_relay_add_one_compute_tile 2>&1 | tee /tmp/p2.txt`; Read it. Expected PASS (or clean SKIP if fixture absent — if SKIP, also run the A5/E4 tests to confirm no regression and note the gating).

- [ ] **Step 7: Full lib tests.** `cargo test --lib 2>&1 | tee /tmp/p2_full.txt`; Read it. Expected: all green, existing E2/E3/A5/E4 edge-count tests unaffected (CoreLockRelay is additive).

- [ ] **Step 8: Commit.**
```bash
git add src/device/stream_switch/
git commit -m "feat(#140): CoreLockRelay edge (lock-pair INTERSECT buffer-contact INTERSECT ordering)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P3: Dump integration — load compute ELF + serialize the new edge

**Files:**
- Modify: `examples/dump_config_json.rs`
- Modify: `tools/config_extract/dump_model.py` (accept the new kind)
- Regenerate + commit: `tools/config_extract/fixtures/add_one_using_dma.config.json`

**Interfaces:**
- Consumes: the xclbin's compute-core ELF sections; `crate::parser::elf::Elf::parse` + `elf.load_into(tile)`; `state.resolve_route_graph()`.
- Produces: a dump JSON whose `route_graph.edges` includes `{"kind":"core_lock_relay", ...}` for add_one.

- [ ] **Step 1: Load compute core ELFs in the dump before resolving.** In `dump_config_json.rs`, after CDO + insts.bin are applied and before `resolve_route_graph()`, parse each compute-core ELF from the xclbin and `load_into` the matching tile. Find the existing xclbin parse in the example; the ELF sections are per-core (named by col/row). If the example currently only applies CDO, add: for each `(col,row)` compute tile, locate its ELF bytes in the partition, `Elf::parse(bytes)?.load_into(state.array.tile_mut(col,row))`. (Reuse whatever the interpreter/bridge path uses to map ELF→tile; grep for `load_into` callers.)

- [ ] **Step 2: Accept the new kind in dump_model.** In `tools/config_extract/dump_model.py`, `RouteEdge.kind`'s comment enumerates kinds; add `"core_lock_relay"`. No parse change needed if `kind` is a free `str`; confirm `load_dump` doesn't validate against a closed set (it doesn't per extraction). Add a one-line test asserting a `core_lock_relay` edge round-trips through `load_dump`.

- [ ] **Step 3: Regenerate the fixture.**
```bash
cd /home/triple/npu-work/xdna-emu
cargo run --example dump_config_json -- \
  /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin \
  /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/insts.bin \
  > tools/config_extract/fixtures/add_one_using_dma.config.json 2>/tmp/p3_dump.txt
```
Read `/tmp/p3_dump.txt` for errors. Then verify the fixture has a `core_lock_relay` edge:
```bash
grep -c core_lock_relay tools/config_extract/fixtures/add_one_using_dma.config.json
```
Expected: ≥1.

- [ ] **Step 4: Run the dump_model round-trip test.**
```bash
cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_config_extract_dump_model.py -v 2>&1 | tee /tmp/p3_py.txt
```
(If no `test_config_extract_dump_model.py` exists, add the round-trip assertion to the nearest existing config_extract test file.) Read it. Expected PASS.

- [ ] **Step 5: Commit.**
```bash
git add examples/dump_config_json.rs tools/config_extract/dump_model.py tools/config_extract/fixtures/add_one_using_dma.config.json
git commit -m "feat(#140): dump loads compute ELF so CoreLockRelay edges enter the config dump

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P4: `program_path` predicate in the engine (ledger + rules)

**Files:**
- Modify: `tools/inference/ledger.py`
- Modify: `tools/inference/rules.py`
- Test: `tools/test_inference_ledger.py`, `tools/test_inference_rules.py`

**Interfaces:**
- Consumes: `Fact`, `KB.by_predicate`, `Structural(cite)` (facts.py — unchanged).
- Produces: ledger entries with `kind:"program"` map to `program_path(a,b,cite)` facts; `try_derives` consumes both `config_path` and `program_path` for orientation.

- [ ] **Step 1: Write failing ledger test.** In `test_inference_ledger.py`:
```python
def test_program_kind_maps_to_program_path():
    from inference.ledger import ledger_facts
    led = {"program:p--via-core-->c": {"cite": "program:p--via-core-->c", "a": "p", "b": "c", "kind": "program"}}
    facts = ledger_facts(led)
    assert any(f.predicate == "program_path" and f.args[:2] == ("p", "c") for f in facts)
    assert not any(f.predicate == "config_path" for f in facts)
```

- [ ] **Step 2: Run — fails** (`unknown kind 'program'` from `_KINDS`, or maps to config_path).
```bash
cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_inference_ledger.py::test_program_kind_maps_to_program_path -v 2>&1 | tee /tmp/p4_a.txt
```
Read it.

- [ ] **Step 3: Implement.** In `ledger.py`: line 24 → `_KINDS = {"route", "bd", "lock", "identity", "program"}`. Line 46 `ledger_facts` mapping:
```python
        if e["kind"] == "identity":
            pred = "identity"
        elif e["kind"] == "program":
            pred = "program_path"
        else:
            pred = "config_path"
```
Update the module docstring schema to list `program -> program_path(a, b, cite)` and the new kind.

- [ ] **Step 4: Run — passes.** Re-run the Step-2 command; Read; expect PASS.

- [ ] **Step 5: Write failing rules test** (engine derives via program_path):
```python
def test_try_derives_consumes_program_path(tmp_path):
    # measured: child correlates to a stochastic parent; only a program_path gives orientation
    from inference.facts import KB
    from inference.ledger import install_ledger
    # build runs where child=parent+offset, parent stochastic (reuse the engine test's _runs helper pattern)
    ...
    kb = KB.empty()
    install_ledger(kb, {"program:S--via-core-->C": {"cite": "...", "a": "S", "b": "C", "kind": "program"}})
    d = try_derives(run_dirs, kb, "C", "S")
    assert d is not None and d.predicate == "derives"
```
(Mirror the existing `test_engine_reconstructs_placement` run-construction; assert the derive fires from a `program_path`-only ledger.)

- [ ] **Step 6: Run — fails** (try_derives only queries config_path).
```bash
cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_inference_rules.py::test_try_derives_consumes_program_path -v 2>&1 | tee /tmp/p4_b.txt
```
Read it.

- [ ] **Step 7: Implement.** In `rules.py` `try_derives`, change the orientation query (lines 40-41) to scan both predicates:
```python
    cp = next((f for f in (kb.by_predicate("config_path") + kb.by_predicate("program_path"))
               if f.args[0] == parent and f.args[1] == child), None)
```
(Same `a=parent, b=child` contract; both predicates carry it. Keep them as distinct facts in the KB — only the orientation *query* unions them, so audits/reports can still filter by predicate.)

- [ ] **Step 8: Run — passes.** Re-run Step-6 command; Read; expect PASS.

- [ ] **Step 9: Full inference suite (no regression).**
```bash
cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_inference_*.py -v 2>&1 | tee /tmp/p4_full.txt
```
Read it. Expected: all green (52+ baseline + 2 new).

- [ ] **Step 10: Commit.**
```bash
git add tools/inference/ledger.py tools/inference/rules.py tools/test_inference_ledger.py tools/test_inference_rules.py
git commit -m "feat(#140): real program_path predicate (distinct from config_path, shared orientation rule)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P5: Generator — program-taint reachability + `program` emission + audit

**Files:**
- Modify: `tools/config_extract/reachability.py`
- Modify: `tools/config_extract/generator.py`
- Test: `tools/test_config_extract_generator.py`, `tools/test_config_extract_reachability.py`

**Interfaces:**
- Consumes: `ConfigDump` (with `core_lock_relay` edges), `Reachability`.
- Produces: `generate_ledger` emits `{"kind":"program", "cite":"program:<parent>--via-core-->{child}", ...}` for pairs reachable ONLY when `core_lock_relay` edges are included; `audit_ledger` validates both kinds.

- [ ] **Step 1: Write failing reachability test** (config-only excludes program edges):
```python
def test_program_only_reachability_split():
    from config_extract.reachability import Reachability
    # 2 edges: A->B config (e.g. inter_tile), B->C only via core_lock_relay
    edges = [mk_edge(A, B, "inter_tile"), mk_edge(B, C, "core_lock_relay")]
    full = Reachability(edges)
    cfg  = Reachability([e for e in edges if e.kind != "core_lock_relay"])
    assert full.reachable(A, C) and not cfg.reachable(A, C)   # A->C needs the program edge
    assert cfg.reachable(A, B)                                 # A->B is config-only
```

- [ ] **Step 2: Run — fails or passes-trivially.** If `Reachability` already takes an arbitrary edge list, this may pass immediately (it just confirms the split strategy works). Run:
```bash
cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_config_extract_reachability.py::test_program_only_reachability_split -v 2>&1 | tee /tmp/p5_a.txt
```
Read it. If it passes with no code change, good — `Reachability` is already edge-list-parametric; the split lives in the generator (Step 3). If it fails, make `Reachability` accept the filtered list.

- [ ] **Step 3: Write failing generator test** (program pair emitted as program kind):
```python
def test_generates_program_path_for_through_core_pair():
    dump = load_dump(FIX)  # regenerated fixture has the core_lock_relay edge
    led = generate_ledger(dump, FIRED, start_col=START_COL)
    progs = [e for e in led["entries"] if e["kind"] == "program"]
    assert progs, "expected >=1 program entry for the through-core pair"
    for e in progs:
        assert e["cite"].startswith("program:")
        assert "--via-core-->" in e["cite"]
    # config pairs still emitted as route
    assert any(e["kind"] == "route" for e in led["entries"])
```
(`FIRED`/`START_COL` are the existing module constants; the regenerated fixture from P3 is required.)

- [ ] **Step 4: Run — fails.** `cd tools && python -m pytest test_config_extract_generator.py::test_generates_program_path_for_through_core_pair -v 2>&1 | tee /tmp/p5_b.txt`; Read it.

- [ ] **Step 5: Implement in `generator.py`.**
  - Add `_make_program_cite(parent, child) -> f"program:{parent}--via-core-->{child}"` and `_RE_PROGRAM_CITE = re.compile(r"^program:(?P<parent>.+?)--via-core-->(?P<child>.+)$")`.
  - In `generate_ledger`: build `full = Reachability(all_edges)` and `cfg = Reachability(non_core_lock_relay_edges)`. For each fired pair: if `cfg.reachable(parent, child)` → emit `kind:"route"` (existing path). Else if `full.reachable(parent, child)` → emit `kind:"program"` with `_make_program_cite`. Else decline (unchanged soundness).
  - Keep orientation `a=parent, b=child` for both (the P4 rule contract).

- [ ] **Step 6: Run — passes.** Re-run Step-4 command; Read; expect PASS.

- [ ] **Step 7: Extend `audit_ledger`.** Currently it hard-errors on `kind != "route"`. Branch: for `kind == "program"`, validate the cite against `_RE_PROGRAM_CITE` and that `m.group("parent")==a`, `m.group("child")==b` (mirror the route branch). For `kind == "route"`, unchanged. Add a test `test_audit_accepts_program_and_catches_program_cite_mismatch` (clean on a good program entry; bites on a corrupted program cite).

- [ ] **Step 8: Run audit test + full config_extract suite.**
```bash
cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_config_extract_*.py -v 2>&1 | tee /tmp/p5_full.txt
```
Read it. Expected: all green (75+ baseline + new).

- [ ] **Step 9: Commit.**
```bash
git add tools/config_extract/reachability.py tools/config_extract/generator.py tools/test_config_extract_*.py
git commit -m "feat(#140): generator emits program_path for through-core-only-reachable pairs

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P6: Two-witness runtime validation (E4-style superset gate)

**Files:**
- Modify: `src/interpreter/execute/control.rs` (core lock acquire-grant + release witnesses)
- Modify: `src/interpreter/execute/memory/mod.rs` (core buffer load/store witnesses)
- Modify: tile/array recorder plumbing (mirror E4's `enable_lock_recording`/`take_lock_events_by_tile`)
- Test: inline in `route_graph.rs` (mirror `static_graph_covers_enacted_lock_handoffs_add_one`)

**Interfaces:**
- Consumes: the runtime execution path; `InterpreterEngine::new_npu1()`; the static graph from `resolve_route_graph`.
- Produces: a recorder of core lock handoffs + core buffer touches; a test asserting every enacted through-core handoff is covered by a static `CoreLockRelay` edge with matching orientation.

- [ ] **Step 1: Add the two recorders (gated, zero-cost when off).** Mirror E4: add `Option<Vec<CoreLockEvent>>` and `Option<Vec<CoreBufEvent>>` recorders reachable per compute tile (on the tile, or on a per-tile struct the array exposes). `CoreLockEvent { cycle, lock_local_id, op: Acquire|Release, col, row }`; `CoreBufEvent { cycle, local_off, is_store, col, row }`. Add `enable_core_relay_recording()` / `take_core_relay_events()` on the array. Default `None` (no overhead).

- [ ] **Step 2: Hook the witnesses (two independent executor paths).**
  - In `control.rs`: at the `LockResult::Success` arm of `SemanticOp::LockAcquire` (~line 224) and the own-tile `defer_core_lock_release` site (~line 308), if the recorder is on, push a `CoreLockEvent` (resolve raw lock id to local via the tile's lock space; record `op`, `ctx.cycles`, `col`, `row`).
  - In `memory/mod.rs`: at each `get_address`/`get_store_address` hook site (5 sites listed in extraction), if the recorder is on AND the address is local (`addr >> 16 == 7`), push `CoreBufEvent { local_off: addr & 0xFFFF, is_store, cycle: ctx.cycles, col, row }`.
  These are genuinely separate paths (control vs memory) — the independence the gate needs.

- [ ] **Step 3: Write the failing validation test.**
```rust
#[test]
fn static_graph_covers_enacted_core_relays_add_one() {
    use crate::interpreter::engine::InterpreterEngine;
    // 1. static CoreLockRelay set from a fully-loaded state (CDO + ELFs)
    let Some(state) = load_npu1_state_with_core_elfs(PATH) else { println!("SKIP"); return; };
    let static_set: HashSet<(PhysKey,PhysKey)> = state.resolve_route_graph().edges.iter()
        .filter(|e| e.kind == EdgeKind::CoreLockRelay).map(|e| (phys(&e.src), phys(&e.dst))).collect();
    // 2. run with both recorders on
    let mut engine = InterpreterEngine::new_npu1(); // loads ELFs
    engine.device_mut().array.enable_core_relay_recording();
    // ... run to completion (mirror E4 loop) ...
    let (locks, bufs) = engine.device_mut().array.take_core_relay_events();
    // 3. reconstruct enacted through-core handoffs per compute tile:
    //    a core ACQUIRE of L_in (released by an S2MM) ordered before a core RELEASE of L_out
    //    (acquired by an MM2S), with a local LOAD in in1-range and a local STORE in out1-range
    //    between them (cycle order). Resolve to (S2MM master port -> MM2S slave port).
    // 4. assert every enacted handoff is in static_set (static ⊇ dynamic), orientation matching.
    for h in enacted { assert!(static_set.contains(&(h.src, h.dst)), "uncovered enacted relay {:?}", h); }
    assert!(!enacted.is_empty(), "add_one must enact >=1 through-core relay");
}
```

- [ ] **Step 4: Run — fails** (recorder/method missing). `cargo test --lib route_graph::tests::static_graph_covers_enacted_core_relays_add_one 2>&1 | tee /tmp/p6.txt`; Read it.

- [ ] **Step 5: Implement the reconstruction + assertion** in the test (the recorders land in Steps 1-2). Pair within a compute tile by cycle order: for each core ACQUIRE event of a lock that some S2MM releases, find a later core RELEASE of a lock some MM2S acquires, with an in1-range LOAD and out1-range STORE event in between. Map to ports via `tile.stream_switch.dma_master/dma_slave` (as the E4 test does). This is the runtime mirror of the static rule — it would FAIL if the static orientation were backwards (genuine oracle).

- [ ] **Step 6: Run — passes.** Re-run; Read; expect PASS (or clean SKIP if fixture absent). Also confirm the existing A5 + E4 tests still pass (recorders are additive/gated).

- [ ] **Step 7: Full lib tests.** `cargo test --lib 2>&1 | tee /tmp/p6_full.txt`; Read it. Expected all green.

- [ ] **Step 8: Commit.**
```bash
git add src/interpreter/ src/device/
git commit -m "test(#140): two-witness runtime validation of CoreLockRelay (static superset of enacted)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task P7: End-to-end engine derive + full regression

**Files:**
- Test: `tools/test_config_extract_generator.py` (or a new `test_program_path_e2e.py`)

**Interfaces:**
- Consumes: the regenerated fixture (P3), the program_path predicate (P4), the generator (P5), `run_engine`.
- Produces: a test proving the engine DERIVES the through-core pair from the GENERATED ledger (the C4/E6 lesson: derives ≠ loads), over captured HW runs.

- [ ] **Step 1: Write the end-to-end derive test.** Using the captured runs under `build/experiments/infer-smoke` (the Plan-1 capture; guard skip-if-absent) and the regenerated fixture:
```python
def test_engine_derives_through_core_relay_from_generated_ledger(tmp_path):
    dump = load_dump(FIX)
    fired = fired_keys_from_run(RUN_DIR, ANCHOR)         # existing helper
    led = generate_ledger(dump, fired, start_col=START_COL)
    p = tmp_path / "gen.ledger.json"; p.write_text(json.dumps(led))
    # the through-core pair must be a program entry
    assert any(e["kind"] == "program" for e in led["entries"])
    # engine derives it (not just loads it): run_engine over captured runs + generated ledger
    rep = run_engine(RUN_DIRS, str(p), candidate_pairs_from(led))
    assert rep["provenance_ok"] is True
    assert rep["replication_violations"] == []
    # the previously-residual shim-DMA / through-core pair now appears in derives
    assert any(<through-core child/parent/offset> in rep["derives"])  # fill exact keys from the run
```
(Resolve the exact event keys + expected offset from the captured run during implementation; the spec notes the residual is the shim-DMA causality / STREAM_STARVATION pair the compute relay now bridges.)

- [ ] **Step 2: Run — fails or reveals the real derive keys.** `cd tools && python -m pytest test_config_extract_generator.py::test_engine_derives_through_core_relay_from_generated_ledger -v 2>&1 | tee /tmp/p7.txt`; Read it. Use the output to pin the exact derived tuple, then finalize the assertion.

- [ ] **Step 3: Run — passes.** Re-run; Read; expect PASS (or documented SKIP if `build/experiments/infer-smoke` absent — if absent, assert at least that the generated ledger loads + `provenance_ok` over a synthetic run set, and note the HW-capture gating).

- [ ] **Step 4: Full regression — both suites.**
```bash
cargo test --lib 2>&1 | tee /tmp/p7_rust.txt
cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_config_extract_*.py test_inference_*.py -v 2>&1 | tee /tmp/p7_py.txt
```
Read both. Expected: all green; config_path derivation unchanged (Global Constraint R4).

- [ ] **Step 5: Commit.**
```bash
git add tools/
git commit -m "test(#140): engine derives through-core relay from generated ledger (E2E, derives not loads)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Self-Review (against the spec)

**Spec coverage:**
- §1 ceiling X (static derive + runtime validate, defer trace/HW) → P0–P6 derive+validate; P7 notes HW-capture gating; trace/HW explicitly out. ✓
- §1 non-goals (no value-dependence; narrow coverage) → Global Constraints + EdgeKind doc + `relay_ordered`/reaching-def block-on-ambiguity. ✓
- §3 derivation = lock ∩ buffer ∩ dominance → P2 `core_lock_relay_edges` (all three) + P1 `relay_ordered`. ✓
- §4 static analysis (reaching-def + CFG-free ordering, address reconciliation, block+report) → P0 spike + P1. ✓
- §5 shared route graph + real program_path predicate → P2 (edge in `resolve_route_graph`) + P4 (ledger/rules). ✓
- §6 two-witness validation + config-vs-ELF independence → P6 (control vs memory paths) + the config-range (BD) vs ELF-access cross-check baked into the rule. ✓
- §7 testing (Rust unit + validation + Python derives-not-loads) → P2/P6 Rust, P4/P5/P7 Python. ✓

**Placeholder scan:** the `<through-core child/parent/offset>` in P7 and the closure bodies in P2 Step 4 are intentionally resolved-at-implementation (they depend on the captured run / are mechanical fills of a shown pattern) — each has the exact shape + the helper to fill it. The `relay_ordered` predicate is pinned by P0's findings, not left vague. No "TBD/add error handling" placeholders.

**Type consistency:** `CoreLockUsage`/`CoreLockOp`/`CoreBufAccess` (P1) consumed identically in P2; `local_off` (tile-local, `& 0xFFFF`) matches `BdConfig.base_addr` tile-local ranges throughout; `EdgeKind::CoreLockRelay` ⇒ serde `"core_lock_relay"` used consistently in P3 dump_model / P5 generator filter; `program` kind ⇒ `program_path` predicate ⇒ `program:...--via-core-->...` cite consistent across P4/P5/P7; orientation `a=parent,b=child` consistent P4/P5.

**Known risk carried into execution:** P0 is a hard gate — if buffer pointers or ordering are not statically recoverable on the real ELF, STOP and report (do not force P1+). This is the single most important sequencing rule.
