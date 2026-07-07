# Firmware MMU/DTLB Data Path -- Translation-Authoritative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make translation authoritative for every CPU memory access in the Xtensa firmware interpreter, route on `(physical address, I/D side)` instead of the virtual-address `is_local_data` bypass, and expose one canonical translation-aware data accessor shared by the executor and probes -- retiring the memory-alias tax.

**Architecture:** Add a side-explicit physical Harvard API to `Bus` (`data_*` = D-side, `inst_*` = I-side; the image-vs-DRAM fork lives only in the low range), add a canonical `Cpu::data_*` accessor (`translate` then D-side route), migrate every consumer file to it, then delete the ambiguous bare `load32/store32/load8/store8`. Additive-first, migrate-per-file, remove-bare-last -- the tree compiles and every test passes at every commit.

**Tech Stack:** Rust; existing `src/firmware/` interpreter. No new dependencies.

**Design spec:** `docs/superpowers/specs/2026-07-06-firmware-mmu-data-path-design.md` (read it for the full rationale, the characterization finding, and the adversarial-review fixes baked into the tasks below).

## Global Constraints

- **The MMU model (`mmu.rs`) is NOT re-architected** -- only the one autorefill PTE-fetch call is renamed. Lookup/autorefill/permission logic is untouched.
- **Behavior-preserving on the real boot:** the firmware boot suite stays **433 passed / 0 failed / 1 ignored** after every task (`m2c_boot_completion_advances_past_recursion` stays `#[ignore]`d).
- **No net executor behavior change:** every existing `mem`/`mmu`/`fastpath`/`interp`/`mmio`/`host_mailbox`/`psp_map` test stays green. Where a call is renamed, its asserted values must not change.
- **Low-address migration is by INTENT, not sed:** both `data_*` and `inst_*` compile at a low-address site, so each is audited for image (I-side) vs DRAM (D-side). High-address sites are unambiguous.
- **The Harvard I/D fork exists only for `paddr < LOCAL_DATA_END`** (`0x0400_0000`). For any high paddr, `data_*` and `inst_*` route identically (region backing).
- **No emoji.** Comments state the hardware *why*. Commit messages end with `Generated using Claude Code.`
- **Run tests bare** (never piped through head/tail/grep). `cargo test --lib` after every task. The boot-suite tests skip cleanly when the firmware binary is absent, so a machine without it still validates everything except the two boot integration tests.

---

## File Structure

| File | Responsibility change |
|------|----------------------|
| `src/firmware/mmio.rs` | Add side-explicit `data_*`/`inst_*`/`data_fill`; later remove bare `load32/store32/load8/store8` and fold `load_local*/store_local*/fill_local/fill_pattern` into `data_*` internals. |
| `src/firmware/xtensa/interp/mod.rs` | Add canonical `Cpu::data_read{8,32}` / `data_write{8,32}`. |
| `src/firmware/xtensa/interp/mem.rs` | Executor ops call `Cpu::data_*`; delete `is_local_data` bypass + `assert_low_window_identity`; `l32r` -> `inst_load32`; preserve `store16` two-phase. |
| `src/firmware/xtensa/interp/fastpath.rs` | Translate every chunk; fill via `bus.data_fill`; drop the `is_local_data` special case. |
| `src/firmware/host_mailbox.rs` | External-actor physical calls renamed to `data_*` (behavior identical). |
| `src/firmware/xtensa/mmu.rs` | Autorefill PTE fetch + test PTE stores -> `data_*`. |
| `src/firmware/psp_map.rs` | Test `load32` -> `data_load32`. |
| `src/firmware/mod.rs` | Boot-test probes migrated by intent; Task 1 characterization test added. |
| `src/firmware/xtensa/coverage_scan.rs` | Verify only `peek8` is used (expected: no migration). |

---

## Task 1: Characterization lock (structural way-6 ei0 facts)

Assert the facts the whole design rests on, so a future change that breaks low-window coverage fails loudly here.

**Files:**
- Modify: `src/firmware/mod.rs` (add a test in `mod boot_tests`)

**Interfaces:**
- Consumes: `Mmu::new_with_varway56`, `Mmu::lookup`, `Mmu::translate`, `Mmu::dtlb` (all public), `FirmwareProcessor::load_m2c`, `firmware_path`, `Mmu::write_rasid`.
- Produces: nothing consumed downstream (pure regression lock).

- [ ] **Step 1: Write the failing test.** In `src/firmware/mod.rs`, in `mod boot_tests`, add:

```rust
/// Characterization lock (MMU data-path design, 2026-07-06): the
/// translation-authoritative data path depends on the low DRAM window being
/// TLB-covered from reset through steady state. Assert the STRUCTURAL facts,
/// not just a point sample: way-6 ei0 is the reset identity region, the
/// prologue clears entry 1 (code region) not entry 0, asid resolves ring 0,
/// and every low-window data probe still translates identity at steady state.
#[test]
fn low_window_dram_is_translation_covered_from_reset() {
    use crate::firmware::xtensa::mmu::Mmu;
    // (a) Reset populates way-6 ei0 = VPN 0 -> paddr 0, asid 1, attr 3, variable.
    let fresh = Mmu::new_with_varway56(true);
    let e0 = fresh.dtlb[6][0];
    assert_eq!(e0.vaddr, 0);
    assert_eq!(e0.paddr, 0);
    assert_eq!(e0.asid, 1);
    assert_eq!(e0.attr, 3, "RWX -- grants low-window read AND write");
    assert!(e0.variable);
    // (c) asid 1 always resolves to ring 0 (write_rasid forces the ring-0 byte).
    let mut r = Mmu::new_with_varway56(true);
    r.write_rasid(0x08070605);
    assert_eq!(r.lookup(0x0000_f9e0, true).expect("low window resolves").ring, 0);

    // (b) + (d) need the real boot; skip cleanly without the binary.
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary absent -- structural (a)/(c) still checked");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    for _ in 0..300_000 {
        if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
            break;
        }
    }
    // (b) prologue invalidated way-6 entry 1 (code region), left entry 0 (low window).
    assert_eq!(proc.cpu.mmu.dtlb[6][0].asid, 1, "low-window entry 0 still live");
    assert_eq!(proc.cpu.mmu.dtlb[6][1].asid, 0, "code-region entry 1 invalidated");
    // (d) every low-window data address the firmware touches translates identity.
    for a in [0x0000_f9e0u32, 0x0000_9070, 0x0000_2278, 0x0000_2250, 0x0000_22bc] {
        let t = proc.cpu.mmu.translate(&mut proc.bus, a, 0 /*load*/, 0).expect("resolves");
        assert_eq!(t.paddr, a, "low-window data translates identity");
    }
}
```

- [ ] **Step 2: Run it.** Run: `cargo test --lib low_window_dram_is_translation_covered_from_reset -- --nocapture`
  Expected: PASS (structural facts already hold; the boot arm passes if the binary is present, skips otherwise).
- [ ] **Step 3: Commit.**

```bash
git add src/firmware/mod.rs
git commit -m "test(#140): characterization lock -- low DRAM window TLB-covered from reset

Generated using Claude Code."
```

---

## Task 2: Bus side-explicit physical Harvard API (additive)

Add the D-side and I-side physical accessors alongside the existing methods. Nothing is removed yet.

**Files:**
- Modify: `src/firmware/mmio.rs`

**Interfaces:**
- Consumes: existing private helpers `read_le32`/`write_le32`/`byte_at`/`set_byte_at`/`fill_mem`, `Region`, `record_stub`, the backing `Vec`s (`rom`/`ram`/`mailbox`/`page_table`/`local_data`), `load_offset`, `LOCAL_DATA_END`.
- Produces (public API later tasks consume):
  - `fn data_load32(&mut self, paddr: u32) -> u32`
  - `fn data_load8(&mut self, paddr: u32) -> u8`
  - `fn data_store32(&mut self, paddr: u32, v: u32)`
  - `fn data_store8(&mut self, paddr: u32, v: u32)`
  - `fn data_fill(&mut self, paddr: u32, pattern: &[u8], byte_len: usize)`
  - `fn inst_load8(&mut self, paddr: u32) -> u8`
  - `fn inst_load32(&mut self, paddr: u32) -> u32`

**Design notes for the implementer:**
- **D-side routing:** `paddr < LOCAL_DATA_END` -> `local_data` (DRAM, offset == paddr); else the *exact current `load32`/`store32` region behavior* (Ram/Mailbox/PageTable backing; Array/System stubbed with `record_stub`; Rom write dropped+logged). Reuse the existing region match arms -- do not re-derive them.
- **I-side routing:** `paddr < LOCAL_DATA_END` -> `rom` image (`byte_at`/`read_le32` at `paddr + load_offset`); else same region behavior as D-side. `inst_load8` is the body `fetch8` already calls via `load8` for the non-overlay path -- have `fetch8`'s fallthrough call `inst_load8`.
- **`data_fill` contract (adversarial finding 1):** byte-identical to `byte_len/pattern.len()` successive `data_store8` at consecutive paddrs. Split the range at `LOCAL_DATA_END` **and at every region boundary**; for the `local_data` sub-span reuse `fill_local` (preserving its zero-pattern no-grow optimization); for other sub-spans reuse `fill_pattern`. A simple correct implementation: walk the range, at each step compute the distance to the next boundary (`LOCAL_DATA_END` if below it, else the next region end), fill that sub-chunk via the matching helper, advance.

- [ ] **Step 1: Write failing tests.** In `mmio.rs` `mod tests` add:

```rust
#[test]
fn data_side_low_hits_dram_inst_side_low_hits_image() {
    // Low paddr: D-side -> local_data (DRAM), I-side -> rom (image). Distinct backings.
    let mut bus = Bus::new(vec![0x11, 0x22, 0x33, 0x44]); // image bytes at paddr 0..4
    assert_eq!(bus.inst_load32(0x0), 0x4433_2211, "I-side low reads the image");
    bus.data_store32(0x0, 0xdead_beef);
    assert_eq!(bus.data_load32(0x0), 0xdead_beef, "D-side low reads/writes DRAM");
    assert_eq!(bus.inst_load32(0x0), 0x4433_2211, "image untouched by the D-side store");
}

#[test]
fn data_and_inst_agree_on_high_addresses() {
    // No Harvard split above LOCAL_DATA_END: both families -> the same region backing.
    let mut bus = Bus::new(vec![]);
    bus.data_store32(0x08b0_0100, 0xcafe_babe); // RAM aperture
    assert_eq!(bus.data_load32(0x08b0_0100), 0xcafe_babe);
    assert_eq!(bus.inst_load32(0x08b0_0100), 0xcafe_babe, "high I-side == high D-side");
}

#[test]
fn data_load_high_records_stub_like_load32() {
    // Mailbox/Array/System D-side reads still record a StubAccess (probe fidelity).
    let mut bus = Bus::new(vec![]);
    bus.arm_probe();
    bus.data_load32(0x2701_0d00); // Mailbox
    bus.data_store32(0x0400_0000, 1); // Array
    assert_eq!(bus.take_probe().len(), 2, "D-side high accesses are probe-recorded");
}

#[test]
fn data_fill_is_byte_identical_to_per_byte_stores_across_boundary() {
    // Adversarial finding 1: a NON-ZERO fill spanning LOCAL_DATA_END must route
    // each side exactly as data_store8 would -- DRAM below, Array (dropped) above,
    // with NOTHING mis-written into local_data above the boundary.
    let mut fill = Bus::new(vec![]);
    let mut loop_ = Bus::new(vec![]);
    let start = LOCAL_DATA_END - 0x800;
    let len = 0x1000usize; // 0x800 DRAM + 0x800 Array
    fill.data_fill(start, &[0xcd], len);
    for i in 0..len as u32 {
        loop_.data_store8(start + i, 0xcd);
    }
    for i in 0..len as u32 {
        let a = start + i;
        assert_eq!(fill.data_load8(a), loop_.data_load8(a), "byte {a:#x} matches per-store");
    }
    // The array side is dropped: reads back 0, and nothing leaked into local_data.
    assert_eq!(fill.data_load8(LOCAL_DATA_END), 0, "array-side byte dropped, not in DRAM");
    assert_eq!(fill.load_local8(LOCAL_DATA_END), 0, "no mis-route into DRAM above the boundary");
}

#[test]
fn data_fill_zero_does_not_grow_dram_backing() {
    // The boot's 128 MiB zero-memset must not allocate: zero fill into never-written
    // DRAM space is a no-op that reads 0 without growing the backing.
    let mut bus = Bus::new(vec![]);
    let before = bus.local_data_len_for_test();
    bus.data_fill(0x0100_0000, &[0u8], 0x0010_0000); // 16 MiB zero fill, low window
    assert_eq!(bus.local_data_len_for_test(), before, "zero fill must not grow DRAM");
    assert_eq!(bus.data_load8(0x0100_0000), 0);
}
```

- [ ] **Step 2: Run to verify they fail** (methods don't exist yet). Run: `cargo test --lib firmware::mmio -- --nocapture`  Expected: FAIL to compile (undefined methods).
- [ ] **Step 3: Implement the seven methods** per the design notes above, reusing the existing region match arms and fill helpers. `data_load32`/`data_store32`/`data_load8`/`data_store8`: `if paddr < LOCAL_DATA_END { local_data path } else { current region match }`. `inst_load8`/`inst_load32`: `if paddr < LOCAL_DATA_END { rom+load_offset path } else { current region match }`. `data_fill`: boundary-split walk.
- [ ] **Step 4: Run all mmio tests.** Run: `cargo test --lib firmware::mmio`  Expected: PASS (new tests green; all existing bare-method tests still green -- nothing removed).
- [ ] **Step 5: Commit.**

```bash
git add src/firmware/mmio.rs
git commit -m "feat(#140): Bus side-explicit Harvard API (data_*/inst_*/data_fill)

Generated using Claude Code."
```

---

## Task 3: Canonical `Cpu::data_*` accessor (additive)

**Files:**
- Modify: `src/firmware/xtensa/interp/mod.rs`

**Interfaces:**
- Consumes: `Cpu::translate`, `Access::{Load,Store}`, `Bus::data_load32/8`, `Bus::data_store32/8`.
- Produces:
  - `fn data_read32(&mut self, bus: &mut Bus, vaddr: u32) -> Result<u32, Step>`
  - `fn data_read8(&mut self, bus: &mut Bus, vaddr: u32) -> Result<u8, Step>`
  - `fn data_write32(&mut self, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step>`
  - `fn data_write8(&mut self, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step>`

- [ ] **Step 1: Write failing tests.** In `interp/mod.rs` `mod tests`:

```rust
#[test]
fn data_accessor_low_hits_dram_high_translates() {
    use crate::firmware::mmio::Bus;
    let mut cpu = Cpu::new(0);
    let mut bus = Bus::new(vec![]);
    // Low window is covered by the varway56 way-6 ei0 identity region.
    cpu.mmu = crate::firmware::xtensa::mmu::Mmu::new_with_varway56(true);
    cpu.data_write32(&mut bus, 0x0000_2278, 0x9040).expect("low D-side write");
    assert_eq!(cpu.data_read32(&mut bus, 0x0000_2278).expect("low D-side read"), 0x9040);
    assert_eq!(bus.load_local32(0x2278), 0x9040, "landed in DRAM, identity paddr");
    // High: map a data page and round-trip through translation.
    cpu.mmu.write_tlb(true, 0x08b0_0000 | 0x3, 0x4000_0000 | 0);
    cpu.data_write32(&mut bus, 0x4000_0010, 0xbeef).expect("high write translates");
    assert_eq!(cpu.data_read32(&mut bus, 0x4000_0010).expect("high read"), 0xbeef);
    assert_eq!(bus.data_load32(0x08b0_0010), 0xbeef, "translated to RAM paddr");
}

#[test]
fn data_accessor_fault_propagates_as_exception() {
    use crate::firmware::mmio::Bus;
    let mut cpu = Cpu::new(0);
    cpu.vecbase = 0x4000_0000;
    let mut bus = Bus::new(vec![]);
    // Unmapped high page -> DTLB miss -> Step::Exception, pc not advanced by the op.
    match cpu.data_read32(&mut bus, 0x5000_0000) {
        Err(Step::Exception { cause, .. }) => assert_eq!(cause, 24),
        other => panic!("expected LOAD_STORE_TLB_MISS, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run to verify they fail.** Run: `cargo test --lib firmware::xtensa::interp::tests::data_accessor -- --nocapture`  Expected: FAIL to compile.
- [ ] **Step 3: Implement** in `impl Cpu`:

```rust
/// Canonical translation-aware CPU data read: translate (DTLB, load), then
/// D-side physical route. The SOLE entry for "what the CPU reads at virtual
/// data address V" -- executor ops and probes share it, so probe-vs-CPU
/// disagreement is structurally impossible. Faults propagate as Step::Exception
/// exactly as Cpu::translate raises them.
pub fn data_read32(&mut self, bus: &mut Bus, vaddr: u32) -> Result<u32, Step> {
    let paddr = self.translate(bus, vaddr, Access::Load)?;
    Ok(bus.data_load32(paddr))
}
pub fn data_read8(&mut self, bus: &mut Bus, vaddr: u32) -> Result<u8, Step> {
    let paddr = self.translate(bus, vaddr, Access::Load)?;
    Ok(bus.data_load8(paddr))
}
pub fn data_write32(&mut self, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step> {
    let paddr = self.translate(bus, vaddr, Access::Store)?;
    bus.data_store32(paddr, v);
    Ok(())
}
pub fn data_write8(&mut self, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step> {
    let paddr = self.translate(bus, vaddr, Access::Store)?;
    bus.data_store8(paddr, v);
    Ok(())
}
```

- [ ] **Step 4: Run.** Run: `cargo test --lib firmware::xtensa::interp`  Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/firmware/xtensa/interp/mod.rs
git commit -m "feat(#140): canonical Cpu::data_* translation-aware accessor

Generated using Claude Code."
```

---

## Task 4: Migrate the executor (`mem.rs`) to the accessor

The core behavior change: delete the `is_local_data` bypass; route all data ops through `Cpu::data_*`; `l32r` reads the image via `inst_load32`; delete `assert_low_window_identity`; preserve `store16`'s no-half-write guarantee.

**Files:**
- Modify: `src/firmware/xtensa/interp/mem.rs`

**Interfaces:**
- Consumes: `Cpu::data_read{8,32}`, `Cpu::data_write{8,32}` (Task 3), `Bus::inst_load32` (Task 2), `Cpu::translate`, `Access`, `Bus::is_local_data`.

**Migration detail:**
- The exec match arms currently call the free fns `data_load32/data_load8/data_store32/data_store8(cpu, bus, vaddr[, val])`. Replace those calls with the `Cpu` methods: `cpu.data_read32(bus, vaddr)?`, `cpu.data_write32(bus, vaddr, val)?`, `cpu.data_read8`, `cpu.data_write8`. Keep the "read AR before the `&mut cpu` borrow" ordering already present.
- Delete the free fns `data_load32`/`data_load8`/`data_store32`/`data_store8` and `assert_low_window_identity` (the bypass and its guard are gone).
- `l32r_load`: keep `let paddr = cpu.translate(bus, target, Access::Load)?;` then `Ok(bus.inst_load32(paddr))` (was `bus.load32(paddr)` -- I-side image, unchanged semantics; the iter12 regression test still passes).
- `load16`/`store16`: rewrite `load16` to use `cpu.data_read8`; keep `store16`'s TWO-PHASE order -- translate/validate BOTH byte destinations before writing either (adversarial finding 3):

```rust
fn store16(cpu: &mut Cpu, bus: &mut Bus, addr: u32, v: u16) -> Result<(), Step> {
    // No-half-write: validate both byte destinations (translate) before writing
    // either, so a page-straddling store16 whose high byte faults never applies
    // the low byte. Under translation-authoritative BOTH bytes translate/fault.
    let (lo, hi) = (addr, addr.wrapping_add(1));
    cpu.translate(bus, lo, Access::Store)?;
    cpu.translate(bus, hi, Access::Store)?;
    cpu.data_write8(bus, lo, (v & 0xFF) as u32)?;
    cpu.data_write8(bus, hi, (v >> 8) as u32)?;
    Ok(())
}
```

- [ ] **Step 1: Write the tests.** Two guards in `mem.rs` `mod tests` -- a
  straddle-fault regression lock (`store16`'s two-phase order holds today; this
  keeps a naive `data_write8(lo)?; data_write8(hi)?` rewrite from silently
  dropping it) and the D-side equivalence invariant (probe == CPU):

```rust
#[test]
fn store16_high_byte_fault_leaves_low_byte_unwritten() {
    use crate::firmware::mmio::Bus;
    // s16i a4,a7,0 with a7 at the last byte of a mapped page; the high byte spills
    // into the next (unmapped) page -> fault, and the low byte must NOT be applied.
    let rom = vec![0x42, 0x57, 0x00]; // s16i a4,a7,0
    let mut bus = Bus::new(rom);
    let mut cpu = Cpu::new(0);
    cpu.vecbase = 0x4000_0000;
    cpu.mmu.write_tlb(false, 0x0 | 0x1, 0x0 | 0); // code page R+X
    // Map data page 0x10000 -> RAM 0x08b00000, but NOT the next page.
    cpu.mmu.write_tlb(true, 0x08b0_0000 | 0x3, 0x0001_0000 | 0);
    cpu.regs.write_ar(7, 0x0001_0fff); // last byte of the mapped page
    cpu.regs.write_ar(4, 0xABCD);
    match cpu.step(&mut bus) {
        Step::Exception { cause, .. } => assert_eq!(cause, 24), // LOAD_STORE_TLB_MISS on the high byte
        other => panic!("expected straddle fault, got {other:?}"),
    }
    assert_eq!(bus.data_load8(0x08b0_0fff), 0, "low byte must NOT be applied on high-byte fault");
}

#[test]
fn executor_result_equals_probe_read_dside() {
    use crate::firmware::mmio::Bus;
    // Equivalence invariant (D-side loads only; l32r is I-side by design and excluded):
    // a store executed by the CPU is read back identically by cpu.data_read32.
    let rom = vec![0x69, 0xc7]; // s32i.n a6,a7,0x30
    let mut bus = Bus::new(rom);
    let mut cpu = mapped_cpu(0);
    map_data(&mut cpu, 0x08b00000);
    cpu.regs.write_ar(7, 0x08b0_0000);
    cpu.regs.write_ar(6, 0x1234_5678);
    assert!(matches!(cpu.step(&mut bus), Step::Ran));
    assert_eq!(cpu.data_read32(&mut bus, 0x08b0_0030).unwrap(), 0x1234_5678, "probe == CPU");
}
```

- [ ] **Step 2: Run the new tests.** Run: `cargo test --lib firmware::xtensa::interp::mem`  Expected: both compile and pass against the current code (they are regression/equivalence locks, not red-first tests). They must STAY green through the Step 3 migration -- that is their job: the straddle test fails if the migration drops `store16`'s two-phase order; the equivalence test fails if the executor and `data_read32` ever diverge.
- [ ] **Step 3: Implement the migration** per the Migration detail above (rewire exec arms, delete the four free fns + `assert_low_window_identity`, `l32r` -> `inst_load32`, `store16` two-phase).
- [ ] **Step 4: Run mem + the boot suite.** Run: `cargo test --lib firmware::xtensa::interp::mem` then `cargo test --lib firmware::boot_tests`  Expected: PASS (all existing mem tests, the two new tests, and the boot suite 433/0/1).
- [ ] **Step 5: Commit.**

```bash
git add src/firmware/xtensa/interp/mem.rs
git commit -m "feat(#140): executor data ops route through Cpu::data_* (drop is_local_data bypass)

Generated using Claude Code."
```

---

## Task 5: Migrate the fill fast-path (`fastpath.rs`)

**Files:**
- Modify: `src/firmware/xtensa/interp/fastpath.rs`

**Interfaces:**
- Consumes: `Mmu::translate` (non-raising, unchanged), `Bus::data_fill` (Task 2), `Cpu::translate` (fault-replication arm, unchanged).

**Migration detail:**
- Delete the `if Bus::is_local_data(vaddr) { ... fill_local ... } else { ... fill_pattern ... }` split (`fastpath.rs:82-114`). Replace with a single loop that, for each chunk, translates via `cpu.mmu.translate(bus, vaddr, 1, 0)` (store), computes the page-bounded chunk, and calls `bus.data_fill(t.paddr, pattern, chunk)`. `data_fill` now owns the region/`LOCAL_DATA_END` sub-splitting (Task 2 contract), so the fast-path no longer special-cases the low window. The fault-replication arm (translate via `cpu.translate` to raise, reconstruct ptr/lcount) is unchanged.
- Because way-6 ei0 covers the low window (Task 1), the low-window fill now translates like every other chunk -- no bypass needed.

- [ ] **Step 1: Write the failing test** -- the fast-vs-grind check the old spanning test could not catch (adversarial finding 1), asserting `local_data` state ABOVE `LOCAL_DATA_END` is identical fast vs grind after a non-zero straddling fill:

```rust
#[test]
fn fastpath_nonzero_straddle_no_dram_leak_above_boundary() {
    // A non-zero byte fill crossing LOCAL_DATA_END must, fast AND grind, leave
    // local_data ABOVE the boundary untouched (the array side is dropped, not
    // mis-routed into DRAM). The pre-existing spanning test reads the array side
    // via the region path and cannot see a DRAM leak; this one reads local_data.
    const CODE: u32 = 0x08b0_0000;
    const DEST: u32 = crate::firmware::mmio::LOCAL_DATA_END - 0x800;
    const N: u32 = 0x1000; // 0x800 local + 0x800 array
    const BOUNDARY: u32 = crate::firmware::mmio::LOCAL_DATA_END;

    let run = |fast: bool| -> Vec<u8> {
        let mut cpu = Cpu::new(CODE);
        cpu.mmu = crate::firmware::xtensa::mmu::Mmu::new_with_varway56(true);
        cpu.fastpath_enabled = fast;
        cpu.mmu.write_tlb(false, (CODE & 0xfff0_0000) | 0x1, (CODE & 0xfff0_0000) | 4);
        let mut bus = Bus::new(vec![]);
        let lend = place_byte_fill_body(&mut bus, CODE);
        cpu.pc = CODE;
        cpu.regs.lbeg = CODE;
        cpu.regs.lend = lend;
        cpu.regs.lcount = N - 1;
        cpu.regs.write_ar(5, DEST);
        cpu.regs.write_ar(3, 0xcd);
        for _ in 0..(N * 4 + 16) {
            if cpu.pc == lend { break; }
            match cpu.step(&mut bus) { Step::Ran | Step::Exception { .. } => {}, o => panic!("{o:?}") }
        }
        // local_data across and ABOVE the boundary.
        (DEST..DEST + N).map(|a| bus.load_local8(a)).collect()
    };
    let fast = run(true);
    let grind = run(false);
    assert_eq!(fast, grind, "DRAM state identical fast vs grind, including above the boundary");
    // Below the boundary: filled; at/above: DRAM untouched (0), not 0xcd.
    let split = (BOUNDARY - DEST) as usize;
    assert!(fast[..split].iter().all(|&b| b == 0xcd), "DRAM side filled");
    assert!(fast[split..].iter().all(|&b| b == 0), "no 0xcd leaked into DRAM above the boundary");
}
```

- [ ] **Step 2: Run to verify it fails** against the current code path if the split is wrong, or passes if already correct; either way it must pass after the migration. Run: `cargo test --lib firmware::xtensa::interp::fastpath -- --nocapture`
- [ ] **Step 3: Implement** the single-loop migration per the Migration detail.
- [ ] **Step 4: Run fastpath + boot suite.** Run: `cargo test --lib firmware::xtensa::interp::fastpath` then `cargo test --lib firmware::boot_tests`  Expected: PASS (all existing fastpath tests, the new leak test, boot 433/0/1).
- [ ] **Step 5: Commit.**

```bash
git add src/firmware/xtensa/interp/fastpath.rs
git commit -m "feat(#140): fill fast-path translates uniformly via data_fill

Generated using Claude Code."
```

---

## Task 6: Migrate `host_mailbox.rs` (external actors)

Behavior-preserving rename to the side-explicit physical API.

**Files:**
- Modify: `src/firmware/host_mailbox.rs`

**Migration detail (each is behavior-identical -- low offsets are identity paddrs, high mailbox regs keep region routing + `record_stub`):**
- `bus.load_local32(SCHED_CURRENT_TASK)` -> `bus.data_load32(SCHED_CURRENT_TASK)`
- `bus.store_local32(done, 1)` -> `bus.data_store32(done, 1)`
- `bus.load32(I2X_*_REG)` / `bus.store32(I2X_*_REG, ..)` -> `bus.data_load32` / `bus.data_store32`
- In the test module: `store_local32`/`load_local32` -> `data_store32`/`data_load32`; `store32`/`load32` -> `data_store32`/`data_load32`.

- [ ] **Step 1: Apply the renames** (production + test module). No new test needed -- the existing tests are the behavior-preservation gate.
- [ ] **Step 2: Run.** Run: `cargo test --lib firmware::host_mailbox`  Expected: PASS (values unchanged).
- [ ] **Step 3: Commit.**

```bash
git add src/firmware/host_mailbox.rs
git commit -m "refactor(#140): host_mailbox uses side-explicit data_* bus API

Generated using Claude Code."
```

---

## Task 7: Migrate `mmu.rs` + `psp_map.rs` (physical PTE traffic)

**Files:**
- Modify: `src/firmware/xtensa/mmu.rs`, `src/firmware/psp_map.rs`

**Migration detail:**
- `mmu.rs` autorefill PTE fetch `Some(bus.load32(t.paddr))` (`get_pte`) -> `Some(bus.data_load32(t.paddr))` (a physical D-side read of the page table).
- `mmu.rs` test module PTE stores `bus.store32(pte_addr, ..)` -> `bus.data_store32(pte_addr, ..)`; any `bus.load32` in those tests -> `bus.data_load32`.
- `psp_map.rs` test module `bus.load32(pt_phys)` -> `bus.data_load32(pt_phys)`. (`write_page_table_word` is unchanged.)

- [ ] **Step 1: Apply the renames.**
- [ ] **Step 2: Run.** Run: `cargo test --lib firmware::xtensa::mmu` then `cargo test --lib firmware::psp_map`  Expected: PASS.
- [ ] **Step 3: Commit.**

```bash
git add src/firmware/xtensa/mmu.rs src/firmware/psp_map.rs
git commit -m "refactor(#140): mmu autorefill + psp_map PTE traffic use data_* bus API

Generated using Claude Code."
```

---

## Task 8: Migrate `mod.rs` boot-test probes (by intent)

The largest by count, and the one requiring judgment: each probe is classified as CPU-view (-> `cpu.data_read*`) or deliberately-physical (-> explicit `inst_*`/`data_*`).

**Files:**
- Modify: `src/firmware/mod.rs`
- Verify (expected no change): `src/firmware/xtensa/coverage_scan.rs`

**Migration detail (classify each site from the grep in the design survey):**
- **CPU-view** ("what does the CPU see at virtual data addr V"): the poll/exec-trace/completion probes reading firmware data structures via `bus.load_local32(a)` where a `proc.cpu` is in hand -> `proc.cpu.data_read32(&mut proc.bus, a)` (returns `Result`; on the diagnostic path, `.unwrap_or(0)` is acceptable since these are steady-state identity reads). The done-flag reads/writes (`load_local32(0x9070)`, `store_local32(done_addr, 1)`, `store_local8(0xf9e0 + ..)`) -> D-side via the CPU accessor where a CPU is present, else `bus.data_*` where only a Bus is in hand.
- **Deliberately-physical:** the image-vs-overlay diff (`mod.rs:2498-99`): `bus.load32(a)` (image) -> `bus.inst_load32(a)`, `bus.load_local32(a)` (DRAM) -> `bus.data_load32(a)` -- now self-documenting. `peek8` disassembly sites: unchanged. The mailbox-register test/harness writes (`store32(HEAD/TAIL/INTR, ..)`, `load32(TAIL)`) -> `data_store32`/`data_load32` (high, region-routed, identical).
- **`fetch8` sites:** unchanged (I-side + overlay; not a bare method).
- Confirm `coverage_scan.rs:223` uses only `peek8` (no bare `load*`/`store*`); expected no edit.

- [ ] **Step 1: Migrate every bare `load32/store32/load8/store8` and `load_local*/store_local*` site in `mod.rs`** per the classification. Where the intent is genuinely ambiguous, prefer the CPU accessor if a `proc.cpu` is in scope (it is the authoritative "what the CPU sees").
- [ ] **Step 2: Verify `coverage_scan.rs`.** Run: `rg -n '\.load32\(|\.store32\(|\.load8\(|\.store8\(|load_local|store_local' src/firmware/xtensa/coverage_scan.rs`  Expected: no matches (only `peek8`, which does not match). If any exist, migrate by intent.
- [ ] **Step 3: Run the boot suite and probes.** Run: `cargo test --lib firmware::boot_tests`  Expected: PASS (433/0/1). Spot-check a probe with `XDNA_FW_PROBE=1 cargo test --lib firmware::boot_tests::m2c_probe_poll_watch -- --nocapture` if the binary is present.
- [ ] **Step 4: Commit.**

```bash
git add src/firmware/mod.rs
git commit -m "refactor(#140): boot-test probes migrate to intent-explicit data_*/inst_* API

Generated using Claude Code."
```

---

## Task 9: Remove bare `load32/store32/load8/store8`; fold the local/pattern helpers

Now that every consumer is migrated, delete the ambiguous surface. The compiler proves completeness.

**Files:**
- Modify: `src/firmware/mmio.rs`

**Migration detail:**
- Migrate `mmio.rs`'s own test module: the region-routing tests that exercise bare `load32/store32/load8/store8` (`routes_addresses_to_regions`, `ram_round_trips`, `rom_store_is_logged_and_ignored`, `array_store_is_stubbed...`, `system_access_*`, `byte_access_*`, `rom_access_honors_psp_load_offset`, `preload_ram_*`, `fill_pattern_*`, `page_table_aperture_*`, etc.) -> rewrite against `data_*`/`inst_*` by intent (e.g. `rom_reads_little_endian_from_image` and `rom_access_honors_psp_load_offset` assert IMAGE reads -> `inst_load32`; `ram_round_trips`/`mailbox_round_trips` are high D-side -> `data_*`). `fetch_overlay_*` uses `fetch8`/`load8` -- `load8` there is a low-window image read -> `inst_load8`.
- Delete `pub fn load32/store32/load8/store8`.
- Fold `fill_pattern` into `data_fill` (remove the standalone `fill_pattern` if now unused; `data_fill` calls the private `fill_mem` directly for non-local sub-spans). Keep `load_local*/store_local*/fill_local` **only if** still referenced internally by `data_*`/`data_fill`; otherwise inline them. `preload_ram`, `write_page_table_word`, `fetch8`, `peek8`, `arm_probe`/`take_probe`/`set_probe_pc`, `add_rom_overlay`, `is_local_data`, `local_data_len_for_test` all STAY.
- Note: `is_local_data` may still be referenced by `data_*` internals as the low-range predicate; keep it if so, else remove.

- [ ] **Step 1: Delete the bare methods and fold the helpers**; migrate the `mmio.rs` test module by intent.
- [ ] **Step 2: Compile-check completeness.** Run: `cargo build`  Expected: clean build (any surviving bare-method call is a hard error naming the exact site -- migrate it by intent, do not blindly pick a side).
- [ ] **Step 3: Full library test run.** Run: `cargo test --lib`  Expected: PASS across the whole suite, including boot 433/0/1.
- [ ] **Step 4: Commit.**

```bash
git add src/firmware/mmio.rs
git commit -m "refactor(#140): remove ambiguous bare Bus load/store; Harvard API is authoritative

Generated using Claude Code."
```

---

## Final verification (whole branch)

- [ ] `cargo test --lib` -- entire suite green, boot 433/0/1.
- [ ] `rg -n 'is_local_data' src/firmware/xtensa/interp/mem.rs` -- no bypass in the executor (should be empty; the predicate, if kept, lives only in `mmio.rs`).
- [ ] `rg -n 'fn load32|fn store32|fn load8\b|fn store8\b' src/firmware/mmio.rs` -- bare methods gone.
- [ ] `rg -n 'assert_low_window_identity' src/firmware/` -- deleted.
- [ ] Grep the diff for low-address `data_*`/`inst_*` sites and re-confirm each side choice by intent (the finding-2 audit).
- [ ] Dispatch the final whole-branch code review (subagent-driven-development's final review), then `superpowers:finishing-a-development-branch`.

## Self-review notes (author)

- **Spec coverage:** every design component (Bus API, Cpu accessor, mem/fastpath/host_mailbox/mmu/psp_map/probes migration, bare-method removal, the three adversarial fixes, Task 1 characterization) maps to a task. The five adversarial findings are addressed in Tasks 2 (finding 1 `data_fill` boundary + zero no-grow), 4 (finding 2 by-intent low migration, finding 3 `store16` two-phase, finding 4 equivalence D-side-only), 8 (finding 5 record_stub note deferred to a possible `data_peek`, only if needed), and 1 (structural characterization).
- **Type consistency:** `Cpu::data_read{8,32}`/`data_write{8,32}` and `Bus::data_load{8,32}`/`data_store{8,32}`/`data_fill`/`inst_load{8,32}` are used with identical signatures across Tasks 2-9.
- **Ordering:** additive (1-3) -> per-file migrate (4-8) -> remove-bare (9); the tree compiles and tests pass after every task.
