# M2c Local-Memory Harvard Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Model the firmware core's low virtual window as Xtensa Harvard local memory -- instruction fetches read the image (local IRAM), data loads/stores use a separate writable backing (local DRAM) -- so the boot's 128 MiB region-zeroing memset stops erasing `.text` and the syscall stack store persists, advancing the boot past `0x2000e035`.

**Architecture:** A data load/store whose *virtual* address is `< 0x04000000` routes to a new writable `local_data` backing on `Bus`, MMU-bypassed; instruction fetches and every access at vaddr `>= 0x04000000` are unchanged. The split is keyed on vaddr (not paddr, which collides between the code region and the low window) and lives in the interp data path (`mem.rs`) and the memset fast-path (`fastpath.rs`).

**Tech Stack:** Rust, existing `src/firmware/` Xtensa interpreter (bus in `mmio.rs`, data ops in `interp/mem.rs`, fill fast-path in `interp/fastpath.rs`).

**Spec:** `docs/superpowers/specs/2026-07-05-m2c-local-memory-harvard-model-design.md`

## Global Constraints

- No emoji anywhere (code, comments, commit messages).
- Never pipe `cargo build`/`cargo test` through `tail`/`head`/`grep` -- run bare; output is auto-clipped.
- `cargo test --lib` must pass after every task (a pass that regresses is a regression to fix before moving on).
- `LOCAL_DATA_END = 0x0400_0000` -- a VIRTUAL-address predicate (`is_local_data(vaddr) = vaddr < LOCAL_DATA_END`), applied before translation.
- Local accessors are keyed on the local OFFSET, which equals the vaddr (the window is 0-based). They never consult `Bus::region` and never touch `rom`.
- The MMU (`mmu.rs`), `translate`, autorefill, PTE synthesis (`psp_map.rs`), and all system-region routing stay untouched.
- Fetches (`Access::Fetch`) are NOT rerouted -- only data loads/stores in the low window change.
- `debug_assert` identity net uses the NON-raising `cpu.mmu.translate` and the LENIENT form (pass on a TLB miss, fail only on a non-identity hit).
- Commit messages end with a blank line then:
  `Generated using Claude Code.`
  `Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/firmware/mmio.rs` | The routed bus + backings | Add `local_data` backing, `LOCAL_DATA_END`, `is_local_data`, `load_local32/8`, `store_local32/8`, `fill_local` (zero-fill allocation cap) |
| `src/firmware/xtensa/interp/mem.rs` | Data load/store execute | Route `is_local_data(vaddr)` accesses to `*_local` via four `data_*` helpers; identity `debug_assert` net |
| `src/firmware/xtensa/interp/fastpath.rs` | memset fast-path | Route the local-window portion of a fill to `fill_local`; shared-`off` invariant |
| `src/firmware/mod.rs` | Boot harness + probes | Remove the three investigation probes; add the past-`0x2000e035` integration gate |

Task order is a dependency chain: Task 1 (bus surface) -> Task 2 (grind data path) -> Task 3 (fast-path) -> Task 4 (cleanup + gate).

---

## Task 1: Bus `local_data` backing and surface

> **AMENDED 2026-07-05:** `local_data` is an **image-backed overlay**, not blank
> zero-init. The blank bet failed in Task 2 (the reset prologue reads `l32r`
> literals from low image addresses; blank returns 0 -> boot dies at PC 0). The
> initial blank implementation landed in commit `6f98de0a`; a follow-up commit
> adds the eager preload (`local_data[i] = rom[i + load_offset]`, capped at
> `LOCAL_DATA_END`) in `new_with_load_offset`, so an unwritten low read mirrors
> the image and a write overrides it, with `rom` never touched. See the spec's
> "local_data backing (image-backed overlay)" section. The accessors and
> `fill_local` zero-cap below are unchanged; only the constructor preloads.

**Files:**
- Modify: `src/firmware/mmio.rs`
- Test: `src/firmware/mmio.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing new.
- Produces (all on `Bus`):
  - `pub const LOCAL_DATA_END: u32 = 0x0400_0000;`
  - `pub fn is_local_data(vaddr: u32) -> bool` (associated fn, like `Bus::region`)
  - `pub fn load_local32(&self, off: u32) -> u32`
  - `pub fn load_local8(&self, off: u32) -> u8`
  - `pub fn store_local32(&mut self, off: u32, v: u32)`
  - `pub fn store_local8(&mut self, off: u32, v: u32)` (stores the low byte of `v`)
  - `pub fn fill_local(&mut self, off: u32, pattern: &[u8], byte_len: usize)`

- [ ] **Step 1: Write the failing tests**

Add to `src/firmware/mmio.rs`'s `mod tests`:

```rust
#[test]
fn is_local_data_boundary() {
    assert!(Bus::is_local_data(0x0000_1000));
    assert!(Bus::is_local_data(0x03ff_ffff));
    assert!(!Bus::is_local_data(0x0400_0000)); // array aperture starts here
    assert!(!Bus::is_local_data(0x2000_0000)); // code region
}

#[test]
fn local_data_round_trips_and_starts_blank() {
    let mut bus = Bus::new(vec![]);
    // Blank on first read.
    assert_eq!(bus.load32(0), 0); // note: paddr path, unrelated
    assert_eq!(bus.load_local32(0x1000), 0);
    assert_eq!(bus.load_local8(0x1000), 0);
    // Round-trips.
    bus.store_local32(0x1000, 0xdead_beef);
    assert_eq!(bus.load_local32(0x1000), 0xdead_beef);
    bus.store_local8(0x2000, 0xab);
    assert_eq!(bus.load_local8(0x2000), 0xab);
}

#[test]
fn store_local_does_not_touch_the_image() {
    // The anti-aliasing invariant: a local-data store at offset X leaves the
    // rom image byte X (read via the paddr Rom path) untouched. Before the
    // Harvard split, a low write corrupted the shared rom backing.
    let mut bus = Bus::new(vec![0x11, 0x22, 0x33, 0x44]); // rom bytes at paddr 0..4
    bus.store_local32(0x0, 0xffff_ffff); // local offset 0
    // The rom image (paddr 0) is unchanged.
    assert_eq!(bus.load32(0x0), 0x4433_2211);
    // The local backing has the write.
    assert_eq!(bus.load_local32(0x0), 0xffff_ffff);
}

#[test]
fn fill_local_nonzero_fills_and_zero_does_not_grow() {
    let mut bus = Bus::new(vec![]);
    // Non-zero fill grows and repeats the pattern (little-endian store order).
    bus.fill_local(0x1000, &0xdead_beefu32.to_le_bytes(), 8);
    assert_eq!(bus.load_local32(0x1000), 0xdead_beef);
    assert_eq!(bus.load_local32(0x1004), 0xdead_beef);
    // A zero fill into never-written space is a no-op that reads back 0
    // WITHOUT allocating (the tail past current len reads 0 by default).
    let before = bus.local_data_len_for_test();
    bus.fill_local(0x0100_0000, &[0u8], 0x1000); // 16 MiB offset, all-zero
    let after = bus.local_data_len_for_test();
    assert_eq!(after, before, "zero fill must not grow the backing");
    assert_eq!(bus.load_local8(0x0100_0000), 0);
    // A zero fill DOES clear an already-written prefix.
    bus.store_local8(0x1000, 0x77);
    bus.fill_local(0x1000, &[0u8], 4);
    assert_eq!(bus.load_local8(0x1000), 0);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib firmware::mmio::tests`
Expected: FAIL -- `is_local_data`, `load_local32`, `store_local32`, `fill_local`, `local_data_len_for_test` are not defined.

- [ ] **Step 3: Add the `local_data` field**

In `src/firmware/mmio.rs`, add a field to `struct Bus` (after `page_table`):

```rust
    // Local data memory (Xtensa DRAM): a writable backing for low-window data
    // accesses (vaddr < LOCAL_DATA_END), offset-keyed from 0, blank-init, grown
    // lazily. Physically distinct from `rom` (the image / local IRAM): the
    // firmware's boot memset zeroes this, not its own code. See the M2c
    // local-memory Harvard model spec.
    local_data: Vec<u8>,
```

Initialize it (`Vec::new()`) in BOTH `new_with_load_offset` (the only real constructor; `new` delegates to it) -- add `local_data: Vec::new(),` to the struct literal.

- [ ] **Step 4: Add the constant, predicate, and accessors**

Add near `ROM_END` (top of file):

```rust
/// End (exclusive) of the low virtual window that maps to local memory. A DATA
/// access below this vaddr goes to the Harvard local data memory (`local_data`),
/// not the image; an instruction fetch below it still reads the image (local
/// IRAM). Coincides numerically with `ROM_END`, but is a VIRTUAL-address
/// predicate applied before translation. See the M2c Harvard-model spec.
pub const LOCAL_DATA_END: u32 = 0x0400_0000;
```

Add these methods to `impl Bus` (place them after `region`):

```rust
    /// True iff `vaddr` is a low-window virtual address whose DATA accesses go
    /// to local memory (`local_data`). A vaddr predicate, applied before
    /// translation -- the local/image split cannot be made on the physical
    /// address, because the code region and the low window collide there.
    pub fn is_local_data(vaddr: u32) -> bool {
        vaddr < LOCAL_DATA_END
    }

    /// Read a little-endian 32-bit word from local data memory at `off` (== the
    /// low-window vaddr). Blank (0) past the written extent.
    pub fn load_local32(&self, off: u32) -> u32 {
        read_le32(&self.local_data, off)
    }

    /// Read a byte from local data memory at `off`.
    pub fn load_local8(&self, off: u32) -> u8 {
        byte_at(&self.local_data, off)
    }

    /// Write a little-endian 32-bit word to local data memory at `off`, growing
    /// the backing to fit.
    pub fn store_local32(&mut self, off: u32, v: u32) {
        write_le32(&mut self.local_data, off, v);
    }

    /// Write the low byte of `v` to local data memory at `off`, growing to fit.
    pub fn store_local8(&mut self, off: u32, v: u32) {
        set_byte_at(&mut self.local_data, off, v as u8);
    }

    /// Bulk fill of local data memory: `pattern` (1/2/4 bytes, little-endian
    /// store order) repeated to cover `byte_len` bytes at `off`. Byte-identical
    /// to that many `store_local8`/`16`/`32`s. Zero-pattern optimization: a
    /// zero fill never GROWS the backing (unwritten offsets already read 0); it
    /// only clears the already-populated prefix. This keeps the boot's 128 MiB
    /// zero-memset from allocating ~64 MiB every boot.
    pub fn fill_local(&mut self, off: u32, pattern: &[u8], byte_len: usize) {
        debug_assert!(matches!(pattern.len(), 1 | 2 | 4));
        debug_assert_eq!(byte_len % pattern.len(), 0);
        if pattern.iter().all(|&b| b == 0) {
            let o = off as usize;
            let end = o + byte_len;
            let cap = end.min(self.local_data.len());
            if cap > o {
                self.local_data[o..cap].fill(0);
            }
        } else {
            fill_mem(&mut self.local_data, off, pattern, byte_len);
        }
    }
```

- [ ] **Step 5: Add the test-only length accessor**

The zero-fill test needs to see the backing length. Add to `impl Bus`, guarded so it is test-only:

```rust
    /// Test-only: current length of the local-data backing (to assert the
    /// zero-fill allocation cap does not grow it).
    #[cfg(test)]
    pub fn local_data_len_for_test(&self) -> usize {
        self.local_data.len()
    }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --lib firmware::mmio::tests`
Expected: PASS (all mmio tests, including the four new ones).

- [ ] **Step 7: Run the full library suite**

Run: `cargo test --lib`
Expected: PASS (no regressions).

- [ ] **Step 8: Commit**

```bash
git add src/firmware/mmio.rs
git commit -m "feat(#140): M2c Harvard model -- Bus local_data backing + surface

$(printf 'Adds the writable local data memory backing and its vaddr-keyed\naccessors (is_local_data, load/store_local*, fill_local with a\nzero-fill allocation cap). Foundation for the low-window Harvard\nsplit; no interp routing yet.\n\nGenerated using Claude Code.\nClaude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh')"
```

---

## Task 2: Route interp data load/store to local memory

**Files:**
- Modify: `src/firmware/xtensa/interp/mem.rs`
- Test: `src/firmware/xtensa/interp/mem.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `Bus::is_local_data`, `Bus::{load_local32,load_local8,store_local32,store_local8}` (Task 1).
- Produces: four private helpers in `mem.rs` that every data arm routes through:
  - `fn data_load32(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32) -> Result<u32, Step>`
  - `fn data_load8(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32) -> Result<u8, Step>`
  - `fn data_store32(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step>`
  - `fn data_store8(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step>`

- [ ] **Step 1: Write the failing tests**

Add to `src/firmware/xtensa/interp/mem.rs`'s `mod tests` (which already imports `mapped_cpu, Cpu, Step` and `Bus`):

These tests reuse the repo's already-verified instruction vectors -- `69 c7` =
`s32i.n a6,a7,0x30` (from `executes_s32i_n_stores_to_bus`) and `48 45` =
`l32i.n a4,a5,0x10` (from `executes_l32i_n_loads_from_bus`) -- and only change
the base register to point the effective address into the low window. Do NOT
hand-encode new bytes.

```rust
#[test]
fn low_window_store_lands_in_local_data_not_image() {
    // s32i.n a6,a7,0x30 (`69 c7`) with a7 = 0x1000 -> effective vaddr 0x1030
    // (low window). The store must land in local_data and NOT corrupt the image.
    let mut bus = Bus::new(vec![0x69, 0xc7]);
    let mut cpu = mapped_cpu(0);
    cpu.regs.write_ar(7, 0x1000); // low-window data base
    cpu.regs.write_ar(6, 0x1122_3344); // value
    assert!(matches!(cpu.step(&mut bus), Step::Ran));
    // Landed in local_data, read back by the same low vaddr (0x1000 + 0x30).
    assert_eq!(bus.load_local32(0x1030), 0x1122_3344);
    // The image (paddr Rom path) is untouched -- anti-aliasing.
    assert_eq!(bus.load32(0x1030), 0);
}

#[test]
fn low_window_load_reads_local_data_blank_then_stored() {
    // l32i.n a4,a5,0x10 (`48 45`) with a5 = 0x1000 -> effective vaddr 0x1010.
    // Reads local_data: blank (0) until a prior store, then the stored value.
    let mut cpu = mapped_cpu(0);
    let mut bus = Bus::new(vec![0x48, 0x45]);
    cpu.regs.write_ar(5, 0x1000);
    assert!(matches!(cpu.step(&mut bus), Step::Ran));
    assert_eq!(cpu.regs.read_ar(4), 0, "blank local_data reads 0, not the image");
    // Store into local_data at the same vaddr and re-load.
    bus.store_local32(0x1010, 0xcafe_babe);
    cpu.pc = 0;
    assert!(matches!(cpu.step(&mut bus), Step::Ran));
    assert_eq!(cpu.regs.read_ar(4), 0xcafe_babe);
}

#[test]
fn high_window_data_still_uses_translate_and_image() {
    // Regression: a data access at vaddr >= LOCAL_DATA_END is unchanged -- it
    // translates and hits the paddr backing (RAM here), NOT local_data. This is
    // the existing `executes_l32i_n_loads_from_bus` path, guarding that the new
    // local branch does NOT capture high addresses.
    let mut bus = Bus::new(vec![0x48, 0x45]); // l32i.n a4,a5,0x10
    bus.store32(0x08b0_0010, 0xdead_beef); // RAM aperture, paddr path
    let mut cpu = mapped_cpu(0);
    let page = 0x08b0_0000u32 & 0xffff_f000;
    cpu.mmu.write_tlb(true, page | 0x3, page | 0); // DTLB identity, way 0
    cpu.regs.write_ar(5, 0x08b0_0000);
    assert!(matches!(cpu.step(&mut bus), Step::Ran));
    assert_eq!(cpu.regs.read_ar(4), 0xdead_beef);
    assert!(!Bus::is_local_data(0x08b0_0010));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib firmware::xtensa::interp::mem::tests`
Expected: FAIL -- the low-window store currently drops (Rom read-only) so `load_local32` reads 0, and the low-window load reads the image path.

- [ ] **Step 3: Add the four `data_*` helpers**

Add to `src/firmware/xtensa/interp/mem.rs` (after `store16`, before `#[cfg(test)]`). These encapsulate the local-vs-translate routing so each arm is one call:

```rust
/// Route a 32-bit data LOAD: a low-window vaddr reads local data memory
/// (MMU-bypassed); anything else translates and reads the paddr backing.
fn data_load32(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32) -> Result<u32, Step> {
    if Bus::is_local_data(vaddr) {
        assert_low_window_identity(cpu, bus, vaddr, 0 /*load*/);
        Ok(bus.load_local32(vaddr))
    } else {
        let paddr = cpu.translate(bus, vaddr, Access::Load)?;
        Ok(bus.load32(paddr))
    }
}

/// Route a byte data LOAD (see [`data_load32`]).
fn data_load8(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32) -> Result<u8, Step> {
    if Bus::is_local_data(vaddr) {
        assert_low_window_identity(cpu, bus, vaddr, 0 /*load*/);
        Ok(bus.load_local8(vaddr))
    } else {
        let paddr = cpu.translate(bus, vaddr, Access::Load)?;
        Ok(bus.load8(paddr))
    }
}

/// Route a 32-bit data STORE (see [`data_load32`]).
fn data_store32(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step> {
    if Bus::is_local_data(vaddr) {
        assert_low_window_identity(cpu, bus, vaddr, 1 /*store*/);
        bus.store_local32(vaddr, v);
        Ok(())
    } else {
        let paddr = cpu.translate(bus, vaddr, Access::Store)?;
        bus.store32(paddr, v);
        Ok(())
    }
}

/// Route a byte data STORE, storing the low byte of `v` (see [`data_load32`]).
fn data_store8(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step> {
    if Bus::is_local_data(vaddr) {
        assert_low_window_identity(cpu, bus, vaddr, 1 /*store*/);
        bus.store_local8(vaddr, v);
        Ok(())
    } else {
        let paddr = cpu.translate(bus, vaddr, Access::Store)?;
        bus.store8(paddr, v & 0xFF);
        Ok(())
    }
}

/// Safety net for the local-memory bypass: the low window is the varway56
/// way-6 identity across the whole probed boot (no autorefill, no fault), and
/// this task runs PAST that boot. Assert the low-window vaddr still identity-
/// maps so a FUTURE non-identity remap fails loudly instead of silently reading
/// the wrong backing. Lenient: passes on a TLB miss (unit-test CPUs don't map
/// the low window) and on an identity hit; fails only on a non-identity hit.
/// Uses the non-raising `cpu.mmu.translate` so the speculative check never
/// perturbs pc/epc1/exccause; compiled out in release.
fn assert_low_window_identity(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32, mode: u8) {
    debug_assert!(
        !matches!(cpu.mmu.translate(bus, vaddr, mode, 0), Ok(t) if t.paddr != vaddr),
        "low-window data vaddr {vaddr:#x} translates to non-identity paddr -- local-memory bypass may be wrong"
    );
}
```

- [ ] **Step 4: Rewire the data arms and the 16-bit helpers**

Replace the load/store bodies in `exec` and in `load16`/`store16` to call the helpers. The new `exec` data arms:

```rust
        Op::L32iN { t, s, imm } | Op::L32i { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match data_load32(cpu, bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v);
        }
        Op::L32r { t, target } => {
            let v = match data_load32(cpu, bus, *target) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v);
        }
        Op::L8ui { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match data_load8(cpu, bus, vaddr) {
                Ok(v) => v as u32,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v);
        }
        Op::L16ui { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match load16(cpu, bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v as u32);
        }
        Op::L16si { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match load16(cpu, bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v as i16 as i32 as u32);
        }
        Op::S8i { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let val = cpu.regs.read_ar(*t); // read before the &mut cpu borrow
            if let Err(step) = data_store8(cpu, bus, vaddr, val) {
                return Some(step);
            }
        }
        Op::S16i { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = cpu.regs.read_ar(*t) as u16;
            if let Err(step) = store16(cpu, bus, vaddr, v) {
                return Some(step);
            }
        }
        Op::S32iN { t, s, imm } | Op::S32i { t, s, imm } | Op::S32ri { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let val = cpu.regs.read_ar(*t); // read before the &mut cpu borrow
            if let Err(step) = data_store32(cpu, bus, vaddr, val) {
                return Some(step);
            }
        }
```

And rewrite `load16`/`store16` to route each byte through the helpers (preserving the per-byte independent-translation semantics -- a low byte in the window and a high byte across the boundary each route correctly):

```rust
fn load16(cpu: &mut Cpu, bus: &mut Bus, addr: u32) -> Result<u16, Step> {
    let lo = data_load8(cpu, bus, addr)? as u16;
    let hi = data_load8(cpu, bus, addr.wrapping_add(1))? as u16;
    Ok(lo | (hi << 8))
}

fn store16(cpu: &mut Cpu, bus: &mut Bus, addr: u32, v: u16) -> Result<(), Step> {
    // Preserve the original no-half-write guarantee: validate both byte
    // destinations before writing either, so a fault on the high byte never
    // leaves the low byte's store applied. A local byte never faults; a
    // non-local byte is validated by a translate probe (raises on fault, no
    // write). Only after both are known routable do we write.
    let (lo, hi) = (addr, addr.wrapping_add(1));
    if !Bus::is_local_data(lo) {
        cpu.translate(bus, lo, Access::Store)?;
    }
    if !Bus::is_local_data(hi) {
        cpu.translate(bus, hi, Access::Store)?;
    }
    data_store8(cpu, bus, lo, (v & 0xFF) as u32)?;
    data_store8(cpu, bus, hi, (v >> 8) as u32)?;
    Ok(())
}
```

The `Access` import stays (used by `data_*` and the `store16` probe). The
non-local path translates twice (probe + `data_store8`); the extra TLB churn is
architecturally unobservable (per the fast-path's own scope note) and the path
is rare -- keeping the guarantee is worth it. A 16-bit store straddling
`LOCAL_DATA_END` (low byte local, high byte array) writes the low byte to
`local_data` and drops the high byte in the Array stub, exactly as two `s8i`s
would.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib firmware::xtensa::interp::mem::tests`
Expected: PASS -- the three new tests plus all existing mem tests (the existing tests use RAM/paddr addresses `>= 0x04000000`, so they take the unchanged translate path).

- [ ] **Step 6: Run the full library suite**

Run: `cargo test --lib`
Expected: PASS. (The boot-observation test `m2c_boot_advances_into_c_runtime` may now advance further, but its assertion is only `instrs_executed > 20_000`, so it still passes.)

- [ ] **Step 7: Commit**

```bash
git add src/firmware/xtensa/interp/mem.rs
git commit -m "feat(#140): M2c Harvard model -- route low-window data to local memory

$(printf 'Data loads/stores whose vaddr < LOCAL_DATA_END now go to the writable\nlocal_data backing (MMU-bypassed) via four data_* helpers, instead of\nthe read-only image. Fetches and data at vaddr >= the boundary are\nunchanged. A lenient debug_assert identity net catches any future\nlow-region remap loudly. The syscall stack store now persists.\n\nGenerated using Claude Code.\nClaude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh')"
```

---

## Task 3: Route the fill-loop fast-path to local memory

**Files:**
- Modify: `src/firmware/xtensa/interp/fastpath.rs`
- Test: `src/firmware/xtensa/interp/fastpath.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `Bus::is_local_data`, `Bus::fill_local`, `crate::firmware::mmio::LOCAL_DATA_END` (Task 1); `data_store8` routing from Task 2 (so per-store grinding of a low-window fill also lands in `local_data`, which the fast-path must match).
- Produces: no new public interface; `try_fill_loop` gains a local-window branch.

- [ ] **Step 1: Write the failing tests**

Add to `src/firmware/xtensa/interp/fastpath.rs`'s `mod tests`. Reuse the existing `place_byte_fill_body` helper (body at a fetchable RAM address) but point the fill DESTINATION into the low window:

```rust
#[test]
fn fastpath_local_window_fill_matches_grind() {
    // A non-zero byte fill whose DEST is in the low window (< LOCAL_DATA_END)
    // must fill local_data, byte-identical fast vs grind. Body stays in RAM so
    // it is fetchable; only the fill target is local.
    const CODE: u32 = 0x08b0_0000; // RAM: body fetchable
    const DEST: u32 = 0x0020_0000; // low window (< 0x04000000)
    const N: u32 = 5000;

    let run = |fast: bool| -> (Vec<u8>, u32, u32) {
        let mut cpu = Cpu::new(CODE);
        cpu.fastpath_enabled = fast;
        // Map only the body (fetch) region; the local DEST needs no mapping.
        cpu.mmu.write_tlb(false, (CODE & 0xfff0_0000) | 0x1, (CODE & 0xfff0_0000) | 4);
        let mut bus = Bus::new(vec![]);
        let lend = place_byte_fill_body(&mut bus, CODE);
        cpu.pc = CODE;
        cpu.regs.lbeg = CODE;
        cpu.regs.lend = lend;
        cpu.regs.lcount = N - 1;
        cpu.regs.write_ar(5, DEST); // ptr
        cpu.regs.write_ar(3, 0xab); // fill byte
        for _ in 0..(N * 4 + 16) {
            if cpu.pc == lend {
                break;
            }
            match cpu.step(&mut bus) {
                Step::Ran | Step::Exception { .. } => {}
                other => panic!("unexpected {other:?}"),
            }
        }
        let filled: Vec<u8> = (DEST..DEST + N).map(|a| bus.load_local8(a)).collect();
        (filled, cpu.regs.read_ar(5), cpu.regs.lcount)
    };

    let (fast_mem, fast_ptr, fast_lc) = run(true);
    let (grind_mem, grind_ptr, grind_lc) = run(false);
    assert_eq!(fast_mem, grind_mem, "local fill must be byte-identical fast vs grind");
    assert!(fast_mem.iter().all(|&b| b == 0xab));
    assert_eq!(fast_ptr, grind_ptr);
    assert_eq!(fast_ptr, DEST + N);
    assert_eq!(fast_lc, 0);
    assert_eq!(grind_lc, 0);
}

#[test]
fn fastpath_fill_spanning_local_boundary_matches_grind() {
    // A byte fill starting below LOCAL_DATA_END and running across it: the
    // local portion fills local_data; the portion at/above 0x04000000 lands in
    // the Array aperture (dropped, reads back 0). Fast == grind on both sides.
    const CODE: u32 = 0x08b0_0000;
    // Start 0x800 below the boundary; N carries it 0x800 past into the array.
    const DEST: u32 = crate::firmware::mmio::LOCAL_DATA_END - 0x800;
    const N: u32 = 0x1000; // 0x800 local + 0x800 array
    const BOUNDARY: u32 = crate::firmware::mmio::LOCAL_DATA_END;

    let run = |fast: bool| -> (Vec<u8>, Vec<u8>, u32, u32) {
        let mut cpu = Cpu::new(CODE);
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
            if cpu.pc == lend {
                break;
            }
            match cpu.step(&mut bus) {
                Step::Ran | Step::Exception { .. } => {}
                other => panic!("unexpected {other:?}"),
            }
        }
        let local: Vec<u8> = (DEST..BOUNDARY).map(|a| bus.load_local8(a)).collect();
        let array: Vec<u8> = (BOUNDARY..DEST + N).map(|a| bus.load8(a)).collect();
        (local, array, cpu.regs.read_ar(5), cpu.regs.lcount)
    };

    let (f_local, f_array, f_ptr, f_lc) = run(true);
    let (g_local, g_array, g_ptr, g_lc) = run(false);
    assert_eq!(f_local, g_local, "local side identical");
    assert_eq!(f_array, g_array, "array side identical");
    assert!(f_local.iter().all(|&b| b == 0xcd), "local side filled");
    assert!(f_array.iter().all(|&b| b == 0), "array side dropped (reads 0)");
    assert_eq!(f_ptr, g_ptr);
    assert_eq!(f_ptr, DEST + N, "pointer advanced across the boundary");
    assert_eq!(f_lc, g_lc);
    assert_eq!(f_lc, 0);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib firmware::xtensa::interp::fastpath::tests`
Expected: FAIL -- the fast run still routes the low-window fill through `cpu.mmu.translate` -> `bus.fill_pattern(paddr)`, which drops (Rom), while grind (Task 2) now fills `local_data`. So `fast_mem` is all 0 and `grind_mem` is `0xab` -> mismatch.

- [ ] **Step 3: Add the local-window branch to `try_fill_loop`**

In `src/firmware/xtensa/interp/fastpath.rs`, extend the imports and the chunk loop. Add to the `use` line:

```rust
use crate::firmware::mmio::{Bus, LOCAL_DATA_END};
```

Replace the chunk loop (the `while off < total { ... }` block) with:

```rust
    let mut off = 0u64;
    while off < total {
        let vaddr = start.wrapping_add(off as u32);
        if Bus::is_local_data(vaddr) {
            // Local data memory: MMU-bypassed (the low window never faults),
            // chunked up to the window boundary. Both this branch and the
            // paddr branch advance the SAME `off`, so the fault reconstruction
            // below counts the local bytes correctly.
            let window_left = (LOCAL_DATA_END - vaddr) as u64;
            let chunk = window_left.min(total - off);
            bus.fill_local(vaddr, pattern, chunk as usize);
            off += chunk;
        } else {
            match cpu.mmu.translate(bus, vaddr, 1 /*store*/, 0) {
                Ok(t) => {
                    let psize = t.page_size as u64;
                    let page_left = psize - (vaddr as u64 & (psize - 1));
                    let chunk = page_left.min(total - off);
                    bus.fill_pattern(t.paddr, pattern, chunk as usize);
                    off += chunk;
                }
                Err(_) => {
                    cpu.regs.write_ar(ptr_reg, vaddr);
                    cpu.regs.lcount = (n - 1 - off / w as u64) as u32;
                    let step = cpu
                        .translate(bus, vaddr, Access::Store)
                        .expect_err("mmu.translate just faulted at this vaddr");
                    return Some(step);
                }
            }
        }
    }
```

(The `Ok`/`Err` arms are the existing code verbatim; only the outer `if Bus::is_local_data(vaddr)` branch and the shared-`off` comment are new. `window_left` is a multiple of `w`: `start` is `w`-aligned, `off` advances by `w`-multiples, and `LOCAL_DATA_END` is `w`-aligned, so `fill_local`'s `byte_len % pattern.len() == 0` debug_assert holds and pattern phase stays 0.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib firmware::xtensa::interp::fastpath::tests`
Expected: PASS -- the two new tests plus all existing fast-path tests (their DESTs are in RAM `>= 0x04000000`, unchanged path).

- [ ] **Step 5: Run the full library suite**

Run: `cargo test --lib`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/firmware/xtensa/interp/fastpath.rs
git commit -m "feat(#140): M2c Harvard model -- fast-path fills local memory

$(printf 'try_fill_loop routes the low-window portion of a fill to fill_local\n(MMU-bypassed), so the boot memset zeroes local DRAM, not the image,\nand stays byte-identical to per-store grinding. A boundary-crossing\nfill fills local below 0x04000000 and drops above, matching grind.\nShared off accumulator keeps the fault reconstruction correct.\n\nGenerated using Claude Code.\nClaude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh')"
```

---

## Task 4: Remove investigation probes and add the integration gate

**Files:**
- Modify: `src/firmware/mod.rs`
- Test: `src/firmware/mod.rs` (the `m2c_boot_advances_into_c_runtime` test)

**Interfaces:**
- Consumes: the full Harvard routing (Tasks 1-3).
- Produces: no new interface; tightens the boot-observation gate and removes dead scaffolding.

- [ ] **Step 1: Tighten the boot gate to assert past the iter10 wall**

In `src/firmware/mod.rs`, in `m2c_boot_advances_into_c_runtime`, add an assertion after the existing `assert!(report.instrs_executed > 20_000 ...)` that the boot no longer walls at the iter10 `break`:

```rust
        // Harvard model (2026-07-05): the syscall stack store now persists in
        // local data memory, so `main` no longer unwinds to the crt0 "main
        // returned" break. The boot must clear the iter10 wall at 0x2000e035.
        assert_ne!(
            report.unknown_op.map(|(pc, _)| pc),
            Some(0x2000_e035),
            "boot still walls at the iter10 break 0x2000e035 -- the local-memory \
             Harvard split did not take effect (last_pc={:#x})",
            report.last_pc,
        );
```

- [ ] **Step 2: Run the boot gate to verify it passes**

Run: `cargo test --lib firmware::boot_tests::m2c_boot_advances_into_c_runtime -- --nocapture`
Expected: PASS, and the printed `last_pc` is NOT `0x2000e035` (the boot advanced to a deeper wall or idle). Record the new `last_pc` -- it is the next iteration's starting point.

- [ ] **Step 3: Remove the three investigation probes**

Delete these three test functions in full from `src/firmware/mod.rs` (they were iter10/iter11 scaffolding; their findings are captured in the finding docs, and `m2c_probe_syscall_service`'s `bus.load32`/`peek8` reads would now go blind to `local_data` -- see the spec's Components note):

- `m2c_probe_pc_regions`
- `m2c_probe_tlb_writes`
- `m2c_probe_syscall_service`

Leave the kept instruments untouched: `m2c_probe_trace_to_wall`, `m2c_probe_peripheral_reads`, and the `m2c_boot_advances_into_c_runtime` gate.

- [ ] **Step 4: Verify no dangling references**

Run: `cargo build --tests 2>&1 | grep -c "cannot find\|unused" || true`
Then run the full suite bare:

Run: `cargo test --lib`
Expected: PASS with no unused-import/dead-code warnings introduced by the removals. If a helper (e.g. a probe-only import) is now unused, remove it.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/mod.rs
git commit -m "feat(#140): M2c Harvard model -- integration gate + probe cleanup

$(printf 'Assert the boot clears the iter10 break wall at 0x2000e035 (the\nsyscall stack store now persists). Remove the iter10/iter11\ninvestigation probes (pc_regions, tlb_writes, syscall_service); the\nlatter read via the paddr Bus path and would go blind to local_data.\n\nGenerated using Claude Code.\nClaude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh')"
```

---

## Final verification (after all tasks)

- [ ] `cargo test --lib` -- full suite green.
- [ ] `cargo build` -- clean (no warnings from the change).
- [ ] The boot advances past `0x2000e035`; the new wall PC is recorded from Task 4 Step 2 for the next M2c iteration / finding.
- [ ] Success criteria (from the spec): syscall stack store persists; memset fills `local_data` not the image; low vectors + code region survive; boot past `0x2000e035`; fast-path local fills byte-identical to grind; suite green; probes removed.

## Notes for the implementer

- **Zero-init bet / boundary risk fallback:** if the boot instead walls on a low-window data *read* returning 0 that expected an image constant (a `l32i`/`l32r` at a low vaddr), that is the zero-init bet failing. The fallback (do NOT do it pre-emptively) is to preload `local_data` from the image for the affected range. Record the wall and raise it before implementing the fallback -- it is not expected.
- **Do not touch** `mmu.rs`, `psp_map.rs`, or the translate/autorefill/PTE code. The whole design is `mem.rs` routing + `fastpath.rs` routing + the `mmio.rs` backing.
