# Firmware MMU/DTLB Data Path -- Translation-Authoritative Design

**Date:** 2026-07-06
**Status:** Design (approved shape; awaiting spec review)
**Branch:** `feat/m2c-mapping-boot-to-idle`
**Motivation issue:** #140 (firmware-emulation dream); this is a foundation piece.

## Goal

Make address translation the single authoritative decision for every CPU memory
access in the Xtensa firmware interpreter, and expose one canonical
translation-aware data accessor that probes, tools, and the executor all share --
retiring the "memory-alias tax" where raw bus access and the CPU's real data path
silently disagree.

## Architecture (2-3 sentences)

Today memory is routed *before* translation, keyed on the *virtual* address (the
`is_local_data(vaddr)` predicate picks DRAM-vs-image and bypasses the MMU for the
low window). This design inverts that: every CPU access translates first, and the
resulting *physical* address plus an instruction/data **side** tag selects the
backing store. The virtual code-region/low-window collision that forced the
vaddr-keyed hacks dissolves, because after translation `(paddr, side)` is
unambiguous.

## Tech stack

Rust; the existing `src/firmware/` Xtensa interpreter (`mmu.rs`, `interp/`,
`mmio.rs`, `psp_map.rs`, `host_mailbox.rs`, `fastpath.rs`). No new dependencies.

---

## Global Constraints

- **Derive from the toolchain / real hardware.** The MMU model
  (`mmu.rs`, QEMU-derived MMU-v3) is already faithful and is **NOT** changed by
  this work. This design only re-plumbs how a *translated* physical address
  reaches its backing store.
- **Behavior-preserving on the real boot path.** The firmware boot suite must
  stay green at **433 passed / 0 failed / 1 ignored** after every task. The
  ignored test (`m2c_boot_completion_advances_past_recursion`) stays ignored.
- **No net behavior change for the executor.** Every existing `mem`/`mmu`/
  `fastpath`/`interp` unit test must stay green (they are the regression gate).
  Where a test's *call* changes (a bus method rename), its *asserted values* must
  not.
- **No emoji anywhere.** Comments explain the *why*. Commit messages end with
  `Generated using Claude Code.`
- **Faithful-knowledge comments.** Attribute behavior to the hardware fact
  (Harvard I/D split, varway56 low-window identity region), not to tool
  internals.

---

## Background: the alias tax, concretely

The firmware interpreter has three ways memory is reached, and they disagree for
low addresses (`< 0x0400_0000`, `LOCAL_DATA_END`):

| Path | Low-address backing | Keyed on |
|------|--------------------|----------|
| CPU data load/store (`mem.rs`) | `local_data` (DRAM overlay) | **virtual** addr, MMU-bypassed |
| CPU fetch / `l32r` | `rom` (image / IRAM) | translated phys, reads image |
| Raw `bus.load32/store32(addr)` | `rom` (image) via `region()` | raw addr as **physical** |

So `bus.store32(0xf9e0, v)` targets the read-only image (logged + dropped) while
the CPU reads `0xf9e0` from `local_data` (DRAM). Same number, different memory.
There is **no single function** that answers "read/write what the CPU sees at
virtual data address V", so probes hand-rolled the split and got it wrong
repeatedly across the iter18 boot RE (the `store_local8`/`load_local32` helpers
are the empirically-correct low-window path).

This is fundamentally a **Harvard architecture** fact: the low address range is
backed by two physical memories -- instruction memory (the read-only image) and
data memory (the writable DRAM the boot memset targets) -- and which one an
access hits depends on its **side** (fetch/`l32r` = I-side -> image; load/store =
D-side -> DRAM). The current model encodes that split as scattered, virtual-
address-keyed special cases instead of one physical-address-plus-side decision.

## Characterization finding (settles the one design risk)

Under a translation-authoritative model, low-window *data* now translates where
the bypass never did -- which only works if the low window is TLB-covered during
boot. **Measured directly** (2026-07-06, throwaway probe over the real boot at
300k-instr steady state):

- Every low-window data address the firmware touches (`0xf9e0` scheduler poll,
  `0x9070` done-flag, `0x2250`/`0x2278`/`0x22bc` SCHED struct, generic
  `0x1000`/`0x100000`) is **resident in DTLB way 6, entry 0**, translating
  **identity** (paddr == vaddr), ring 0.
- Way-6 ei 0 = `vaddr 0 -> paddr 0, asid 1, attr 3 (RWX), variable` -- the
  varway56 reset identity region for `0x0..0x1fff_ffff`, **still live** after the
  prologue. By contrast ei 1 (the `0x2000_0000` code region) shows **asid 0** --
  the prologue's `idtlb 0x20000006` invalidated the code region D-side but
  deliberately left the low DRAM window mapped.
- attr 3 = RWX grants **both read and write**, so low-window loads *and* stores
  translate cleanly (no `STORE_PROHIBITED`).

**Consequence:** the translation-authoritative path is **transparent** on the
real boot -- translate-first resolves low-window data identity via way-6 ei 0 and
routes D-side to DRAM, the same result as today's bypass, now authoritative. **No
added low-window mapping is required.** Task 1 therefore *locks this in as an
assertion* rather than discovering it.

---

## Two access domains

The consumer survey (`mem.rs`, `fastpath.rs`, `host_mailbox.rs`, `psp_map.rs`,
`mmu.rs`, and the `boot_tests` probes) splits cleanly:

1. **CPU / MMU domain -- translation-aware.** The executor, the fill fast-path,
   and instruction fetch. A CPU (hence an MMU) is in hand; access goes
   `vaddr -> translate -> (paddr, side) -> backing`. Uses the **canonical
   `Cpu` accessor**.

2. **Physical / external domain -- no translation.** The DMA/mailbox/completion
   agents (`host_mailbox.rs`, which hold only `&mut Bus`), the PSP page-table
   install, autorefill's own PTE fetch, and probes that deliberately inspect a
   specific backing. Real DMA/completion hardware writes physical DRAM directly,
   *not* through the Xtensa DTLB -- so these correctly do **not** translate; they
   address physical memory through **side-explicit** bus methods.

This is exactly the "go through it, *or* an explicit split with clear semantics"
choice from the project note: it is **both**, partitioned by whether a CPU is in
hand.

---

## Component design

### 1. `Bus` -- side-explicit physical Harvard API (`mmio.rs`)

Replace the ambiguous `load32/store32/load8/store8` (whose low-address meaning is
silently I-side) with two side-named physical families. The **low-range fork
lives in exactly one place** per family.

```rust
// D-side (data unit): low (< LOCAL_DATA_END) -> DRAM (local_data);
// else -> region backing (ram/mailbox/array/system/page_table, with their
// existing stub/drop semantics and probe recording).
fn data_load32(&mut self, paddr: u32) -> u32
fn data_load8(&mut self, paddr: u32) -> u8
fn data_store32(&mut self, paddr: u32, v: u32)
fn data_store8(&mut self, paddr: u32, v: u32)
fn data_fill(&mut self, paddr: u32, pattern: &[u8], byte_len: usize)

// I-side (instruction unit): low -> image (rom, +load_offset);
// else -> region backing.
fn inst_load8(&mut self, paddr: u32) -> u8
fn inst_load32(&mut self, paddr: u32) -> u32
```

- The existing `load_local{8,32}` / `store_local{8,32}` / `fill_local` become the
  **low branch** of the D-side family (folded in, not a parallel surface).
- `data_fill` unifies the two bulk-fill paths the fast-path currently juggles
  (`fill_local` + `fill_pattern`). **Contract:** `data_fill(paddr, pattern, len)`
  MUST be byte-identical to `len / pattern.len()` successive `data_store8` calls
  at consecutive paddrs -- so it splits its range at `LOCAL_DATA_END` **and at
  every region boundary**, routing each sub-span to the backing a single D-side
  store would hit (low -> DRAM; Array/System -> dropped; Ram/Mailbox/PageTable ->
  backing). It preserves `fill_local`'s zero-pattern **no-grow** optimization on
  the DRAM sub-span -- critical: it keeps the boot's 128 MiB zero-memset from
  allocating ~64 MiB. The fast-path translates each page and calls `data_fill`
  for the whole chunk; the boundary knowledge lives in the bus, not the caller.
  (Adversarial finding 1: a single per-call routing decision would mis-route a
  cross-`LOCAL_DATA_END` non-zero fill into DRAM, invisibly to the current
  spanning test.)
- `fetch8(vaddr, phys)` **stays**, layering its vaddr-keyed file-offset overlay
  (`rom_overlays`) on top of `inst_load8`. That overlay is a load-*placement*
  quirk, orthogonal to Harvard side routing -- untouched.
- `peek8` **stays** (side-effect-free image read for disassembly; explicitly
  I-side).
- Bare `load32/store32/load8/store8` are **removed**; every call site picks a
  side. The compiler enforces complete migration.
- **The I/D distinction only bites in the low range.** For any high address the
  `data_*` and `inst_*` families route identically (both -> region backing) --
  the Harvard split is a property of the low window alone, where the image and
  DRAM physically coexist. The families differ *only* on `paddr < LOCAL_DATA_END`.
- **Low-address bare call sites migrate by INTENT, not mechanically.** Both
  `data_*` and `inst_*` compile at every site, so the compiler cannot tell which
  side a low-address `load32`/`store32` meant -- "the compiler enforces complete
  migration" is true for *compilation*, not *semantics*. Each low-address site is
  audited: image (I-side, e.g. the anti-aliasing assert at `mem.rs:597`, which
  reads the image to prove a store did NOT reach it -> keep `inst_load32`) vs DRAM
  (D-side). High-address sites are unambiguous (single backing) and rename freely.
  (Adversarial finding 2.)

### 2. `Cpu` -- the canonical translation-aware data accessor (`interp/mod.rs`)

```rust
// translate(Load/Store) then D-side physical route. The one path for
// "what the CPU reads/writes at virtual data address V". Faults propagate as
// Step::Exception exactly as Cpu::translate already does.
fn data_read32(&mut self, bus: &mut Bus, vaddr: u32) -> Result<u32, Step>
fn data_read8(&mut self,  bus: &mut Bus, vaddr: u32) -> Result<u8,  Step>
fn data_write32(&mut self, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step>
fn data_write8(&mut self,  bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step>
```

- `data_read32` = `self.translate(bus, vaddr, Access::Load)?` then
  `bus.data_load32(paddr)`. `data_write32` = `translate(..., Access::Store)?`
  then `bus.data_store32(paddr, v)`. Byte variants identical with `data_load8`/
  `data_store8`.
- These are the **sole** entry for CPU data access. Probes with a `Cpu` in hand
  call them, so probe-vs-CPU disagreement becomes structurally impossible.

### 3. `mem.rs` -- executor ops call the accessor

- `data_load32/8`, `data_store32/8` drop their `if is_local_data { ... } else {
  translate ... }` branch and call the `Cpu::data_*` accessor.
- `load16`/`store16` stay byte-composed and page-safe. `store16` MUST keep its
  **no-half-write** guarantee: translate BOTH byte destinations before writing
  either, so a page-straddling `store16` whose high byte faults never applies the
  low byte. Under translation-authoritative *both* bytes can now fault, so the
  validate-both-then-write order matters more, not less. A new page-straddle-fault
  test pins it (there is none today). (Adversarial finding 3.)
- `assert_low_window_identity` is **deleted** -- the bypass it guarded is gone;
  translation is now authoritative, so the guard is obsolete.
- `l32r_load` keeps its quirk: `translate(target, Access::Load)?` (the DTLB, as a
  load) then `bus.inst_load32(paddr)` (reads the **image** literal, I-side) --
  preserving the iter12 behavior and its regression test.
- Fetch in `step()` is unchanged in spirit: `translate(pc, Access::Fetch)?` then
  `bus.fetch8(vaddr, phys)` (I-side + overlay).

### 4. `fastpath.rs` -- uniform translation

The fill fast-path drops its `Bus::is_local_data` special case: it translates
every page-bounded chunk via the non-raising `Mmu::translate` (unchanged) and
fills via `bus.data_fill(paddr, ...)`, which routes low -> DRAM / high -> region.
The fault-replication arm (translate through `Cpu::translate` to raise) is
unchanged. The low-window fill now flows through translation like everything else
(the characterization finding guarantees it resolves).

### 5. `host_mailbox.rs` -- external actors, side-explicit physical

The completion/consumer agents hold only `&mut Bus` (no CPU) and address physical
memory directly, which is correct (DMA does not use the DTLB). Migrate their
calls to the side-named physical API with **no behavior change**:
`load_local32/store_local32` -> `data_load32/data_store32` (the SCHED struct /
done-flag live in low DRAM); `load32/store32` on the `0x2720_xxxx` mailbox
registers -> `data_load32/data_store32` (high -> region backing, same result).
The rename makes the D-side intent explicit.

### 6. `mmu.rs` / `psp_map.rs` -- physical PTE traffic

Autorefill's PTE fetch `bus.load32(t.paddr)` -> `bus.data_load32(t.paddr)` (a
physical D-side read of the page table); the `write_tlb`-test PTE stores and
`write_page_table_word` are physical writes -> `data_store32` /
`write_page_table_word` (unchanged). No semantic change.

### 7. `boot_tests` probes (`mod.rs`) -- migrate the CPU-view ones

- Probes meaning "what the CPU sees at virtual data address V" move to
  `cpu.data_read*` (they have a `proc.cpu` in hand). This is the durable payoff:
  no probe hand-rolls the split again.
- Probes that **deliberately** read a specific backing keep explicit physical
  helpers, now self-documenting: the image-vs-overlay diff (mod.rs:2498-99)
  reads `inst_load32` (image) vs `data_load32` (DRAM); `peek8` disassembly stays.
- **Side-effect note (adversarial finding 5):** `cpu.data_read32` on a high
  (Mailbox/Array/System) address fires `record_stub` (advances `probe_seq`,
  appends a `StubAccess`); `peek8` is I-side/image only, so it is *not* a
  side-effect-free D-side peek. A probe reading a high data address while the
  stub-probe is armed would perturb its own log. If that combination is needed,
  add a side-effect-free `data_peek` (mirrors `peek8` on the D-side) -- only if a
  probe actually requires it (YAGNI otherwise).

---

## Data flow (after)

```
CPU data load:   op -> cpu.data_read32(bus, vaddr)
                       -> cpu.translate(vaddr, Load) -> paddr
                       -> bus.data_load32(paddr)
                          -> paddr < LOCAL_DATA_END ? local_data(DRAM) : region backing
Probe (CPU view): cpu.data_read32(bus, vaddr)         # identical path
External DMA:     bus.data_store32(paddr, v)          # no translation, physical
CPU fetch/l32r:   translate(...) -> bus.fetch8/inst_load32(paddr)  # I-side -> image
```

One path per intent; the low-range image-vs-DRAM fork exists in exactly two
places (the D-side and I-side bus families) and nowhere else.

## Error handling

- Translation faults propagate as `Step::Exception` via the existing
  `Cpu::translate` chokepoint (EPC1/EXCVADDR/EXCCAUSE set as today). No new fault
  surface: the characterization finding shows low-window data resolves identity
  with RWX, so the newly-translating low path does not introduce faults on the
  real boot.
- External-domain accesses (no CPU) cannot fault -- they write physical memory
  directly, matching DMA hardware.

## Testing strategy / success criteria

- **Task 1 (characterization lock, TDD first):** assert the STRUCTURAL facts the
  design rests on, not just a point sample (adversarial recommendation 5): (a)
  `Mmu::new_with_varway56(true)` populates DTLB way-6 ei0 = `VPN 0 -> paddr 0,
  asid 1, attr 3 (RWX), variable` at reset, before instruction 0; (b) the
  prologue's `idtlb 0x20000006` invalidates way-6 **entry 1** (code region),
  leaving **entry 0** (low window) live; (c) asid 1 always resolves to ring 0
  (`write_rasid` forces the ring-0 byte); (d) at boot steady state every
  low-window data probe still translates identity via way-6 ei0. Together these
  close the early-boot gap: low-window data is covered from reset through steady
  state, never transiently cleared. Skips cleanly when the firmware binary is
  absent (like the other boot tests).
- **Headline invariant -- equivalence:** `cpu.data_read32(V)` ==
  the executor op's result == a probe's read at `V`. A dedicated test pins
  "probe == CPU" for a low (DRAM) and a high (RAM) address.
- **Bus Harvard API units:** `data_*` vs `inst_*` for low (DRAM vs image) and
  high (single region backing), including the anti-aliasing invariant (a D-side
  low store does not touch the image; an I-side low read does not see it).
- **`Cpu::data_*` units:** low, high, and fault propagation (unmapped page ->
  `Step::Exception`, pc not advanced).
- **Regression gate:** all existing `mem`/`mmu`/`fastpath`/`interp`/`mmio`/
  `host_mailbox` tests stay green (call-site renames only, asserted values
  unchanged).
- **Integration proof:** the firmware boot suite stays **433/0/1** -- the
  end-to-end evidence the rewrite is behavior-preserving on the real path.

## Non-goals (out of scope)

- **The MMU walk itself** (`mmu.rs` lookup/autorefill/permission decode) -- it is
  already faithful and is not modified beyond the one PTE-fetch call rename.
- **Modeling a non-identity low-window remap** -- proven absent on this boot
  (way-6 ei 0 survives). If a future image remaps the low window, the
  translation-authoritative path already handles it correctly for free; no
  speculative machinery is built now.
- **The fetch file-offset overlay** (`rom_overlays`) and multi-segment load model
  -- orthogonal to Harvard side routing, untouched.
- **Boot-to-idle progress** -- this design unblocks that work by making memory RE
  reliable; it does not itself advance the boot wall.

## Risks

1. **Wide surface (eight files:** `mmio.rs`, `interp/mod.rs`, `interp/mem.rs`,
   `interp/fastpath.rs`, `host_mailbox.rs`, `mmu.rs`, `psp_map.rs`, `mod.rs`
   probes**).** Adversarial review confirmed the blast radius is fully contained
   in `src/firmware/` -- **zero** `Bus` consumers elsewhere (no `crates/`,
   `tests/`, `xrt-plugin/`, FFI, `src/visual/`, `src/interpreter/`) -- ~25
   production + ~198 test edits, all owned. Also verify `coverage_scan.rs` (uses
   only the preserved `peek8`; expected no migration). Mitigated: after the
   accessor + bus API land, each migration is a compiler-checked rename gated by
   the full existing test suite (with the by-intent audit of finding 2); tasks
   are sequenced so the tree compiles and tests pass at every commit.
2. **A missed probe intent** (migrating a deliberately-physical probe to the CPU
   view, or vice-versa). Mitigated: probes are inspected case-by-case in their
   own task, and the explicit `data_*`/`inst_*` names make the intended backing
   legible at the call site.

## Reference

- Alias-tax evidence and the iter18 boot RE:
  `docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`.
- Existing Harvard / MMU model design:
  `docs/superpowers/specs/2026-07-04-m2c-mapping-boot-to-idle-design.md`,
  `docs/superpowers/specs/2026-07-04-m2b-mmu-mechanism-design.md`.
