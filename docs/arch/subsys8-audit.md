# Subsystem 8 -- Parser Audit

**Subsystem:** 8 of 8 (Phase 1b of the device-family refactor, final)
**Spec:** [../superpowers/specs/2026-04-23-subsys8-parser-design.md](../superpowers/specs/2026-04-23-subsys8-parser-design.md)
**Plan:** [../superpowers/plans/2026-04-23-subsys8-parser-plan.md](../superpowers/plans/2026-04-23-subsys8-parser-plan.md)

## Baseline (pre-subsystem, at phase1-subsys-isa-execute tag / HEAD)

- `cargo test --lib`: 2686 passed; 0 failed; 5 ignored
- `cargo test -p xdna-archspec --lib`: 320 passed; 0 failed; 2 ignored
- `cargo build --release`: (capture from Step 1 scratch file)
- Bridge smoke (`--no-hw -v add_one_cpp_aiecc`): green
- Full bridge pass/fail summary: (capture from /tmp/claude-1000/subsys8-baseline-bridge.log tail)
- ISA test pass/fail summary: (capture from /tmp/claude-1000/subsys8-baseline-isa.log tail)

Known pre-existing failures (carry through):
- `bd_chain_repeat_on_memtile` EMU deadlock (bridge suite).
- `dma_task_large_linear` + `objectfifo_repeat/init_values_repeat` Peano EMU timeouts.

## Audit methodology

Per the spec, this audit runs nine sections. Sections 1--6 are per-area
deep dives (XCLBIN, AIE Partition, CDO syntax, CDO semantics, device-state
consumer, ELF consumer). Sections 7--9 are cross-cutting (control-packet
overlap, design note output, trait decision).

Per-area subsection template:

- **Files audited.** Exact paths + LOC.
- **AIE2 hardcode count.** Grep for literal `"AIE2"`, `AIE_ML_*`,
  `aie2`/`Aie2` identifiers, arch-branded constants, hardcoded offsets
  that appear in archspec (drift candidates).
- **Arch variance evidence.** From aie-rt per-arch headers, real XCLBINs
  on disk, llvm-aie TableGen (for ELF), AM025 register DB.
- **Prescribed migration.** `move-to-archspec` / `read-archspec-via-accessor`
  / `leave-alone`.
- **Estimated LOC impact.** Lines changing xdna-emu-side + lines added
  archspec-side.

---

## 1. XCLBIN section-kind classification

Files: `src/parser/xclbin.rs` (456 LOC).

(Filled in by Task 2 Step 1.)

## 2. AIE Partition wrapper

Files: `src/parser/aie_partition.rs` (372 LOC).

(Filled in by Task 2 Step 2.)

## 3. CDO syntax (byte format)

Files: `src/parser/cdo.rs` lines 1--412 approximately (framing + `CdoOpcode` +
`RawCdoHeader` + `Cdo` + `CdoCommandIterator`).

(Filled in by Task 2 Step 3.)

## 4. CDO semantics (device effect per command)

Files: `src/parser/cdo.rs` lines 219--285 approximately (`CdoCommand` enum),
plus `src/device/state/cdo.rs` (`apply_command`).

(Filled in by Task 2 Step 4.)

## 5. Device-state consumer

Files: `src/device/state/cdo.rs`, `src/device/state/mod.rs`.

(Filled in by Task 2 Step 5.)

## 6. ELF consumer

Files: `src/parser/elf.rs` (618 LOC) + 8 consumer files
(see plan §File Structure).

(Filled in by Task 2 Step 6.)

## 7. Control-packet parser overlap

Files: `src/device/control_packets/parser.rs`, compared against
`src/parser/*`.

(Filled in by Task 2 Step 7.)

## 8. Design note output

`docs/arch/binary-loader-model.md` -- "what would AIE1 look like?"
for the parser layer.

(Filled in by Task 5.)

## 9. Trait-or-no-trait decision

(Filled in by Task 2 Step 9.)

---

## Closing summary

(Filled in by Task 2 Step 9.)

### Data migration list

### Trait decision (populated / anchor / none) + reasoning

### Refined `DeviceOp` enum proposal

### AIE1 projection (one paragraph)

---

## Completion

(Filled in at the end of Stage 8a, in Task 6.)
