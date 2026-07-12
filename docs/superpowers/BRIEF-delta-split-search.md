# Brief: does an (offset δ, split-point) reframing of [0x8c98,0x8d52) dissolve the 0x8cae collision?

## The question (one sentence)

The emulated Phoenix mgmt firmware boots clean, builds `_NPU` in local memory, but
walls at the runtime "collision" at identity-mapped PA `0x8cae`: the **publisher**
call graph (AT-framed, reaches `waiti 0x5645`, builds `_NPU`) and the **service**
call graph (BASE-framed, reaches `0x8c6c`, is the path that copies `_NPU` to device
SRAM `0x030bb000` + writes `FW_ALIVE_OFF 0x030bf000`) require *different* file bytes
at the *same* physical byte — **determine whether that collision is genuine, or an
artifact of the reconstruction only ever testing two whole-region file offsets
(BASE=+0x5c, AT=+0x100)**, by searching (region file-offset δ × instruction-boundary
split-point) over `[0x8c98,0x8d52)` for an assignment under which BOTH call graphs
still reach their landmarks without contradicting on a shared byte.

## Why this is tractable and in-scope (read first)

This is a bounded search **in code we own** (`src/firmware/`), NOT a re-run of walled
escalations. Two things are CLOSED — do NOT reopen:

- **PSP-loader RE** (recover the firmware's physical scatter/placement map from
  extractable off-chip PSP firmware): CLOSED NEGATIVE. Placement logic is in on-chip
  mask ROM or absent from extraction. Do not disassemble PSP/BIOS blobs.
- **Self-modifying code / runtime byte rewrite** at `0x8cae`: CLOSED NEGATIVE THIS
  SESSION. `m2c_probe_overlay_store_conflicts` was run to the collision wall
  (instrs=53659, pc=0x8cb1 — the full publisher AND service window) and captured
  every store landing in a `ground_truth()` overlay range, which INCLUDES
  `(0x8c98,0x8d52)`. Result: **zero stores** ever touch `0x8cae..0x8cbc` or the
  literal `0x354c`. The byte at `0x8cae` is genuinely STATIC and single-valued. Do
  not chase a time-varying-byte / loader-second-stage mechanism; it does not exist.

## The exact reframe this task must test

The "genuine collision" verdict (findings `2026-07-11-alive-sram-overlay-collision.md`
and `-8c6c-service-path-is-real.md`) rests on TWO asserted instruction boundaries:

- publisher: `Bgeu @0x8cac` (3 bytes, so its 3rd byte occupies `0x8cae`), branch taken
  to `0x8cb4`, then AT `MoviN @0x8cb4`, ... reaching `0x5645`.
- service: `Addi a8,a8,0x60 @0x8cae` (starts on `0x8cae`), then `Addmi @0x8cb1`,
  `Addmi @0x8cb4`, `Wsr @0x8cb7`, `RetwN @0x8cba`, returning coherently to `0x7fe4`.

Both were accepted because each side "executes coherently to a landmark." **But with
variable-length Xtensa instructions, coherence to a landmark does NOT pin a unique
framing.** The framing search (`m2c_probe_execution_guided_framing_search`, line ~883)
only ever tried the shared cell at two whole-region offsets ({BASE,AT} × {code,literal}
= 4 assignments) and its "free section variables: []" conclusion is *conditioned on
those two offsets*. It never searched:

1. a **third file-offset δ** for the region (the true scatter map — in mask ROM —
   could place `[0x8c98,0x8d52)` or a sub-range at any δ, not only +0x5c/+0x100), and
2. a **split-point** inside `[0x8c98,0x8d52)` that assigns different δ to sub-ranges,
   which SHIFTS the induced instruction boundaries on each side (so `0x8cac` may not
   frame as a 3-byte `Bgeu`, and `0x8cae` may not be a `Addi` start — the shared-byte
   contradiction can simply evaporate under a different boundary set).

The publisher's AT framing is strongly validated (builds `_NPU`, reaches `waiti`), so
treat the publisher landmark as a hard constraint. The suspect is the *service-side*
boundary set near `0x8c8b..0x8cba` (short 5-instruction coherent run = weak pin), but
the search must not assume that — it must re-derive BOTH boundary sets under each
candidate (δ, split) and check both landmark predicates.

## Search space and predicates

- **Region:** `[0x8c98, 0x8d52)` (the AT overlay). Consider also the immediately
  upstream service bytes `[0x8c6c, 0x8c98)` if a split there shifts the `Bbci @0x8c8b`
  target/boundary that delivers control to `0x8cae`.
- **δ candidates:** at minimum {+0x5c, +0x100}; ALSO test a swept/third δ — the map is
  a per-region file-offset, and other regions in `load_m2c` may already use deltas
  other than these two (enumerate the actual deltas present in the reconstructed map
  and include them as candidates). A δ is only admissible if the resulting bytes decode
  as valid instructions along the executed path.
- **Split-points:** every byte boundary in `[0x8c98,0x8d52)` (bounded: <0xba positions).
  A split assigns δ_lo to `[0x8c98, split)` and δ_hi to `[split, 0x8d52)`.
- **Publisher landmark predicate (hard):** builds `_NPU`
  (local memory `[0x14820] == 0x55504e5f`) AND natural execution reaches `0x5645`.
- **Service landmark predicate (the goal):** the service path performs the device-SRAM
  publish — a store composing the 16-word struct at `0x030bb000` AND the pointer write
  to `FW_ALIVE_OFF 0x030bf000`. Use `m2c_probe_alive_device_sram_struct` (line ~2261)
  as the acceptance oracle.
- **Admissibility:** derive-not-force. The shared physical byte(s) hold ONE value;
  a candidate is valid only if that single value, under the re-derived boundaries,
  satisfies both cones. No per-path byte swap, no state injection.

## Code anchors (all in `src/firmware/`)

- `FirmwareProcessor::load_m2c` + overlay tuples: `mod.rs:120-260`; constants
  `mod.rs:380-512` (`PSP_LOAD_OFFSET=0x5c` BASE, `LOW_VMA_FILE_OFFSET=0x100` AT,
  `SEG_B_*`). This is where any resulting (δ, split) correction lands.
- Bus / overlay selector / `add_rom_overlay` / `remove_rom_overlay`: `mmio.rs`
  (VMA-keyed selector ~`:200-227`, `peek8` BASE-only ~`:462-475`).
- Existing probes to GENERALIZE (do not invent a new harness):
  - `m2c_probe_execution_guided_framing_search` (`coherence_mapper.rs:883`) — already
    pins both backward cones and tests the 4 {BASE,AT} assignments. Extend it to sweep
    (δ, split) and re-derive boundaries under each.
  - `m2c_probe_alive_device_sram_struct` (`:2261`) — BAR2-dump acceptance oracle for
    the service landmark.
  - `m2c_probe_overlay_store_conflicts` (`:1083`) — the store audit that closed the
    self-mod hatch (note: its `assert_eq!(pc,0x5645)` is now STALE post-`75796dc2`;
    the boot walls at `0x8cb1`, not `0x5645` — ignore/adjust that assertion).
- `ground_truth()` (`:326`) — the overlay range list; `(0x8c98,0x8d52)` is entry ~28.

## Reproduce

```
XDNA_FW_PROBE=1 cargo test --lib m2c_probe_execution_guided_framing_search -- --nocapture
XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib m2c_probe_alive_device_sram_struct -- --nocapture
cargo test --lib                       # full suite must stay green (4113+ tests)
```

Firmware auto-detected: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`
(248592 B, Xtensa LE, SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`).

## Deliverables

1. **Verdict:** does a (δ, split) assignment over `[0x8c98,0x8d52)` (optionally with the
   upstream `[0x8c6c,0x8c98)` boundary) exist under which BOTH landmark predicates hold?
   - **If YES:** the exact (δ, split), the re-derived instruction boundaries on both
     sides showing the shared byte is no longer contradictory, and a proposed `load_m2c`
     map correction as a diff, with `m2c_probe_alive_device_sram_struct` now reaching the
     device-SRAM publish (`_NPU` at `0x030bb000` + pointer at `0x030bf000`). Present the
     diff for review — do NOT commit; integration is Opus/Maya's call.
   - **If NO:** a proof of exhaustion — for every admissible (δ, split), which cone's
     landmark breaks and at which byte — establishing that byte `0x8cae` genuinely must
     hold two values under any framing that satisfies both cones. That EARNS the
     exotic-fetch / dual-instruction-memory conclusion the arc previously assumed, and
     is a real, valuable result.

## Ground rules (fidelity — important)

- **Match real hardware; do not force/shim firmware state.** Any fix is a faithfulness
  correction to the EMULATOR's file↔VMA map (a δ/split in `load_m2c`), never a
  firmware-state injection, per-path byte swap, or hardcoded advance. Prior arc lesson:
  forcing scheduler/memory state CORRUPTS the boot.
- The `_NPU` local build is real and correct (`[0x14820]==0x55504e5f`) — don't regress
  it. Full `cargo test --lib` must stay green.
- Honest-negative is valuable. A rigorous exhaustion proof (deliverable 2, "NO") is a
  genuine result that finally justifies the external-mechanism pin — say so precisely.
