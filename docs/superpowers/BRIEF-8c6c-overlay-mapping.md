# Brief: is VMA 0x8c6c..0x8c98 sourced from the wrong file bytes? -- the 0x8cb1 dead-end is a load_m2c overlay-mapping question

## Why this brief exists (the arc pivot -- read carefully)

For many sessions the boot wall was framed as a runtime "view collision /
below-CPU instruction bank" at `0x8cae/0x8cb1`, presumed to need PSP-loader RE.
This session **relocated the wall into our own image loader**, and it is now a
bounded, derive-only question. The verified chain:

1. The plain, un-forced boot (`m2c_probe_alive_device_sram_struct`, plain
   `FirmwareProcessor::load_m2c`, NO `HARNESS_VIEW`) stops at
   `n=53659 pc=0x8cb1 word=0x61a800` (Unknown). This is NOT a probe artifact --
   it is the natural boot.
2. Ground-truth Xtensa disassembly (two independent tools, see Anchors) of the
   raw image shows `0x8caf = J 0x8d50` (3 bytes), so `0x8cb1` is **mid-instruction**
   in the correct alignment -- not a real boundary.
3. Our decoder is NOT broken: a static linear disasm from `0x8c60` matches
   ground truth exactly (`Bgeu@0x8cac`, `J@0x8caf`, `L32iN@0x8cb2`). So the
   dynamic boot reached `0x8cb1` by executing a DIFFERENT byte-stream than the
   raw image.
4. Root cause: `load_m2c` (`src/firmware/mod.rs:120-165`) uses two address
   mappings -- a **default** `file = VMA + 0x5c` (the PSP load offset) and
   hand-added `+0x100` **ROM overlays** (`LOW_VMA_FILE_OFFSET`, via
   `add_rom_overlay`) for specific low-VMA code blocks (`LOW_TEXT_BLOCK`,
   `SYSCALL_BLOCK`, `CTXSW_CALLEE`, `IPC_PRIMITIVE`, ...). **VMA `0x8c6c..0x8c98`
   is not covered by any `+0x100` overlay**, so it falls through to the default
   `+0x5c`:
   - `+0x5c` (current): VMA `0x8c6c` -> file `0x8cc8` -> bytes `36 41 00` =
     `Entry` + a loop that branches (`Bbci bit3`) to `0x8cae`, then falls into
     `0x8cb1 = 0x61a800` (undecodable) -> **dead-end**.
   - `+0x100` (overlay scheme): VMA `0x8c6c` -> file `0x8d6c` -> bytes `22 2a 00`
     = a clean `l32i/extui/beqz` function.
   The two mappings re-converge at `0x8c98` (both give file byte `36 81 00`),
   which is evidently an overlay boundary. So `0x8c6c..0x8c98` looks like a
   **gap (or mis-range) in the overlay table**.

## The question

**For VMA `0x8c6c..0x8c98`, which file-byte source is CORRECT: the default
`+0x5c` (file `0x8cc8`) or a missing `+0x100` overlay (file `0x8d6c`)?**

This must be **DERIVED from the image's own structure and the code's semantics**,
NOT fitted to "whatever makes the boot advance." The determination criteria, in
order:

1. **Disassembly coherence (ground truth).** Disassemble BOTH candidate sources
   for `0x8c6c..0x8c98` with the two independent Xtensa disassemblers and
   cross-check. Which source is a single coherent function with valid internal
   branch/loop targets (not landing mid-instruction, not dead-ending)?
2. **Call/entry semantics.** The firmware reaches `0x8c6c` via `Call8 0x8c6c`
   (from `0x7fe1`, observed ~n=53639). A windowed `Call8` target must begin with
   an `Entry` instruction. Which source begins with a valid `Entry` whose frame
   size is consistent with the callee's stack use? (Note: the `+0x5c` source
   starts with `Entry` but dead-ends; the `+0x100` source starts with `l32i`
   (no `Entry`). Resolve this tension -- e.g. is `0x8c6c` really the call target,
   or is the true target elsewhere and our default mapping mis-sizes an
   upstream instruction so PC lands at `0x8c6c` wrongly? Follow the evidence.)
3. **Consistency with the existing overlay pattern.** Read the overlay
   constants and `psp_load_map` (`src/firmware/mod.rs`, `src/firmware/psp_map.rs`).
   Do the existing `+0x100` overlays form a pattern (contiguous low-VMA code
   sections) that `0x8c6c..0x8c98` naturally belongs to? What determines which
   VMA ranges are `+0x100` vs `+0x5c` -- is there a derivable rule (section
   table, a boundary in the image) rather than a hand-maintained list?
4. **Downstream coherence.** Under the correct source, does execution from
   `0x8c6c` proceed past `0x8cb1` to a sensible continuation (ideally toward the
   device-SRAM copy-out / `FW_ALIVE_OFF` publish)? This is a CONSEQUENCE to
   verify, not the fitting criterion.

## If the fix is a load_m2c change

Unlike prior briefs, **you MAY modify production `load_m2c`** here, because the
fix -- if the source is wrong -- is a loader correction. But:

- The change must be a **derived** overlay-range correction (an `add_rom_overlay`
  with a range justified by the image structure / disasm coherence), NOT a
  hardcoded byte swap, NOT a per-VMA special-case to dodge `0x8cb1`, NOT a forced
  branch. If you cannot derive the correct range from the image, say so and do
  NOT invent one.
- Prefer a **rule** over a hand-added constant if the image structure supports
  one (e.g. if a section table or a consistent boundary dictates which ranges are
  `+0x100`). A rule that subsumes the existing hand-added overlays is the ideal
  outcome; a single new justified overlay range is acceptable.
- After the change: `cargo test --lib` MUST stay green (4091 baseline; the
  env-gated probes are +N), AND report where the boot now goes (does it pass
  `0x8cb1`? reach the copy-out? a new frontier?). Do NOT commit.

## Scan for the same signature elsewhere

Once you understand the `0x8c6c` case, **scan the low-VMA space for other
regions with the same signature**: VMA ranges where `+0x5c` and `+0x100` decode
to different byte-streams and the `+0x5c` (default) stream is incoherent /
dead-ends / branches mid-instruction. List any other suspected overlay gaps with
their ranges and evidence. (These are the likely next walls if `0x8c6c` alone
isn't the whole story.)

## Deliverables

1. **Ground-truth disasm of both candidate sources** for `0x8c6c..0x8c98`,
   cross-checked between the two Xtensa disassemblers, side by side with our
   decoder's current output.
2. **The correct-source determination**, justified by criteria 1-3 above (image
   structure + call/branch semantics), with the specific evidence. If it's a
   missing `+0x100` overlay: the exact range (LO..HI) and the derivation. If the
   `+0x5c` source is actually correct: then explain the `0x8cb1 = 0x61a800`
   dead-end some other way (a real decoder gap? an upstream mis-size? name it).
3. **The proposed load_m2c change** (if any), derived not fitted, with
   `cargo test --lib` green and a report of the new boot frontier.
4. **Same-signature scan results** -- other overlay-gap candidates.
5. **Ranked single next step** -- derive-only.

## What "done" looks like

A written finding
(`docs/superpowers/findings/2026-07-13-8c6c-overlay-mapping.md`) with the five
deliverables, plus any production `load_m2c` diff (uncommitted) and probe
changes. Present for review -- do NOT commit.

## Execution discipline (READ -- prior runs hung here)

- A prior job in this arc soft-hung 40+ minutes in a `wait` loop on a self-
  dispatched independent-review subagent. Do NOT dispatch an independent
  reviewer and do NOT enter any `wait`/collaboration poll loop. Verify inline,
  write the finding, TERMINATE. Your final message is the deliverable.
- If a boot/probe run is long CPU-bound, background-and-block in ONE shell
  command (`cmd & wait $!`); do not poll in a `check -> sleep` loop.
- Reconstruct context FIRST by reading this brief + the committed findings +
  the `load_m2c`/`psp_map`/`image` source named in Anchors.

## Ground rules

- **Derive, do NOT calibrate.** The correct byte-source is derived from the
  image structure + ground-truth disasm + call/branch semantics. Do NOT fit the
  overlay to make the boot pass, hardcode a byte, force a branch, or special-case
  `0x8cb1`. "The boot advances" is a consequence to verify, never the criterion.
- Cross-check every decode against BOTH independent Xtensa disassemblers plus
  our own decoder; do not trust a single tool.
- `cargo test --lib` stays green (4091 baseline) after any change.
- Do NOT re-open PSP-loader RE mechanism work, the CPU-self-modifying-code path,
  or a below-CPU-bank hunt -- this is now a load_m2c file-offset question, which
  supersedes those framings.
- Ground everything in Phoenix `1502_00`
  (`../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`, SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`). The
  `17f1_10` sibling is untrusted different-generation.
- Present findings for review. Do NOT commit.

## Anchors

- Firmware: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`.
- The dead-end: plain boot `m2c_probe_alive_device_sram_struct` (in
  `src/firmware/boot_tests/coherence_mapper.rs`) stops at
  `n=53659 pc=0x8cb1 word=0x61a800`.
- Candidate sources for VMA `0x8c6c`: file `0x8cc8` (`+0x5c` default, bytes
  `36 41 00 30 62 00 ...`) vs file `0x8d6c` (`+0x100` overlay, bytes
  `22 2a 00 20 28 74 ...`).
- Loader: `src/firmware/mod.rs:120-165` (`load_m2c`, `add_rom_overlay`,
  `LOW_VMA_FILE_OFFSET`, the overlay constants) and `src/firmware/psp_map.rs`
  (`psp_load_map`, segment offsets), `src/firmware/image.rs` (base-0 image).
- Committed findings: `docs/superpowers/findings/2026-07-13-alive-gate-provenance.md`,
  `2026-07-13-alive-publish-reconciliation.md`.
- Our disasm probe: `XDNA_FW_DISASM=<lo>:<hi> XDNA_FW_PROBE=1 cargo test --lib
  m2c_probe_disasm_range -- --nocapture`.
- Ground-truth Xtensa disassemblers (both present):
  - `xtensa-lx106-elf-objdump -D -b binary -m xtensa --adjust-vma=<vma> blob.bin`
  - `llvm-objdump-20 -D -b binary --triple=xtensa --adjust-vma=<vma> blob.bin`
    (also `llvm-mc-18/19/20` have the `xtensa` target). Extract a raw blob with
    python (`d[file_lo:file_hi]`).
- Base commit: `59d7ab32`.
