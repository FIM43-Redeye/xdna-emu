# Brief: consolidate the memory/overlay architecture — orient, don't grind

## Why this brief exists (change of mode)

Your last run (finding `2026-07-12-upstream-cone-framing-and-reload.md`, committed
`9cd8ddc5`) proved the `0x8cae` collision is a framing artifact and found a
SECOND instance of the same phenomenon at `0x26d4`: two independent VMAs that
hold different code at different times. That is now a confirmed *pattern*, not a
one-off — a genuine dynamic low-VMA overlay.

We have been reverse-engineering this core's memory architecture through
execution, byte by byte, for many sessions — WITHOUT ever consolidating it into
one coherent map. Before any more tactical search, step back and build that
picture. **This is an orientation/consolidation pass. You have wide latitude to
roam. The deliverable is a MODEL and a list of HOLES, not another negative
sweep.** Do not force firmware state, do not propose a `load_m2c` diff, do not
commit. Read, reason, reconcile.

## The tension to resolve (the core hole)

Two established facts are in apparent conflict:
- Prior arc (ITLB probe, finding `2026-07-11-itlb-not-the-code-view-selector.md`):
  `0x8cae` is identity-mapped PA, way-6, byte-identical across publisher and
  service — "silicon has ONE physical byte at 0x8cae."
- This arc: `0x8cae` AND `0x26d4` each require *different* code bytes at the same
  VMA at different times.

Identity-mapped-single-physical-byte and two-different-code-views cannot both be
literally true unless something reloads the backing between the two times, or the
"identity map" conclusion was scoped too narrowly. Reconcile them. Which
assumption has the crack?

## Deliverables (a model + holes, roam to get there)

1. **Draw the memory map.** From the boot/reset code this firmware actually
   executes (the Xtensa MPU/cache/IRAM/region setup it runs early), plus the
   segment layout in `load_m2c`, reconcile the address windows into ONE model:
   - the low window (`<0x10000`: the `0x8cxx`, `0x26xx`, `0x55xx`, `0x7fxx` code),
   - the `0x08b00000` Segment-B RAM region,
   - the `0x2000xxxx` "AT" high region (the publisher root is `0x2000324c`; the
     high-alias code lives at `0x20008xxx`).
   For each: is it physical RAM, an alias of another window, ROM, or an
   overlay/bank region? In particular — is the low window an *alias* of
   `0x2000xxxx` (low = high − 0x20000000), an independent IRAM the firmware loads,
   or something else? The `+0x100` / `+0x5c` file-offset deltas we sweep are
   clues to how the loader placed each.

2. **Mine the Zephyr/MERT open-source angle — this is the "documented vs mask-ROM"
   fork.** This firmware is Zephyr 3.7.1 + AMD MERT. Zephyr has *open-source,
   documented* code-overlay mechanisms: code relocation (`__ramfunc`, reloc
   sections, `CONFIG_CODE_DATA_RELOCATION`), XIP, and demand paging
   (`CONFIG_DEMAND_PAGING`, the Xtensa MMU/MPU backend). Determine whether any of
   these explains a low-VMA region being re-viewed with different code over time.
   If the overlay is a Zephyr feature, the mechanism is RE-able from the Zephyr
   side (design is public) rather than being HW mask-ROM. Check the Xtensa-specific
   Zephyr paths. Does the observed pattern (per-dispatch code re-view, keyed near
   the `0x7fe7 -> 0x26d4` context switch) match Zephyr code relocation / demand
   paging behavior?

3. **Answer the observability question head-on (take this on — Maya cannot answer
   it faster than you).** For device SRAM (`0x030bb000`) we read it back from the
   host as our oracle. This is the MANAGEMENT Xtensa core's *instruction* memory.
   Can the host read that core's IRAM / low code window through the MMIO/debug
   aperture the way we read device SRAM? Investigate via: the xdna-driver source
   (`/home/triple/npu-work/xdna-driver`), the aie-rt / debug apertures, the BAR
   layout, any mgmt-core memory window exposed to the host. If YES: an IRAM dump
   at two time points (early boot vs post-dispatch) IS the overlay, ground truth —
   say how to capture it. If NO: state the ceiling — we would be inferring the
   mechanism from the firmware image alone, and say what that limits.
   HARDWARE SAFETY: reading is fine, but NEVER sustained-poll the BAR0 PSP/SMU
   mgmt aperture (a 30s host MMIO read loop hard-reset the machine on 2026-07-07).
   Single/occasional one-shot reads only. If a capture needs the running firmware
   halted or a specific trigger, describe it — do not attempt a risky HW loop.

4. **Audit our own assumptions for holes.** List what we asserted but did not
   independently verify. The identity-map-vs-two-views tension (deliverable's top)
   is the sharpest. Others to scrutinize: the `+0x100`/`+0x5c` offset model, the
   "flat single segment, no scatter" container claim, whether the `0x2000xxxx`
   region and the low window are the same physical bytes. For each hole, name the
   evidence (firmware bytes, toolchain, Zephyr source, or a HW read) that would
   close it.

## What "done" looks like

A written finding (`docs/superpowers/findings/2026-07-12-memory-architecture-model.md`)
containing: (a) the consolidated memory map with each window classified; (b) the
Zephyr-mechanism verdict (does code-reloc/XIP/demand-paging fit — yes/no/which);
(c) the observability answer (can we read mgmt IRAM from host — yes+how / no+why);
(d) the holes list with the evidence that closes each; (e) a recommended next
concrete step, ranked. No commit — present for review.

## Ground rules

- Orientation, not forcing. No `load_m2c` diff, no firmware-state injection, no
  per-path byte swap. Reading and reasoning.
- Latitude to roam — follow what you find. But converge on the model + holes; do
  not spend the budget regenerating sweeps you already have.
- Full `cargo test --lib` stays green if you touch any probe (4090 baseline).
- Sources in priority: the firmware bytes you already execute, the open-source
  toolchain (Zephyr, aie-rt, xdna-driver, llvm-aie Xtensa), then AM0xx docs.
- Present findings for review. Do NOT commit.

## Anchors

- Firmware: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`
  (SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`).
- `load_m2c` map + constants: `src/firmware/mod.rs:120-260, 380-512`.
- Prior findings: `2026-07-12-upstream-cone-framing-and-reload.md` (this arc),
  `2026-07-11-itlb-not-the-code-view-selector.md` (identity-map claim),
  `2026-07-11-8c6c-service-path-is-real.md`,
  `2026-07-11-alive-sram-overlay-collision.md`.
- Driver source for observability: `/home/triple/npu-work/xdna-driver`.
