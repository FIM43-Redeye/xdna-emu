# Firmware boot-to-alive arc: BANKED, blocked on unobtainable section-placement data

Date: 2026-07-13  
Branch: `feat/m2c-mapping-boot-to-idle` (UNMERGED)  
Image: Phoenix `1502_00/npu.dev.sbin`

## Conclusion (why the arc stops here)

The in-tree Xtensa interpreter boots the Phoenix management firmware
(`npu.dev.sbin`) to a coherent scheduler idle. It does **not** reach
host-visible "alive" publication, and this session established that the
remaining gap is **blocked on data we cannot obtain**. This is a genuine wall,
not an unfinished task.

### The wall, precisely

The go-alive path requires two real, independently call-anchored functions to
occupy **overlapping virtual addresses** `[0x8c98, 0x8cbc)`:

- the **service** function (rooted `0x8c6c`, reached by `Call8 0x8c6c`) needs
  base (`file = VMA + 0x5c`) bytes there -- its loop/`RetwN` genuinely extend to
  `0x8cba`;
- the **publisher** (rooted `0x8c98`, the `_NPU`/`FW_ALIVE_OFF` copy-out) needs
  the `+0x100` overlay bytes there.

The emulator's address-only ROM-overlay model (one byte-source per VMA) cannot
serve both. No interval edit fixes it: moving the boundary back breaks the
service, forward breaks the publisher. Full technical detail:
`2026-07-13-8c6c-overlay-mapping.md` (commit `b0ce6349`). The `_NPU` builder
itself runs correctly and builds the descriptor in local memory
(`2026-07-13-alive-gate-provenance.md`, commit `59d7ab32`); only the copy-out
to host-visible SRAM is blocked. See also
`2026-07-13-alive-publish-reconciliation.md` (commit `12a99780`).

### Why it cannot be resolved with available inputs

The root cause is that the `$PS1` firmware container is **flat with no section
table**, so `load_m2c` hand-places every low-VMA code section by walk-and-stub
coherence. `0x8c98` is where two hand-placed sections collide. Resolving it
requires the real per-section load map, and the logic is a closed dilemma:

- **If the sections genuinely overlap on silicon** -> a runtime code bank must
  swap the bytes -> but every bank mechanism was eliminated (MMU-remap ruled out
  by ITLB-identity; memory-copy ruled out by zero `+0x100` stores). Contradiction.
- **If they do not overlap** -> a section is mis-placed -> we need the canonical
  placement map, which is **doubly-walled**:
  1. **PSP-loader RE (mask ROM):** the platform PSP firmware (Framework 16 BIOS,
     held in plaintext ARM32) was exhaustively disassembled and contains **no**
     NPU-firmware scatter/placement consumer. The placement lives in the NPU's
     **on-chip Xtensa mask ROM, which cannot be dumped.**
  2. **HW dump:** the scatter targets (low-VMA windows + `0x08b00000`) live in
     the management Xtensa's **private memory, off every host BAR -- never
     host-readable.**

Both external oracles for the placement map are dead ends. Walk-and-stub
coherence is the only available method, and it is fundamentally insufficient for
the overlapping region.

### The one input that would unblock it

A canonical section-placement artifact for this exact `1502_00` image: a linker
map, pre-signing/unstripped ELF with program headers, or a toolchain load
manifest naming each low-VMA section's real load address. Only AMD can supply
this; it is a benign, non-security ask ("where does each code section load"),
but not currently available.

## Status

- **Boot-to-idle:** works (coherence-derived scatter map; `waiti` idle).
- **Go-alive publication:** BLOCKED on placement data (above). Banked.
- Branch `feat/m2c-mapping-boot-to-idle` stays UNMERGED; the env-gated RE probes
  in `src/firmware/boot_tests/` are read-only and do not affect production.

## If revisited

Do not re-run: PSP-loader RE of the platform image (done, negative), HW-dump of
the placement (walled), the below-CPU-bank *mechanism* hunt (eliminated), or
trace-forward RE to "the next thing past the wall" (always dead-ends at the
`0x8c98` overlap). The only productive re-entry is a genuinely new external
placement artifact. Absent that, the firmware model is at its ceiling.
