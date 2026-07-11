# Go-alive is reached; the ISR "VMA-map wall" was an injection artifact

> **PARTIALLY CORRECTED (2026-07-11, same day).** The reframe below (the ISR
> collision is an injection artifact; there is no section map to recover) STANDS
> and is reinforced. But this doc's "Milestone" framing is WRONG on one point:
> the firmware does **not** park at the 0x5645 waiti "awaiting the completion
> that finalizes go-alive." That waiti is itself a mapping artifact -- the guard
> literal at VMA 0x31a4 is misframed (BASE=0 vs AT=0x27010ac0), so we deref addr
> 0, read garbage, and park; faithful execution reads the real status word (=0)
> and `beqz`-skips the waiti entirely. There is NO async completion to wait for
> or synthesize. The one genuine remaining step is not "deliver the completion"
> but "continue the mapping-first overlay extension." See
> `2026-07-11-goalive-dispatch-target-and-completion.md`.

**Date:** 2026-07-11
**Status:** Reframe (injection artifact / no section map) STANDS; the "awaiting
completion" milestone framing is CORRECTED -- see banner. Supersedes the
*premise* of `2026-07-11-firmware-vma-file-map-not-statically-recoverable.md`.
**Branch:** `feat/m2c-mapping-boot-to-idle` (unmerged).

## Milestone (reproducible)

From the reset entry via `FirmwareProcessor::load_m2c`, the firmware boots its
own code ~52.4k instructions, runs `goalive_runfn`, and **publishes the mgmt
channel**: `local_data[0x14820] == 0x5550_4e5f` ("_NPU"). It then rests at a
`waiti` at pc=0x5645, which is **inside** `goalive_runfn` (+0x4d). Gate:
`guards.rs::m2c_boot_advances_into_c_runtime` (reached_idle, wait=Waiti,
last_pc=0x5645, magic=_NPU).

So go-alive gets far enough to publish the channel, then **parks at a waiti
waiting for the completion that finalizes it**. It has not returned to
steady-state idle; the natural boot stops there only because nothing in the
model wakes it. Delivering that completion faithfully is the one genuine
remaining step for boot-to-alive.

## The reframe: the "VMA-map collision" was our own stimulus

The recent arc chased a static/runtime `VMA->file` map for a line-0
interrupt-service path that walls at VMA 0x8cb1. Three independent results show
that path is **an artifact of injecting a line-0 interrupt**, not real firmware
behavior:

1. **No static map (Codex execution-guided search, committed `815911bc`,
   `m2c_probe_execution_guided_framing_search`).** Both call cones are pinned
   from independent architectural roots with zero free framing variables:
   publish from absolute pointer 0x55f8 -> AT/+0x100; the injected line-0 service
   from the reset/vector spine -> 0x7fc4 -> 0x8c6c BASE/+0x5c. The conflict
   normalizes to one shared code cell [0x8cae,0x8cbc): publish REQUIRES code=AT
   (else walls 0x8c32), service REQUIRES code=BASE (else walls 0x8cb1). No
   vaddr-keyed assignment serves both.

2. **No runtime remap (`m2c_probe_isr_remap_hunt`).** Single-stepping the
   injected line-0 ISR from the waiti to the 0x8cb1 wall: 21 stores, ALL to data
   (exception frame 0x22xx/0x31xx) or the mailbox (0xfae0) -- **zero stores into
   the 0x8c00..0x8e00 code region**, and **no witlb/wdtlb**. So nothing copies or
   remaps the conflict cell before the firmware runs there. (A sub-page 0xa4 shift
   cannot come from a TLB page anyway.)

3. **The whole service chain is never executed naturally
   (`m2c_probe_natural_isr_chain_reachability`).** In the natural boot to the
   waiti (52,390 instrs), first-hit of {0x7fc4, 0x7fe1, 0x8c6c, 0x8c72, 0x8c88,
   0x8cb1, 0x2958} is **None** for every one -- including the general-exception
   handler 0x2958. Only 0x28b4 (the syscall stub) runs, once (n=47437).

Taken together: the real code at VMA 0x8c98+ is the **AT** publish helper (it
runs in the natural boot and yields the correct `_NPU`). The **BASE**
`FUN_00008c68` at 0x8c6c is a phantom -- its bytes are the AT function decoded at
the wrong framing, reached only because our injected interrupt routed the wake
into a region the production overlay ([0x8c98,0x8d52) at AT) does not cover below
0x8c98. There is no map to recover and no remap to model.

## What "100% firmware" now requires

Not a section map -- a **faithful go-alive-completion wake + dispatch**. At the
waiti, `INTENABLE==0x1`: line 0 is the wake line the firmware itself armed. When
we assert it, our exception/interrupt **dispatch** routes to the phantom 0x8c6c
instead of the real service. The real firmware, on that wake, reads a handler
target (a pointer / vector-table entry) and jumps to it; our model computes or
frames that jump wrong. That is an ordinary emulator-fidelity problem, well
scoped, not an unrecoverable-artifact one.

## The core question for the next arc

**What completion does the `goalive_runfn` waiti (pc=0x5645) wait for, and when
line 0 fires, where does the real firmware dispatch -- i.e., which handler
pointer does it read, and from where?** Localize it from the instructions the
firmware runs in the first steps after the waiti wakes (before it reaches the
phantom), and the memory it reads the dispatch target from. Settling that
replaces the phantom path with the real one.

## Instruments (all XDNA_FW_PROBE-gated, `boot_tests/coherence_mapper.rs`)

- `m2c_probe_execution_guided_framing_search` -- the no-static-map proof.
- `m2c_probe_line0_service_returns` -- Gate B, a deliberate RED TDD target
  (goes green when the real dispatch is modeled).
- `m2c_probe_isr_remap_hunt` -- no code-region store / no TLB on the injected ISR.
- `m2c_probe_natural_isr_chain_reachability` -- the service chain is injection-only.
