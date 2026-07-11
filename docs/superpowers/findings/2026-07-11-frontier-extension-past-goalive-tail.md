# Frontier extension past the go-alive tail

**Date:** 2026-07-11  
**Target:** Phoenix/NPU1 `1502_00/npu.dev.sbin`  
**Image SHA-256:** `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`  
**Status:** Q1 resolved; the code-framing frontier is clean through 500,000 instructions. The terminal dispatcher cycle is stable but is not yet proven to be a mailbox-empty idle.

## Mapping model and source anchors

`FirmwareProcessor::load_m2c` installs the low-VMA overlays at
`src/firmware/mod.rs:120-225`. BASE is `PSP_LOAD_OFFSET=0x5c`
(`src/firmware/mod.rs:380-390`); AT is `LOW_VMA_FILE_OFFSET=0x100`
(`src/firmware/mod.rs:502`). `Bus::add_rom_overlay` and the VMA-keyed selector
are at `src/firmware/mmio.rs:200-227`; `peek8` remains BASE-only at
`src/firmware/mmio.rs:462-475`. Segment B uses file start `0x2d100` and physical
base `0x08b00000` (`src/firmware/mod.rs:392-400`). The L32R target formula used
below is implemented and exhaustively checked at
`src/firmware/xtensa/decode/mem.rs:21-32`.

The executable probe and all trial tuples are in
`src/firmware/boot_tests/coherence_mapper.rs:1170` (`m2c_probe_add_31a4_overlay_frontier`).

## Starting anchor: go-alive status literal `0x31a4`

This gap was resolved before this walk and is retained as the starting tuple.

| VMA | BASE file/value | AT file/value | Executed discriminator | Tuple |
|---:|---:|---:|---|---|
| `0x31a4` | `0x3200` / `0x00000000` | `0x32a4` / `0x27010ac0` | `pc=0x562b l32r a7,[0x31a4]`; AT makes `pc=0x563e l32i.n` read the real zero at `0x27010ac0`, so `pc=0x5640` branches over the phantom `waiti` | `(0x31a4, 0x31a8, 0x100)` |

## Q1: `callx8 [0x32c8]` targets Segment B `0x08b04428`

AT bytes at file `0x574a` are `81 1f f7`. At VMA `0x564a`, the in-tree L32R
formula decodes them as `L32r { t: 8, target: 0x32c8 }`; the next instruction at
`0x564d` is `Callx8 a8`.

| VMA | BASE file/value | AT file/value | Executed discriminator | Tuple |
|---:|---:|---:|---|---|
| `0x32c8` | `0x3324` / `0x00015ff0` | `0x33c8` / `0x08b04428` | BASE reaches `pc=0x15ff0`, `Unknown word=0`. AT enters Segment B at file `0x31528`, executes `36 c1 00` (`entry a1,0x60`), takes syscall 0x6b through the shared exception path, returns to `0x5650`, and exposes `0x7c5c`. | `(0x32c8, 0x32cc, 0x100)` |

The real target is therefore **VMA `0x08b04428`**. Segment-B relocation gives
`0x2d100 + (0x08b04428 - 0x08b00000) = file 0x31528`. This refutes all four Q1
alternatives: the L32R is correctly anchored; no third delta is involved;
`0x15ff0` is not missing code; and no runtime store installs the pointer. The
minimal evidence-backed overlay is one word, `0x32c8..0x32cc`.

## Ordered mapping gaps resolved after Q1

Each row was installed at the active execution frontier and the boot rerun. A
clean static decode was not accepted without the executed edge or consumed
value shown here.

| VMA / exposing edge | BASE view | AT view | Execution evidence and exact tuple |
|---|---|---|---|
| `0x7c5c`, from `0x5652 call8` | file `0x7cb8`: `ff1b2256f3fb8602`; immediate `Unknown 0x56221bff` | file `0x7d5c`: `364100206200a50e`, `entry a1,0x20`; helper cluster ends before `0x7cee` | AT executes the scheduler helper chain. `(0x7c5c, 0x7cee, 0x100)` |
| `0x7d4c`, from `0x7c62 call8` | file `0x7da8`: `20cc43f020002512`, a plausible `min; nop; call8` sequence in another BASE body | file `0x7e4c`: `36410021aaed42a0`, `entry a1,0x20`; returns at `0x7dcc` and `0x7e26`, next entry `0x7e28` | BASE creates a long coherent-looking false branch; AT enters the scheduler/interrupt path and exposes `0xd864`. `(0x7d4c, 0x7e28, 0x100)` |
| `0xd864`, from `0x29dd callx4 a4`, `a4=0xd864` | file `0xd8c0`: `6988e6a8c687022c`, middle of BASE `wake_tasks_by_event_mask` | file `0xd964`: `36410050d3034106`, `entry a1,0x20`; the existing `SYSCALL_BLOCK` begins at `0xd8a7` (`src/firmware/mod.rs:442-443`) | BASE corrupts the windowed return and later lands in zero-filled `0x15f01`; AT completes the missing prefix. `(0xd864, 0xd8a7, 0x100)` |
| literal `0x353c`, consumed at `0x8952 l32r a8,[0x353c]` | file `0x3598`: `0x27200310` | file `0x363c`: `0x000117c0` | BASE makes `0x895a` compute `a6=0x030a1000+0x27200310=0x2a2a1310`; `0x8964` then faults `STORE_PROHIBITED`. AT computes `a6=0x030b27c0`, and the executed `0x8964` store succeeds there. `(0x353c, 0x3540, 0x100)` |
| `0xc6b0`, from `0xdc25 call8` | file `0xc70c`: `0229712961295129`, begins `l32i`, not an ABI entry; this body later emits the false `0xc710 call8 0x26d4` | file `0xc7b0`: `3641002062003173`, `entry a1,0x20`; CFG analysis bounds the live function at `0xc730` and names calls `0x2630`, `0xc48c`, `0xc894` | AT executes repeatedly; by 500k it has 132 visits and `0x26d4` has zero. `(0xc6b0, 0xc730, 0x100)` |
| literals `0x3c88..0x3c90`, consumed by the AT `0xc6b0` CFG | files `0x3ce4/0x3ce8`: `0x06194010`, `0x0603450c` | files `0x3d88/0x3d8c`: `0x0000c6b0`, `0x0001186c` | `0xc6b9 l32r` consumes `0x3c88`; `0xc713 l32r` consumes `0x3c8c`. These extend the existing one-word pool tuple at `0x3c84`. `(0x3c88, 0x3c90, 0x100)` |

The exact new tuples, excluding Maya's already-confirmed `0x31a4` tuple, are:

```rust
(0x0000_32c8, 0x0000_32cc, LOW_VMA_FILE_OFFSET),
(0x0000_7c5c, 0x0000_7cee, LOW_VMA_FILE_OFFSET),
(0x0000_7d4c, 0x0000_7e28, LOW_VMA_FILE_OFFSET),
(0x0000_d864, 0x0000_d8a7, LOW_VMA_FILE_OFFSET),
(0x0000_353c, 0x0000_3540, LOW_VMA_FILE_OFFSET),
(0x0000_c6b0, 0x0000_c730, LOW_VMA_FILE_OFFSET),
(0x0000_3c88, 0x0000_3c90, LOW_VMA_FILE_OFFSET),
```

The last tuple may instead be integrated by extending the existing
`(0x3c84,0x3c88)` tuple to `(0x3c84,0x3c90)`.

## Why `0x26d4` is not a tuple

With `0x353c` fixed but `0xc6b0` still BASE-framed, execution is:

```text
0xdc25 call8 0xc6b0
0xc6b0 ... BASE body ...
0xc710 call8 0x26d4
0x26d4 Unknown 0x0039ffa0
```

`0x26d4` already lies inside the broad existing AT `CTXSW_CALLEE` overlay, so
the unknown word is AT file `0x27d4`. BASE file `0x2730` happens to begin a
different coherent `entry`. Shrinking `CTXSW_CALLEE` at `0x26d4` is not valid:
a trial made the pre-go-alive boot fail at `0x2000e035` before `_NPU` publication.
The actual error is the caller. Once `0xc6b0..0xc730` is AT-framed, the 500k
trace never executes `0x26d4`. No BASE override, contextual overlay, or
`0x26d4` tuple belongs in the fix.

The same pruning rule removes the earlier exploratory `0xaf8c`, `0x9e5c`,
`0xd7a8`, `0x4c04`, `0x6f28`, `0xe620`, `0x4e68`, and `0xaef8` candidates: none
is reached by the corrected path.

## Terminal state: mapping-clean, but not yet proven idle

With exactly the tuples above plus `0x31a4`, a 500,000-instruction run has no
`Unknown`, `Wait`, or `PollSpin`. `_NPU` remains published at
`local_data[0x14820] == 0x55504e5f`. The first 40 captured post-go-alive
exceptions are all expected syscall cause `1` transitions through the shared
exception vector; no repeating MMU-fault cycle appears in the execution tail.

The execution is periodic, but it is not honest to call it mailbox-empty idle:

- `goalive_runfn` entry `0x55f8` executes 131 times; tail `0x5645` executes 132
  times through the cutoff.
- The corrected `0xdc25 -> 0xc6b0` edge executes 132 times; the false `0x26d4`
  edge executes zero times.
- `FW_ALIVE_OFF` at `0x030bf000` remains zero.
- The one post-tail modeled SRAM-band store is the now-correct
  `0x8964 -> 0x030b27c0`, value zero; no store reaches `FW_ALIVE_OFF`.

Thus the code-framing frontier is exhausted for this run, but Q2's stronger
idle/alive discriminator is not yet met. The exact next unresolved seam is the
repeated dispatch of the same go-alive run function after its successful return,
not another undecoded VMA. The settling observation is a queue-ownership trace
across the `0xc6b0 -> 0x2630` yield/context-switch path: record the work-item
head/tail and current-task stores before the first and second `0x55f8` entries.
That will distinguish an intentional periodic MERT worker from a queue item that
the emulator fails to retire. Separately, the already-known host-SRAM visibility
seam still explains why local `_NPU` publication does not become
`FW_ALIVE_OFF`; it is not a code-framing gap.

Reproduction:

```text
XDNA_FW_PROBE=1 XDNA_FW_MAX=500000 cargo test --lib \
  m2c_probe_add_31a4_overlay_frontier -- --nocapture
```
