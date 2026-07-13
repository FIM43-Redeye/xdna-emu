# Brief: is the 0x8cae contradiction created UPSTREAM, by a mis-framed region in one of the two cones?

## Where we are (your own prior results)

You proved two things about the `0x8cae` collision:
- The local `(δ,split)` fix at `[0x8c98,0x8d52)` is impossible: every static offset
  in `{0,+0x5c,+0x100,+0x244}` × split either sends the publisher's `0x8cac` to a
  garbage branch (`[87 ba d2]`→`0x8c82`) or walls the service at `Unknown 0x8c32`.
- It is NOT a genuine external wait (cause #3 falsified): under a coherent
  publisher/BASE-service control the path progresses (`[0x2c]=1`, returns 1,
  advances to `0x7fe7`). The firmware waits on nothing external.

So the collision is real *given the current cone reconstruction*, and unfixable
*locally*. The finding explicitly redirects upstream: "the recoverable error, if
any, is upstream of `0x8cae` in the cone reconstruction."

## The north star (Maya's framing — keep this honest)

The firmware demonstrably goes alive on real silicon, so **there is a solution** —
the code genuinely wants *some* consistent view. If no static map exists, the
mechanism is *dynamic* (the same VMA holds different code at different times), and
we must identify it, not conclude "impossible." Do not stop at a bare negative.

## The question (one sentence)

Determine whether a mis-framed region UPSTREAM of `0x8cae` — in the publisher cone
(`0x55f8→0x50d4→0x8f44→0x9045→0x8c98→...→0x8cac`) or the service cone
(`[0x1187c]=0x8770→0xc530→0x7fc4→0x8c6c→...→0x8c8b Bbci→0x8cae`) — is what creates
the shared-byte contradiction at `0x8cae`: i.e. is there a `(region, δ)` correction
somewhere along either cone under which BOTH cones stay coherent to their landmarks
AND the instruction boundaries approaching `0x8cae` no longer collide — and if no
static correction exists anywhere, characterize precisely what dynamic view-switch
the silicon must be using.

## Why upstream can matter (the mechanism to exploit)

"Cone root is pinned" (absolute pointer / live callback — you established both) does
NOT mean every instruction boundary from root to `0x8cae` is correct. With
variable-length Xtensa instructions, a wrong file-offset on an *intermediate* region
produces a coherent-but-shifted boundary stream, which changes:
- WHERE the publisher's instruction covering `0x8cae` starts (maybe `0x8cac` isn't
  really the `Bgeu`/`Bgeu`-like instr whose 3rd byte lands on `0x8cae`), and/or
- WHETHER the service's `Bbci @0x8c8b → 0x8cae` edge and the `Addi @0x8cae` start are
  the true boundaries (maybe the service never actually starts an instruction at
  `0x8cae` under correct upstream framing).

If either cone's *approach* to `0x8cae` is mis-framed, the "both need different bytes
at `0x8cae`" premise can dissolve without any local change.

## Search / diagnosis (you have latitude — experiment)

Primary:
- Enumerate every region each cone traverses from root to `0x8cae`, with its current
  `load_m2c` file-offset. Publisher intermediates to scrutinize: the functions at
  `0x8f44`, `0x9045`, and `0x8c98`. Service intermediates: `0xc530`, `0x7fc4`, and
  `0x8c6c`'s body up to the `0x8c8b` branch.
- For each, test alternate offsets/splits (extend the generalized search to these
  UPSTREAM regions, not just `[0x8c98,0x8d52)`), keeping the landmark predicates:
  publisher builds `_NPU` (`local[0x14820]==0x55504e5f`) + reaches `0x5645`; service
  reaches the device-SRAM publish (`m2c_probe_alive_device_sram_struct`). A correction
  is admissible only if it keeps BOTH cones coherent to those landmarks.
- Pay special attention to whether a corrected upstream framing changes the
  instruction boundary that lands on `0x8cae` on either side.

Fallback (if no static correction exists ANYWHERE — deliver this, do not stop at a
negative): characterize the dynamic mechanism. The two views are then genuinely
different code at the same identity-mapped PA at different times. Software-visible
mechanisms already eliminated: ITLB remap (identity), CPU self-modifying store to
`0x8cae` (store audit = zero across full boot). The ONE dynamic mechanism NOT yet
excluded: a bulk code-page reload that rewrites the `0x8cxx` region between the
publisher phase and the service phase via something other than a tracked CPU store —
e.g. a firmware-programmed DMA/copy, or a context-switch code-overlay load from the
`0x08b00000` RAM image into the low VMA window. Check for it: does any DMA descriptor,
block-copy routine, or context-switch path target the low code window? If found,
that IS the mechanism (model it faithfully). If provably absent too, state precisely
what remains (HW fetch banking / mask-ROM) with the evidence that earns it.

## Code anchors

- `load_m2c` map + constants: `src/firmware/mod.rs:120-260, 380-512`.
- Generalized `(δ,split)` search harness: `m2c_probe_execution_guided_framing_search`
  (`src/firmware/boot_tests/coherence_mapper.rs:580`) — extend to upstream regions.
- Acceptance oracle: `m2c_probe_alive_device_sram_struct` (`coherence_mapper.rs:2261`).
- Store audit (proves no CPU store hits `0x8cae`): `m2c_probe_overlay_store_conflicts`
  (`coherence_mapper.rs:1083`) — note its `assert_eq!(pc,0x5645)` is stale; boot walls
  at `0x8cb1` now.
- Cone provenance already established: findings `2026-07-11-8c6c-service-path-is-real.md`
  (service root/chain) and `2026-07-11-alive-sram-overlay-collision.md` (publisher).
- Prior data on disk: `build/experiments/firmware-re/delta-split-*.tsv`,
  `delta-split-trace-100-0-8cae-5c.log`.

## Reproduce

```
XDNA_FW_PROBE=1 cargo test --lib m2c_probe_execution_guided_framing_search -- --nocapture
XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib m2c_probe_alive_device_sram_struct -- --nocapture
cargo test --lib     # full suite stays green (4088 baseline)
```

Firmware: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`
(SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`).

## Deliverables

1. **Verdict:** (a) an upstream `(region, δ/split)` correction that dissolves the
   `0x8cae` collision with both cones coherent to their landmarks — presented as a
   `load_m2c` diff, with `m2c_probe_alive_device_sram_struct` advancing toward the
   publish; OR (b) no static correction exists upstream either, plus the dynamic-
   mechanism characterization (DMA/context-switch code reload found → model it; or
   provably absent → the earned HW-banking/mask-ROM statement with evidence).
2. Which specific upstream region (if any) was mis-framed, and how correcting it
   moves the `0x8cae` boundary.

## Ground rules (fidelity)

- **Match real hardware; do not force/shim.** A fix is a `load_m2c` map correction or
  a faithful model of a real dynamic mechanism — never firmware-state injection, a
  per-path byte swap, or a hardcoded advance.
- Don't regress the `_NPU` local build or the publisher landmark. Full `cargo test
  --lib` stays green.
- Do NOT reopen (closed): PSP-loader RE, CPU self-modifying-code to `0x8cae`. (A
  DMA/bulk code reload is a DIFFERENT, un-excluded mechanism — that IS in scope.)
- Honest-negative with the dynamic-mechanism characterization is the required
  deliverable if no static fix exists — a bare "impossible" is not acceptable, per
  the north star: it wants something; say what.
- Present diffs for review — do NOT commit.
