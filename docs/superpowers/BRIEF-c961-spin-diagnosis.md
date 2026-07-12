# Brief: why does the service path spin in the 0xc961 loop (never publishing) once the 0x8cae collision is crossed?

## Where we are (your own prior result)

Your (δ,split) search proved the 0x8cae "collision" is dissolvable locally. The
framing `delta_lo=+0x100, delta_hi=0, split=0x8cae, literal=+0x5c` over
`[0x8c98,0x8d52)` gives **`publisher_pass=true, service_entered=true`** — the
publisher still builds `_NPU` AND the service path now executes *past* `0x8cae`
for the first time in this arc. Real progress.

But under that framing the service does NOT publish. It stops
`stop_kind=budget, stop_pc=0xc969` — it runs the full 1,000,000 instructions
**spinning in the zero-overhead loop at `0xc961`** (body `0xc964..0xc972`) and
never writes the device-SRAM descriptor (`0x030bb000`) or `FW_ALIVE_OFF`
(`0x030bf000`). All 14 publisher-valid candidates converge to this same spin
(4 stop at `0xc969`, 10 at `0xc96f`). Across the full 2,226-map sweep:
1,488 `state-cycle`, 710 `unknown` (both = wrong framings, red herrings),
28 `budget` (the ones that cross the collision and hit this loop).

**So the blocker is NOT an unimplemented opcode.** The loop body decodes
cleanly in fully-implemented instructions:

```text
0xc961 Loop  {s:5, end:0xc972}     # LCOUNT = a5; LBEG=0xc964, LEND=0xc972
0xc964 Extui {r:5, t:3, mask:1}    # a5 = a3 & 1
0xc967 AddN  {r:4, s:5, t:4}       # a4 = a5 + a4   (running popcount)
0xc969 Extui {r:5, t:4, mask:8}    # a5 = a4 & 0xff
0xc96c Bgeui {s:5, imm:2, ->0xc979}# exit if a5 >= 2
0xc96f Srli  {r:3, t:3, imm:1}     # a3 >>= 1
```

It exits when 2+ set bits have been counted in `a3`, OR when the `Loop` LCOUNT
expires. 1M instructions / ~6 per iter ≈ 166k iterations — far more than a
32-bit popcount (≤32) — so either LCOUNT is enormous, or an OUTER loop re-enters
this one many times, or the loop never terminates.

## The question (one sentence)

Determine which of three causes makes the `0xc961` loop spin to budget instead
of terminating and letting the service reach the device-SRAM publish:

1. **Xtensa `LOOP`/`LCOUNT` interpreter bug** — our zero-overhead-loop semantics
   over-iterate (e.g. LCOUNT not decremented, LBEG/LEND off-by-one, mishandling
   of branching OUT of the loop body via the `Bgeui` at `0xc96c` while LCOUNT/
   LBEG/LEND state persists, or re-arming on a later backward branch to LBEG).
   High value: a LOOP bug could affect many firmware paths, not just this one.
2. **Wrong upstream data** — `a3` (the scanned value) or `a5` (the LCOUNT seed)
   is garbage because the dissolving framing is *locally* valid but *globally*
   subtly wrong; the spin is then the SYMPTOM that this isn't the true map. If
   so, identify which region/framing feeds the bad value.
3. **Genuine firmware wait-loop** — the loop legitimately re-checks an internal
   condition (a status word, a queue/bitmap the emulator populates wrong or not
   at all) that should become true and doesn't. A state analog of the old
   go-alive wait. If so, name the exact address/value it awaits.

## You have latitude — experiment

This is a diagnosis, not a script. Feel around. Some angles, not exhaustive and
not mandatory — follow what the evidence opens up:

- Instrument register values (`a3`, `a4`, `a5`) and the loop registers
  (LCOUNT/LBEG/LEND) at first entry to `0xc961` and across the first several
  iterations. Is LCOUNT sane? Does `a3` look like real data or garbage?
- Audit our `LOOP`/`Loopnez` EXECUTION semantics (`src/firmware/xtensa/interp/`)
  against the Xtensa ISA: LCOUNT decrement timing, the LEND-triggered loopback,
  and — critically — what happens when control leaves the loop body early via a
  taken branch (`Bgeui @0xc96c`). Compare a suspect decode/exec against an
  independent Xtensa reference if useful.
- Trace UPSTREAM: what code computes `a3` and sets the `Loop` count register
  before `0xc961`? Is that code correctly framed? Walk back to where `a3` is
  loaded — if it comes from a memory structure (a channel bitmap, a queue,
  a descriptor), is that structure populated the way real firmware expects?
- Try ALTERNATE framings for the `0xc9xx` region and the upstream cone (not just
  the `[0x8c98,0x8d52)` region you already swept) — if a different upstream
  framing feeds `a3` a value that terminates the loop AND still reaches publish,
  that's the real map.
- Consider whether the loop is genuinely waiting (hypothesis 3): does `a3` (or
  whatever it scans) come from a device/mailbox/SRAM read that the emulator
  returns constant for? The driver trace proved go-alive is autonomous, so any
  real wait must be on INTERNAL state that should self-satisfy — trace why it
  doesn't.

If one angle dead-ends, try another. A precise "it's cause #N because X, and
here's the instruction/address where it goes wrong" is the goal — including an
honest "the framing is globally wrong (cause #2), here's the evidence" if that's
where it lands.

## Code anchors

- Xtensa LOOP: decode `src/firmware/xtensa/decode/control.rs` (LEND = `pc+4+imm8`
  asymmetry is documented there ~`:140,:264`); EXEC in
  `src/firmware/xtensa/interp/` (`control.rs`, `mod.rs`, `fastpath.rs`).
- Generalized (δ,split) probe / harness: `m2c_probe_execution_guided_framing_search`
  (`src/firmware/boot_tests/coherence_mapper.rs:580`) — this is where the
  dissolving framing is configured; extend/instrument here.
- Acceptance oracle: `m2c_probe_alive_device_sram_struct`
  (`coherence_mapper.rs:2261`) — struct at `0x030bb000` + pointer at `0x030bf000`.
- `load_m2c` map + constants: `src/firmware/mod.rs:120-260, 380-512`.
- Your data on disk (reuse, don't regenerate blindly):
  `build/experiments/firmware-re/delta-split-search-publisher-valid-1m.tsv`
  (the 14 budget-stoppers with boundaries + tail instruction streams),
  `delta-split-trace-100-0-8cae-5c.log` (full trace of the dissolving framing).

## Reproduce

```
XDNA_FW_PROBE=1 cargo test --lib m2c_probe_execution_guided_framing_search -- --nocapture
XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib m2c_probe_alive_device_sram_struct -- --nocapture
cargo test --lib     # full suite must stay green (4085 passed baseline)
```

Firmware: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`
(SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`).

## Deliverables

1. **Verdict:** cause #1 (LOOP/LCOUNT bug), #2 (wrong upstream data → framing
   globally wrong), or #3 (genuine internal wait) — with the exact
   instruction/address/register/value evidence that pins it.
2. **If #1 (interpreter bug):** the faithful fix as a diff, with the boot now
   advancing past the loop toward the publish (`m2c_probe_alive_device_sram_struct`
   showing a store reaching `0x030bb000`/`0x030bf000`, or at minimum getting
   materially further). Present for review — do NOT commit.
3. **If #2 (framing globally wrong):** which region/upstream framing feeds the
   bad `a3`/count, and the corrected framing if you find it — else a precise
   statement of where the true map diverges from the local dissolving one.
4. **If #3 (genuine wait):** the exact internal condition (address + expected
   value) the loop awaits, and why the emulator doesn't satisfy it.

## Ground rules (fidelity)

- **Match real hardware; do not force/shim.** A fix is either a faithful
  interpreter-semantics correction (LOOP/LCOUNT) or a `load_m2c` map correction
  (δ/split) — never a firmware-state injection, register poke, or hardcoded
  advance. Prior arc lesson: forcing scheduler/memory state CORRUPTS the boot.
- Don't regress the `_NPU` local build (`local[0x14820]==0x55504e5f`) or the
  publisher landmark. Full `cargo test --lib` stays green.
- Do NOT reopen (both closed): PSP-loader RE (mask ROM, unobtainable) and the
  self-modifying-code hypothesis (proven negative — byte at `0x8cae` is static).
- Honest-negative is valuable. "The dissolving framing is globally wrong and
  here's the proof" is a real result, not a failure.
