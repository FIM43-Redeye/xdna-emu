# Held-level trace encoding: the skip-token model (design, supersedes snapshot)

**Date:** 2026-06-08
**Status:** Design. Supersedes the *mechanism* of
[2026-06-07 held-level design](2026-06-07-lock-stall-level-emission-design.md)
sections 1 and the trace-unit emission parts of the
[plan](../plans/2026-06-07-trace-level-event-emission.md). The interpreter-side
edge detection and the archspec level classifier are unchanged and reused.
**Why:** The committed M1 held-level mechanism emits a *snapshot* mask aimed at our
own `tools/trace_decoder/decode.py`, but the regression comparison decodes with
**upstream `aie.utils.trace.parse_trace`**, whose held-level model is different.
Under upstream, our snapshot emission renders a genuine multi-thousand-cycle hold as
a string of `dur=1` spans. Full evidence:
[2026-06-08 falling-edge finding](../findings/2026-06-08-lock-stall-falling-edge-depends-on-concurrent-levels.md)
(note: that finding's *mechanism* analysis was done under our decoder's semantics and
is corrected here -- the real root cause is snapshot-vs-skip-token, not the
`active==0` early-return).

## The decoder we must match

`test.py` produces both `trace_hw.json` and `trace_emu.json` via upstream
`parse_trace` (`test.py:40,141`). Its mode-0 walk (`parse.py`
`convert_commands_to_json` / `deactivate_events` / `activate_event`):

- Each command does `timer += 1`, then (Single/Multiple) `timer += cycles`.
- **A frame with `cycles > 0` deactivates ALL currently-active events** (emits `E`
  at the pre-`cycles` timer), then activates the events named in that frame.
- A frame with `cycles == 0` deactivates only active events *not* named in the
  frame -- so events named in a `cycles==0` frame **persist**.
- `Repeat0(n)` / `Repeat1(n)` after a frame whose `cycles == 0` just do
  `timer += n` (linear extension, **no deactivation**); after `cycles > 0` they
  replay the deactivate/activate loop.

Consequence: a held level is encoded as a `cycles==0` frame **plus Repeat tokens**,
never as a mask carried in subsequent `cycles>0` frames.

## HW reference encoding (ground truth, distribute_lateral core tile)

Decoding HW's raw `trace_hw.txt` to commands, the first long LOCK_STALL (dur 6354):

```
[1] Single0  [7]  cycles=0          rising edge: activate LOCK_STALL at timer=1  -> B=1
[2..8] Repeat1 1023,1023,1023,1023,1023,1023,215   sum=6353 -> timer=6354
[9] Single0  [5]  cycles=5          timer+=1 -> 6355 deactivates LOCK_STALL -> E=6355 (dur 6354)
                                    then timer+=5 -> 6360 activates INSTR_LOCK_ACQUIRE_REQ
```

Concurrent levels join the same way (second stall region):

```
[13] Single0   [7]    cycles=0      LOCK_STALL rising
[14] Repeat1   896                  hold
[15] Multiple0 [3,7]  cycles=0      PORT_RUNNING joins; LOCK_STALL persists (cycles=0)
[16] Repeat1   1023                 hold {3,7}
```

## Mode-0 byte opcodes (upstream `utils.py::convert_to_commands`)

Our trace unit already emits Single0/1/2, Multiple0/1/2, Start, Pad correctly. The
two **missing** mode-0 emitters:

- `Repeat0`: `0b1110_RRRR`            (repeats 0..15)
- `Repeat1`: `0b110110_RR RRRRRRRR`   (repeats 0..1023)

(Same bit patterns as the existing mode-2 `encode_repeat0/1`, but emitted
byte-aligned into `byte_buffer` rather than through the mode-2 bit accumulator.)

## Encoder algorithm

Track `last_active` (event set as of the last emitted frame) and `last_emit_cycle`.
On any change at cycle C (a pulse fires, or a level edges) to `new_active`:

1. `gap = C - last_emit_cycle`.
2. **If `last_active` is non-empty** (levels were held across the gap): the gap must
   be carried by Repeat tokens so survivors are not deactivated. Emit Repeat1(1023)
   chunks + a final Repeat for `gap - 1`, then emit the change-frame
   (Single/Multiple of `new_active`) with **`cycles = 0`**.
3. **If `last_active` is empty** (nothing held): carry the gap in the change-frame's
   `cycles` field (current behavior) -- Single/Multiple with `cycles = gap - 1`.
4. A pure pulse is `last_active`-empty before and (after the pulse cycle) returns to
   whatever levels remain; it closes naturally at the next frame, giving `dur~1` as
   today.
5. Falling edge of a level: no dedicated frame. It is carried by the next change
   frame whose `new_active` drops the bit. When the *last* held level deasserts and
   `new_active` becomes empty with no coincident pulse, the close defers to the next
   frame / segment end -- which is exactly what HW does (no empty-frame encoding
   exists).

Open detail to settle empirically during TDD: the exact `-1` offsets and the
"level asserts after an empty gap > 0" case (rising frame needs `cycles==0` for the
following Repeat; may require a gap-carrying frame then a `cycles=0` hold-prep frame,
mirroring HW `[12]->[13]`). Round-trip tests pin these down.

## Test strategy (upstream is the oracle)

1. **Unit (fast, `cargo test --lib`):** drive a `TraceUnit` through assert / hold /
   deassert and assert the **byte stream shape** matches HW's skip-token pattern
   (`Single0 cycles=0` + `Repeat1` run + closing frame). These are faithful because
   they match HW's actual bytes (above), not a reimplemented decoder.
2. **Round-trip (oracle):** a Python helper under `tools/` feeds emitted bytes
   through upstream `parse_trace` and asserts the rendered spans (B/E, duration).
   Used during TDD and kept as a regression check.
3. **Integration gate (the real one):** re-run the distribute_lateral trace
   comparison (already upstream-decoded) and confirm the LOCK_STALL long spans
   surface with HW-matching structure. Residual duration/placement is the documented
   DMA-fill-timing axis.

## Impact on the milestone plan

- M1 (LOCK_STALL) is re-opened: the *interpreter* edge plumbing stays; the
  *trace-unit byte encoding* is replaced (snapshot -> skip-token). The M1 unit test
  (`held_level_emits_one_span_not_per_cycle`) currently asserts byte counts against
  our snapshot decoder and must be rebased onto the byte-shape / round-trip checks.
- M2-M4 (more levels) ride the same skip-token encoder once it is correct; they
  become "classify + drive edges," with no further encoding work.
- The snapshot path in our `decode.py` is now a *second* decoder that disagrees with
  upstream on held levels. Either align it to the skip-token model or mark it
  non-authoritative; upstream `parse_trace` is the oracle. (Maya flagged we may
  upstream-PR subtle decoder fixes later -- out of scope here.)
