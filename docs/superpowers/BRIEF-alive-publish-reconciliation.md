# Brief: does the firmware actually go alive, or is "alive published" over-claimed? -- reconcile two findings

## Why this brief exists

Two findings in this arc now frame the boot wall differently, and they must be
reconciled before either is banked:

- **Predecessor (committed `d5824e21`), `2026-07-12-a7-reject-provenance.md`:**
  the service sequencer `0x7fc4` rejects `a7=6` at `0x7fc7`
  (`Bgeui a7,6 -> 0x7fec`); `a7=6` is genuine scheduler state (the current
  task's slot ID). Headline: the firmware self-rejects; the missing state is an
  in-range service context. It framed the reject as the wall.
- **Successor (UNCOMMITTED on disk), `2026-07-12-slot6-selector-provenance.md`:**
  slot-6 selection is intended (clean mapped provenance, fork 2). It goes
  further and claims the go-alive worker **already publishes alive** on the
  clean pre-counterfactual path -- the run-function at `0x55f8` writes
  little-endian `0x030bb000` to firmware VA `0..3` at n=52119-52122 -- and that
  the `a7=6` reject is downstream counterfactual noise reached only after the
  probe's `HARNESS_VIEW` forcing, temporally after publication. Headline: there
  is no self-reject wall; the only gap is where VA `0..3` lands in the emulator.

These are not a flat contradiction: the successor is the answer to the
predecessor's own ranked next step ("trace the caller that supplies selector
6"), and it demotes the predecessor's "self-reject is the wall" headline. But
the successor's central interpretive claim -- **"alive already published"** --
is unverified, and its own independent-review pass never completed. This
session decides whether the successor supersedes the predecessor or over-claimed.

## Already verified -- do NOT redo this

The successor's empirical spine was independently re-run and confirmed this
session (log: `build/experiments/firmware-re/slot6-reverify.log`). All four hold
byte-for-byte; treat them as GIVEN and build on them, do not re-derive:

1. **Publisher stores:** n=52119-52122, `0x50c6..0x50cf`, four `S8i` writing
   bytes `00 b0 0b 03` to local EA `0,1,2,3` = little-endian `0x030bb000`,
   region `ordinary-local-data`.
2. **Single go-alive dispatch:** exactly one `Entry` to `0x55f8` at n=49925
   (`DISPATCH_SEQUENCE ... goalive_entries=[49925]`). Not the old ~131x loop.
3. **Dispatch sequence:** `selectors=[6, 0] service_slots=[0, 6]`.
4. **Queue-count gate `0x24c4`:** `0 -> 1` enqueue at n=47383; `=1` first
   scheduler pass n=47668; `1 -> 0` retirement n=49736; `=0` second pass
   n=53141. Single selection is genuinely queue-count-gated.

Also given: the reject is reached only after the probe's own view forcing --
n-ordering publish 52119 << first `HARNESS_VIEW` counterfactual 53640 << BASE
`0x26d4` view 53784 << reject 53873. The publisher path itself runs un-forced
(52119 precedes every counterfactual).

## The one derivable question that decides it

"Alive published" hinges on a single unproven link. The firmware writes the
*value* `0x030bb000` into its own **local** word `VA 0..3`. But `0x030bb000` is
the **device** address where the HW oracle (finding `1a7bda8d`) located the
alive struct -- a different memory space from management-core local RAM. And the
host does not read management local `0..3`; host visibility is the established
`FW_ALIVE_OFF` destination. So writing a pointer-value into local word 0 is not
self-evidently "publishing alive."

**Trace both halves of the consumption/publication link, derive-only:**

1. **Consumer of local VA `0..3`.** After n=52122, does any mapped-firmware
   instruction on the clean (un-forced) path *read back* local `0..3` and act on
   it -- e.g. use it as a base/pointer for a subsequent store, a DMA descriptor,
   or a device/BAR write that reaches host-visible space? Or is the four-byte
   write terminal on the clean path (nothing consumes it before the trace ends /
   before forcing)? Give the ordered consumer chain or prove its absence.
2. **Writer of the host-visible destination.** Where does the established
   host-visible alive destination (`FW_ALIVE_OFF`, and/or device
   `0x030bb000`/BAR-space, and/or the struct the HW oracle found) actually get
   written in the trace, on ANY path? Classify each such write by path:
   - **clean/un-forced** (n < 53640, before any `HARNESS_VIEW`), or
   - **post-forcing / on the `a7` service path** (n >= 53640, reachable only via
     the probe's counterfactual views, i.e. past the reject).

## The discriminator (state the verdict this way)

- **If** the host-visible destination is written on the clean un-forced path (or
  VA `0..3` provably maps to / feeds the host-visible slot through mapped
  firmware) **->** the successor is right: alive is genuinely achieved and the
  only residual is an emulator address-map gap. The predecessor's "self-reject
  wall" is superseded (reject is post-publication noise). Name the exact
  address-map fix locus (which VA -> which host aperture) as the next step.
- **If** the host-visible destination is written ONLY past the reject / only on
  the forced `a7` service path, and nothing on the clean path consumes local
  `0..3` into host-visible space **->** the successor over-claimed: the clean
  path builds a pointer but does NOT publish where the host reads, the real
  publish lives on the service path the reject blocks, and the predecessor's
  reject-as-wall framing stands (now with the added fact that a pointer is
  pre-staged in local `0..3`). Name what the service path would do with `a7<6`
  that local `0..3` alone does not.
- **If** genuinely indeterminate from the trace, say so and name the single
  observation (derive-only, no forcing) that would settle it.

## Deliverables

1. **Consumer chain for local VA `0..3`** after n=52122 on the clean path --
   ordered, each edge VERIFIED (executed) / CLAIMED (inferred), with addresses
   and memory classes -- or a proof of non-consumption.
2. **Host-visible-destination writer inventory** -- every write to
   `FW_ALIVE_OFF` / device `0x030bb000` / the host-read alive slot, each tagged
   clean-path vs post-forcing-path, with n and PC.
3. **The reconciliation verdict** -- successor supersedes predecessor, or
   successor over-claimed, decided by (1)+(2) above, with the specific evidence.
   Do not hedge; pick the branch the trace supports.
4. **Ranked single next step** -- derive-only. If supersede: the address-map fix
   locus. If over-claim: the derivable observation of what the blocked service
   path publishes.

## What "done" looks like

A written finding
(`docs/superpowers/findings/2026-07-13-alive-publish-reconciliation.md`) with the
four deliverables. Extend the same env-gated probe if you need new observations;
`cargo test --lib` green (4091 baseline). Present for review -- do NOT commit.

## Execution discipline (read this -- the last run hung here)

- The previous job in this arc soft-hung for 40+ minutes in a `wait` loop on a
  self-dispatched independent-review subagent, after all deliverables were
  already written. **Do NOT dispatch an independent read-only reviewer and do
  NOT enter a `wait`/collaboration poll loop.** Do your own inline verification,
  write the finding, and terminate. Your final message is the deliverable.
- If a probe run is long CPU-bound, background-and-block in ONE shell command
  (`cmd & wait $!`); do not poll in a `check -> sleep -> "still running"` loop.
- Reconstruct context FIRST by reading the two findings + the reverify log named
  above; the slot6 finding is uncommitted but present on disk.

## Ground rules

- Derive, do NOT calibrate. No fitted constants, no hardcoded value, no forced
  branch, no "make it pass" shim. The deliverable is the identified link (or its
  absence), not an injected result.
- Read-only. No forcing firmware state, no `load_m2c` diff, no per-path byte
  swap. Instrument and reason. (The existing `HARNESS_VIEW` transports are the
  probe's; do not add new forcing.)
- Test-only probe code; production `load_m2c`/`mod.rs`/`mmio.rs`/`system.rs`
  behavior unchanged. `cargo test --lib` stays green (4091 baseline).
- Do NOT re-open PSP-loader RE, the CPU-self-modifying-code-to-`0x8cae` path, or
  the below-CPU bank *mechanism* hunt (all closed).
- Ground everything in Phoenix `1502_00`
  (`../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`, SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`). The
  `17f1_10` sibling is untrusted different-generation -- no importing its
  semantics without a byte-level match against Phoenix.
- Present findings for review. Do NOT commit.

## Anchors

- Committed predecessor: `docs/superpowers/findings/2026-07-12-a7-reject-provenance.md`.
- Uncommitted successor (on disk):
  `docs/superpowers/findings/2026-07-12-slot6-selector-provenance.md`.
- Fresh reverify trace: `build/experiments/firmware-re/slot6-reverify.log`.
- HW oracle for the alive struct at device `0x030bb000`: finding `1a7bda8d`
  (`2026-07-11-alive-state-sram-hw-dump.md`).
- Probe family to extend: `m2c_probe_26d4_cache_pageroot_timeline` in
  `src/firmware/boot_tests/coherence_mapper.rs`.
- `FW_ALIVE_OFF` / host-visible alive destination: grep the emulator source for
  the established constant; do not invent one.
- Base commit: `d5824e21`.
