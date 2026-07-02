# SP-5c Phase 2: `d_h` shim-row characterization (block-mask replication) -- execution plan

> **SUPERSEDED (2026-07-02) -- the shim-row premise this plan rests on was refuted on
> silicon.** The block-shim-row `d_h` capture confined all floods to their source column
> (the AIE-ML shim row does not forward tile-to-tile E/W broadcast; the reset uses a
> fabric detour instead). `d_h` is taken from the free-flood fabric measurement. This
> plan is kept for its Sec. 0 block-mask register derivation (still valid, toolchain-
> cited) and as the record of the approach. See
> `findings/2026-07-02-sp5c-phase2-shim-row-topology.md` and the design rev5 banner.

> **Status: DRAFT (2026-07-02).** First HW-phase plan of SP-5c, drawn after Phase
> 0/1 (pure-SW slice, merged `63fc8dd0`) landed the per-capture identifiability
> proofs, per the design-doc deferral (`specs/2026-07-01-sp5c-skew-characterization-design.md`
> rev4 Sec.5 verdict). Governing design: that rev4 doc (Sec.1 pt2, Sec.2, Sec.5
> Phase 2, Sec.6 gates 2/5). Mechanism decision (architect, 2026-07-02):
> **Option A -- block-mask replication** (keep measured tiles as AIE cores; shape the
> flood with `Event_Broadcast_Block` masks), NOT shim-row tile placement.
>
> **Purpose.** Measure `d_h` on the *shim-row* path the real timer reset uses
> (IF-1), scoped **`d_h`-only** (the block-routed capture cannot identify `d_v` --
> Sec.1 trap, proven in Phase 0). Nothing here flips `calibrated`; the flip is
> Phase 6.

## 0. Derived foundation (from aie-rt `xlnx_rel_v2025.2`, verified)

Source-derivation record: research pass 2026-07-02 against `aie-rt/driver/src/`,
cross-checked vs `mlir-aie/.../aie_registers_aie2.json` (AM025) and the emulator's
own `src/device/events/broadcast.rs`. Every value below is toolchain-cited.

**Block-mask register config** to replicate `_XAie_SetupBroadcastConfig`
(`xaie_timer.c:472-542`, AIE2/`!AIE1` branch):

| Tile type (rows) | Register (local offset) | Value | Meaning |
|---|---|---|---|
| AIE compute (rows 2-5) | mem-mod `Event_Broadcast_Block_East_Set` `0x14080` | `0xC000` | block E on the mem module (real external E port) |
| AIE compute (rows 2-5) | core-mod `Event_Broadcast_Block_West_Set` `0x34060` | `0xC000` | block W on the core module (real external W port) |
| Mem-tile (row 1) | switch-A `Event_Broadcast_A_Block_East_Set` `0x94080` | `0xC000` | block E |
| Mem-tile (row 1) | switch-A `Event_Broadcast_A_Block_West_Set` `0x94060` | `0xC000` | block W |
| Shim (row 0) | -- none -- | -- | left at reset default (unblocked) |

- **`0xC000` = bits 14|15**: blocks BOTH ch14 (s2) and ch15 (s1). The Set register
  is sticky-OR (bit N = channel N, `[15:0]`), so the combined write is safe.
- **Switch A only** (`_XAie_SetupBroadcastConfig` hardcodes `XAIE_EVENT_SWITCH_A`;
  core/mem modules have no switch B). Residual risk: switch-A/B independence is
  undocumented -- we follow the silicon-validated reset precedent. Disclosed.
- **The CORE-WEST / MEM-EAST asymmetry (AM020 ch2:349-363):** the core module has
  only N/S/W *external* ports (its "east" is an internal, unblockable wire to the
  co-located mem module's "west"); the mem module has only N/S/E external ports.
  Blocking mem-E + core-W severs the two real external horizontal ports; the
  internal core<->mem wire stays open (which is what lets one arrival arm both
  modules). Do NOT block mem-W or core-E -- those are no-ops (internal wire).
- **The two-channel trap (`xaie_timer.c:823-904`):** vanilla `XAie_SyncTimer`
  blocks only the reset `BcastId` (one channel); the `+1` channel is left
  *unblocked* as a shim-row relay trigger. Our two-*independent*-source R3b sources
  ch15 and ch14 directly, so BOTH must carry the full shaped block -- hence
  `0xC000`, not `0x4000`. Copying the vanilla single-channel block would let the
  ch15 flood free-route and corrupt `d_h`.

**Which tiles get the block:** EVERY AIE tile and memtile the flood can traverse
horizontally -- for `npu1_3col` that is cols 0-2 x rows 1-5 (12 AIE + 3 memtile =
15 tiles). Uniform, no origin special-case (blocking gates egress, not local
injection; the source tile is blocked too, matching `_XAie_SetupBroadcastConfig`).

**Verified interval algebra.** Under this block config, for s1=shim(0,0)/ch15,
s2=core(2,5)/ch14, the per-tile interval reduces to
```
r_X = const + dn_h * d_h ,    dn_h = |s2.col - X.col| - |s1.col - X.col|
```
exactly `reset_routed_coeffs` (`r3b_observe.py:24-33`). The vertical terms cancel:
s2's climb-down (`s2.row * d_v`, constant across tiles) dies in reference
differencing; each tile's climb-up (`X.row * d_v`) is common to both floods and
cancels in the interval. `dn_v = s2.row - s1.row` is constant => `d_v`
unidentifiable here (rank-1), matching the Phase-0 guard
`test_block_routed_capture_is_rank_deficient_for_dh_dv`. **The capture isolates
`d_h`. This is by design, not a defect.**

## 1. Code deltas (all `calibrated=false`, nothing irreversible)

### 1a. New kernel `sp5_skew_r3b_pc_dh` (mlir-aie, branch `xdna-emu-cycle-budget`)
A **copy** of `sp5_skew_r3b_pc/aie.mlir` (do NOT mutate the validated free-flood
kernel -- it is the Phase-3 `d_v` instrument) with the block-mask writes from Sec.0
emitted on all 15 tiles **before** the two `Event_Generate` floods fire. Reuse the
counter-arming (`0x31500=0x797A`, `0x31520=0`) + OP_READ control-packet readback
verbatim -- that path is silicon-validated; only the flood *shape* changes.
- Runtime-seq order: block-masks -> counter config -> both `Event_Generate` ->
  readback (block-masks must precede the floods).
- `geometry.json` gains `"routing": "block_shim_row"` and keeps the horizontal
  spine `(0,3),(1,3),(2,3)` (`dn_h = [2,0,-2]`, the 3-col leverage the accessible
  array allows -- Seam 3 power ceiling, disclosed). Vertical-spine tiles are
  `dn_h=0` here (redundant with `(1,3)`); drop them or keep as ballast, they add no
  `d_h` leverage.
- **Open implementation detail:** confirm how `aiex.npu.write32` targets a specific
  tile (local offset -> full addr with col/row shift) from the existing kernel's
  write pattern; the derivation pinned only the local offsets.
- Build per the existing README recipe; emu smoke-run (no HW) must complete clean
  (`halt_reason=completed`, floods fire, 6 OP_READ responses, 24-byte readback).
  Counters read 0 on emu (expected -- emu doesn't drive the counter from
  broadcasts; that is the silicon proof).

### 1b. `observe_r3b` routing-aware (xdna-emu `tools/calibration/skew/r3b_observe.py`)
Branch on `geometry["routing"]`: `"free_flood"` (default/absent for back-compat with
the bring-up kernel) -> current Manhattan coeffs; `"block_shim_row"` ->
`reset_routed_coeffs`. **Fail loud** on any other value (no silent free-flood
fallback -- a mis-tagged block capture fit against Manhattan coeffs is exactly the
"clean-but-wrong" failure this whole design guards against). Frozen-fixture unit
test per routing.

### 1c. `d_h`-only extractor (xdna-emu `tools/calibration/skew/r3b_extract.py`)
Add `extract_r3b_dh(observations, reference=0)`: single-column design
`A=[[dn_h-ref.dn_h]]`, `b=[r-ref.r]`, `solve_design_matrix(A,b,min_rank=1)` ->
`{"d_h", "fit_residual"}`. (Plain `extract_r3b` fits `{d_h,d_v}` at `min_rank=2` and
CORRECTLY raises `RankDeficientError` on a block-routed capture -- keep that; the
`d_h` capture uses the new `_dh` path.) Unit test: green fit on a synthetic
block-routed spine; `RankDeficientError` if fed to the rank-2 path.

### 1d. Gate `r3b_pc_dh_gate.sh` + tally (xdna-emu `build/experiments/sp5-skew/`)
Fork `r3b_pc_gate.sh` structure. Checks: rc-0, zero TDR/IOMMU delta, 24-byte
readback, **`d_h`-only rank sufficiency** (rank-1 on the `dn_h` column), b-vector
range-0 across runs. No value asserted (numbers are interpreted in Sec.3). Serial
only, no `xrt-smi` inside.

## 2. HW capture protocol (Phoenix, when 1a-1d are green on emu)

Preflight: `xrt-smi validate` exit 0 (alone), dmesg TDR/IOMMU baseline, no
concurrent HW suite. Then:
1. **`d_h` block-routed capture**, `r3b_pc_dh_gate.sh 20` -- the primary number.
2. **Channel sweep (Seam 1, inter-channel only):** re-run with s2 on a different
   measurement channel to detect inter-*measurement*-channel differences. ch15
   hop-cost uniformity stays a **disclosed assumption** (structurally unmeasurable
   -- ch15 is always one leg of the blend; do NOT sell this sweep as retiring it).
3. **b-vector jitter/drift gate:** N>=20 runs **spaced across the drift timescale**
   (minutes, not back-to-back -- the existing gate's back-to-back N=20 covers only
   *jitter*, already green 2026-07-02). Spacing wrapper needed. RED (range >= `d_h`)
   blocks. Re-sampled again at the Phase-6 flip.

## 3. Phase-2 exit gates (design Sec.6)

- **Gate 2 (partial -- `d_h` leg):** `d_h` residual green on the accessible-cols
  geometry, else provenance downgraded to "assumed".
- **Gate 5 (partial):** b-vector range strictly `<` measured `d_h` over N>=20
  *spaced* runs (RED blocks).
- Per-channel result recorded (ch15 disclosed-assumed).
- `d_h` provenance: **measured (shim-row path, IF-1)**, single-instrument, no
  absolute-frame cross-check (Seam 2 -- disclosure is the only mitigation).

`d_v`, the R1 two-sided spine, sign anchors, the held-out kernel, and the flip are
NOT in this plan (Phases 3/5/6). The `d_h` and `d_v` captures co-locate in one
Phoenix window (design Sec.5 dependency note); only the reconciliation is a true
dependency.

## 4. Subagent decomposition + model tiering

Subagent-driven, per the arc discipline (judgment on Opus/me, craft on Sonnet):

| Task | Model | Scope |
|---|---|---|
| A. Kernel delta (`sp5_skew_r3b_pc_dh`) | Sonnet | copy kernel, emit block-masks on 15 tiles, geometry `routing` flag, build xclbin, emu smoke-run. **mlir-aie is a separate repo -- named paths only, never `git add -A`.** |
| B. observe/extract/geometry | Sonnet | routing-aware `observe_r3b`, `extract_r3b_dh`, tests (both routings + rank-deficiency). |
| C. Gate + tally | Haiku | fork `r3b_pc_dh_gate.sh` + `d_h`-only tally. |
| Whole-delta review | Opus (me) | review A+B+C against this plan + the derivation before any HW. |
| HW capture | me | run Sec.2 protocol on Phoenix. |

`cargo test --lib` + `pytest tools/test_skew_*.py` green after B/C. FFI `.so` is
irrelevant to the gate (real-HW path, `env -u XDNA_EMU`).

## 5. Risks carried (design Sec.8)

- Seam 1 (ch14/15 blend, `d_h != d_h_ch15`) -- HIGH, un-retireable, disclosed.
- Seam 2 (`d_h` no absolute cross-check) -- HIGH structural, disclosure only.
- Seam 3 (per-module / non-hop-linear, 3-col leverage) -- power-limited, disclosed.
- Switch-A/B independence undocumented -- follow validated precedent, disclosed.
- Mis-tagged routing -> wrong coeffs -- retired by 1b fail-loud.
