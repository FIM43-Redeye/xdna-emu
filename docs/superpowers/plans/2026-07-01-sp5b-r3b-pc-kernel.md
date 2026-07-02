# SP-5b R3b-PC Kernel + Readback + Gate -- Implementation Plan

> **Supersedes the enriched-geometry plan** `2026-07-01-sp5b-r3b-pc-enriched.md`
> (Tasks 1-3 there are abandoned: the two-flood interval cannot identify
> within-axis direction anisotropy -- proven rank theorem, see
> `docs/superpowers/findings/2026-07-01-r3b-two-source-identifiability-limit.md`).
> This plan builds the R3b-PC instrument for the RESOLVED `{d_h, d_v}` model:
> horizontal direction isotropy is ASSUMED (structurally unmeasurable by any
> two-source interval), vertical anisotropy is reallocated to the two-sided R1
> spine. The observation bridge (`r3b_observe.py`) and extractor
> (`r3b_extract.py`) that consume this kernel's output are **already built and
> merged** (commit `77180706`); this plan does NOT rebuild them.

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

---

## Goal

Build the R3b-PC skew-measurement **apparatus** (SP-5b Phase 2): a hand-authored
MLIR kernel that runs two broadcast floods from two corner tiles, arms a hardware
Performance Counter on each measured tile (START = flood-1 broadcast event, STOP =
flood-2 broadcast event), reads every counter back off-chip into a host buffer,
and a gate that proves the whole path **runs and reproduces** on Phoenix. The
counter buffer feeds the already-merged `{dn_h, dn_v, r}` observation bridge.

SP-5b ships apparatus that **runs and reproduces (range-0)**. It produces **no
skew number and no evidence any number is correct** -- correctness is SP-5c's
separate human causal-vs-HW gate. Every gate check asserts shape,
non-degeneracy, and reproducibility; **never a value**.

## Architecture

The instrument is a post-run **local single-clock interval** read on each measured
tile: a Performance Counter counts from the flood-1 broadcast-event arrival to the
flood-2 broadcast-event arrival, yielding
`r_X = D(s2,X) - D(s1,X) + (T0_2 - T0_1)`. The two floods ride **distinct broadcast
channels** so they arrive as **distinct events**: flood-1 on broadcast channel 15
(received everywhere as `BROADCAST_15` = 122), flood-2 on channel 14 (`BROADCAST_14`
= 121). The counter arms START = 122, STOP = 121, so it distinguishes the two
arrivals with no timer involvement -- the perf counter (`0x31500`/`0x31520`) is a
**separate HW unit from the timer** (`0x34000`), so **no `Timer_Control.Reset_Event`
is written on measured tiles**.

`write32` is write-only, so the counters are read back via a **hand-assembled
control-packet OP_READ path** (NOT the `aiex.control_packet` op -- see Readback
Design below). The critical-path resolution: mirror the working `add_one_ctrl_packet`
/ `debug_halt_probe` template -- a shim-DMA MM2S BD pushes one OP_READ request
packet per measured tile into that tile's `TileControl` port; the tile's control
port returns one response word per read via `TileControl -> shim S2MM` into a
readback BO; `bridge-trace-runner` supplies the request packets via `--ctrlpkt`
and dumps the response BO via `--output`, in `counter_index` order as little-endian
u32 -- exactly the layout `observe_r3b` parses.

## Tech Stack

- Hand-authored MLIR (`aie.device(npu1_3col)`), compiled by `aiecc.py` via mlir-aie.
- Python 3.13 (readback request-packet generator + gate glue; `pytest`).
- Bash (HW gate), mirroring `build/experiments/sp5-skew/r1_gate.sh`.
- `bridge-trace-runner` (the C++ HW host; already supports `--ctrlpkt`/`--output`).
- Consumed unchanged: `tools/calibration/skew/{r3b_observe,r3b_extract,_solve}.py`.

## Global Constraints

- **Derive from the toolchain; hardcode nothing extractable.** Every register
  offset, event id, and target name is looked up / derived and cited at point of
  use (regdb `aie_registers_aie2.json`, events DB `events_database.json`, aie-rt
  `xaiemlgbl_params.h`, `AIETargetNPU.cpp`). Comment the *hardware fact*, not the
  tool internal.
- **SP-5b produces NO number and flips NO flag.** No value assertions anywhere.
- **Resolved model = `{d_h, d_v}` only.** No signed N/S+E/W solver columns, no
  per-direction geometry, no third source. `extract_r3b` fits `min_rank=2`. The
  identifiability guard (`test_skew_r3b_identifiability.py`) already locks this.
- **Verified register anchors (use verbatim, cite the source):**
  - target `npu1_3col` (NOT `npu1_4col`, which does not exist; bare `npu1` is
    4-col) -- `AIEAttrs.td:112-132`.
  - `Performance_Control0` @ `0x31500`: `Cnt0_Start_Event` bits 6:0,
    `Cnt0_Stop_Event` bits 14:8 (core has 4 counters; Cnt0/Cnt1 packed here,
    Cnt1 start 22:16 / stop 30:24) -- regdb 4944-4996, aie-rt
    `xaiemlgbl_params.h:2264` for `Performance_Counter0` = `0x31520`.
  - `Timer_Control` @ `0x34000` -- **NOT written on measured tiles**.
  - `Event_Generate` @ `0x34008` (core/mem/shim); `Event_Broadcast{N}` (core/mem)
    @ `0x34010 + 4*N`; `Event_Broadcast{N}_A` (shim) @ `0x34010 + 4*N`. So
    channel 15 = `0x3404C`, channel 14 = `0x34048` -- regdb 6072-6408.
  - Events (core/mem): `BROADCAST_0..15` = 107..122, `USER_EVENT_2` = 126; shim:
    `BROADCAST_A_0..15` = 110..125, `USER_EVENT_0` = 126 / `USER_EVENT_1` = 127
    (no shim `USER_EVENT_2`); `events_database.json`.
  - `write32` tile addressing: `(col<<25)|(row<<20)|(offset & 0xFFFFF)` for AIE2
    -- but MLIR uses the `{address = <20-bit offset>, column, row}` attribute form
    and the compiler packs it (`AIEXDialect.cpp:699-747`, `AIETargetModel.h:619`).
  - Control-packet header word[1]: `parity<<31 | stream_id<<24 | opcode<<22 |
    beats<<20 | (addr & 0xFFFFF)`, `opcode=1` = READ, `beats = words-1`
    (`AIETargetNPU.cpp:277-344`); routing word[0]: `parity<<31 | (pkt_type&0x7)<<12
    | (pkt_id&0xff)`.
  - **PARITY IS ODD (verified against source, corrected 2026-07-01).**
    `AIETargetNPU.cpp:309-336`: the `parity` lambda returns true iff the header's
    popcount is EVEN, and bit 31 is set to that value -- which forces the FULL
    32-bit word to have an ODD number of set bits. So bit 31 =
    `1 iff popcount(low31 header) is even`, equivalently the complement of the
    XOR of the low 31 bits. A correctly-encoded word satisfies
    `(popcount(word) % 2) == 1`. (The earlier draft of this plan said "even" --
    inverted; do not follow it.)
- **HW safety (Tasks with [HW-GATED], when they run on Phoenix):** `env -u
  XDNA_EMU`; never two HW suites concurrently; no `xrt-smi` during a HW run;
  `pkexec` not `sudo`; TDR recovery `pkexec sh -c 'modprobe -r amdxdna && modprobe
  amdxdna'`; reboot handed to the human; rebuild the FFI `.so` (`cargo build -p
  xdna-emu-ffi`) before any plugin/gate use -- never bare `cargo build`; never pipe
  build/test through `tail`/`grep` (redirect + Read).
- **mlir-aie is a SEPARATE sibling repo** (`/home/triple/npu-work/mlir-aie`, branch
  `xdna-emu-cycle-budget`, messy working tree). Kernel files are tracked THERE, not
  in xdna-emu. **Never `git add -A` in mlir-aie**; add named paths only.
- **Python import root:** tests import `from calibration.skew.<mod>`; run with
  `cd tools && python3 -m pytest`.
- **Commit trailer** ends with `Generated using Claude Code.` +
  `Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh`.
  No emoji. On internal-only work, do not pre-approve commit messages -- just commit.

---

## File Structure

| File | Repo | Responsibility | Task | Gate |
|---|---|---|---|---|
| `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/geometry.json` | mlir-aie | Two corner sources + measured tiles (>=3 collinear per axis, rank-2) with `counter_index` | 1 | SW-NOW |
| `xdna-emu/tools/test_skew_r3b_pc_geometry.py` | xdna-emu | Loads shipped geometry.json, asserts observe+extract is rank-2 non-degenerate | 1 | SW-NOW |
| `xdna-emu/tools/calibration/skew/r3b_ctrlpkt.py` | xdna-emu | Generate the OP_READ request-packet binary (`--ctrlpkt`) from geometry.json + regdb | 2 | SW-NOW |
| `xdna-emu/tools/test_skew_r3b_ctrlpkt.py` | xdna-emu | Frozen-fixture test of the request-packet byte layout + parity + address | 2 | SW-NOW |
| `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/aie.mlir` | mlir-aie | Hand-authored kernel: two floods, perf-counter config, readback plumbing | 3 | HW-GATED (compile+emu-smoke NOW) |
| `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/README.md` | mlir-aie | Build recipe + derived-fact provenance + readback mechanism | 3 | SW-NOW |
| `xdna-emu/build/experiments/sp5-skew/r3b_pc_gate.sh` | xdna-emu | N serial runs: rc-0, no TDR/IOMMU, rank-sufficient, non-inversion, range-0 | 4 | HW-GATED (Python glue dry-run NOW) |
| `xdna-emu/build/experiments/sp5-skew/r3b_pc_tally.py` | xdna-emu | observe->extract per run; range-0 b-vector + non-inversion; no value assertion | 4 | SW-NOW |

**Executability boundary.** Tasks 1-2 and the Python glue of Task 4 are fully
executable now (pure software/data). Task 3's kernel and Task 4's HW loop are
**authored now against the cited templates**; their *silicon* verification is
SP-5c. Task 3 gets a compile + emulator-smoke-run as its now-verification.

**Consumed unchanged (do NOT modify):** `tools/calibration/skew/r3b_observe.py`,
`r3b_extract.py`, `_solve.py`, `test_skew_r3b_{observe,extract,identifiability}.py`
(all merged, commit `77180706`).

---

## Readback Design (the critical path -- resolved, not a placeholder)

> **ERRATA (Task-3 execution, 2026-07-01): the 2-word "blockwrite BD + in-band
> routing header" mechanism below is WRONG and was corrected during
> implementation.** Proven on the emulator: the shim control-packet DMA applies
> its OWN stream header, so an in-band routing word (word[0]) arrives at the tile
> as payload and is mis-parsed as a control opcode -- every push routed to one
> tile and the kernel wedged. The correct mechanism (per
> `add_one_ctrl_packet_4_cores/aie.mlir`, and now implemented in the committed
> `sp5_skew_r3b_pc/aie.mlir` + README) routes per-tile via a
> `packet = <pkt_id, pkt_type=1>` attribute on a packet-mode `dma_memcpy_nd`, and
> the pushed data is a bare 1-word control header. The Task-2 `--ctrlpkt` binary
> is consumed unchanged (only its control word `2k+1` is pushed; its routing word
> `2k` is now dead -- optional Task-2 cleanup, not required). Two more Task-3
> findings: (1) readback is recovered via `--trace-out`, not `--output` (see Task
> 4); (2) packet IDs must be <=31 (5-bit field, `AIEDialect.cpp:2307`) and be
> route-search-derived (contiguous IDs false-match the pathfinder's merged-flow
> mask) -- the geometry's `pkt_in`/`pkt_out` were revised accordingly. The kernel
> README is the authoritative readback description; the steps below are retained
> for the request-header field layout only.

**What does NOT work:** the high-level `aiex.control_packet` op + its
`-aie-ctrl-packet-to-dma` pass is a **write-only config-delivery pipeline**. A
READ (`opcode=1`) lowered through it produces only the request push -- no
`packet_flow` back from `TileControl`, no `shim_dma_allocation(S2MM,...)`, no
receive arm. The read response is dropped. (`AIECtrlPacketToDma.cpp:63-218`
hardcodes `dir=1` MM2S and never inspects the opcode.) So we do **not** use that
op.

**What works (the resolved mechanism):** hand-assemble the OP_READ path with
low-level `aiex.npu.*` ops, exactly as the two working template kernels do:

- `mlir-aie/test/npu-xrt/add_one_ctrl_packet/aie.mlir` (canonical upstream; reads
  tile data memory back to a host BO).
- `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir` (local prior xdna-emu work;
  reads an actual core register, `Core_Status` @ `0x32004`, into a readback BO --
  same address class as `Performance_Counter0` @ `0x31520`).

**The wire mechanism, per measured tile k:**
1. Inbound route: `aie.packet_flow(IN_k) { source<shim, DMA:ch_in>; dest<tile_k,
   TileControl:0> }`.
2. Outbound route: `aie.packet_flow(OUT_k) { source<tile_k, TileControl:0>;
   dest<shim, DMA:ch_out> }`, with one `aie.shim_dma_allocation @perf_resp(shim,
   S2MM, ch_out)` receiving all N responses.
3. The request packet for tile k is 2 words: `word[0] = routing header
   (pkt_id=IN_k)`, `word[1] = control header (stream_id=OUT_k, opcode=1 READ,
   beats=0 -> 1 word, addr = 0x31520 & 0xFFFFF)`, each with even parity at bit 31
   per `AIETargetNPU.cpp:309-336`. The read TARGET address (`0x31520`) is a
   tile-local 20-bit offset; the DEST tile is selected purely by `pkt_id`/routing,
   so the address word is identical for every tile.
4. Runtime sequence (load-bearing order, from both templates + design Sec.5.1):
   configure all counters (write32 `0x31500` + zero `0x31520`) and map both floods
   (`Event_Broadcast{15,14}`) -> **arm the S2MM receive** (`dma_memcpy_nd(%readback,
   metadata=@perf_resp)`, sized N words) -> `Event_Generate(s1)` -> `Event_Generate(s2)`
   -> per tile: `blockwrite` BD template once, then N x (`address_patch` selecting
   the k-th request packet in the `--ctrlpkt` BO by `arg_plus=k*8`, doorbell
   `write32` to the MM2S task queue, `sync`) -> `dma_wait @perf_resp`.
   **Counter config MUST precede `Event_Generate(s1)`** or the START event is
   missed (counter reads 0/garbage). **The readback MUST follow both generates.**
5. Host binding: the `--ctrlpkt <bin>` file (generated by `r3b_ctrlpkt.py`, Task 2)
   fills the request BO; `--output <path>` dumps the response BO
   (`bridge-trace-runner.cpp:2267-2282`), auto-sized by
   `discover_arg_sizes_from_insts` (commit `65c3f852`) from the compiled shim BD
   length -- so the output-only BO is not under-allocated (the IOMMU-fault class
   the finding `2026-07-01-bridge-runner-output-bo-underallocation.md` fixed).
   Response words land in issue order = `counter_index` order, little-endian u32,
   which is exactly what `observe_r3b` reads at `counter_index*4`.

**Named fallbacks (build only if the primary is disproven on HW at SP-5c):**
- (a) If `bridge-trace-runner`'s `--ctrlpkt` arg classification mis-binds, embed
  the request packets as an MLIR `memref.global` constant (like
  `@blockwrite_data_0`) and DMA from it, removing the `--ctrlpkt` dependency.
- (b) If a control-packet OP_READ of a *perf-counter* register does not return the
  value on silicon (the templates read data-memory + `Core_Status`; `0x31520` is
  the same core-module class, high confidence but unverified for perf registers),
  fall back to a post-run core `LDA 0x31520 -> store -> DMA out` (heavier,
  core-program shape; design Sec.5.1). Record which path was used.

**Genuinely unresolved until HW (flag for SP-5c, do not block SP-5b):** whether a
control-packet OP_READ of `Performance_Counter0` returns the live counter value on
Phoenix silicon. It is the same register address class as `Core_Status`, which the
`debug_halt_probe` template reads successfully, so confidence is high -- but no test
in the tree reads a *perf-counter* register this way. The emulator smoke-run
(Task 3) validates the plumbing shape; the silicon read is the SP-5c gate's first
real proof, with fallback (b) ready.

---

## Task 1: Kernel geometry + rank-sufficiency guard  [SW-NOW]

**Files:**
- Create: `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/geometry.json` (mlir-aie repo)
- Create: `xdna-emu/tools/test_skew_r3b_pc_geometry.py`

**Design.** Two sources at opposite corners of an `npu1_3col` partition (virtual
frame: cols 0..2; rows 0 shim / 1 memtile / 2..5 core). `s1 = (0,0)` (shim corner,
flood on broadcast channel 15 -> `BROADCAST_15`=122), `s2 = (2,5)` (core corner,
flood on channel 14 -> `BROADCAST_14`=121). Measured tiles are all `core` (uniform
kind), giving >=3 collinear per axis:
- horizontal at row 3: `(0,3),(1,3),(2,3)` -> `dn_h` in {2, 0, -2}
- vertical at col 1: `(1,2),(1,3),(1,4),(1,5)` -> `dn_v` in {1, -1, -3, -5}

`(1,3)` is shared. `dn_h = |col-2| - |col-0|`, `dn_v = |row-5| - |row-0|`
(matching `r3b_observe._hops`). After referencing against tile 0, the design
matrix is rank 2 (identifiable) with >=3 collinear same-kind tiles per axis (so a
per-hop non-uniformity would raise `fit_residual`).

- [ ] **Step 1: Write `geometry.json`** (mlir-aie repo path):

```json
{
  "target": "npu1_3col",
  "comment": "R3b-PC skew instrument. s1 flood on broadcast ch15 (BROADCAST_15=122), s2 on ch14 (BROADCAST_14=121). Perf-counter arms START=122 STOP=121. Model {d_h,d_v} only; horizontal isotropy ASSUMED (see r3b_extract.py header + identifiability finding). Virtual frame: rows 0 shim / 1 memtile / 2-5 core.",
  "sources": {
    "s1": {"col": 0, "row": 0, "broadcast_channel": 15, "arrival_event": "BROADCAST_15", "generate_event": "USER_EVENT_0"},
    "s2": {"col": 2, "row": 5, "broadcast_channel": 14, "arrival_event": "BROADCAST_14", "generate_event": "USER_EVENT_2"}
  },
  "tiles": [
    {"col": 0, "row": 3, "kind": "core", "counter_index": 0, "role": "h_west",   "pkt_in": 16, "pkt_out": 32},
    {"col": 1, "row": 3, "kind": "core", "counter_index": 1, "role": "interior", "pkt_in": 17, "pkt_out": 33},
    {"col": 2, "row": 3, "kind": "core", "counter_index": 2, "role": "h_east",   "pkt_in": 18, "pkt_out": 34},
    {"col": 1, "row": 2, "kind": "core", "counter_index": 3, "role": "v_row2",   "pkt_in": 19, "pkt_out": 35},
    {"col": 1, "row": 4, "kind": "core", "counter_index": 4, "role": "v_row4",   "pkt_in": 20, "pkt_out": 36},
    {"col": 1, "row": 5, "kind": "core", "counter_index": 5, "role": "v_row5",   "pkt_in": 21, "pkt_out": 37}
  ]
}
```

> `pkt_in`/`pkt_out` are the per-tile control-packet routing ids (distinct, 8-bit)
> the kernel's `packet_flow`s and `r3b_ctrlpkt.py` must agree on. Values are
> placeholders the executor confirms against the compiler's packet-id allocator;
> keep them distinct and out of the flood/trace id range.

- [ ] **Step 2: Write the guard test** (`xdna-emu/tools/test_skew_r3b_pc_geometry.py`):

```python
"""R3b-PC shipped-geometry rank-sufficiency guard (#140 SP-5b).

Loads the kernel dir's geometry.json and asserts the {d_h, d_v} extractor is
rank-2 non-degenerate on it, with >=3 collinear same-kind tiles per axis. Guards
against a geometry edit silently making the instrument unidentifiable before it
ever reaches hardware."""
import json
import os
import struct
import pytest
from calibration.skew.r3b_observe import observe_r3b
from calibration.skew.r3b_extract import extract_r3b

GEOM_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "mlir-aie", "test", "npu-xrt", "sp5_skew_r3b_pc", "geometry.json"))


def _load():
    with open(GEOM_PATH) as f:
        return json.load(f)


def test_geometry_file_exists():
    assert os.path.exists(GEOM_PATH), GEOM_PATH


def test_target_is_npu1_3col():
    assert _load()["target"] == "npu1_3col"  # npu1_4col does not exist


def test_counter_indices_are_dense_and_unique():
    tiles = _load()["tiles"]
    idx = sorted(t["counter_index"] for t in tiles)
    assert idx == list(range(len(tiles))), idx


def test_extract_is_rank_two_on_synthetic_reading():
    geom = _load()
    n = max(t["counter_index"] for t in geom["tiles"]) + 1
    # Synthesize a readback from a known truth via the bridge's own coefficients.
    d_h, d_v, const = 2.0, 3.0, 500.0
    obs0 = observe_r3b(struct.pack("<%dI" % n, *([0] * n)), geom)
    vals = [0] * n
    for t, o in zip(geom["tiles"], obs0):
        vals[t["counter_index"]] = int(round(const + o["dn_h"] * d_h + o["dn_v"] * d_v))
    obs = observe_r3b(struct.pack("<%dI" % n, *vals), geom)
    r = extract_r3b(obs)  # min_rank=2; raises RankDeficientError if under-spanned
    assert r["fit_residual"] < 1e-6


def test_three_collinear_per_axis():
    tiles = _load()["tiles"]
    # >=3 tiles sharing a row (horizontal axis) and >=3 sharing a col (vertical).
    from collections import Counter
    rows = Counter(t["row"] for t in tiles)
    cols = Counter(t["col"] for t in tiles)
    assert max(rows.values()) >= 3, rows
    assert max(cols.values()) >= 3, cols
```

- [ ] **Step 3: Run** `cd tools && python3 -m pytest test_skew_r3b_pc_geometry.py -v`.
  Expected: 5 passed. If `RankDeficientError`, the geometry is under-spanned -- add
  a spanning tile, **do not weaken the solver**.

- [ ] **Step 4: Commit** (two repos -- named paths only in mlir-aie):

```bash
# xdna-emu
git add tools/test_skew_r3b_pc_geometry.py
git commit -m "test(#140): R3b-PC shipped-geometry rank-sufficiency guard"   # + trailer

# mlir-aie (sibling repo; NEVER git add -A here)
git -C ../mlir-aie add test/npu-xrt/sp5_skew_r3b_pc/geometry.json
git -C ../mlir-aie commit -m "feat(#140): R3b-PC kernel geometry (npu1_3col, {d_h,d_v})"  # + trailer
```

---

## Task 2: Readback request-packet generator  [SW-NOW]

**Files:**
- Create: `xdna-emu/tools/calibration/skew/r3b_ctrlpkt.py`
- Create: `xdna-emu/tools/test_skew_r3b_ctrlpkt.py`

**Interface.** `build_ctrlpkt(geometry, counter_offset=0x31520) -> bytes`: emit the
`--ctrlpkt` request binary -- 2 words per measured tile in `counter_index` order,
each `[routing_header(pkt_id=tile.pkt_in), control_header(stream_id=tile.pkt_out,
opcode=1 READ, beats=0, addr=counter_offset & 0xFFFFF)]`, little-endian u32, even
parity at bit 31. `counter_offset` defaults to `0x31520` but the docstring cites
the derivation (regdb `Performance_Counter0` / aie-rt `xaiemlgbl_params.h:2264`);
the executor MAY parse it from the regdb JSON rather than pass the literal.

The parity function must match `AIETargetNPU.cpp` (even parity over the low 31
bits). Encode both header words per the Global-Constraints layout.

- [ ] **Step 1: Write the failing test** (`xdna-emu/tools/test_skew_r3b_ctrlpkt.py`):

```python
"""R3b-PC control-packet request-binary tests (#140 SP-5b): byte layout, parity,
opcode, and target address of the OP_READ request stream, against a frozen fixture
and the header formula in AIETargetNPU.cpp:277-344."""
import struct
from calibration.skew.r3b_ctrlpkt import build_ctrlpkt, _parity, _read_header

GEOM = {
    "sources": {"s1": {"col": 0, "row": 0}, "s2": {"col": 2, "row": 5}},
    "tiles": [
        {"col": 0, "row": 3, "counter_index": 0, "pkt_in": 16, "pkt_out": 32},
        {"col": 1, "row": 3, "counter_index": 1, "pkt_in": 17, "pkt_out": 33},
    ],
}


def test_two_words_per_tile_in_counter_index_order():
    buf = build_ctrlpkt(GEOM, counter_offset=0x31520)
    assert len(buf) == len(GEOM["tiles"]) * 2 * 4  # 2 words/tile
    words = struct.unpack("<%dI" % (len(buf) // 4), buf)
    # tile 0 occupies words[0:2], tile 1 words[2:4] (counter_index order).
    assert (words[1] & 0xFFFFF) == 0x31520          # control header addr field
    assert (words[3] & 0xFFFFF) == 0x31520


def test_control_header_encodes_read_opcode_and_stream_id():
    buf = build_ctrlpkt(GEOM)
    words = struct.unpack("<%dI" % (len(buf) // 4), buf)
    ctrl0 = words[1]
    assert (ctrl0 >> 22) & 0x3 == 1          # opcode = READ
    assert (ctrl0 >> 20) & 0x3 == 0          # beats = 0 (single word read)
    assert (ctrl0 >> 24) & 0x7F == 32        # stream_id = pkt_out of tile 0


def test_routing_header_carries_pkt_in():
    buf = build_ctrlpkt(GEOM)
    words = struct.unpack("<%dI" % (len(buf) // 4), buf)
    assert (words[0] & 0xFF) == 16           # pkt_id = pkt_in of tile 0
    assert (words[2] & 0xFF) == 17


def test_odd_parity_bit_31_on_every_word():
    # AIETargetNPU.cpp:328/336 sets bit 31 so the FULL word has ODD popcount.
    buf = build_ctrlpkt(GEOM)
    for w in struct.unpack("<%dI" % (len(buf) // 4), buf):
        assert bin(w).count("1") % 2 == 1, hex(w)   # odd parity over the full 32 bits


def test_read_header_matches_formula():
    # _read_header(stream_id, opcode, beats, addr) == parity | fields.
    h = _read_header(stream_id=32, opcode=1, beats=0, addr=0x31520)
    assert (h >> 24) & 0x7F == 32 and (h >> 22) & 0x3 == 1 and (h & 0xFFFFF) == 0x31520
```

- [ ] **Step 2: Run** `cd tools && python3 -m pytest test_skew_r3b_ctrlpkt.py -v`.
  Expected: FAIL (`No module named ...r3b_ctrlpkt`).

- [ ] **Step 3: Write** `xdna-emu/tools/calibration/skew/r3b_ctrlpkt.py` implementing
  `_parity`, `_read_header`, `_routing_header`, and `build_ctrlpkt` per the header
  formula (`AIETargetNPU.cpp:277-344`). Header comment must cite: `0x31520` from
  regdb `Performance_Counter0` / aie-rt `xaiemlgbl_params.h:2264`; the 2-word
  packet form from `add_one_ctrl_packet/aie.mlir`; opcode=1=READ from
  `debug_halt_probe/test.cpp:157`. **Parity is ODD** (see Global Constraints):
  `_read_header`/`_routing_header` set bit 31 = `1 iff popcount(low31 header) is
  even` (the complement of the XOR of the low 31 bits), so the full word has an
  odd popcount -- matching `AIETargetNPU.cpp:328/336`. A helper `_parity(w)`
  returning `bin(w).count("1") % 2` should be `1` for every emitted word.

- [ ] **Step 4: Run** the test again. Expected: 5 passed.

- [ ] **Step 5: Commit** (xdna-emu):

```bash
git add tools/calibration/skew/r3b_ctrlpkt.py tools/test_skew_r3b_ctrlpkt.py
git commit -m "feat(#140): R3b-PC OP_READ request-packet generator"   # + trailer
```

---

## Task 3: Hand-authored MLIR kernel  [HW-GATED; compile + emu-smoke NOW]

> **Authoring-against-reference, not fill-in-code.** Register-level MLIR must be
> validated by the compiler, not pre-guessed. Every value, offset, event id, and
> ordering step is fixed below and in Global Constraints; the executor writes the
> MLIR against the two cited template kernels and verifies it **compiles +
> emulator-smoke-runs** now. Silicon is SP-5c.

**Files:**
- Create: `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/aie.mlir` (mlir-aie repo)
- Create: `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/README.md` (build recipe + provenance)

**Templates to copy from (read first):** `add_one_ctrl_packet/aie.mlir` (OP_READ
plumbing), `debug_halt_probe/aie.mlir` (reads a real register; distinct MM2S
channel; runtime-seq ordering comment). Flood register pattern:
`AIEInsertTraceFlows.cpp:676-763` (`Event_Broadcast{N}` map -> `Event_Generate`).

- [ ] **Step 1: Skeleton.** `aie.device(npu1_3col) { ... }`. Declare `aie.tile` for
  the shim source `(0,0)`, the core source `(2,5)`, all 6 measured cores, and the
  shim `(0,0)` used for readback DMA. Declare the readback `aie.shim_dma_allocation
  @perf_resp(%shim, S2MM, <ch_out>)` and, per measured tile, the inbound/outbound
  `aie.packet_flow(pkt_in)`/`(pkt_out)` from geometry.json (Readback Design step 1-2).
  Add the ctrl-in and readback BOs as `runtime_sequence` args (bind order matches
  `bridge-trace-runner`'s classifier -- verify against `debug_halt_probe`'s arg
  layout).

- [ ] **Step 2: Counter config (per measured tile, before any generate).** For each
  tile: `aiex.npu.write32 {address = 0x31520, column, row, value = 0}` (zero the
  counter) then `aiex.npu.write32 {address = 0x31500, column, row, value = 0x797A}`
  where `0x797A = 122 | (121 << 8)` = `Cnt0_Start_Event=BROADCAST_15(122)`,
  `Cnt0_Stop_Event=BROADCAST_14(121)`. Comment the field derivation (regdb
  `Performance_Control0` bits 6:0 / 14:8; events DB `BROADCAST_15`=122,
  `BROADCAST_14`=121). **No `Timer_Control` (0x34000) write on any measured tile.**

- [ ] **Step 3: Map both floods (config, before generate).**
  - s1 `(0,0)` shim: `write32 {address = 0x3404C, column=0, row=0, value = 126}`
    (`Event_Broadcast15_A` <- `USER_EVENT_0`=126).
  - s2 `(2,5)` core: `write32 {address = 0x34048, column=2, row=5, value = 126}`
    (`Event_Broadcast14` <- `USER_EVENT_2`=126).

- [ ] **Step 4: Arm the readback S2MM receive.** `aiex.npu.dma_memcpy_nd(%readback[...]
  [1,1,1,N][...]) {id=..., issue_token=true, metadata=@perf_resp}` sized to N=6
  response words. Must precede the first OP_READ push.

- [ ] **Step 5: The two floods (load-bearing order).**
  `write32 {address = 0x34008, column=0, row=0, value = 126}` (`Event_Generate`
  on s1 shim) then `write32 {address = 0x34008, column=2, row=5, value = 126}`
  (`Event_Generate` on s2 core). The program-order gap between these two is
  `T0_2 - T0_1` (cancels in `r_X - r_Y`).

- [ ] **Step 6: Readback pushes** (per measured tile, `counter_index` order).
  `blockwrite` the MM2S BD template once (`@blockwrite_data_0`, word[0]=2 for a
  2-word request packet, per `add_one_ctrl_packet`); then per tile: set the MM2S
  `CONTROLLER_ID`/BD packet id to route this push, `address_patch {addr=0x1d004,
  arg_idx=<ctrl-in>, arg_plus=k*8}` (select the k-th 2-word request packet),
  doorbell `write32` to the MM2S task queue (`0x1d214`), `sync`. Then
  `aiex.npu.dma_wait {symbol = @perf_resp}`. Comment: request address `0x31520` and
  routing come from the `--ctrlpkt` binary (Task 2), not the MLIR.

- [ ] **Step 7: Write `README.md`** with the exact build recipe (Step 8), the
  derived-fact provenance table (every offset/event id + its source), and the
  readback mechanism summary + the two fallbacks. Mirror `sp5_skew_r1/README.md`.

- [ ] **Step 8: Compile** (bare, not piped; from the kernel dir):

```bash
cd ../mlir-aie/test/npu-xrt/sp5_skew_r3b_pc
env -u XDNA_EMU aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts \
  --no-compile-host --alloc-scheme=basic-sequential \
  --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie.mlir
```
  Expected: `aie.xclbin` + `insts.bin`, no error. (`--alloc-scheme=basic-sequential`
  because we hand-build BDs, per `add_one_ctrl_packet/run.lit`.)

- [ ] **Step 9: Emulator smoke-run (structure only, no HW).** Rebuild the FFI:
  `cargo build -p xdna-emu-ffi` (NOT bare `cargo build`). Run the compiled kernel
  through the emulator with the Task-2 `--ctrlpkt` binary and confirm: no panic,
  the two `Event_Generate` writes + 6 counter configs appear, and the readback BO
  is non-empty and 24 bytes (6 x u32). Redirect output to a file and Read it; never
  pipe through `tail`/`grep`. This checks plumbing shape, NOT counter values.

- [ ] **Step 10: Commit** (mlir-aie sibling repo; named paths only):

```bash
git -C ../mlir-aie add test/npu-xrt/sp5_skew_r3b_pc/aie.mlir test/npu-xrt/sp5_skew_r3b_pc/README.md
git -C ../mlir-aie commit -m "feat(#140): R3b-PC kernel -- two floods + perf-counter + OP_READ readback"  # + trailer
```

---

## Task 4: HW runnability gate  [HW-GATED; Python glue dry-run NOW]

> **Authored now; runs green only on Phoenix (SP-5c).** Mirror
> `build/experiments/sp5-skew/r1_gate.sh` (the more defensive, proven-20/20
> template) and `r1_tally.py`.

**Files:**
- Create: `xdna-emu/build/experiments/sp5-skew/r3b_pc_gate.sh`
- Create: `xdna-emu/build/experiments/sp5-skew/r3b_pc_tally.py`

- [ ] **Step 1: Write `r3b_pc_gate.sh`** (bash, N=20 serial runs), mirroring
  `r1_gate.sh`:
  - `tdr_count()` / `iommu_fault_count()` verbatim from `r1_gate.sh:67-81` (grep
    `aie2_tdr_work` / `IO_PAGE_FAULT`); per-run before/after deltas; any non-zero
    delta => `clean=0`, run not trusted.
  - Preflight: fail loud if xclbin/insts/`--ctrlpkt` binary missing; clear stale
    `run_*` dirs.
  - Per run: generate the `--ctrlpkt` binary via `r3b_ctrlpkt.py`, then
    `env -u XDNA_EMU XDNA_EMU_RUNTIME=release "$RUNNER" --xclbin "$XCLBIN" --instr
    "$INSTS" --ctrlpkt "$CTRLPKT" --trace-out "$rd/trace.bin" --trace-size 256
    >"$rd/runner.log" 2>&1`; check `rc==0`; the 6 counter words are the **first
    24 bytes** of `trace.bin` (`head -c24`, or the tally slices `[:24]`).
    **READBACK SOURCE = `--trace-out`, NOT `--output`** (Task-3 finding): the
    kernel places `%readback` last so it lands in the runner's trace slot;
    `--output` writes only the XRT-declared 8-byte pointer size, so it cannot
    recover the 24-byte readback. Confirmed against r1's 8-byte `out.bin`.
  - Overall `clean` flag gates whether the tally runs (do not tally a dirty batch).
  - No `xrt-smi` anywhere. `dmesg` bare (no pkexec). Serial only.

- [ ] **Step 2: Write `r3b_pc_tally.py`** consuming the N `counters.bin` files +
  `geometry.json`:
  - Per run: `observe_r3b(counters, geometry)` -> `extract_r3b(obs)`. A
    `RankDeficientError` fails the gate loud.
  - **Non-inversion:** assert every counter is non-zero and not obviously garbage
    (a zero/garbage counter = `s1`-before-`s2` inversion or a missed START).
  - **Range-0 b-vector:** `ranges = [max(v)-min(v) for v in per_tile_across_runs]`;
    assert all `<= tol` (default `0`), mirroring `r1_tally.py`. This is the real
    correctness proxy (a systematically-contaminated-but-nonzero counter is often
    non-reproducible).
  - Print `{d_h, d_v, fit_residual}` for visibility. **No value assertions** --
    printing is not asserting; the numbers are SP-5c's.
  - *(Optional, report-only)* If the executor armed spare counters (core has 4):
    print the cross-column broadcast-arrival-jitter range (a counter armed
    START=`s1`/STOP=`s1`) and a channel-uniformity delta, per design Sec.5.4 rev3.
    These are diagnostics for the Phase-3 go/no-go, not gate-blocking, and require
    extra `Performance_Control0` Cnt1 / `Performance_Control1` writes in Task 3 --
    add only if cheap; do not block the core gate on them.

- [ ] **Step 3: Dry-run the Python glue offline (no HW).** Feed a synthetic
  `counters.bin` (built from a known `{d_h, d_v}` via `observe_r3b`'s own
  coefficients, as in Task 1's test) through `r3b_pc_tally.py` for 3 fake runs and
  confirm it parses, reports `{d_h, d_v, fit_residual}`, passes non-inversion +
  range-0, and raises loud on a rank-deficient / short buffer. Expected: prints the
  triple, no exception; a deliberately-corrupted short buffer raises `ValueError`.

- [ ] **Step 4: Commit** (`build/` is gitignored -> force-add, `r1_gate.sh`
  precedent):

```bash
git add -f build/experiments/sp5-skew/r3b_pc_gate.sh build/experiments/sp5-skew/r3b_pc_tally.py
git commit -m "feat(#140): R3b-PC HW gate -- rc/TDR/IOMMU, rank, non-inversion, range-0 (no value assertions)"  # + trailer
```

---

## Self-Review

**Spec coverage (design rev3 Sec.5 + errata + NEXT-STEPS resolution):**
- `{d_h, d_v}` model, horizontal isotropy assumed, no signed columns / third source
  -> Tasks 1, 4 (consume merged `extract_r3b`, `min_rank=2`). Matches errata +
  identifiability finding.
- Two floods on distinct channels -> distinct arrival events (122/121) -> Task 3
  Steps 2-3, 5. No `Timer_Control` on measured tiles -> Step 2.
- Perf-counter config START=s1/STOP=s2, config-before-generate ordering -> Step 2, 5.
- Readback via hand-assembled control-packet OP_READ (NOT `aiex.control_packet`) +
  `--ctrlpkt`/`--output` host binding -> Readback Design + Tasks 2, 3.
- Geometry spans both axes, >=3 collinear per axis, rank-2 -> Task 1.
- `npu1_3col` (not 4col); generic control_packet read (opcode 1, no named enum) ->
  Global Constraints + Tasks 1-3.
- Gate: rc-0, no TDR, no IOMMU, rank-sufficient, s1-before-s2 non-inversion,
  range-0 b-vector; **no value assertions** -> Task 4.
- Arrival-jitter + channel-uniformity pre-checks -> Task 4 Step 2 (optional,
  report-only, honestly scoped as non-blocking; they need spare counters).

**Deferred to SP-5c (correctly out of this plan):** the `calibrated` flip and its
pre-flip gates; R3b-`LDA_TM` Phase-3 go/no-go; the actual skew number; the silicon
proof that a control-packet OP_READ returns the live perf-counter value (with
fallback (b) ready).

**Placeholder scan:** Tasks 1-2 and Task 4's Python glue carry complete code/data.
Task 3 (register-level MLIR) is authoring-against-reference by necessity -- every
value, offset, event id, and ordering step is fixed, with a concrete compile +
emulator-smoke verification. The Readback Design section is the resolved mechanism,
not a placeholder: the working path is specified end-to-end (op sequence, header
formula, host binding), with two named fallbacks and one honestly-flagged
HW-gated unknown.

**SW-NOW vs HW-GATED:** Tasks 1, 2, and Task 4 Steps 2-3 (Python glue) run today
with zero hardware. Task 3 compiles + emu-smoke-runs today; its silicon proof and
Task 4's HW loop are SP-5c. `cargo test --lib` stays green throughout (no emulator
source changes -- the seam is untouched).
