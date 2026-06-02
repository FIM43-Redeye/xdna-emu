# aiesim Part II-B: bridge data path + lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: drive INLINE on the dev box (these
> tasks need aietools + the cluster, cannot be sandbox-unit-tested). Each task
> ends with an explicit out-of-sandbox verification gate. Steps use checkbox
> (`- [ ]`) syntax.

**Goal:** Finish the in-process aiesim bridge: the `ess_*()` PS seam bound to the
cluster, CDO/NPU op-stream replay, the elaborate-once service-thread lifecycle,
and the tier-2/tier-3 bring-up — so `XDNA_BACKEND=aiesim` runs real xclbins
end-to-end.

**Architecture:** The II.1-II.3 spine already stands: `libxdna_aiesim_bridge.so`
constructs an E513-free cluster in-process (see the spine commits + the
feasibility findings doc). Part II-B adds the data path on top of that proven
cluster handle. Parser-driven direct replay: the Rust side serializes ops; the
bridge decodes them and drives the cluster through our own `ess_*()` PS bridge
(no embedded HAL).

**Tech stack:** C++17, aietools SystemC 2.3.1 (QuickThreads), xtlm AXI-MM TLM,
the closed `MathEngineBase` cluster interface.

---

## Scouting results (2026-06-01) — the grounding, read before executing

Live-source scout of the cluster interface + the PS reference. These supersede
the original Part II (II.4-II.8) guesses in
`2026-06-01-aiesim-cpp-bridge-backend.md`, which were written before the cluster
was proven and are **not** to be executed as-written.

**Sources (read-only references; we write original code):**
- `mlir-aie/aie_runtime_lib/AIE2/aiesim/genwrapper_for_ps.cpp` — `PSIP_ps_i3`,
  the exact PS-bridge template (the 6 `ess_*` + their TLM bodies).
- `<aietools>/.../aie_cluster_v1_0_0/include/math_engine_base.h` — the
  authoritative `MathEngineBase` interface (the factory return type).
- `<aietools>/.../aie_xtlm/.../aie_xtlm.cpp` — how the real flow binds the
  cluster's sockets (`create_cluster`, lines ~266-660).
- `aie-rt/driver/src/io_backend/ext/xaie_sim.c` — the HAL SIM backend (what
  `ess_*` the *HAL* consumes — relevant only to the excluded phase-2).

**Grounded facts (corrections to the old plan in bold):**

1. **The `ess_*` set we provide is the genwrapper 6:** `ess_Write32`,
   `ess_Read32`, `ess_Write128`, `ess_Read128`, `ess_WriteGM`, `ess_ReadGM`.
   **`ess_WriteCmd` is NOT ours** — only `xaie_sim.c` (the HAL's NPI command
   path) calls it; our design is parser-driven direct replay, so we never link
   the HAL. (The old plan's "II.4 Step 1: verify ess_WriteCmd" is resolved here:
   out of scope.) probe4 already confirmed the cluster needs none of these at
   load — they are outbound PS->cluster calls we define + invoke.

2. **`ess_*` signatures (from genwrapper):**
   - `void ess_Write32(uint64_t addr, uint32_t data)`
   - `uint32_t ess_Read32(uint64_t addr)`
   - `void ess_Write128(uint64_t addr, uint32_t* data)` (4 words)
   - `void ess_Read128(uint64_t addr, uint32_t* data)`
   - `void ess_WriteGM(uint64_t addr, const void* data, uint64_t size)`
   - `void ess_ReadGM(uint64_t addr, void* data, uint64_t size)`
   Each delegates to a process-singleton PS object; the bodies are TLM
   `aximm_payload` transactions (`set_command`/`set_address`/`b_transport`),
   `SC_ZERO_TIME` delay. GM is chunked into <=4096-byte AXI transactions.

3. **`MathEngineBase` socket accessors (the bind targets):**
   - host->cluster (config/MMIO, the `ess_Write32/Read32` path):
     `std::vector<xtlm::xtlm_aximm_target_socket*>& get_ss_aximm_rd()` /
     `get_ss_aximm_wr()`. `[0]` is the NPI/config port (per aie_xtlm). The PS
     bridge's **initiator** sockets bind to these **target** sockets.
   - cluster->host (DDR, the `ess_WriteGM/ReadGM` + shim-DMA path):
     `xtlm::xtlm_aximm_initiator_socket* shim_dma_rd_socket(unsigned col)` /
     `shim_dma_wr_socket(unsigned col)`. These are cluster **initiators**; the PS
     bridge provides a **target** DDR model they bind to.

4. **Factory signature (corrected):**
   `MathEngineBase* create_math_engine(sc_module_name nm, const char* file,
   bool is_pm_write, bool is_dm_write)`. The 3rd/4th args are **is_pm_write /
   is_dm_write** (program/data-memory backdoor-write enables), NOT
   "is_fast_pm/is_fast_dm" as the spike guessed. `aiesim_top.cpp` passes
   `false,false` (works for instantiation); II-B.3 revisits them for memory load.

5. **No `transport_dbg_cb` method** on the interface. Tier-2 backdoor reads
   (`aiesim_read_reg`) use **TLM-2.0 `transport_dbg`** on the aximm socket
   (zero sim-time), not a MathEngine callback. (Corrects the old plan's
   "transport_dbg_cb from the MathEngine vtable".)

6. **No `run`/`reset` methods.** RUN = drive the SystemC kernel (`sc_start()` to
   quiescence / budget). Cores are enabled by CDO register writes. Topology query
   = `get_num_cols()` (+ `get_num_noc_tiles()`, `get_num_aximm_interfaces()`,
   `get_num_streams_*`); **rows are fixed per-arch** (no `get_num_rows()`).
   `reset` between submissions = re-apply CDO (or destroy+recreate) — II-B.3.

7. **Open detail to resolve in II-B.1 (not a blocker):** genwrapper routes
   `writeGM` through the SAME PS aximm initiator as `write32` (cluster models its
   addressable memory behind `ss_aximm`), while `aie_xtlm` separately binds the
   cluster's `shim_dma_*` initiators to the NoC/DDR. Decide during II-B.1 whether
   our `ess_WriteGM` writes via the config aximm (genwrapper style) or a PS-side
   DDR target bound to `shim_dma_*` — resolve empirically against a real buffer
   round-trip (II-B.1 Step 5).

---

## File structure

```
aiesim-bridge/src/
  ps_bridge.{h,cpp}        II-B.1  ess_*() seam + sockets, bound in aiesim_top
  cdo_replay.{h,cpp}       II-B.2  decode wire format -> ess_*
  service_thread.{h,cpp}   II-B.3  elaborate-once lifecycle + command queue
  aiesim_top.cpp           modified: bind ps_bridge (II-B.1), topology (II-B.3)
  c_abi.cpp                modified: route entries through the queue (II-B.3)
tests/aiesim_bringup.rs    II-B.4  gated tier-2 integration (or scripts/ harness)
scripts/emu-bridge-test.sh II-B.5  optional aiesim third runtime
```

---

## Task II-B.1: `ps_bridge` — the `ess_*()` seam + socket bindings

> **STATUS: config path DONE (commit c822f11).** ps_bridge (the 6 ess_*, TLM
> bodies, backdoor R/W) is bound to the cluster's config ss_aximm[0]; an
> env-gated backdoor write+read sweep round-trips 6/6 AIE2 array addresses. The
> **cluster->host DDR path (shim_dma_*_socket, GM routing fact 7) is deferred to
> II-B.3** (it needs the functional-run lifecycle: sc_start + full socket
> stubbing). Step 5's GM round-trip moves there.

**Files:** Create `aiesim-bridge/src/ps_bridge.{h,cpp}`; modify `aiesim_top.cpp`,
`CMakeLists.txt`.

Model on `PSIP_ps_i3` (genwrapper). A process-singleton `sc_module` holding the
PS-side AXI-MM initiator socket(s) to the cluster's config `ss_aximm`, plus a DDR
target model for shim-DMA. Provide the 6 `ess_*()` free functions delegating to
the singleton.

- [ ] **Step 1: `ps_bridge` class** — initiator socket util pair
  (`xtlm_aximm_initiator_rd/wr_socket_util` + `xtlm_aximm_mem_manager`), exactly
  as `PSIP_ps_i3` sets up (genwrapper lines ~210-225). Methods `write32`,
  `read32`, `write128`, `read128`, `writeGM`, `readGM` reproduce the genwrapper
  TLM bodies (payload acquire/set_command/set_address/`b_transport`/release; GM
  chunked at 4096). Singleton accessor (`instance()`).

- [ ] **Step 2: the 6 `ess_*()` free functions** in `ps_bridge.cpp`, delegating
  to `instance()`, signatures per scouting fact 2. **Do not** define
  `ess_WriteCmd`.

- [ ] **Step 3: tier-2 backdoor read** — `ps_bridge::read32_backdoor(addr)` issues
  a TLM `transport_dbg` on the read socket (zero sim-time), distinct from the
  timed `read32`. (`aiesim_read_reg` wires to this in II-B.3.)

- [ ] **Step 4: bind in `aiesim_top`** — after `create_math_engine`, construct
  `ps_bridge` and bind its initiator to `me->get_ss_aximm_wr()[0]` /
  `get_ss_aximm_rd()[0]` (config/NPI), and a DDR target to `me->shim_dma_rd/wr_
  socket(col)` for each col. (Reference aie_xtlm `create_cluster` ~360-400.)

- [ ] **Step 5 (gate, out of sandbox):** drive `aiesim_create("aie2",
  VC2802.json)`, then one `ess_Write32` to a known register via the bridge, read
  it back via `read32_backdoor`; assert the value matches. Resolve scouting
  fact 7 (GM routing) with a small `ess_WriteGM`/`ess_ReadGM` round-trip.

- [ ] **Step 6: commit** `aiesim-bridge: ps_bridge ess_*() seam + socket bindings (II-B.1)`.

## Task II-B.2: `cdo_replay` — decode the op-stream, drive `ess_*()`

**Files:** Create `aiesim-bridge/src/cdo_replay.{h,cpp}`; modify `c_abi.cpp`
(`aiesim_load_cdo` + `aiesim_exec_npu` call it), `CMakeLists.txt`.

Decoder twin of the Rust `encode_cdo`/`encode_npu` wire format
(`crates/xdna-emu-ffi/src/aiesim/backend.rs`). **Matched pair** — keep tag
constants identical; cross-reference in comments.

- [ ] **Step 1: confirm the wire format** from the Rust encoder (tags
  `OP_WRITE32=1`, `OP_BLOCKWRITE=2`, `OP_MASKPOLL=3`; the separate `npu_tag`
  namespace with `DdrPatch`/`Sync`). Copy the exact field order into a header
  comment.

- [ ] **Step 2: the decoder** — `cdo_replay(const uint8_t* ops, size_t len)`:
  `OP_WRITE32 -> ess_Write32`; `OP_BLOCKWRITE -> ess_Write128`/per-word
  `ess_Write32`; `OP_MASKPOLL -> ` poll `ess_Read32` against mask/value with a
  bounded `sc_start` step cap. Unknown tag -> error (encoder/decoder drift).

- [ ] **Step 3: NPU op-stream** — `exec_npu` decoder resolves `DdrPatch`
  (host-buffer addr + arg_plus -> patched value -> `ess_Write32`) against the
  registered host buffers (II-B.3 add_host_buffer); `Sync` -> DMA-wait
  (`sc_start` until the channel's done bit, bounded).

- [ ] **Step 4: wire into `c_abi.cpp`** — `aiesim_load_cdo`/`aiesim_exec_npu`
  call the decoder (both via the queue once II-B.3 lands).

- [ ] **Step 5 (gate):** load a small real CDO op-stream (Rust encoder from a
  parsed fixture xclbin); confirm expected registers read back via
  `read32_backdoor`.

- [ ] **Step 6: commit** `aiesim-bridge: cdo_replay decodes op-stream, drives ess_*() (II-B.2)`.

## Task II-B.3: `service_thread` — elaborate-once lifecycle + command queue

**Files:** Create `aiesim-bridge/src/service_thread.{h,cpp}`; modify `c_abi.cpp`
(all entries marshal onto the queue), `sc_bootstrap.cpp` (sc_main runs the
service loop, not a one-shot), `CMakeLists.txt`.

Replace the II.3 synchronous-start scaffold with the spec §6 model: the first
`aiesim_create` spawns a thread running `sc_elab_and_sim`; `sc_main` constructs
`aiesim_top` once, then loops pulling commands; C-ABI calls enqueue + block for
the reply. Process-singleton; one SystemC sim per process.

- [ ] **Step 1: command queue** — thread-safe queue of tagged commands
  (`LOAD_CDO`/`EXEC_NPU`/`WRITE_GM`/`READ_GM`/`READ_REG`/`ADD_HOST_BUF`/
  `CLEAR_HOST_BUF`/`RUN`/`RESET`) with reply slots (condvar handshake).

- [ ] **Step 2: service thread owns `aiesim_top`** — `sc_main` constructs it,
  then services commands. `RUN` -> `sc_start()` until `plio_complete` /
  completion or the cycle budget -> reply `{halt, cycles}`. `READ_REG` ->
  `read32_backdoor` (zero sim-time, serviced without advancing). `RESET` ->
  re-apply CDO (or destroy+recreate aiesim_top; decide here).

- [ ] **Step 3: route `c_abi.cpp` through the queue** — each `aiesim_*` enqueues
  + blocks. `aiesim_create` spawns the thread + waits for elaboration, returns
  the singleton handle. `aiesim_destroy` parks the thread.

- [ ] **Step 4: topology** — fill cols/rows from `me->get_num_cols()` (+ per-arch
  rows). Decide whether to extend the C ABI to hand topology back to the Rust
  selector (which currently hardcodes 5x6 in `select_backend`) or document it as
  bridge-internal + the Rust geometry as informational.

- [ ] **Step 5 (gate):** full submission `create -> load_cdo -> write_gm -> run
  -> read_gm`, asserting the kernel advances (cycles > 0) + completes. First
  end-to-end batch through the service thread.

- [ ] **Step 6: commit** `aiesim-bridge: elaborate-once service thread + command queue (II-B.3)`.

## Task II-B.4: tier-2 bring-up (hello-cluster + vs-swap exact-match)

**Files:** Create `tests/aiesim_bringup.rs` (gated) or a `scripts/` harness;
`aiesim-bridge/README.md`.

- [ ] **Step 1: hello-cluster through the FFI** — with the bridge built +
  `--features aiesim`, run a trivial xclbin via `XDNA_BACKEND=aiesim` + the real
  `AiesimBackend`/`select_backend` seam; assert correct output. (Re-confirms
  II-B.3 through the real Rust path, not the C ABI directly.)

- [ ] **Step 2: in-process vs. the proven ELF-swap path** — same kernel, both
  aiesim routes (in-process vs. `docs/aiesimulator.md` swap); assert exact-match
  output. Isolates "wired correctly" from "aiesim correct".

- [ ] **Step 3: grid** — {Peano, Chess} core ELFs x aiesim-in-process, correct
  output (the 2x3 grid through the real seam).

- [ ] **Step 4: commit** the gated tests + the README (build + run: env, device
  JSON, `XDNA_BACKEND=aiesim`).

## Task II-B.5: tier-3 oracle — differential vs. interpreter + third runtime

**Files:** Modify `scripts/emu-bridge-test.sh`; create a differential harness.

- [ ] **Step 1: differential interpreter-vs-aiesim** on a small corpus through
  the same FFI seam: compare output + tier-2 register/memory at sync points
  (`read_reg`/`read_gm`). Triage divergences -> the interpreter "never fall
  behind" backlog. (Phoenix-survival corpus is the natural input once that thread
  executes — link, don't block.)

- [ ] **Step 2: aiesim as a third `emu-bridge-test.sh` runtime** — HW vs EMU vs
  aiesim, gated on the bridge `.so` present (absent -> silently skip the aiesim
  column, never fail). The "third bridge-test runtime" unlock.

- [ ] **Step 3: commit** `aiesim: third bridge-test runtime + interpreter differential (II-B.5)`.

---

## Out of scope (per spec §10, unchanged)

- **Phase-2 HAL-driven independent replay** (`hal_driver`, the `ess_WriteCmd`
  path) — a second, independent CDO interpretation as a stricter oracle.
- **Custom device-model generation** (our own device JSON; NPU1 5x6) — gated on
  decoding the binary `XbV18.3` format. Until then we use the shipped device JSON.
- **Interpreter tier-3 feature backlog** — closing gaps aiesim exposes.

## Self-review notes

- **Encoder/decoder are a matched pair** (Rust `encode_cdo`/`encode_npu` <->
  `cdo_replay`). Tag constants + field order identical; cross-reference both.
- **`ess_WriteCmd` stays unprovided** — it is the HAL path (excluded). Providing
  symbols the cluster doesn't need only invites confusion.
- **GM routing (scouting fact 7)** is the one genuinely open data-path question;
  resolve it empirically in II-B.1 Step 5 before building on it.
- **Topology hardcode (5x6)** in the Rust selector is the stopgap resolved in
  II-B.3 Step 4.
