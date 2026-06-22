# Program-Path: Through-Core Dataflow Edges (Follow-on to Config-Path Tier E)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.
> This is a **stub** — flesh out the tasks (TDD, bite-sized) before execution. It exists so
> the through-core relay is *slated, not dropped* (Maya, 2026-06-21).

**Status:** SLATED — to begin **immediately after** config-path Tier E (E1–E5) is "ready".

**Goal:** Derive the one dataflow relay that config alone cannot reach — the **compute tile**
relay `S2MM → core(compute) → MM2S`, mediated by the core's **ELF lock instructions** — so
compute-tile trace events can be oriented the same way memtile DMA events now are.

**Why separate from Tier E.** Tier E is *config-derived* (CDO + instruction stream + DMA/BD
config). The core's `aie.use_lock(...)` acquire/release operations are **program** behavior
compiled into the core ELF, a different source. Keeping them in a distinct `program_path`
layer preserves the clean source taxonomy the inference engine depends on
("what the binary's *config* contains" vs "what the *program* does").

**The relay to model (add_one ground truth).** Compute tile (0,2):
`S2MM0` writes `in1` buffer (rel compute-lock1 cons) → **core**: `acq lock1`, `+1`,
`rel lock3` → `MM2S0` reads `out1` buffer (acq lock3). Pure BD lock-pairing (Tier E) cannot
link `S2MM0 → MM2S0` here because **no BD** acquires lock1 or releases lock3 — the **core
program** does. So this layer must source the core's lock acq/rel.

**Not exercised by add_one's current trace events** (compute-tile ports aren't instrumented in
the add_one trace). A kernel that traces compute-tile ports is needed to validate end-to-end —
identify or add one during planning.

## Open design questions (resolve during brainstorming/planning)

1. **Source of core lock ops:** parse the core ELF for lock instructions (static, preferred,
   "config-like") vs observe lock acq/rel during an emulated core run (dynamic). Decide and
   justify against the engine's soundness keystone.
2. **Edge representation:** a new `EdgeKind::CoreLockRelay` (or `program_*`) in the route graph,
   or a separate program-path graph layered at generation time? Tier E put edges in the Rust
   route graph; weigh consistency vs keeping config/program sources cleanly separable.
3. **Validation:** A5/E4-style — static core-lock-relay edges ⊇ runtime-enacted core lock
   handoffs, observed by extending the lock-handoff recorder to core-initiated acq/rel.
4. **Decode source:** llvm-aie TableGen for the lock-instruction encoding (DERIVE FROM TOOLCHAIN).

## Provisional task shape (to be expanded)

- P1: Locate/decode core lock acquire/release in the compute ELF (TableGen-driven).
- P2: Capture core lock ops into a structured per-tile form reachable from `DeviceState`.
- P3: Emit through-core relay edges (S2MM-consumed-buffer → core-lock-chain → MM2S-produced-buffer).
- P4: A5-style validation vs runtime core-lock enactment.
- P5: Validate on a compute-tile-traced kernel; generator derives the compute-tile dataflow edge.
