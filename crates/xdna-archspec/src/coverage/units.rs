//! Behavioral units: the explicit override registry (spec Section 3) and the
//! single hand-curated CapabilityDomain spine (spec Section 6). Seeded coarse;
//! Phase 2 refines. Co-located on purpose (one location, spec Section 6).

use crate::coverage::verdict::{Verdict, Verification};
use crate::coverage::CoverageNode;
use crate::types::Architecture;

/// What fine nodes a behavioral unit claims (spec Section 3). A unit either
/// claims explicit nodes (override) or is the derived bucket for a category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Claims {
    /// Explicit override: this exact set of nodes (spec Section 3).
    Nodes(Vec<CoverageNode>),
}

/// A behavioral unit: a cluster of fine nodes with one verdict (spec Section 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehavioralUnit {
    pub id: String,
    pub arch: Architecture,
    pub claims: Claims,
    pub verdict: Verdict,
    /// Free-text human narrative for the Section-3 no-silent-shadow rule:
    /// when an override pulls a node off the toolchain-derived path, this
    /// records why for a human reader. NEVER substring-matched for
    /// enforcement -- see `shared_from` for the typed soundness gate.
    pub shadows_derived: Option<String>,
    /// Typed cross-arch provenance (spec Section 7). Some(other_arch) means
    /// this verdict was shared from other_arch. enforce_coverage panics on
    /// Verified + shared_from.is_some() -- verification never transfers
    /// across silicon. Typed, so phrasing can neither bypass nor trip it.
    pub shared_from: Option<Architecture>,
}

/// A top-level hardware capability the manual names (spec Section 6), now
/// carrying the retired index's per-subsystem detail as seeded data (spec
/// Section 2). Each must be claimed by >= 1 behavioral unit per applicable
/// arch, or the build panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityDomain {
    pub id: String,
    pub arches: Vec<Architecture>,
    /// Authoritative source (aie-rt path / AM025 / device model). Non-empty,
    /// enforced.
    pub source_ref: String,
    /// Our emulator src/ path(s). Non-empty unless an OOS/MISSING narrative is
    /// present (enforced).
    pub src_locations: Vec<String>,
    /// Human-readable notes and known gaps for this domain.
    pub narrative: String,
    /// The domain's own asserted coverage verdict (spec Section 1/2).
    pub verdict: Verdict,
    /// Documents a deliberate own-vs-rollup divergence (spec Section 3); a
    /// silent material drift is a hard failure, an annotated one is allowed.
    pub drift_rationale: Option<String>,
}

impl CapabilityDomain {
    pub fn applies_to(&self, arch: Architecture) -> bool {
        self.arches.contains(&arch)
    }
}

/// The explicit override registry, per arch. Phase 2 adds entries here; the
/// sole current entry Accepts the `SemanticOp::Intrinsic(_)` catch-all (#104).
pub fn override_registry(arch: Architecture) -> Vec<BehavioralUnit> {
    match arch {
        Architecture::Aie2 => {
            vec![intrinsic_catchall_accepted(), dma_ops_verified(), vector_ops_verified()]
        }
        _ => Vec::new(),
    }
}

/// Flip the empirically silicon-verified Vector-category ops to Verified (#126).
///
/// The vector differential fuzzer (#112/#114) silicon-matched its full 218-key
/// universe against real NPU1 (0 divergent). An empirical executed-op audit --
/// `fuzz_recorder` captures the exact SemanticOps the emulator dispatches per
/// case, banked as `executed.json`; current replay of the silicon-matched
/// corpus is 24/24 match, and dispatch is a deterministic function of (xclbin,
/// emulator) so a current output-match locks the banked executed set as current
/// -- gives the precise set of Vector SemanticOps that ran in matched runs:
/// these 20. Method + the full per-op evidence:
/// `docs/superpowers/findings/2026-06-12-vector-semanticop-empirical-audit.md`.
///
/// The audit corrected the static guess: integer `aie::min`/`aie::max`/`abs`/
/// `neg`/`maxdiff` lower to the hardware's FUSED compare-select instructions, so
/// the verified ops are `MaxLt`/`MinGe`/`AbsGtz`/`NegGtz`/`NegLtz`/`MaxDiffLt` --
/// NOT `Min`/`Max` (which are dead for this corpus and stay perishable).
///
/// `Pack`/`Unpack` were added by #127: re-fuzzing pack8 (I16->I8) and unpack16
/// (I8->I16) on real NPU1 with pool-bearing replayable seeds confirmed current
/// silicon match + current dispatch. Note pack16 (I32->I16) and unpack32
/// (I16->I32) lower to `Srs`/`Ups` (accumulator<->vector narrowing/widening),
/// already claimed -- only the I16<->I8 couplers hit the dedicated Pack/Unpack ops.
///
/// Deliberately NOT claimed (stay perishable -- under-claim is safe): the
/// never-executed variants (`MatMulSub`, `NegMatMul`, `AddMac`, `SubMac`,
/// `NegMul`, `NegAdd`, `AccumNegAdd`, `AccumNegSub`, `VectorPush`,
/// `VectorPushHi`, `SubLt`, `SubGe`, `Min`, `Max`).
///
/// Provenance stays `AietoolsModeled` (the compute semantics were reimplemented
/// from the aietools models); only verification moves to `Verified`. This does
/// NOT green `clean_release(Aie2)`: the unclaimed Vector ops + the stream/cascade
/// SideEffect ops (tenants 4/5) keep the perishable queue non-empty.
fn vector_ops_verified() -> BehavioralUnit {
    use crate::aie2::isa::SemanticOp;
    use crate::coverage::verdict::{Provenance, Verdict};
    let ops = [
        SemanticOp::Mac,
        SemanticOp::MatMul,
        SemanticOp::Srs,
        SemanticOp::Ups,
        SemanticOp::Shuffle,
        SemanticOp::Convert,
        SemanticOp::VectorBroadcast,
        SemanticOp::VectorExtract,
        SemanticOp::VectorInsert,
        SemanticOp::VectorSelect,
        SemanticOp::VectorClear,
        SemanticOp::MaxDiffLt,
        SemanticOp::MaxLt,
        SemanticOp::MinGe,
        SemanticOp::AbsGtz,
        SemanticOp::NegGtz,
        SemanticOp::NegLtz,
        SemanticOp::Accumulate,
        SemanticOp::AccumSub,
        SemanticOp::Align,
        SemanticOp::Pack,
        SemanticOp::Unpack,
    ];
    BehavioralUnit {
        id: "aie2.vector_ops.verified".into(),
        arch: Architecture::Aie2,
        claims: Claims::Nodes(
            ops.iter()
                .map(|op| CoverageNode::Semantic { arch: Architecture::Aie2, op: op.clone() })
                .collect(),
        ),
        verdict: Verdict {
            provenance: Provenance::AietoolsModeled,
            verification: Verification::Verified {
                evidence: "Vector differential fuzzer (#112/#114) vs real NPU1: full 218-key universe \
                    silicon-matched, 0 divergent. Empirical executed-op audit (fuzz_recorder -> \
                    executed.json over the silicon-matched corpus; current replay 28/28 match, \
                    deterministic dispatch) shows these 22 Vector-category SemanticOps dispatched in \
                    matched runs. Note the fused compare-select lowering: min/max/abs/neg/maxdiff \
                    execute as MaxLt/MinGe/AbsGtz/NegGtz/NegLtz/MaxDiffLt (Min/Max are dead). Pack/Unpack \
                    (#127): re-fuzzed pack8 (I16->I8) and unpack16 (I8->I16) on real NPU1 with \
                    pool-bearing replayable seeds (60/60 HW match, executed.json shows Pack/Unpack); \
                    note pack16 (I32->I16) and unpack32 (I16->I32) lower to Srs/Ups, not Pack/Unpack. See \
                    docs/superpowers/findings/2026-06-12-vector-semanticop-empirical-audit.md."
                    .into(),
            },
        },
        shadows_derived: Some(
            "These 22 ops otherwise derive to the Vector category default (AietoolsModeled/Unverified \
             -> perishable). Claimed ONLY where the empirical audit shows the op executed in a \
             currently-replay-confirmed silicon match; the never-executed Vector variants \
             (Min/Max/MatMulSub/VectorPush/...) are NOT claimed and stay perishable."
                .into(),
        ),
        shared_from: None,
    }
}

/// Flip `DmaStart` + `DmaWait` to silicon-Verified (#113 axis-2).
///
/// The diff-fuzzing framework's DMA/data-movement tenant drove the DMA
/// access-pattern engine against real NPU1 silicon to ledger completion: the
/// full 81-key universe `{feature}/{engine}/{dir}/{dtype}` (33 shim + 48
/// memtile; linear/strided-2d/3d/4d/transpose/overlap/packet/pad-{before,
/// after,both}) was silicon-matched at depth ~10 seeds/key -- 810 HW cases, 0
/// emulator-vs-silicon divergences, 0 crashes, 0 wedges. Every n-dimensional BD
/// program the generator can express lands byte-for-byte identically on the
/// emulator and the hardware. In SemanticOp terms that is exactly `DmaStart`
/// (issue + walk the BD) and `DmaWait` (await channel completion).
///
/// Scope is deliberately TWO ops, not the whole `SideEffect` category. The
/// remaining five SideEffect ops -- `CascadeRead`/`CascadeWrite` (cascade) and
/// `StreamRead`/`StreamWrite`/`StreamWritePacketHeader` (core-side stream) --
/// are the contention/cascade tenants (4/5) and are NOT exercised by the DMA
/// tenant, so they keep their honest `DocSpecified`/`Unverified` default.
///
/// Provenance stays `DocSpecified` (the access-pattern semantics are
/// AM025/aie-rt described); only the verification axis moves to `Verified`,
/// mirroring the modeled-then-silicon-checked transition. This Verified flip
/// does NOT green `clean_release(Aie2)`: the perishable queue still holds the
/// vector ops and the unclaimed SideEffect ops, so the gate stays honestly red
/// until the vector and tenant-4/5 work lands.
fn dma_ops_verified() -> BehavioralUnit {
    use crate::aie2::isa::SemanticOp;
    use crate::coverage::verdict::{Provenance, Verdict};
    BehavioralUnit {
        id: "aie2.dma_ops.verified".into(),
        arch: Architecture::Aie2,
        claims: Claims::Nodes(vec![
            CoverageNode::Semantic { arch: Architecture::Aie2, op: SemanticOp::DmaStart },
            CoverageNode::Semantic { arch: Architecture::Aie2, op: SemanticOp::DmaWait },
        ]),
        verdict: Verdict {
            provenance: Provenance::DocSpecified,
            verification: Verification::Verified {
                evidence: "Diff-fuzzing framework DMA/data-movement tenant vs real NPU1: the full \
                    81-key access-pattern universe ({feature}/{engine}/{dir}/{dtype}, 33 shim + 48 \
                    memtile, linear/strided-2d/3d/4d/transpose/overlap/packet/pad-*) silicon-matched \
                    at depth ~10 seeds/key -- 810 HW cases, 0 divergent, 0 crash, 0 wedge. The \
                    emulator DMA model is indistinguishable from NPU1 silicon across the whole \
                    n-dimensional BD access-pattern space. See \
                    docs/superpowers/plans/2026-06-11-framework-step3a-dma-tenant.md (Outcome) and \
                    docs/superpowers/specs/2026-06-11-dma-data-movement-domain.md."
                    .into(),
            },
        },
        shadows_derived: Some(
            "DmaStart/DmaWait otherwise derive to the SideEffect category default \
             (DocSpecified/Unverified -> perishable). This override claims ONLY these two ops -- \
             the silicon evidence is the DMA-engine access-pattern campaign, which does not touch \
             the cascade (CascadeRead/Write) or core-side stream (StreamRead/Write/ \
             WritePacketHeader) ops; those stay on their derived default for tenants 4/5."
                .into(),
        ),
        shared_from: None,
    }
}

/// Accept the `SemanticOp::Intrinsic(_)` catch-all node (#104).
///
/// The node `SemanticOp::Intrinsic(0)` represents "an intrinsic we could not
/// classify to a concrete SemanticOp." It derives to the NeedsTriage category
/// default (`Unspecified` / `Unverified`) and so registers as a comprehension
/// gap. It is closed here by explicit Accept rather than left open, on these
/// confirmed facts:
///
///   * It is **never constructed** by the live pipeline. Every intrinsic-backed
///     instruction is resolved to a concrete SemanticOp at build time
///     (`build_helpers/extract.rs:semantic_from_intrinsic`), and the ISA build
///     asserts 100% semantic coverage (`aie2/isa/mod.rs`), so no decoded
///     instruction can carry `Intrinsic`. The runtime classifier
///     `SemanticOp::from_intrinsic` that would mint `Intrinsic(idx)` is dead
///     code (never called outside its own unit tests).
///   * It is **fail-loud, not silent-wrong**: the surface probe classes
///     `Intrinsic(_)` as `Absent`, so any value reaching execution hits a hard
///     `ExecuteResult::Error`, never a wrong result.
///   * The concrete vector SemanticOps that intrinsics actually resolve to
///     (`Srs`/`Ups`/`Pack`/`Convert`/`Mac`/`MatMul`) are differentially
///     verified bit-exact against the genuine aietools models (#103 Half A;
///     `docs/superpowers/findings/2026-06-08-vector-compute-audit-half-a-rollup.md`).
///
/// Provenance stays `Unspecified` -- the catch-all genuinely has no model; it is
/// Accepted because it is unconstructible and fail-loud, not because it is
/// understood. Silicon verification of the *resolved* ops is Half B (HW-gated)
/// and is tracked separately in the perishable queue, so this Accept does not
/// green the release gate.
fn intrinsic_catchall_accepted() -> BehavioralUnit {
    use crate::aie2::isa::SemanticOp;
    use crate::coverage::verdict::{Provenance, Verdict};
    BehavioralUnit {
        id: "aie2.intrinsic_catchall.accepted".into(),
        arch: Architecture::Aie2,
        claims: Claims::Nodes(vec![CoverageNode::Semantic {
            arch: Architecture::Aie2,
            op: SemanticOp::Intrinsic(0),
        }]),
        verdict: Verdict {
            provenance: Provenance::Unspecified,
            verification: Verification::Accepted {
                rationale: "SemanticOp::Intrinsic(_) catch-all: never constructed by the live \
                    pipeline (every intrinsic-backed instruction resolves to a concrete SemanticOp \
                    at build time via semantic_from_intrinsic; the ISA build asserts 100% semantic \
                    coverage; the runtime from_intrinsic classifier is dead code). Fail-loud: the \
                    surface probe classes Intrinsic(_) as Absent -> hard ExecuteResult::Error, never \
                    a wrong value. The concrete ops it resolves to (Srs/Ups/Pack/Convert/Mac/MatMul) \
                    are differentially verified bit-exact vs the genuine aietools models (#103 Half \
                    A). Accepted as unconstructible + fail-loud, not understood; silicon \
                    verification of the resolved ops is Half B (perishable queue)."
                    .into(),
            },
        },
        shadows_derived: Some(
            "Intrinsic(0) otherwise derives to the NeedsTriage default (Unspecified/Unverified -> \
             comprehension gap). This Accept shadows no real unmodeled behavior: the catch-all is \
             never constructed (build-time 100% concrete-SemanticOp coverage) and is fail-loud."
                .into(),
        ),
        shared_from: None,
    }
}

/// The single hand-curated capability spine (spec Section 6). Seeded once from
/// the AM020 ToC + aie-rt module tree; maintained ONLY here. Deliberately
/// COARSER than `SubsystemKind` (spec Section 6: architectural domains, not a
/// 1:1 register-taxonomy mirror). Documented folds so a Phase-2 author does
/// not recreate a domain or think one is missing:
///   - `core` covers `SubsystemKind::Processor` (the VLIW compute core)
///   - `program_counter` covers `SubsystemKind::ProgramCounter` (PC sampling)
///   - `events_trace` covers `SubsystemKind::Trace` AND `::Event`
///   - `locks` covers `SubsystemKind::Lock` AND `::LockRequest`
///   - `debug_halt` covers `SubsystemKind::Debug`
///   - `control_packets` covers on-chip control-packet reassembly, packet
///     handler status, and the NPU host instruction stream (spec Appendix)
///   - `clock_control` covers module/column/tile clock + reset control
///   - `tile_isolation` covers Tile_Control isolation bits / N-S-E-W gates
///   - `binary_load` covers CDO/ELF/XCLBIN ingest, SS-routing reconstruction,
///     and array-topology construction (spec Appendix N1 resolutions)
/// `SubsystemKind::Unknown` is a classifier placeholder, NOT a hardware
/// capability -- intentionally excluded. Coarse, arch-invariant; AIE2 is the
/// only wired arch today (Plan 1). Phase 1 auto-claims every domain via the
/// derived shim (Task 6); the SubsystemKind<->spine partition is Phase 2.
pub fn capability_spine() -> Vec<CapabilityDomain> {
    use crate::coverage::verdict::{Completeness::*, Provenance::*, Verification::*};
    let aie2 = || vec![Architecture::Aie2];
    let d = |id: &str,
             source_ref: &str,
             locs: &[&str],
             narrative: &str,
             v: Verification,
             drift_rationale: Option<&str>| CapabilityDomain {
        id: id.into(),
        arches: aie2(),
        source_ref: source_ref.into(),
        src_locations: locs.iter().map(|s| s.to_string()).collect(),
        narrative: narrative.into(),
        verdict: Verdict { provenance: ToolchainDerived, verification: v },
        drift_rationale: drift_rationale.map(|s| s.to_string()),
    };
    vec![
        // Order must match SPINE_DOMAIN_IDS exactly (spec Section 6: one location).
        d("core", "aie-rt core/, llvm-aie TableGen; AM025 Core_Control/Core_Status/Error_Halt_*",
          &["src/interpreter/", "src/device/core_debug/"],
          "VLIW core, control (enable/done/reset), and error-halt path. 100% ISA decode; SemanticOp coverage ~33%, rest in legacy handlers. Generic error_halt fires INSTR_ERROR (event 69) on every CoreStatus::Error; ECC fires ECC_ERROR_STALL. Saturation/watchdog error sources not yet detected.",
          Modeled { completeness: Full },
          Some("Own verdict reflects the actually-modeled subsystem; the rolled-up verdict is the pessimistic Phase-1 coarse category default (NeedsTriage=Unspecified/Unverified), a bootstrap floor, not a ceiling. Per-op refinement that would raise the rollup is Phase-2 override work (spec Section 3).")),
        d("program_memory", "AM025",
          &["src/parser/elf.rs"],
          "16KB program memory; ELF load -> run.", Modeled { completeness: Full }, None),
        d("program_counter", "AM025 PC_Event0..3 (0x32020/4/8/C); aie-rt xaiemlgbl_params.h",
          &["src/interpreter/", "src/device/core_debug/"],
          "PC sampling + PC_Event0..3 / Core_PC_Range matching; drives event-halt selector.",
          Modeled { completeness: Full }, None),
        d("data_memory", "aie-rt memory/, AM025",
          &["src/device/banking.rs", "src/device/state/memtile.rs", "src/interpreter/timing/memory.rs"],
          "64KB compute (8 banks x 128-bit) / 512KB memtile; conflict detection done; per-bank MEM_CONFLICT_DM_BANK_N fired. ECC: status bit readable, no scrubber/fault-injection -- accepted out of scope unless workloads require.",
          Modeled { completeness: Full }, None),
        d("dma", "aie-rt dma/, AM025 (112/433/144 reg)",
          &["src/device/dma/"],
          "BDs per tile-type 16/48/16 (memtile 48 not 64 -- aie-rt xaiemlgbl_reginit.c), channels 2/6/2, address dims 3/4/3; n-d addressing, padding, lock coupling, packet header, compression. Repeat / out-of-order BD execution: verify. Stream-side back-pressure is modeled functionally and as observable status: S2MM stalls in Transferring on empty input FIFO (transfer_s2mm returns Stalled, status emits Stalled_Stream_Starvation at bit 4 of DMA_S2MM_Status_0); MM2S stalls in Transferring when stream_out reaches the local-slave FIFO depth (output_fifo_capacity() = STREAM_LOCAL_SLAVE_FIFO_DEPTH per AM020 ch2, status emits Stalled_Stream_Backpressure at bit 4 of DMA_MM2S_Status_0). The S2MM and MM2S stall bits share bit position but use distinct field names in the regdb and are loaded separately so a future register split would surface as a build failure (audit landed 2026-05-24, see docs/superpowers/findings/2026-05-24-bridge-sweep-objectfifo-wedges.md).",
          Modeled { completeness: Full },
          Some("Own verdict reflects the actually-modeled subsystem; the rolled-up verdict is the pessimistic Phase-1 coarse category default (SideEffect=DocSpecified/Unverified), a bootstrap floor, not a ceiling. Per-op refinement that would raise the rollup is Phase-2 override work (spec Section 3).")),
        d("locks", "aie-rt locks/, AM025",
          &["src/device/tile/locks.rs"],
          "Counts per tile-type 16/64/16 (aie-rt xaiemlgbl_reginit.c; the 192 in AieMlMemTileDmaMod.NumLocks is the cross-tile reference range, not slot count). acquire/release/get/set, semaphore semantics, round-robin arbiter.",
          Modeled { completeness: Full }, None),
        d("stream_switch", "aie-rt stream_switch/, AM025 (160/119/149 reg)",
          &["src/device/stream_switch/"],
          "Circuit + packet, FIFOs, port events, packet-header matching. Parse-time routing reconstruction is a binary_load concern (spec Appendix N1). Per-port data-available / space-available signals propagate to DMA consumers/producers: S2MM stalls on empty per-channel input FIFO; MM2S stalls on local-slave-FIFO-full (paired with dma side).",
          Modeled { completeness: Full },
          Some("Own verdict reflects the actually-modeled subsystem; the rolled-up verdict is the pessimistic Phase-1 coarse category default (SideEffect=DocSpecified/Unverified), a bootstrap floor, not a ceiling. Per-op refinement that would raise the rollup is Phase-2 override work (spec Section 3).")),
        d("events_trace", "aie-rt events/ + trace/, AM025 (128/161/51 events)",
          &["src/device/events/", "src/device/trace_unit/"],
          "Events (broadcast 16ch, combo, group, port), cross-tile broadcast network, trace unit modes 0/1/2, pipelined start/stop + multi-tile timer sync. Combo/edge generator boundary cases need targeted tests; L2 broadcast propagation verify.",
          Modeled { completeness: Full }, None),
        d("performance_counters", "aie-rt perfcnt/, AM025 (compute/memtile/shim 4/11/6 reg)",
          &["src/device/perf_counters/"],
          "4 counters, threshold events. DMA/stream FIFO-size events not emitted (cycle-accuracy gap, tracked in cycle-accuracy-mission.md).",
          Modeled { completeness: Full }, None),
        d("timer", "aie-rt timer/, AM025 (5 reg)",
          &["src/device/timer.rs"],
          "Free-running 64-bit per-module; Reset_Event consumed via pending_reset latch; multi-tile timer-sync modeled. Trig_Event_Low/High_Value write effect: verification follow-up.",
          Modeled { completeness: Full }, None),
        d("watchpoint", "AM025 Compute WatchPoint0/1 (2), MemTile WatchPoint0..3 (4)",
          &["src/interpreter/execute/cycle_accurate.rs"],
          "Compute 2 / memtile 4 slots; WriteStrobes==0xF gate, direction + address comparator, AXI/DMA/quadrant origin filters, scalar+vector+DMA-engine paths, modifier-register effective address. Locked by 17+ unit tests.",
          Modeled { completeness: Full }, None),
        d("debug_halt", "AM025 Debug_*; aie-rt core/",
          &["src/device/core_debug/", "src/interpreter/engine/coordinator.rs"],
          "Halt + status bits, synchronous PC-event breakpoints (before-commit pre-execute seam, G1 silicon-derived 2026-05-18), async halt paths, and debug-register read/write routing all modeled. Count-step (Debug_Control0[5:2] Single_Step_Count) is a live N-committed-bundle budget that halts before the (N+1)th bundle commits (G2 silicon-derived 2026-05-19). Event-driven single-step boundary is the principled split: PC-wired (SSTEP_EVENT==Core_PC_0..3) halts before-commit via the same seam; watchpoint/mem/lock/range-wired stays after-commit (documented modeling decision -- no coherent before-commit point). Section 8 close-out (2026-05-19): Core_Status RESET-bit divergence and OUTBUF_ADDR/TRAP_PC probe fragility RESOLVED; G1 bounded-escalation tracker retired (contingency never fired). Open (tracked, spec section 8, HW-budget-gated): count-step finer silicon characterization (decrement cadence / larger-N / 0x11-on-silicon -- only N=4 observed) and resume hardware-verification.",
          Modeled { completeness: Full }, None),
        d("cascade", "aie-rt, aietools events",
          &["src/interpreter/execute/cascade.rs"],
          "Tile<->tile cascade read/write. Deadlock detection is a placeholder (deadlock.rs) -- promote to real detection or remove (verification follow-up).",
          Modeled { completeness: Full },
          Some("Own verdict reflects the actually-modeled subsystem; the rolled-up verdict is the pessimistic Phase-1 coarse category default (SideEffect=DocSpecified/Unverified), a bootstrap floor, not a ceiling. Per-op refinement that would raise the rollup is Phase-2 override work (spec Section 3).")),
        d("interrupt", "aie-rt interrupt/, AM025 (shim interrupt 23 reg: 18 L1 + 5 L2)",
          &["src/device/interrupts/l1.rs", "src/device/interrupts/l2.rs",
            "src/device/interrupts/mod.rs", "src/device/tile/registers.rs",
            "src/device/tile/mod.rs", "src/device/state/effects.rs",
            "src/device/state/dispatch.rs", "src/interpreter/core/interpreter.rs"],
          "Full AIE2 shim interrupt path MODELED. All 23 registers (18 L1 over 2 switches 0x35000-0x35050, 5 L2 0x15000-0x15010) read- and write-routed with exact semantics (write-1-to-clear status, enable/disable->mask, read-only mask). Stimulus path wired: event/error -> EventModule -> L1 (both switches, per-event-slot independent match + enable gating) -> broadcast network (block-mask honored) -> L2 sink -> NPI line, driven to a fixed point so an L1 output reaches L2 within one dispatch. Hardware errors enter via raise_instr_error -> EventModule. Privilege gating (only the L2 interrupt-routing register) is a driver-side concern, scoped out unrestricted per the noc/shim_mux precedent. Host-visible delivery is firmware-mediated (the AIE interrupt never reaches the x86 host directly) -- Tier B Spec 1 (firmware async-event mailbox plumbing + INSTR_ERROR producer) is shipped: cache + per-column rings + push callback + driver-mirror categorization tables + DRM_AMDXDNA_HW_LAST_ASYNC_ERR ioctl wired through the plugin. Per-class detection follow-ups (DMA / parity / ECC / lock / stream) remain tracked. Tier C (wedge-recovery / context-restart) is shipped: per-context state model (Connected/Stopped/Failed) with completed_counter; device-side TdrDetector classifying engine run state into Progressing/NaturalCompletion/MaskPollUnsatisfied/Wedged per cycle; xdna_emu_run consumes the classifier and returns the new XdnaEmuHaltReason::WedgeRecovered halt code on wedge, which the plugin translates to an EIO-shaped XRT command state. Plugin xdna_emu_reset_context resolution upgraded to fail-loud (required). Plumbed for multi-context throughout (single ContextId(0) today, all APIs take ContextId). Multi-context engine scheduling and lifecycle ioctls are a separate spec. See docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md and docs/superpowers/specs/2026-05-19-interrupt-tier-b-firmware-mailbox-design.md and docs/superpowers/findings/2026-05-19-interrupt-tier-c-tdr.md and docs/superpowers/specs/2026-05-19-interrupt-tier-c-tdr-design.md.",
          Modeled { completeness: Full }, None),
        d("noc", "AM025 (NoC_Interface_AIE_to_NoC 4 reg, AIE_AXIMM_Config); aie-rt npi/; hardware spec",
          &[],
          "SCOPE: this domain is the AIE-array <-> SoC-data-fabric (Infinity Fabric) edge ONLY -- the off-chip handoff where shim DMA reaches DDR. It is NOT the on-chip interconnect: tile-to-tile data movement (stream_switch), the broadcast/event network (interrupt), cascade forwarding (cascade), and the memory-mapped control bus (control_packets) are all separately MODELED/Full. Three pieces: (1) Functional data path -- shim DMA <-> DDR is emulated (data arrives correctly via host_memory); no separate NoC model is needed for functional correctness. (2) Fabric timing (AXI latency / NoC arbitration / DDR egress rate) -- DELIBERATELY ABSTRACTED: the data fabric is a shared SoC IP, not the NPU; its latency is nondeterministic from the NPU's view (depends on CPU/GPU/DDR-controller traffic) and has no AM025-style register spec. First-order egress is folded into the calibrated shim-DMA constants (shim_ddr_cold_start_*, words_per_cycle); cycle-level re-calibration is gated on a trace-independent oracle (Perf_Counter readback, currently blocked) -- cycle-accuracy-mission.md item #5 (DEFERRED). (3) Config surface -- AIE_AXIMM_Config (incl. SLVERR_Block) and the NoC_Interface registers are AM025-derivable on demand but unused by every observed binary: control-packet SLVERR is modeled for the reset-default (SLVERR-enabled) config every binary uses; honoring SLVERR_Block would only matter for a binary that programs it (none do). NPI privileged register access is driver-side privilege -- emulator gives unrestricted access, out of scope.",
          Accepted { rationale: "AIE <-> SoC-data-fabric (Infinity Fabric) edge: a deliberate abstraction boundary, not unbuilt NPU work. Functional path is covered by shim DMA + host_memory; the on-chip interconnect (stream_switch / interrupt / cascade / control_packets) is separately Full. Fabric timing is a shared non-NPU SoC IP with no register-level spec, folded into the calibrated shim-DMA constants (cycle-calibration gated on the perf-counter oracle -- cycle-accuracy-mission.md #5). The AXIMM config surface (SLVERR_Block) is AM025-derivable on demand but unused by all observed binaries.".into() }, None),
        d("shim_mux", "aie-rt, AM025 (shim Mux/Demux 2 reg); aie-rt pl/ (PL Interface)",
          &["src/device/stream_switch/"],
          "Shim master/slave NoC-facing mux/demux. PL Interface (Upsizer/Downsizer) is Versal-FPGA stream-width adaptation -- NPU1 exposes no programmable PL: accepted out of scope.",
          Modeled { completeness: Full }, None),
        d("control_packets", "AM025 Control_Packet_Handler_Status (0x3FF30/0xB0F30); aie-rt xaiegbl_params.h:7761; XRT host protocol",
          &["src/device/control_packets/", "src/device/tile/mod.rs", "src/device/state/ctrl_access.rs", "src/interpreter/engine/coordinator.rs", "src/npu/"],
          "Control-packet headers, reassembly, register read/write effects, response packets MODELED. NPU host instruction stream (WRITE32/BLOCKWRITE/BLOCKSET/MASKWRITE/MASKPOLL/CONFIG_SHIMDMA_*/DDR_PATCH) MODELED. All four Tile_Control_Packet_Handler_Status sticky bits have faithful detecting paths: First/Second-header parity and Tlast via the reassembler, SLVERR_On_Access via undecoded-address decode at the dispatch boundary, all with poll-only sticky-continue semantics (aie-rt/AM025: latch + continue, no interrupt, no abort). The AIE_AXIMM_Config.SLVERR_Block config-suppression refinement is a tracked NoC-gated goal -- see the noc domain. Keystone subsystem.",
          Modeled { completeness: Full }, None),
        d("clock_control", "AM025 Module_Clock_Control / Column_Clock_Control / Reset_Control_1 / AIE_Tile_Column_Reset; aie-rt pm/xaie_clock.c",
          &["src/device/clock_control/", "src/device/state/dispatch.rs", "src/device/array/mod.rs", "src/device/array/routing.rs", "src/device/array/dma_ops.rs", "src/interpreter/engine/coordinator.rs"],
          "Three-tier clock gating MODELED: Column_Clock_Control, per-module Module_Clock_Control (compute/memtile/shim_0/shim_1 layouts decoded from regdb), and adaptive clock gating (idle counters + abort_period + wake-on-event). Step loop consults all three tiers before DMA / StreamSwitch / Core stepping. Adaptive counters tick once per data-movement cycle, gated by per-module clocks (silicon-accurate); column or module re-ungate transitions reset the relevant counter; Phase-5 tick consults DmaEngine::any_channel_has_pending_work so a queued-but-FSM-Idle channel is treated as not-yet-idle. Wake-on-event paths (cycle-accuracy-mission.md item #8): (Wake 1) register-bus access to the Dma/Lock/DataMemory subsystem wakes the DMA counter, StreamSwitch subsystem wakes the SS counter, via DeviceState::wake_adaptive_for_subsystem invoked from write_register/mask_write_register/dma_write; (Wake 2) stream beat arriving at a slave port sets cycle_active, which the existing Phase-5 SS branch converts to tick_adaptive_ss(active=true); (Wake 3) lock-value changes reduce to Wake 1 cross-tile (control-packet decode into Lock register write) plus the Phase-5 has_pending_work check for same-tile (AcquiringLock implies is_active(), so the gate cannot engage). Silicon-accurate boot default: all tiles gated, ungate-via-CDO is the documented path; ungate_all() test helper exercises the real register-write path. Gated-tile non-clock-control writes serve-and-warn (dedup per (col,row,offset); XDNA_EMU_WARN_GATED_ACCESS=0 silences). AIE_Tile_Column_Reset (shim 0xFFF28) partition teardown MODELED: asserting bit 0 tears down every non-shim tile in the column (cores, DMAs, locks, stream switches, adaptive counters) via Array::reset_column, preserving tile memory and exempting the shim row, on the assert edge (aie-rt pm/xaie_reset.c); Reset_Control_1/0xFFF14 (shim-NoC reset) is intentionally a no-op pending the NoC stub. Clock-gating counter-freeze MODELED (\"no clock, no tick\"): a clock-gated module advances neither its free-running timer (core/mem module timers) nor its performance counters. Phase 3e consults the Core module clock (compute core bank) or Memory module clock (compute/memtile mem bank), falling back to the column clock for the shim-PL bank (no separate shim-PL module gate in the model), before ticking -- so gating a module or column freezes Performance_Counter0..3 and the module timers, and they resume from the held value on re-ungate (clock-gating preserves state; the separate AIE_Tile_Column_Reset path is what clears them). FULL: the AIE2 Core module has no adaptive clock gating to model -- Module_Clock_Control (core, 0x60000) exposes only DMA_Adaptive_Clock_Gate (bit 5), Stream_Switch_Adaptive_Clock_Gate (bit 4), and the plain Core_Module_Clock_Enable (bit 2); the AM025 register DB and aie-rt define no Core idle counter, no Core abort_period register, and no cascade wake event. A core waiting on cascade input stalls at the FSM level (Cascade_Stall_MCD/SCD in Core_Status, already modeled via the cascade queues + core stall) -- it does not clock-gate-and-wake -- so the former \"Wake 4\" cascade-arrival item is a non-feature (hardware has no such mechanism), not deferred work. Privilege enforcement is out of scope (no privilege model in the emulator yet) -- a candidate for later bug-probing, not a modeling gap. See docs/superpowers/specs/2026-05-24-clock-control-design.md.",
          Modeled { completeness: Full }, None),
        d("tile_isolation", "aie-rt pm/xaie_tilectrl.c, AM025 (Tile_Control compute 0x36030 / memtile 0x96030)",
          &["src/device/tile/mod.rs", "src/device/state/effects.rs", "src/device/array/routing.rs", "src/interpreter/execute/memory/neighbor.rs", "src/interpreter/engine/coordinator.rs"],
          "Tile_Control low 4 bits (S/W/N/E) snapshotted on register write. Inter-tile stream transfers, cross-tile NeighborMemory snapshots/reads/buffered writes, and NeighborLocks slices all consult the destination/own isolation byte. Shim isolation snapshotted; only memtile->shim south-bound routing gate consults it today. Clock-gating bits of Tile_Control pass through unmodeled (see clock_control).",
          Modeled { completeness: Full }, None),
        d("binary_load", "XRT container / CDO / ELF formats; mlir-aie device model (tools/aie-device-models.json)",
          &["src/parser/xclbin.rs", "src/parser/cdo/", "src/parser/elf.rs", "src/parser/stream_switch_topology.rs", "src/device/array/"],
          "XCLBIN container, CDO framing/syntax/semantics -> DeviceOps, per-core ELF load, all MODELED. Stream-switch routing reconstruction from CDO writes (parse-side, distinct from the runtime stream_switch subsystem -- spec Appendix N1). Tile array topology (5x6 NPU1) constructed from the device model at load: folded here as the array-constructed-from-binary concern (spec Appendix N1 rationale), not a reachability-forced tag.",
          Modeled { completeness: Full }, None),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Architecture;

    #[test]
    fn override_registry_has_intrinsic_accept() {
        // #104: one Phase-2 override is the Accept of the SemanticOp::Intrinsic(_)
        // catch-all (never constructed, fail-loud; the concrete ops it resolves
        // to are differentially verified -- #103). Located by id, not index, so
        // it survives the registry growing (the DMA-ops Verified override, #113,
        // is the sibling entry).
        use crate::coverage::verdict::Verification;
        let regs = override_registry(Architecture::Aie2);
        let intr = regs
            .iter()
            .find(|u| u.id == "aie2.intrinsic_catchall.accepted")
            .expect("intrinsic-catchall Accept present");
        assert!(matches!(intr.verdict.verification, Verification::Accepted { .. }));
        assert!(intr.shadows_derived.is_some(), "Accept pulls Intrinsic off its derived NeedsTriage default");
    }

    #[test]
    fn override_registry_has_dma_ops_verified() {
        // #113 axis-2: the DMA/data-movement tenant of the diff-fuzzing
        // framework silicon-verified the DMA access-pattern engine against
        // real NPU1 (81/81 keys, 810 HW cases at depth ~10, 0 divergent),
        // which exercises exactly the DmaStart + DmaWait SemanticOps. The
        // other five SideEffect ops (CascadeRead/Write, StreamRead/Write/
        // WritePacketHeader) are the core-side stream/cascade tenants 4/5 --
        // deliberately NOT claimed here. Under-claim is safe; over-claim is a
        // correctness bug.
        use crate::aie2::isa::SemanticOp;
        use crate::coverage::verdict::Verification;
        let regs = override_registry(Architecture::Aie2);
        let dma = regs
            .iter()
            .find(|u| u.id == "aie2.dma_ops.verified")
            .expect("DMA-ops Verified override present");
        assert!(matches!(dma.verdict.verification, Verification::Verified { .. }));
        let Claims::Nodes(nodes) = &dma.claims;
        let want_start = CoverageNode::Semantic { arch: Architecture::Aie2, op: SemanticOp::DmaStart };
        let want_wait = CoverageNode::Semantic { arch: Architecture::Aie2, op: SemanticOp::DmaWait };
        assert!(nodes.contains(&want_start), "claims DmaStart");
        assert!(nodes.contains(&want_wait), "claims DmaWait");
        assert_eq!(
            nodes.len(),
            2,
            "claims DmaStart + DmaWait ONLY -- not the whole SideEffect category (cascade/stream are tenants 4/5)"
        );
        assert!(
            dma.shadows_derived.is_some(),
            "Verified pulls DMA ops off their DocSpecified/Unverified default"
        );
        assert!(dma.shared_from.is_none(), "earned on AIE2 silicon directly, not shared");
    }

    #[test]
    fn override_registry_has_vector_ops_verified() {
        // #126/#127 axis-1: the vector differential fuzzer (#112/#114) silicon-matched
        // 218/218 keys; an empirical executed-op audit over the silicon-matched
        // corpus (docs/superpowers/findings/2026-06-12-vector-semanticop-empirical-audit.md)
        // shows exactly these 22 Vector-category SemanticOps dispatched in
        // currently-replay-confirmed matched runs. #127 re-fuzzed pack8/unpack16
        // on real NPU1 with pool-bearing (replayable) seeds, adding Pack/Unpack.
        // The never-executed variants are deliberately NOT claimed.
        use crate::aie2::isa::SemanticOp::*;
        use crate::coverage::verdict::Verification;
        let regs = override_registry(Architecture::Aie2);
        let vec_unit = regs
            .iter()
            .find(|u| u.id == "aie2.vector_ops.verified")
            .expect("vector-ops Verified override present");
        assert!(matches!(vec_unit.verdict.verification, Verification::Verified { .. }));
        let Claims::Nodes(nodes) = &vec_unit.claims;
        let claimed: std::collections::HashSet<_> = nodes
            .iter()
            .map(|n| match n {
                CoverageNode::Semantic { op, .. } => op.clone(),
                other => panic!("vector override claims a non-semantic node: {other:?}"),
            })
            .collect();
        let expected: std::collections::HashSet<_> = [
            Mac,
            MatMul,
            Srs,
            Ups,
            Shuffle,
            Convert,
            VectorBroadcast,
            VectorExtract,
            VectorInsert,
            VectorSelect,
            VectorClear,
            MaxDiffLt,
            MaxLt,
            MinGe,
            AbsGtz,
            NegGtz,
            NegLtz,
            Accumulate,
            AccumSub,
            Align,
            Pack,
            Unpack,
        ]
        .into_iter()
        .collect();
        assert_eq!(claimed, expected, "claims exactly the 22 empirically-verified Vector ops");
        // Guard the never-executed ops are NOT in the claim.
        for not_claimed in [Min, Max, MatMulSub, VectorPush] {
            assert!(!claimed.contains(&not_claimed), "{not_claimed:?} must NOT be claimed (perishable)");
        }
        assert!(vec_unit.shadows_derived.is_some());
        assert!(vec_unit.shared_from.is_none());
    }

    #[test]
    fn capability_spine_seeded_for_aie2() {
        let spine = capability_spine();
        assert!(spine.iter().any(|d| d.id == "dma"));
        assert!(spine.iter().any(|d| d.id == "locks"));
        assert!(spine.iter().any(|d| d.id == "stream_switch"));
        // Every seeded domain applies to AIE2 (the only wired arch today).
        assert!(spine.iter().all(|d| d.applies_to(Architecture::Aie2)));
        // Spine extended to 20 (spec Section 2 hybrid decision).
        assert_eq!(spine.len(), 20);
        for id in ["control_packets", "clock_control", "tile_isolation", "binary_load"] {
            assert!(spine.iter().any(|d| d.id == id), "missing new domain {id}");
        }
        // Every domain carries non-empty source_ref.
        assert!(spine.iter().all(|d| !d.source_ref.is_empty()));
        // src_locations may be empty only for OOS/MISSING states (Stub/Partial/Accepted).
        // Domains with Modeled{Full} + Verified must name where they live in src/.
        // This mirrors what enforce_coverage block-4 gates on (spec Section 2 N2).
        use crate::coverage::verdict::{Completeness, Verification};
        for d in &spine {
            let oos_or_missing = match &d.verdict.verification {
                Verification::Accepted { .. } => true,
                Verification::Modeled { completeness } => {
                    matches!(completeness, Completeness::Stub | Completeness::Partial { .. })
                }
                Verification::NotApplicable | Verification::Verified { .. } | Verification::Unverified => {
                    false
                }
            };
            if !oos_or_missing {
                assert!(
                    !d.src_locations.is_empty(),
                    "domain '{}' is not OOS/MISSING but has empty src_locations",
                    d.id
                );
            }
        }
    }

    #[test]
    fn capability_domain_arch_applicability_is_explicit() {
        let dma = capability_spine().into_iter().find(|d| d.id == "dma").unwrap();
        assert!(dma.applies_to(Architecture::Aie2));
    }

    #[test]
    fn capability_spine_matches_the_leaf_id_list() {
        // The leaf list (build.rs-visible) and the rich spine MUST agree --
        // they are the same source of truth, just two views (spec Section 6
        // one location; Plan 2 cycle note).
        use crate::coverage::spine_ids::SPINE_DOMAIN_IDS;
        let rich: Vec<String> = capability_spine().into_iter().map(|d| d.id).collect();
        let leaf: Vec<String> = SPINE_DOMAIN_IDS.iter().map(|s| s.to_string()).collect();
        assert_eq!(rich, leaf);
    }
}
