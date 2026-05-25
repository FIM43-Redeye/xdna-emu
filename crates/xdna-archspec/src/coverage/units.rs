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

/// The explicit override registry, per arch. Empty in Phase 1 (spec Section 5):
/// coarse category defaults carry the bootstrap; Phase 2 adds entries here.
pub fn override_registry(_arch: Architecture) -> Vec<BehavioralUnit> {
    Vec::new()
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
          "BDs per tile-type 16/48/16 (memtile 48 not 64 -- aie-rt xaiemlgbl_reginit.c), channels 2/6/2, address dims 3/4/3; n-d addressing, padding, lock coupling, packet header, compression. Repeat / out-of-order BD execution: verify. Stream-side back-pressure not modeled: S2MM channels do not consume a stream-data-available signal before advancing, and MM2S channels do not consume a stream-space-available signal -- both currently advance without checking the stream switch FIFO state. This is an inter-subsystem gap surfaced 2026-05-24 by the objectfifo / dynamic_object_fifo bridge wedges (docs/superpowers/findings/2026-05-24-bridge-sweep-objectfifo-wedges.md).",
          Modeled { completeness: Partial { missing: "S2MM/MM2S stream back-pressure (TVALID/TREADY handshake with stream_switch FIFOs); repeat / out-of-order BD execution verification".to_string() } },
          Some("Own verdict reflects the actually-modeled subsystem; the rolled-up verdict is the pessimistic Phase-1 coarse category default (SideEffect=DocSpecified/Unverified), a bootstrap floor, not a ceiling. Per-op refinement that would raise the rollup is Phase-2 override work (spec Section 3).")),
        d("locks", "aie-rt locks/, AM025",
          &["src/device/tile/locks.rs"],
          "Counts per tile-type 16/64/16 (aie-rt xaiemlgbl_reginit.c; the 192 in AieMlMemTileDmaMod.NumLocks is the cross-tile reference range, not slot count). acquire/release/get/set, semaphore semantics, round-robin arbiter.",
          Modeled { completeness: Full }, None),
        d("stream_switch", "aie-rt stream_switch/, AM025 (160/119/149 reg)",
          &["src/device/stream_switch/"],
          "Circuit + packet, FIFOs, port events, packet-header matching. Parse-time routing reconstruction is a binary_load concern (spec Appendix N1). Per-port data-available / space-available signals are tracked internally but not exposed to DMA consumers/producers as a back-pressure handshake (paired gap on the dma side -- see dma narrative).",
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
          "Direct NoC control / AIE_AXIMM_Config / NoC fabric latency-arbitration are not modeled (NoC fudged; impacts cycle-accuracy more than functional correctness; cycle-accuracy-mission.md tracks calibration). AIE_AXIMM_Config.SLVERR_Block (unmapped-access SLVERR suppression) is the tracked control_packets-fidelity dependency gated here: control-packet SLVERR is modeled for the reset-default (SLVERR-enabled) configuration every observed binary uses; honoring SLVERR_Block needs this NoC register plumbed. NPI privileged register access is driver-side privilege -- emulator gives unrestricted access: accepted out of scope. No emulator src for the unmodeled NoC surface.",
          Modeled { completeness: Stub }, None),
        d("shim_mux", "aie-rt, AM025 (shim Mux/Demux 2 reg); aie-rt pl/ (PL Interface)",
          &["src/device/stream_switch/"],
          "Shim master/slave NoC-facing mux/demux. PL Interface (Upsizer/Downsizer) is Versal-FPGA stream-width adaptation -- NPU1 exposes no programmable PL: accepted out of scope.",
          Modeled { completeness: Full }, None),
        d("control_packets", "AM025 Control_Packet_Handler_Status (0x3FF30/0xB0F30); aie-rt xaiegbl_params.h:7761; XRT host protocol",
          &["src/device/control_packets/", "src/device/tile/mod.rs", "src/device/state/ctrl_access.rs", "src/interpreter/engine/coordinator.rs", "src/npu/"],
          "Control-packet headers, reassembly, register read/write effects, response packets MODELED. NPU host instruction stream (WRITE32/BLOCKWRITE/BLOCKSET/MASKWRITE/MASKPOLL/CONFIG_SHIMDMA_*/DDR_PATCH) MODELED. All four Tile_Control_Packet_Handler_Status sticky bits have faithful detecting paths: First/Second-header parity and Tlast via the reassembler, SLVERR_On_Access via undecoded-address decode at the dispatch boundary, all with poll-only sticky-continue semantics (aie-rt/AM025: latch + continue, no interrupt, no abort). The AIE_AXIMM_Config.SLVERR_Block config-suppression refinement is a tracked NoC-gated goal -- see the noc domain. Keystone subsystem.",
          Modeled { completeness: Full }, None),
        d("clock_control", "AM025 Module_Clock_Control / Column_Clock_Control / Reset_Control_1 / AIE_Tile_Column_Reset; aie-rt pm/xaie_clock.c",
          &["src/device/clock_control/", "src/device/state/dispatch.rs", "src/device/array/routing.rs", "src/device/array/dma_ops.rs", "src/interpreter/engine/coordinator.rs"],
          "Three-tier clock gating MODELED: Column_Clock_Control, per-module Module_Clock_Control (compute/memtile/shim_0/shim_1 layouts decoded from regdb), and adaptive-gate infrastructure (idle counters + abort_period; is_adaptive_dma_engaged/is_adaptive_ss_engaged queries). Adaptive counters tick once per data-movement cycle, gated by per-module clocks (silicon-accurate); column or module re-ungate transitions reset the relevant counter. Silicon-accurate boot default: all tiles gated, ungate-via-CDO is the documented path; ungate_all() test helper exercises the real register-write path. Step loop consults gates before DMA/StreamSwitch/Core stepping; gated-tile non-clock-control writes serve-and-warn (dedup per (col,row,offset); XDNA_EMU_WARN_GATED_ACCESS=0 silences). PARTIAL because: (1) Reset_Control_1 / AIE_Tile_Column_Reset (partition teardown) not modeled; (2) no power/cycle-count effect modeled. Privilege enforcement explicitly out of scope (no privilege model in the emulator yet). See docs/superpowers/specs/2026-05-24-clock-control-design.md.",
          Modeled { completeness: Partial { missing: "Reset_Control_1 / AIE_Tile_Column_Reset partition teardown; power/cycle-count effects".to_string() } }, None),
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
    fn override_registry_is_empty_at_phase1() {
        // Phase 1 ships green on coarse category defaults; overrides are
        // Phase 2 refinement work (spec Section 5).
        assert!(override_registry(Architecture::Aie2).is_empty());
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
