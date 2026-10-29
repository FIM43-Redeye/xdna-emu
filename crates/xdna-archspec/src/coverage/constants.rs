//! Per-constant provenance for the timing model: the numeric half of the
//! derivation backlog, beside the per-op semantic ledger.
//!
//! `TimingModel` (types.rs) holds the depths, latencies and rates that no
//! machine-readable toolchain source supplies. Each field gets exactly one
//! `ConstantUnit` here, keyed by its serde path (`dma.mm2s_egress_fifo_depth`),
//! carrying the same two-axis `Verdict` the semantic ledger uses. The
//! completeness test walks the serialized model, so a new field without a
//! verdict fails the build instead of joining the timing model unlabelled.
//!
//! Verdicts are per architecture: silicon evidence for NPU1 says nothing about
//! AIE2P, so an architecture's entries are written for that architecture.

use crate::coverage::verdict::{Provenance, Verdict, Verification};
use crate::types::{Architecture, TimingModel};

/// One timing-model field and where its value comes from.
#[derive(Debug, Clone)]
pub struct ConstantUnit {
    /// Dotted serde path into `TimingModel`, e.g. `stream_switch.local_slave_fifo_depth`.
    pub path: &'static str,
    pub verdict: Verdict,
    /// Source or rationale for an unverified value, plus reader/caveat facts a
    /// derivation needs (an unread field, a placeholder, a coupling).
    pub note: &'static str,
}

/// The registry for `arch`; empty for an architecture without a timing model.
pub fn timing_constant_registry(arch: Architecture) -> Vec<ConstantUnit> {
    match arch {
        Architecture::Aie2 => aie2_timing_constants(),
        _ => Vec::new(),
    }
}

/// Every numeric/boolean leaf of `t` as (dotted path, value), in serialization
/// order. The attribution block (`source`) is metadata, not a constant.
pub fn timing_leaves(t: &TimingModel) -> Vec<(String, serde_json::Value)> {
    fn walk(prefix: &str, v: &serde_json::Value, out: &mut Vec<(String, serde_json::Value)>) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, child) in map {
                    let path = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    walk(&path, child, out);
                }
            }
            leaf => out.push((prefix.to_string(), leaf.clone())),
        }
    }
    let mut out = Vec::new();
    let mut v = serde_json::to_value(t).expect("TimingModel serializes");
    if let serde_json::Value::Object(map) = &mut v {
        map.remove("source");
    }
    walk("", &v, &mut out);
    out
}

fn unit(
    path: &'static str,
    provenance: Provenance,
    verification: Verification,
    note: &'static str,
) -> ConstantUnit {
    ConstantUnit { path, verdict: Verdict { provenance, verification }, note }
}

fn measured(path: &'static str, evidence: &str, note: &'static str) -> ConstantUnit {
    unit(path, Provenance::HardwareObserved, Verification::Verified { evidence: evidence.into() }, note)
}

fn manual(path: &'static str, note: &'static str) -> ConstantUnit {
    unit(path, Provenance::DocSpecified, Verification::Unverified, note)
}

fn unsourced(path: &'static str, note: &'static str) -> ConstantUnit {
    unit(path, Provenance::Unspecified, Verification::Unverified, note)
}

/// AIE2 (NPU1 Phoenix) entries. Classified 2026-10-29 from the field docs,
/// the model_builder.rs comments, the priv findings they cite, and a grep of
/// every reader; the evidence names the capture, the note what is still open.
fn aie2_timing_constants() -> Vec<ConstantUnit> {
    use Provenance::*;
    const SHIM_2C: &str = "NPU1 trace campaign N=50 on _diag_shim_chain_sweep K={1,2,4,8}, fitted against K=1 \
        (cold) and K>=4 (steady) together (xdna-emu-priv: research/findings/\
        2026-05-27-phase-2c-dispatch-overhead-recalibration.md, 2026-05-27-dispatch-overhead-multirun-structural-variance.md)";
    const SKEW: &str = "NPU1 R3b perf-counter skew sweep, N=20, drift range 0 \
        (crates/xdna-archspec/data/skew_constants.json; xdna-emu-priv: research/findings/2026-07-01-r3b-pc-silicon-bringup.md, \
        2026-07-02-sp5c-phase0-identifiability.md)";
    const HOP_SLOPE: &str = "NPU1 R4-D3 egress race (tools/experiments/r4d3_*.py): a stalled DMA consumer 0/1/2/3 hops away \
        absorbs 48/56/64/72 words on silicon (8 per hop, AM020 ch.2 8-deep external hops) vs 29/43/57/71 in the emulator \
        (14 per hop). The emulator's per-hop budget is ext->ext latency 4 + external master FIFO 4 + crossing depth \
        (slave FIFO 4 + hop latency 2); silicon contradicts the sum, the split between the terms is not yet separated \
        (docs/fidelity-gaps/dma-stream-resources.md, egress staging row)";
    vec![
        // -- lock --
        unsourced("lock.acquire_latency", "Cited to AM020 ch.2 'a new request every clock cycle', but that sentence is the \
            memory-bank arbitration bullet; AM020's lock text gives no cycle count. Live: core lock timing (interpreter/timing/latency.rs)."),
        unsourced("lock.release_latency", "No source. Effectively unread: lock-release timing uses the acquire latency."),
        unsourced("lock.retry_interval", "No source. Read only by DMA lock timing, which production never enables (with_lock_timing is test-only)."),
        // -- dma --
        unsourced("dma.bd_setup_cycles", "Original estimate ('BD register reads and state-machine setup'); never isolated on silicon."),
        unsourced("dma.channel_start_cycles", "Original estimate ('arbitration and initial memory request'); plausible order, never isolated."),
        unit("dma.words_per_cycle", ToolchainDerived, Verification::NotApplicable,
            "aie-rt xaiemlgbl_params.h MEMORY_MODULE_DATAMEMORY_WIDTH 128 (hand-typed, not extracted). Stream-fed DMA is \
            bound at 1 word/cycle, so this rarely bites."),
        measured("dma.shim_words_per_cycle", "NPU1 shim throughput sweep, 12 BD sizes, R^2 0.996/1.000 \
            (xdna-emu-priv: research/findings/2026-05-25-shim-throughput-1-word-per-cycle.md)", ""),
        measured("dma.stream_words_per_cycle", "AM020 ch.2 32-bit AXI4-Stream; NPU1 memtile DMA is stream-bound at 1 word/cycle, \
            strided == contiguous (xdna-emu-priv: research/findings/2026-07-15-memtile-bank-access-width.md)", ""),
        manual("dma.memory_latency_cycles", "AM020 ch.4 core data-memory latency 5, applied to the DMA by analogy; not a DMA measurement."),
        unsourced("dma.lock_acquire_cycles", "No source; the real grant also passes the bank/lock arbiter."),
        unsourced("dma.lock_release_cycles", "No source. Unread in production (only an uncalled estimate function)."),
        unsourced("dma.bd_chain_cycles", "No source. Read only on the repeat-count restart path; linked chains take the prefetch bypass."),
        unsourced("dma.host_memory_latency_cycles", "Placeholder 0: folded into the per-direction shim cold-start values."),
        measured("dma.shim_ddr_cold_start_mm2s_cycles", SHIM_2C, "Fitted, not first-principles; the DDR read path is bimodal on silicon."),
        measured("dma.shim_ddr_cold_start_s2mm_cycles", SHIM_2C, "Fitted."),
        measured("dma.shim_per_task_overhead_mm2s_cycles", SHIM_2C, "Fitted."),
        measured("dma.shim_per_task_overhead_s2mm_cycles", SHIM_2C, "Fitted."),
        measured("dma.shim_warmup_decay_mm2s_permille", "NPU1 K=8 chain durations 1739/804/497/422 fit cold_start * 0.310^i \
            (xdna-emu-priv: research/findings/2026-05-30-phase-2d-warmup-transient-and-gap0-mechanism.md)", "Fitted."),
        measured("dma.shim_warmup_decay_s2mm_permille", "Same campaign: no measurable S2MM tail (excess at i=1 within noise)",
            "A measured zero, not a placeholder."),
        unsourced("dma.memtile_first_bd_startup_cycles", "Placeholder 0 (inert). The cold-start gap it was meant for was later \
            root-caused to the CORE_CONTROL reset, so it may never be needed."),
        unsourced("dma.compute_first_bd_startup_cycles", "Placeholder 0 (inert), as memtile_first_bd_startup_cycles."),
        unit("dma.s2mm_ingress_fifo_depth", HardwareObserved, Verification::Unverified,
            "Silicon only lower-bounds it: memtile recv decodes [16,16,16,16] (>= 16). The exact value is not pinned (the \
            16-vs-28 capture was never taken), and backpressure onset is +16 in the emulator vs +15 on silicon."),
        measured("dma.mm2s_egress_fifo_depth", "NPU1 R4-D3 egress race, 20 route x phase points, zero spread \
            (tools/experiments/r4d3_egress_race.py; DmaTiming field doc)", "Compute tiles only; mem tiles inherit it unverified. \
            Assumes a stalled core holds one word (+/-1)."),
        // -- stream_switch --
        manual("stream_switch.local_slave_fifo_depth", "AM020 ch.2 'local slave ports ... 4-deep FIFO'. Also sizes EXTERNAL \
            slaves (no separate field) and the per-channel MM2S output FIFO; part of the contradicted per-hop sum."),
        manual("stream_switch.local_master_fifo_depth", "AM020 ch.2 'local master ports ... 2-deep FIFO'."),
        manual("stream_switch.external_master_fifo_depth", "AM020 ch.2 'external ports ... 4-deep FIFO' (does not say master or \
            slave); part of the contradicted per-hop sum."),
        manual("stream_switch.local_to_local_latency", "AM020 ch.2 '3-cycle latency and 6-deep FIFO'; the 6-deep half has no field."),
        manual("stream_switch.local_to_external_latency", "AM020 ch.2 '4-cycle, 8-deep'. Also used as the neighbour-tile \
            data-memory latency (interpreter/timing/memory.rs), which AM020 gives no basis for."),
        unit("stream_switch.external_to_external_latency", DocSpecified, Verification::Contradicted { evidence: HOP_SLOPE.into() },
            "AM020 ch.2 '4-cycle, 8-deep'. The switch backpressure budget counts this latency as buffer capacity."),
        manual("stream_switch.external_to_local_latency", "AM020 ch.2. UNREAD: the ext->local path uses the local->local latency \
            (equal value, 3, hides it)."),
        unit("stream_switch.inter_tile_hop_latency", HardwareObserved, Verification::Contradicted { evidence: HOP_SLOPE.into() },
            "Pinned from a single memtile->compute crossing (send headroom 14 = 8 + 6; xdna-emu-priv: research/findings/\
            2026-06-28-send-cadence-pipeline-latency-and-ingress-depth.md); two mechanisms fit that capture. Do not retune \
            without a discriminating multi-hop capture."),
        unsourced("stream_switch.packet_arbitration_overhead", "Original estimate; AM020 ch.2 excludes it from the latency table \
            and no capture exists."),
        // -- broadcast --
        measured("broadcast.per_hop_horizontal", SKEW, "Horizontal isotropy assumed."),
        measured("broadcast.per_hop_vertical", SKEW, "Vertical isotropy assumed (blocked on the cold-start fill gap)."),
        measured("broadcast.intra_tile_core_offset", SKEW, "Gauge-fixed at 0: only the core-memory contrast is observable."),
        measured("broadcast.intra_tile_mem_offset", SKEW, "The add_one core-vs-memory-module signature contrast."),
        measured("broadcast.calibrated", SKEW, "Flag: the constants above are a silicon calibration (SP-5c, reversible)."),
        // -- instruction --
        manual("instruction.data_memory_latency", "AM020 ch.4 '5-cycle latency of data memory'; agrees with llvm-aie \
            AIE2Schedule.td LoadLatency 5 (marked FIXME there)."),
        unsourced("instruction.branch_penalty", "Original estimate (~3 fetch stages); llvm-aie MispredictPenalty is 4 (FIXME). \
            The executor deliberately adds no branch penalty (the delay slots are the flush), so its live effect is unclear."),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::verdict::{Provenance, Verification};
    use crate::runtime::ARCHSPEC_MODELS;
    use std::collections::BTreeSet;

    fn npu1_timing() -> &'static TimingModel {
        ARCHSPEC_MODELS
            .get("npu1")
            .expect("npu1 model")
            .timing
            .as_ref()
            .expect("npu1 timing")
    }

    #[test]
    fn aie2_registry_covers_every_timing_field_exactly_once() {
        let fields: BTreeSet<String> = timing_leaves(npu1_timing()).into_iter().map(|(p, _)| p).collect();
        let reg = timing_constant_registry(Architecture::Aie2);
        let mut seen = BTreeSet::new();
        for u in &reg {
            assert!(seen.insert(u.path.to_string()), "{} registered twice", u.path);
        }
        let missing: Vec<_> = fields.difference(&seen).collect();
        let stale: Vec<_> = seen.difference(&fields).collect();
        assert!(missing.is_empty(), "timing fields without a provenance verdict: {missing:?}");
        assert!(stale.is_empty(), "registry entries naming no timing field: {stale:?}");
    }

    #[test]
    fn timing_leaves_skip_the_attribution_block() {
        assert!(timing_leaves(npu1_timing()).iter().all(|(p, _)| !p.starts_with("source")));
    }

    #[test]
    fn registry_verdicts_are_well_formed() {
        for u in timing_constant_registry(Architecture::Aie2) {
            let v = &u.verdict;
            assert!(
                !(v.provenance == Provenance::Unspecified
                    && matches!(v.verification, Verification::Verified { .. })),
                "{}: Unspecified is never Verified (verdict.rs invariant)",
                u.path
            );
            if let Verification::Verified { evidence } = &v.verification {
                assert!(!evidence.trim().is_empty(), "{}: Verified needs evidence", u.path);
            }
            if v.provenance == Provenance::ToolchainDerived {
                assert_eq!(v.verification, Verification::NotApplicable, "{}", u.path);
            }
        }
    }
}
