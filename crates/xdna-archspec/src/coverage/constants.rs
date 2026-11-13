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
    const HOP_SLOPE: &str = "NPU1 R4-D3 egress race (tools/experiments/r4d3_*.py): a stalled consumer 0/1/2/3 hops away \
        absorbs 26/34/42/50 words from a core source and 48/56/64/72 (+ granule phase) from a DMA source, 8 per hop = \
        AM020 ch.2:92's 8-deep ext->ext crossing (slave 4 + master 4); the emulator reproduces all 25 route x phase \
        points exactly (docs/fidelity-gaps/dma-stream-resources.md)";
    const RTT_SLOPE: &str = "NPU1 core-to-core round trip 32/40/48 cycles at 1/2/3 hops, r1 == r2 \
        (tools/gen_stream_rtt_probe.py): +4 cycles one way per hop = one ext->ext crossing, nothing for the wire; \
        the emulator's slope matches (its intercept is 8 cycles high, docs/fidelity-gaps/dma-stream-resources.md)";
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
        measured("dma.shim_per_task_overhead_mm2s_cycles", SHIM_2C, "Fitted 325, re-read 2026-11-07 as the shim read \
            latency (dma.shim_mm2s_read_latency_cycles); 0 so it is not counted twice."),
        measured("dma.shim_per_task_overhead_s2mm_cycles", SHIM_2C, "Fitted."),
        measured("dma.shim_mm2s_read_latency_cycles", "NPU1 shim-inflight probe 2026-11-07: back-pressured shim MM2S \
            FINISHED 308-722 cycles after the release that issues its final read; 325 = the 2026-05-27 per-task MM2S \
            overhead moved between issue and stream (free-stream timings unchanged)", "Fitted, low end of the HW spread; \
            DDR latency under load is not modelled."),
        measured("dma.shim_mm2s_read_window_bytes", "NPU1 shim-inflight tile-size sweep 2026-11-07 (128 B..4 KiB tiles): \
            reads-to-core-memory span at the final read in [3358, 3584), less the ~164 B stream path", "Per channel; \
            sharing across a shim tile's channels unprobed."),
        measured("dma.shim_warmup_decay_mm2s_permille", "NPU1 K=8 chain durations 1739/804/497/422 fit cold_start * 0.310^i \
            (xdna-emu-priv: research/findings/2026-05-30-phase-2d-warmup-transient-and-gap0-mechanism.md)", "Fitted."),
        measured("dma.shim_warmup_decay_s2mm_permille", "Same campaign: no measurable S2MM tail (excess at i=1 within noise)",
            "A measured zero, not a placeholder."),
        unsourced("dma.memtile_first_bd_startup_cycles", "Placeholder 0 (inert). The cold-start gap it was meant for was later \
            root-caused to the CORE_CONTROL reset, so it may never be needed."),
        unsourced("dma.compute_first_bd_startup_cycles", "Placeholder 0 (inert), as memtile_first_bd_startup_cycles."),
        measured("dma.s2mm_ingress_fifo_depth", "NPU1 S2MM_0_MEMORY_BACKPRESSURE opens 15 cycles into every steady \
            lock-stall window (39 windows, zero variance, xdna-emu-priv: research/findings/2026-07-14-dma-memory-pressure-\
            event-semantics.md; 15 again on the 2026-11-04 rebuild, EMU == HW)", "The FIFO the event watches; the ingress \
            total (16, DMA:0 master port count) is this plus the skid stage."),
        measured("dma.s2mm_ingress_skid_depth", "NPU1 port probe 2026-11-11 (tools/gen_path_onset_probe.py ports, data \
            build/experiments/path-capacity/onset-ports): the DMA:0 master passes 16 words into a held S2MM (EMU was 18) \
            and the source's slave port 22, so the switch holds 6 (AM020)", "16 in all less the 15 the backpressure \
            onset pins; which structure holds the word is not separated."),
        measured("dma.shim_s2mm_ingress_depth", "NPU1 shim absorb sweep (tools/gen_dma_queue_probe.py, absorb family): \
            MM2S 1 -> S2MM 1 on the shim, S2MM parked on a lock 76 words absorbed, 8 with no task, 3 runs identical",
            "A running (unparked) shim S2MM is not separated from a parked one; length 75 is a silicon oddity \
            not modelled."),
        measured("dma.mm2s_egress_fifo_depth", "NPU1 R4-D3 egress race (20 route x phase points, zero \
            spread, tools/experiments/r4d3_egress_race.py) pins the route capacity; the onset length sweep 2026-11-11 \
            (gen_path_onset_probe.py prod_own_nN, data path-capacity/onset-len) puts the MM2S stall boundary at 23 words \
            with the hand-off word inside the MM2S; DmaTiming field doc", "Compute tiles (memtile: \
            memtile_mm2s_egress_fifo_depth). The races pin staging + hand-off + downstream, the boundary pins the split."),
        measured("dma.memtile_mm2s_egress_fifo_depth", "NPU1 memtile start-phase data race 2026-11-10 \
            (tools/gen_pipeline_fill_probe.py race *_loop_pK): 52/51/54/53 vs compute 48/47/50/49, two runs identical; \
            31 with the hand-off word inside the MM2S since 2026-11-11 (onset-len)",
            "Upstream of the stream port (backpressure onset, gen_path_onset_probe.py); deeper staging vs a granule \
            in flight not separated."),
        measured("dma.memtile_mm2s_fetch_latency_cycles", "NPU1 backpressure onset 2026-11-10 \
            (tools/gen_path_onset_probe.py): every memtile-sourced edge one cycle after the compute-sourced one, two runs identical",
            "Seen at task start; a fetch after a mid-transfer stall is assumed to pay it too."),
        measured("dma.mm2s_read_to_offer_cycles", "NPU1 2026-11-11 (tools/gen_path_onset_probe.py prod_own_nN / mem_own_nN, \
            path-capacity/onset-len + onset-long, two runs identical): first word offered at 11 + min(N, 4) while the BD's \
            last read lands at N + 8",
            "Compute value; the memtile pays memtile_mm2s_fetch_latency_cycles on top."),
        unit("dma.mm2s_word_clock_start_cycles", HardwareObserved, Verification::Unverified,
            "Fitted. NPU1 2026-11 (tools/gen_path_onset_probe.py prod_own_nN, two runs identical): FINISHED_BD = \
            START_TASK + N + 8; word 0 at + 7 follows from placing the access after the last word and the finish \
            one cycle later, a placement that is not measured on the host path. Compute tiles, one value for every N."),
        unit("dma.mm2s_word_clock_lead_granules", HardwareObserved, Verification::Unverified,
            "Fitted (model M1g: 4 granules, a partial granule takes a slot, a slot frees when the access is \
            presented). NPU1 2026-11 (tools/gen_dma_lock_probe.py sync and chain families): a long-parked channel \
            holds 4 granules at its grant whatever their size (a 2-word BD before a 14..17-word BD puts the lock \
            request step where a 4-word one does); granules 0..3 at grant + 2..5, granule 4 at + 7, then every 4."),
        unit("dma.mm2s_bd_switch_access_gap_cycles", HardwareObserved, Verification::Unverified,
            "Fitted. NPU1 2026-11 (tools/gen_dma_lock_probe.py sync family): a lock-free 5..8-word BD behind a parked \
            4-word BD finishes 3 cycles after it, not 2. Compute tiles."),
        unit("dma.mm2s_access_to_offer_cycles", HardwareObserved, Verification::Unverified,
            "Inferred. Neither side is a direct observation: the host-path access (7 + min(N, 4)) is the model's \
            placement, and the lock-path offer (grant + 6) is the first word at the receiving DMA (grant + 9, \
            NPU1 2026-11, tools/gen_dma_lock_probe.py) less a route of 3 that is not separated. The host first \
            offer 11 + min(N, 4) (tools/gen_path_onset_probe.py) is measured. Compute tiles."),
        measured("dma.mm2s_access_to_finish_cycles", "NPU1 2026-11 (tools/gen_dma_lock_probe.py): last access grant + 51, \
            FINISHED_BD + 52, + 53 when a core access wins the bank at + 51",
            "Paired with the lock-to-access latency through the conflict-stamp-equals-access assumption."),
        measured("dma.mm2s_lock_to_access_cycles", "NPU1 2026-11 (tools/gen_dma_lock_probe.py, first-BD and chained): \
            first access grant + 2, none at + 0 / + 1; prewake dummy BD FINISHED grant + 3",
            "Compute tiles; the grant itself (release + 2 or + 3 on silicon) is a separate latency. Assumes a conflict \
            event is stamped on the access cycle."),
        unit("dma.mm2s_lock_request_after_grant_cycles", HardwareObserved, Verification::Unverified,
            "Fitted (lock request and grant law). Q >= the previous lock BD's grant + 3; each value off by one breaks most chain rows. NPU1 2026-11 (tools/gen_dma_lock_probe.py sync and chain families, every design with at most four MM2S BDs: 2348 of 2348 rows, about 2102 of them the rows the rule was fitted to and 246 predicted before they were measured)."),
        unit("dma.mm2s_lock_request_trigger_words", HardwareObserved, Verification::Unverified,
            "Fitted (lock request and grant law). The request follows the sequencing of the previous BD's third-to-last word; 2 or 4 differ on about 50 rows only, the weakest-pinned constant with the release slot. NPU1 2026-11 (tools/gen_dma_lock_probe.py sync and chain families, every design with at most four MM2S BDs: 2348 of 2348 rows, about 2102 of them the rows the rule was fitted to and 246 predicted before they were measured)."),
        unit("dma.mm2s_lock_request_trigger_latency_cycles", HardwareObserved, Verification::Unverified,
            "Fitted (lock request and grant law). Q >= trigger word + 3. NPU1 2026-11 (tools/gen_dma_lock_probe.py sync and chain families, every design with at most four MM2S BDs: 2348 of 2348 rows, about 2102 of them the rows the rule was fitted to and 246 predicted before they were measured)."),
        unit("dma.mm2s_lock_release_after_finish_cycles", HardwareObserved, Verification::Unverified,
            "Fitted (lock request and grant law). One cycle, the previous lock BD's FINISHED_BD + 4, neither requests nor grants; only 4-word predecessors show it, read as that BD's own lock release (inferred). NPU1 2026-11 (tools/gen_dma_lock_probe.py sync and chain families, every design with at most four MM2S BDs: 2348 of 2348 rows, about 2102 of them the rows the rule was fitted to and 246 predicted before they were measured)."),
        unit("dma.mm2s_lock_grant_period_cycles", HardwareObserved, Verification::Unverified,
            "Fitted (lock request and grant law). From its request a channel's acquire is grantable in 2 of every 3 cycles; what blocks the third is not known. NPU1 2026-11 (tools/gen_dma_lock_probe.py sync and chain families, every design with at most four MM2S BDs: 2348 of 2348 rows, about 2102 of them the rows the rule was fitted to and 246 predicted before they were measured)."),
        unit("dma.mm2s_lock_grant_blocked_phase", HardwareObserved, Verification::Unverified,
            "Fitted (lock request and grant law). The blocked cycle is request + 2 mod 3. NPU1 2026-11 (tools/gen_dma_lock_probe.py sync and chain families, every design with at most four MM2S BDs: 2348 of 2348 rows, about 2102 of them the rows the rule was fitted to and 246 predicted before they were measured)."),
        measured("dma.mm2s_lock_grant_visible_cycles", "NPU1 2026-11 (tools/gen_dma_lock_probe.py chain family): LOCK_SEL0_ACQ_GE \
            fires exactly once per acquire, one cycle before DMA_MM2S_0_STALLED_LOCK falls, in every waiting row, and also \
            when the lock was free; release to grant is 2 or 3 for core and S2MM releasers",
            "Compute MM2S. The split of release + 2 into a release reaching the requester a cycle late and this cycle is \
            the emulator's; only the sum and ACQ_GE at grant - 1 are observed."),
        unit("dma.mm2s_lock_stall_after_sequencing_cycles", HardwareObserved, Verification::Unverified,
            "Fitted (lock request and grant law). STALLED_LOCK rises no earlier than the BD's first sequenced word + 2. NPU1 2026-11 (tools/gen_dma_lock_probe.py sync and chain families, every design with at most four MM2S BDs: 2348 of 2348 rows, about 2102 of them the rows the rule was fitted to and 246 predicted before they were measured)."),
        unit("dma.mm2s_lock_stall_before_request_cycles", HardwareObserved, Verification::Unverified,
            "Fitted (lock request and grant law). STALLED_LOCK rises no earlier than the request - 1 (a free lock shows one cycle). NPU1 2026-11 (tools/gen_dma_lock_probe.py sync and chain families, every design with at most four MM2S BDs: 2348 of 2348 rows, about 2102 of them the rows the rule was fitted to and 246 predicted before they were measured)."),
        unit("dma.mm2s_lock_stall_before_last_access_cycles", HardwareObserved, Verification::Unverified,
            "Fitted (lock request and grant law). STALLED_LOCK rises no earlier than the previous BD's last access - 1. NPU1 2026-11 (tools/gen_dma_lock_probe.py sync and chain families, every design with at most four MM2S BDs: 2348 of 2348 rows, about 2102 of them the rows the rule was fitted to and 246 predicted before they were measured)."),
        measured("dma.bank_arbiter_period", "NPU1 2026-11 (tools/gen_dma_lock_probe.py arb and ring cases, 222 rows, \
            every row as predicted before the run): a core camping on a bank lets a waiting DMA channel in once in every \
            5 served cycles, never after more than 4 refusals; the producer probe's dense-store captures (45 transfers) \
            fit at period 5 and at no other",
            "Compute tiles, one core port against DMA channels. Why 5 is not known; neighbour cores, wide accesses and \
            memtile banks are unmeasured."),
        unit("dma.bank_arbiter_dma_residue", HardwareObserved, Verification::Unverified,
            "Fitted: only the bank's served count at a probe's window minus the residue is observable. NPU1 2026-11 \
            (tools/gen_dma_lock_probe.py): with the count from reset (configuration-written 16-byte lines plus the core \
            prologue's stores) banks 0 and 1 of one design, whose histories differ by two stores, need the same value, \
            and all four DMA channels share it; the image-line unit is measured by adding initialised words."),
        // -- stream_switch --
        unit("stream_switch.local_slave_fifo_depth", DocSpecified, Verification::Verified { evidence: HOP_SLOPE.into() },
            "AM020 ch.2 'local slave ports ... 4-deep FIFO'. Also sizes EXTERNAL slaves (no separate field)."),
        manual("stream_switch.local_master_fifo_depth", "AM020 ch.2 'local master ports ... 2-deep FIFO'."),
        unit("stream_switch.external_master_fifo_depth", DocSpecified, Verification::Verified { evidence: HOP_SLOPE.into() },
            "AM020 ch.2 'external ports ... 4-deep FIFO'; with the slave's 4, the 8 words silicon adds per hop."),
        manual("stream_switch.switch_fifo_depth", "AM020 ch.2:74 'the switch has one FIFO that is 16-deep'. A route through \
            FIFO:0 absorbs +21 words on silicon (core and DMA sources, r4d3 rig and gen_stream_absorb_probe.py): this 16, \
            the FIFO slave port's 4 and the 1-word FIFO output register; modelled, EMU == HW."),
        manual("stream_switch.slave_port_latency", "AM020 ch.2:78/80 'slave ports are 2-cycle latency'. Splits a \
            crossing's latency between its slave and master slots; only the sums are observed."),
        manual("stream_switch.local_to_local_latency", "AM020 ch.2 '3-cycle latency and 6-deep FIFO'; the 6-deep half has no field."),
        manual("stream_switch.local_to_external_latency", "AM020 ch.2 '4-cycle, 8-deep'. Also used as the neighbour-tile \
            data-memory latency (interpreter/timing/memory.rs), which AM020 gives no basis for."),
        unit("stream_switch.external_to_external_latency", DocSpecified, Verification::Verified { evidence: RTT_SLOPE.into() },
            "AM020 ch.2 '4-cycle, 8-deep'. Latency is time inside the two FIFOs, not storage on top of them."),
        manual("stream_switch.external_to_local_latency", "AM020 ch.2. UNREAD: the ext->local path uses the local->local latency \
            (equal value, 3, hides it)."),
        measured("stream_switch.inter_tile_hop_latency", RTT_SLOPE, "0: the wire adds no latency and no storage. \
            The value 2 it replaces came from one memtile->compute capture that two mechanisms fit."),
        unsourced("stream_switch.packet_arbitration_overhead", "Original estimate; AM020 ch.2 excludes it from the latency table \
            and no capture exists."),
        // -- broadcast --
        measured("broadcast.per_hop_horizontal", SKEW, "Horizontal isotropy assumed."),
        measured("broadcast.per_hop_vertical", SKEW, "Vertical isotropy assumed (blocked on the cold-start fill gap)."),
        measured("broadcast.intra_tile_core_offset", SKEW, "Gauge-fixed at 0: only the core-memory contrast is observable."),
        unit("broadcast.intra_tile_mem_offset", HardwareObserved, Verification::Unverified,
            "Compute memory module against the core (core is the gauge, 0). Started at 2, the raw add_one core-vs-memory event \
            offset, which also held that event pair's wall-clock separation as then placed. 1 since the core memory-access and \
            lock-release placement was corrected: derived from cross-unit event pairs, not from a skew sweep. On silicon \
            LOCK_0_REL (memory unit) minus INSTR_EVENT_0 (core unit) is 5 and MEMORY_STALL (core unit) minus CONFLICT_DM_BANK \
            (memory unit) is 1; each read one cycle off with 2. The median memory-minus-core offset over the suite's qualifying \
            test-tiles (tools/mem_core_offset.py) went from -1 to 0. Few tiles qualify and the rest are host-timing outliers."),
        unsourced("broadcast.intra_tile_memtile_offset", "The value the memory-module offset carried before it was re-measured (2). A memtile has no core module: nothing measures it against a core, so it is held, not calibrated."),
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
