//! Kernarg-role classification from the NPU instruction stream (and
//! optionally the xclbin's CDO).
//!
//! `bridge-trace-runner` and similar host-side tools need to bind buffer
//! objects to kernarg slots in a way that matches what aiecc decided at
//! compile time. The xclbin's EMBEDDED_METADATA only carries generic
//! `boN` names; the semantic role of each slot (control-packet stream,
//! MM2S data input, S2MM data output) is baked into the NPU instruction
//! stream. This module walks the stream and recovers the mapping.
//!
//! ## Classification signals (three tiers)
//!
//! A kernarg is classified as **Ctrlpkt** if either:
//!
//! - **Tier 1** — the BD configuring its first transfer has `EnPkt = 1`
//!   in BD word 2 (bit 30). Reference: aie-rt
//!   `xaiemlgbl_reginit.c:713` (`AieMlShimDmaBdPktProp.EnPkt.Idx = 2`).
//!   This signal catches the "BD-level packet mode" pattern used by
//!   tests like `add_one_ctrl_packet`.
//!
//! - **Tier 2** — the arg_idx has ≥ `CTRLPKT_MIN_PATCHES` `DdrPatch`
//!   records targeting the same BD register, with monotonically
//!   increasing `arg_plus` at a consistent small stride. This is the
//!   compiled fingerprint of aiecc streaming a ctrlpkt blob byte-by-byte
//!   through a non-packet BD while the stream switch handles packet
//!   framing. Catches the `ctrl_packet_reconfig` pattern, where the
//!   shim BD has `EnPkt = 0` but the SS routes MM2S_0 as packet.
//!
//! Otherwise the arg is classified as **DataMm2s** or **DataS2mm** based
//! on which Task_Queue register (`0x1D214` / `0x1D21C` = MM2S 0/1;
//! `0x1D204` / `0x1D20C` = S2MM 0/1) the channel was started on after
//! the patch.
//!
//! An optional **Tier 3** cross-check is available via
//! [`classify_with_topology`]: the caller supplies a
//! [`StreamSwitchTopology`] reconstructed from the xclbin's CDO and we
//! confirm that at least one shim SS slave port in the packet-routing
//! range matches when any arg is classified as Ctrlpkt. Disagreement
//! surfaces as a warning-level log line rather than a classification
//! override, because the two signals are testing slightly different
//! things (run-time payload vs. init-time routing config) and we'd
//! rather flag the conflict for investigation than silently change an
//! answer.

use std::collections::{BTreeMap, HashMap};

use xdna_archspec::types::TileAddr;

use super::{NpuInstruction, NpuInstructionStream};
use crate::parser::stream_switch_topology::StreamSwitchTopology;

/// Kernarg role recovered from the instruction stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KernargRole {
    /// Non-packet BD started on an MM2S channel: host-to-NPU data.
    DataMm2s = 0,
    /// Non-packet BD started on an S2MM channel: NPU-to-host data.
    DataS2mm = 1,
    /// Packet-mode BD or packet-fingerprint DMA stream: control-packet
    /// input bound for tile control ports.
    Ctrlpkt = 2,
    /// BD config or channel direction could not be resolved.
    Unknown = 255,
}

/// Classification for one `arg_idx` referenced by the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernargClassification {
    pub arg_idx: u8,
    pub role: KernargRole,
    /// Full (column-qualified) BD source register address first observed
    /// for this arg, for diagnostics.
    pub bd_reg_addr: u32,
}

// ShimDMA BD layout (AIE-ML / NPU1). 16 BDs, 8 words each, 0x20 stride.
// Reference: aie-rt xaiemlgbl_params.h.
const SHIM_BD_BASE_LOCAL: u32 = 0x0001_D000;
const SHIM_BD_COUNT: u32 = 16;
const SHIM_BD_STRIDE: u32 = 0x20;

// Shim Task Queue register offsets (tile-local). Writing to these
// enqueues a BD on the corresponding channel. Reference:
// xaiemlgbl_params.h lines 18687-18805.
const SHIM_S2MM_0_TASK_QUEUE_LOCAL: u32 = 0x0001_D204;
const SHIM_S2MM_1_TASK_QUEUE_LOCAL: u32 = 0x0001_D20C;
const SHIM_MM2S_0_TASK_QUEUE_LOCAL: u32 = 0x0001_D214;
const SHIM_MM2S_1_TASK_QUEUE_LOCAL: u32 = 0x0001_D21C;

// BD word 2 EnPkt flag (bit 30). Reference: aie-rt xaiemlgbl_reginit.c:713.
const BD_WORD_2_ENPKT_BIT: u32 = 1 << 30;

// Task Queue START_BD_ID field (low 4 bits).
const TASK_QUEUE_BD_ID_MASK: u32 = 0x0000_000F;

// AIE-ML tile stride = 2 MiB; tile-local regs fit in low 21 bits.
const TILE_LOCAL_MASK: u32 = 0x001F_FFFF;

/// Minimum DdrPatch count on the same BD before the monotonic-stride
/// signature qualifies as a ctrlpkt stream. Two patches could plausibly
/// occur on a padded data transfer; three in arithmetic progression is
/// the compiled fingerprint of ctrlpkt content being streamed byte-wise.
const CTRLPKT_MIN_PATCHES: usize = 3;

/// Classify kernarg roles from a parsed instruction stream.
pub fn classify_kernargs(stream: &NpuInstructionStream) -> Vec<KernargClassification> {
    classify_inner(stream, None)
}

/// Same as [`classify_kernargs`] but also accepts a stream-switch
/// topology (reconstructed from the xclbin's CDO). The topology is used
/// as a cross-check: if any arg is classified Ctrlpkt but no shim SS
/// slave has packet_enable set in the topology, that disagreement is
/// logged (not fatal).
pub fn classify_with_topology(
    stream: &NpuInstructionStream,
    topology: &StreamSwitchTopology,
) -> Vec<KernargClassification> {
    classify_inner(stream, Some(topology))
}

/// Convenience: parse raw instruction bytes, then classify.
pub fn classify_bytes(data: &[u8]) -> Result<Vec<KernargClassification>, String> {
    let stream = NpuInstructionStream::parse(data)?;
    Ok(classify_kernargs(&stream))
}

fn classify_inner(
    stream: &NpuInstructionStream,
    topology: Option<&StreamSwitchTopology>,
) -> Vec<KernargClassification> {
    // Collect per-BD BD-word-2 via BlockWrites, and per-arg_idx DdrPatch
    // records. Then fuse them.
    let mut bd_word_2: HashMap<u32, u32> = HashMap::new();
    // For tier 2: per (arg_idx, bd_base_full) list of arg_plus values,
    // in order of appearance.
    let mut patch_log: BTreeMap<(u8, u32), Vec<u32>> = BTreeMap::new();
    // For direction inference: per arg_idx, remember first-seen bd_base_full
    // and the corresponding BD id within its tile, to resolve via a
    // subsequent Task Queue write.
    let mut pending: Vec<PendingPatch> = Vec::new();
    // First-seen role per arg_idx. First-seen wins except Tier 2
    // reclassification (see below).
    let mut roles: BTreeMap<u8, KernargClassification> = BTreeMap::new();

    for instr in stream.instructions() {
        match instr {
            NpuInstruction::BlockWrite { reg_off, values } => {
                if let Some(bd_base_full) = shim_bd_base(*reg_off) {
                    let word_at_reg_off = ((*reg_off & TILE_LOCAL_MASK)
                        - (bd_base_full & TILE_LOCAL_MASK))
                        / 4;
                    if let Some(w2_idx) = 2u32.checked_sub(word_at_reg_off) {
                        if let Some(&w2) = values.get(w2_idx as usize) {
                            bd_word_2.insert(bd_base_full, w2);
                        }
                    }
                }
            }
            NpuInstruction::Write32 { reg_off, value } => {
                if let Some(dir) = task_queue_direction(*reg_off) {
                    let bd_id = *value & TASK_QUEUE_BD_ID_MASK;
                    let tile_prefix = *reg_off & !TILE_LOCAL_MASK;
                    pending.retain(|p| {
                        if p.tile_prefix == tile_prefix && p.bd_id == bd_id {
                            roles.entry(p.arg_idx).or_insert(KernargClassification {
                                arg_idx: p.arg_idx,
                                role: dir,
                                bd_reg_addr: p.bd_src_reg,
                            });
                            false
                        } else {
                            true
                        }
                    });
                }
            }
            NpuInstruction::DdrPatch {
                reg_addr,
                arg_idx,
                arg_plus,
            } => {
                let bd_base_full = reg_addr.saturating_sub(4);
                // Tier 1: BD EnPkt.
                if let Some(&w2) = bd_word_2.get(&bd_base_full) {
                    if w2 & BD_WORD_2_ENPKT_BIT != 0 {
                        roles.entry(*arg_idx).or_insert(KernargClassification {
                            arg_idx: *arg_idx,
                            role: KernargRole::Ctrlpkt,
                            bd_reg_addr: *reg_addr,
                        });
                        patch_log
                            .entry((*arg_idx, bd_base_full))
                            .or_default()
                            .push(*arg_plus);
                        continue;
                    }
                }
                patch_log
                    .entry((*arg_idx, bd_base_full))
                    .or_default()
                    .push(*arg_plus);
                let local = bd_base_full & TILE_LOCAL_MASK;
                let bd_id = (local - SHIM_BD_BASE_LOCAL) / SHIM_BD_STRIDE;
                pending.push(PendingPatch {
                    arg_idx: *arg_idx,
                    bd_src_reg: *reg_addr,
                    tile_prefix: reg_addr & !TILE_LOCAL_MASK,
                    bd_id,
                });
            }
            _ => {}
        }
    }

    // Tier 2 upgrade: any arg with ≥ CTRLPKT_MIN_PATCHES monotonic small-stride
    // patches on the same BD is reclassified as Ctrlpkt, overriding whatever
    // direction was inferred. This catches ctrl_packet_reconfig where the BD
    // has EnPkt=0 but the packet framing is done in the stream switch.
    for ((arg_idx, bd_base_full), patches) in &patch_log {
        if looks_like_ctrlpkt_stream(patches) {
            roles.insert(
                *arg_idx,
                KernargClassification {
                    arg_idx: *arg_idx,
                    role: KernargRole::Ctrlpkt,
                    bd_reg_addr: bd_base_full + 4, // the source-lo word
                },
            );
        }
    }

    // Any pending patches whose direction was never resolved: Unknown.
    for p in pending {
        roles.entry(p.arg_idx).or_insert(KernargClassification {
            arg_idx: p.arg_idx,
            role: KernargRole::Unknown,
            bd_reg_addr: p.bd_src_reg,
        });
    }

    // Tier 3 cross-check (optional): warn if Ctrlpkt classifications
    // disagree with the CDO-reported packet routing. We don't override
    // -- the two signals measure different things (post-init state vs.
    // packet content) and disagreement is information, not a conclusion.
    if let Some(topology) = topology {
        let has_ctrlpkt_class =
            roles.values().any(|c| c.role == KernargRole::Ctrlpkt);
        let cdo_has_shim_packet = topology
            .packet_slaves()
            .any(|(addr, off, _)| is_shim_row(addr) && is_shim_south_slave(off));
        if has_ctrlpkt_class && !cdo_has_shim_packet {
            log::warn!(
                "classify: instruction stream signals ctrlpkt but CDO has no \
                 packet-enabled shim SOUTH slave -- possible false positive"
            );
        }
        if cdo_has_shim_packet && !has_ctrlpkt_class {
            log::warn!(
                "classify: CDO enables shim packet routing but no arg was \
                 classified as Ctrlpkt -- possible false negative"
            );
        }
    }

    roles.into_values().collect()
}

struct PendingPatch {
    arg_idx: u8,
    bd_src_reg: u32,
    tile_prefix: u32,
    bd_id: u32,
}

fn shim_bd_base(reg_off: u32) -> Option<u32> {
    let local = reg_off & TILE_LOCAL_MASK;
    if local >= SHIM_BD_BASE_LOCAL
        && local < SHIM_BD_BASE_LOCAL + SHIM_BD_COUNT * SHIM_BD_STRIDE
    {
        let bd_base_local =
            SHIM_BD_BASE_LOCAL + ((local - SHIM_BD_BASE_LOCAL) / SHIM_BD_STRIDE) * SHIM_BD_STRIDE;
        Some((reg_off & !TILE_LOCAL_MASK) | bd_base_local)
    } else {
        None
    }
}

fn task_queue_direction(reg_off: u32) -> Option<KernargRole> {
    match reg_off & TILE_LOCAL_MASK {
        SHIM_S2MM_0_TASK_QUEUE_LOCAL | SHIM_S2MM_1_TASK_QUEUE_LOCAL => Some(KernargRole::DataS2mm),
        SHIM_MM2S_0_TASK_QUEUE_LOCAL | SHIM_MM2S_1_TASK_QUEUE_LOCAL => Some(KernargRole::DataMm2s),
        _ => None,
    }
}

fn is_shim_row(addr: TileAddr) -> bool {
    addr.row == 0
}

/// Returns true if `offset` is a shim SS SLAVE_SOUTH_n config register
/// in the range that can carry shim DMA MM2S data. Reference:
/// aie-rt xaiemlgbl_params.h (SLAVE_CONFIG_SOUTH_0..7 at 0x3F108..0x3F124).
fn is_shim_south_slave(offset: u32) -> bool {
    let local = offset & TILE_LOCAL_MASK;
    (0x3F108..=0x3F124).contains(&local)
}

/// Decide whether a list of `arg_plus` values on one BD fits the
/// ctrlpkt-streaming fingerprint: ≥ CTRLPKT_MIN_PATCHES entries,
/// monotonically non-decreasing, with at least one gap between entries
/// (so we don't accept a BD rewritten identically multiple times).
fn looks_like_ctrlpkt_stream(patches: &[u32]) -> bool {
    if patches.len() < CTRLPKT_MIN_PATCHES {
        return false;
    }
    let mut prev = patches[0];
    let mut any_gap = false;
    for &p in &patches[1..] {
        if p < prev {
            return false;
        }
        if p > prev {
            any_gap = true;
        }
        prev = p;
    }
    any_gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::ops::DeviceOp;

    fn bd_words(pkt_en: bool) -> Vec<u32> {
        let mut w = vec![0u32; 8];
        if pkt_en {
            w[2] = BD_WORD_2_ENPKT_BIT;
        }
        w
    }

    fn build_stream(instrs: Vec<NpuInstruction>) -> NpuInstructionStream {
        NpuInstructionStream::from_instructions(instrs)
    }

    #[test]
    fn tier1_bd_enpkt_detects_ctrlpkt() {
        let stream = build_stream(vec![
            NpuInstruction::BlockWrite {
                reg_off: 0x0001_D000,
                values: bd_words(true),
            },
            NpuInstruction::DdrPatch {
                reg_addr: 0x0001_D004,
                arg_idx: 2,
                arg_plus: 0,
            },
        ]);
        let roles = classify_kernargs(&stream);
        assert_eq!(roles[0].role, KernargRole::Ctrlpkt);
    }

    #[test]
    fn tier2_pattern_detects_ctrlpkt_when_bd_enpkt_is_zero() {
        // Mimic ctrl_packet_reconfig: BD word 2 = 0, but 4 monotonic
        // patches at stride 24.
        let mut instrs = Vec::new();
        for plus in [0u32, 24, 48, 72] {
            instrs.push(NpuInstruction::BlockWrite {
                reg_off: 0x0001_D000,
                values: bd_words(false),
            });
            instrs.push(NpuInstruction::DdrPatch {
                reg_addr: 0x0001_D004,
                arg_idx: 2,
                arg_plus: plus,
            });
            instrs.push(NpuInstruction::Write32 {
                reg_off: SHIM_MM2S_0_TASK_QUEUE_LOCAL,
                value: 0,
            });
        }
        let roles = classify_kernargs(&build_stream(instrs));
        assert_eq!(roles.len(), 1);
        assert_eq!(roles[0].role, KernargRole::Ctrlpkt);
    }

    #[test]
    fn tier2_does_not_trigger_on_single_patch() {
        let stream = build_stream(vec![
            NpuInstruction::BlockWrite {
                reg_off: 0x0001_D000,
                values: bd_words(false),
            },
            NpuInstruction::DdrPatch {
                reg_addr: 0x0001_D004,
                arg_idx: 0,
                arg_plus: 0,
            },
            NpuInstruction::Write32 {
                reg_off: SHIM_MM2S_0_TASK_QUEUE_LOCAL,
                value: 0,
            },
        ]);
        let roles = classify_kernargs(&stream);
        assert_eq!(roles[0].role, KernargRole::DataMm2s);
    }

    #[test]
    fn tier2_preempts_direction_classification_for_reused_channel() {
        // ctrl_packet_reconfig: same BD, same channel, first 3 ctrlpkt
        // patches then 1 data patch. Both args go through MM2S_0.
        let mut instrs = Vec::new();
        for plus in [0u32, 24, 48] {
            instrs.push(NpuInstruction::BlockWrite {
                reg_off: 0x0001_D000,
                values: bd_words(false),
            });
            instrs.push(NpuInstruction::DdrPatch {
                reg_addr: 0x0001_D004,
                arg_idx: 2,
                arg_plus: plus,
            });
            instrs.push(NpuInstruction::Write32 {
                reg_off: SHIM_MM2S_0_TASK_QUEUE_LOCAL,
                value: 0,
            });
        }
        // Now the data transfer on the same BD with a different arg.
        instrs.push(NpuInstruction::BlockWrite {
            reg_off: 0x0001_D000,
            values: bd_words(false),
        });
        instrs.push(NpuInstruction::DdrPatch {
            reg_addr: 0x0001_D004,
            arg_idx: 0,
            arg_plus: 0,
        });
        instrs.push(NpuInstruction::Write32 {
            reg_off: SHIM_MM2S_0_TASK_QUEUE_LOCAL,
            value: 0,
        });
        let roles = classify_kernargs(&build_stream(instrs));
        let map: HashMap<u8, KernargRole> =
            roles.iter().map(|r| (r.arg_idx, r.role)).collect();
        assert_eq!(map[&2], KernargRole::Ctrlpkt);
        assert_eq!(map[&0], KernargRole::DataMm2s);
    }

    #[test]
    fn data_s2mm_from_task_queue_offset() {
        let stream = build_stream(vec![
            NpuInstruction::BlockWrite {
                reg_off: 0x0001_D020,
                values: bd_words(false),
            },
            NpuInstruction::DdrPatch {
                reg_addr: 0x0001_D024,
                arg_idx: 1,
                arg_plus: 0,
            },
            NpuInstruction::Write32 {
                reg_off: SHIM_S2MM_0_TASK_QUEUE_LOCAL,
                value: 1,
            },
        ]);
        let roles = classify_kernargs(&stream);
        assert_eq!(roles[0].role, KernargRole::DataS2mm);
    }

    #[test]
    fn tier3_cross_check_runs_without_panicking() {
        // Build a minimal topology with a packet-enabled shim SOUTH_3
        // slave, plus a stream whose arg is Ctrlpkt-classified via
        // tier 1. Cross-check should agree (no warning).
        let stream = build_stream(vec![
            NpuInstruction::BlockWrite {
                reg_off: 0x0001_D000,
                values: bd_words(true),
            },
            NpuInstruction::DdrPatch {
                reg_addr: 0x0001_D004,
                arg_idx: 2,
                arg_plus: 0,
            },
        ]);
        let topo = StreamSwitchTopology::from_device_ops(vec![DeviceOp::RegWrite {
            tile: TileAddr::new(0, 0),
            offset: 0x0003_F114, // SLAVE_SOUTH_3
            value: 0xC000_0000,
        }]);
        let roles = classify_with_topology(&stream, &topo);
        assert_eq!(roles[0].role, KernargRole::Ctrlpkt);
    }

    #[test]
    fn multi_column_tile_prefix_preserved() {
        let stream = build_stream(vec![
            NpuInstruction::BlockWrite {
                reg_off: 0x0021_D000,
                values: bd_words(true),
            },
            NpuInstruction::DdrPatch {
                reg_addr: 0x0021_D004,
                arg_idx: 5,
                arg_plus: 0,
            },
        ]);
        let roles = classify_kernargs(&stream);
        assert_eq!(roles[0].bd_reg_addr, 0x0021_D004);
    }

    #[test]
    fn ddr_patch_without_prior_bd_config_falls_to_unknown() {
        let stream = build_stream(vec![NpuInstruction::DdrPatch {
            reg_addr: 0x0001_D004,
            arg_idx: 7,
            arg_plus: 0,
        }]);
        let roles = classify_kernargs(&stream);
        assert_eq!(roles[0].role, KernargRole::Unknown);
    }
}
